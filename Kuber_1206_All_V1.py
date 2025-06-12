import os, sys, time, math
import datetime as dt
import zoneinfo
import pandas as pd
import logging
import yfinance as yf
import json  # Keeping if you use it elsewhere
from twilio.rest import Client
from typing import Union
from nsepython import nsefetch, nse_optionchain_scrapper, index_history

# --- GLOBAL CONFIGURATION AND INITIALIZATION ---
logging.basicConfig(level=logging.INFO,  # Set to logging.DEBUG for very detailed logs
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("nifty_bot.log"),
                        logging.StreamHandler(sys.stdout)
                    ])

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "<YOUR_TWILIO_SID>")
logging.info(f"[INIT] TWILIO_ACCOUNT_SID: {TWILIO_ACCOUNT_SID}")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "<YOUR_TWILIO_TOKEN>")
logging.info(f"[INIT] TWILIO_AUTH_TOKEN: {TWILIO_AUTH_TOKEN}")
if TWILIO_ACCOUNT_SID != "<YOUR_TWILIO_SID>" and TWILIO_AUTH_TOKEN != "<YOUR_TWILIO_TOKEN>":
    twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    logging.info("[INIT] Twilio client initialized")
else:
    twilio_client = None
    logging.warning("[INIT] Twilio credentials not fully set. SMS alerts will be disabled.")

SYMBOL = "NIFTY"
logging.info(f"[INIT] SYMBOL set to {SYMBOL}")
FIXED_ATR_LOW_THRESHOLD = 25.0
ATR_PERCENTILE = 25
TARGET_PCT = 0.20
SL_PCT = 0.20

active_trade = None
real_signal_history = []
UPS_FRONT_KEY = None  # Not used with NSEPython
NIFTY_SPOT_UPSTOX_KEY = None  # Not used with NSEPython

STATE_FILE = "nifty_bot_state.json"
logging.info(f"[INIT] STATE_FILE: {STATE_FILE}")


# --- HELPER FUNCTIONS ---

def send_sms_alert(message: str):
    logging.info(f"[send_sms_alert] Called with message: {message}")
    if twilio_client:
        try:
            message_obj = twilio_client.messages.create(
                to=os.getenv("ALERT_PHONE_NUMBER", "+1234567890"),  # Replace with actual phone number
                from_=os.getenv("TWILIO_PHONE_NUMBER", "+0987654321"),  # Replace with Twilio phone number
                body=message
            )
            logging.info(f"[send_sms_alert] SMS alert sent: {message_obj.sid}")
        except Exception as e:
            logging.error(f"[send_sms_alert] Failed to send SMS alert: {e}", exc_info=True)
    else:
        logging.info(f"[send_sms_alert] Twilio client not initialized. Would have sent SMS: {message}")


def load_state():
    global active_trade, real_signal_history
    logging.info(f"[load_state] Called")
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
                logging.info(f"[load_state] State file loaded: {state}")
                active_trade = state.get('active_trade')
                real_signal_history = state.get('real_signal_history', [])
            logging.info("[load_state] Bot state loaded successfully.")
        except json.JSONDecodeError as e:
            logging.error(f"[load_state] Error decoding state file {STATE_FILE}: {e}. Starting fresh.", exc_info=True)
            active_trade = None
            real_signal_history = []
        except Exception as e:
            logging.error(f"[load_state] Error loading state file {STATE_FILE}: {e}. Starting fresh.", exc_info=True)
            active_trade = None
            real_signal_history = []
    else:
        logging.info("[load_state] No existing state file found. Starting fresh.")
        active_trade = None
        real_signal_history = []


def save_state():
    logging.info("[save_state] Called")
    try:
        state = {
            'active_trade': active_trade,
            'real_signal_history': real_signal_history
        }
        logging.info(f"[save_state] Saving state: {state}")
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=4)
        logging.info("[save_state] Bot state saved successfully.")
    except Exception as e:
        logging.error(f"[save_state] Error saving state file {STATE_FILE}: {e}", exc_info=True)


def generate_final_suggestion(
        pcr: float, skew_val: float, spot: float, mp: int, atr_val: float, ma_15m: float, ATR_LOW_THRESHOLD: float,
        rsi_val: float, macd_vals: tuple[float, float, float], stoch_vals: tuple[float, float],
        daily_ma_val: float, volume_strength_score: int, divergence_score: int
) -> tuple[str, int, int, int]:
    logging.info(f"[generate_final_suggestion] Called with pcr={pcr}, skew_val={skew_val}, spot={spot}, mp={mp}, "
                 f"atr_val={atr_val}, ma_15m={ma_15m}, ATR_LOW_THRESHOLD={ATR_LOW_THRESHOLD}, "
                 f"rsi_val={rsi_val}, macd_vals={macd_vals}, stoch_vals={stoch_vals}, "
                 f"daily_ma_val={daily_ma_val}, volume_strength_score={volume_strength_score}, "
                 f"divergence_score={divergence_score}")

    oi_score = 0
    if pcr < 0.90:
        oi_score -= 1
        logging.info("[generate_final_suggestion] oi_score -- PCR < 0.90")
    elif pcr > 1.10:
        oi_score += 1
        logging.info("[generate_final_suggestion] oi_score ++ PCR > 1.10")
    if skew_val < -3:
        oi_score -= 1
        logging.info("[generate_final_suggestion] oi_score -- Skew < -3")
    elif skew_val > +3:
        oi_score += 1
        logging.info("[generate_final_suggestion] oi_score ++ Skew > 3")
    if spot > mp + 20:
        oi_score -= 1
        logging.info("[generate_final_suggestion] oi_score -- spot > mp+20")
    elif spot < mp - 20:
        oi_score += 1
        logging.info("[generate_final_suggestion] oi_score ++ spot < mp-20")
    logging.info(f"[generate_final_suggestion] oi_score: {oi_score}")

    price_score = 0
    # Note: ATR filter is applied later for final score calculation
    if ma_15m != 0.0:  # Only score if 15m MA was calculated
        if spot < ma_15m:
            price_score -= 1  # Bearish if spot below 15m MA
            logging.info("[generate_final_suggestion] price_score -- spot < ma_15m")
        elif spot > ma_15m:
            price_score += 1  # Bullish if spot above 15m MA
            logging.info("[generate_final_suggestion] price_score ++ spot > ma_15m")
    logging.info(f"[generate_final_suggestion] price_score: {price_score}")

    # --- Momentum Indicator Scoring ---
    momentum_score = 0
    # Only score if RSI is valid
    if rsi_val != 0.0:
        if rsi_val > 60:
            momentum_score -= 1
            logging.info("[generate_final_suggestion] momentum_score -- rsi_val > 60")
        elif rsi_val < 40:
            momentum_score += 1
            logging.info("[generate_final_suggestion] momentum_score ++ rsi_val < 40")

    macd_line, signal_line, _ = macd_vals
    # Only score if MACD values are valid (not all zeros)
    if macd_line != 0.0 or signal_line != 0.0:
        if macd_line > signal_line:
            momentum_score += 1
            logging.info("[generate_final_suggestion] momentum_score ++ macd_line > signal_line")
        elif macd_line < signal_line:
            momentum_score -= 1
            logging.info("[generate_final_suggestion] momentum_score -- macd_line < signal_line")

    stoch_k, stoch_d = stoch_vals
    # Only score if Stochastic values are valid (not all zeros)
    if stoch_k != 0.0 or stoch_d != 0.0:
        if stoch_k > stoch_d:
            momentum_score += 1
            logging.info("[generate_final_suggestion] momentum_score ++ stoch_k > stoch_d")
        elif stoch_k < stoch_d:
            momentum_score -= 1
            logging.info("[generate_final_suggestion] momentum_score -- stoch_k < stoch_d")
    logging.info(f"[generate_final_suggestion] momentum_score: {momentum_score}")

    # --- Additional Scoring Components ---
    additional_score = 0
    if daily_ma_val != 0.0:  # Only score if daily MA was successfully calculated
        if spot > daily_ma_val:
            additional_score += 1
            logging.info("[generate_final_suggestion] additional_score ++ spot > daily_ma_val")
        elif spot < daily_ma_val:
            additional_score -= 1
            logging.info("[generate_final_suggestion] additional_score -- spot < daily_ma_val")

    # These remain 0 unless actual logic is implemented
    additional_score += volume_strength_score
    logging.info(f"[generate_final_suggestion] additional_score += volume_strength_score: {volume_strength_score}")
    additional_score += divergence_score
    logging.info(f"[generate_final_suggestion] additional_score += divergence_score: {divergence_score}")
    logging.info(f"[generate_final_suggestion] additional_score: {additional_score}")

    # Final score calculation based on ATR threshold
    if atr_val > ATR_LOW_THRESHOLD:  # Only apply ATR filter if ATR is meaningful (not 0.0) and above threshold
        final_score = oi_score + price_score + momentum_score + additional_score
        logging.info("[generate_final_suggestion] Using ATR logic, final_score = %d", final_score)
    else:  # If ATR is not meaningful or below threshold, do not use it as a filter (still sum scores)
        final_score = oi_score + price_score + momentum_score + additional_score
        logging.info("[generate_final_suggestion] Not using ATR filter (low/zero ATR), final_score = %d", final_score)

    if final_score <= -2:
        suggestion = "BUY PUT"
        logging.info("[generate_final_suggestion] suggestion = BUY PUT")
    elif final_score >= 2:
        suggestion = "BUY CALL"
        logging.info("[generate_final_suggestion] suggestion = BUY CALL")
    else:
        suggestion = "Stay Neutral"
        logging.info("[generate_final_suggestion] suggestion = Stay Neutral")

    logging.info(
        f"[generate_final_suggestion] Return: suggestion={suggestion}, final_score={final_score}, oi_score={oi_score}, price_score={price_score}, momentum_score={momentum_score}, additional_score={additional_score}")
    return suggestion, final_score, oi_score, price_score, momentum_score, additional_score


def colorize(text: str, sentiment: str) -> str:
    logging.info(f"[colorize] text: {text}, sentiment: {sentiment}")
    s = sentiment.lower()
    if "bullish" in s or "📈" in s:
        colored = f"\033[92m{text}\033[0m"
    elif "bearish" in s or "📉" in s:
        colored = f"\033[91m{text}\033[0m"
    else:
        colored = f"\033[93m{text}\033[0m"
    logging.info(f"[colorize] colored result: {colored}")
    return colored


def clear_screen():
    logging.info("[clear_screen] Called")
    if os.name == 'nt':
        os.system('cls')
    else:
        os.system('clear')


def get_nearest_expiry(sym: str) -> Union[str, None]:
    logging.info(f"[get_nearest_expiry] Called with sym: {sym}")
    try:
        oc = nse_optionchain_scrapper(sym)
        logging.debug(f"[get_nearest_expiry] option chain fetched: {type(oc)}")
        if oc and "records" in oc and "expiryDates" in oc["records"] and oc["records"]["expiryDates"]:
            expiry = oc["records"]["expiryDates"][0]
            logging.info(f"[get_nearest_expiry] Expiry found: {expiry}")
            return expiry
        logging.info("[get_nearest_expiry] Expiry not found, returning None.")
        return None
    except Exception as e:
        logging.error(f"[get_nearest_expiry] Error: {e}", exc_info=True)
        return None


def get_spot_price(sym: str):
    logging.info(f"[get_spot_price] Called with sym: {sym}")
    try:
        data = nsefetch("https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050")
        logging.debug(f"[get_spot_price] nsefetch result: {data}")
        if data and 'data' in data and data['data']:
            price = float(data['data'][0]['lastPrice'])
            logging.info(f"[get_spot_price] Price from nsefetch: {price}")
            return price
        logging.warning("[get_spot_price] nsefetch did not return valid spot price. Trying fallback.")
    except Exception as e:
        logging.warning(f"[get_spot_price] nsefetch error: {e}. Trying fallback.", exc_info=True)
    try:
        oc = nse_optionchain_scrapper(sym)
        logging.debug(f"[get_spot_price] Fallback option chain: {type(oc)}")
        if oc and "records" in oc and "underlyingValue" in oc["records"]:
            price = float(oc["records"]["underlyingValue"])
            logging.info(f"[get_spot_price] Price from fallback: {price}")
            return price
        logging.error("[get_spot_price] All fallbacks failed. Returning 0.0.")
        return 0.0
    except Exception as e2:
        logging.error(f"[get_spot_price] fallback error: {e2}. Returning 0.0.", exc_info=True)
        return 0.0


def get_oi_data(sym: str, expiry: str) -> pd.DataFrame:
    logging.info(f"[get_oi_data] sym: {sym}, expiry: {expiry}")
    try:
        oc = nse_optionchain_scrapper(sym)
        logging.debug(f"[get_oi_data] option chain received: {type(oc)}")
    except Exception as e:
        logging.error(f"[get_oi_data] Failed to fetch option chain: {e}", exc_info=True)
        return pd.DataFrame()
    if not oc or "records" not in oc or "data" not in oc["records"]:
        logging.warning("[get_oi_data] OC data malformed or empty.")
        return pd.DataFrame()
    rows = [r for r in oc["records"]["data"] if r.get("expiryDate") == expiry]
    logging.info(f"[get_oi_data] Filtered rows count: {len(rows)}")
    if not rows:
        logging.info(f"[get_oi_data] No option chain data for expiry {expiry}.")
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    logging.info(f"[get_oi_data] DataFrame shape after load: {df.shape}")
    for leg in ("CE", "PE"):
        df[f"{leg}_OI"] = df[leg].apply(lambda x: x.get("openInterest", 0) if isinstance(x, dict) else 0)
        logging.debug(f"[get_oi_data] {leg}_OI created")
        df[f"{leg}_ChgOI"] = df[leg].apply(lambda x: x.get("changeinOpenInterest", 0) if isinstance(x, dict) else 0)
        logging.debug(f"[get_oi_data] {leg}_ChgOI created")
        df[f"{leg}_LTP"] = df[leg].apply(lambda x: x.get("lastPrice", 0.0) if isinstance(x, dict) else 0.0)
        logging.debug(f"[get_oi_data] {leg}_LTP created")
    df = df.rename(columns={"strikePrice": "Strike"})
    logging.debug(f"[get_oi_data] Columns renamed: {df.columns.tolist()}")
    df = df[["Strike", "CE_OI", "PE_OI", "CE_ChgOI", "PE_ChgOI", "CE_LTP", "PE_LTP"]]
    logging.debug(f"[get_oi_data] Columns filtered: {df.columns.tolist()}")
    df["Bias"] = df.apply(
        lambda r: "Bullish" if r.PE_ChgOI > r.CE_ChgOI else ("Bearish" if r.CE_ChgOI > r.PE_ChgOI else "Neutral"),
        axis=1)
    logging.debug(f"[get_oi_data] Bias column applied")
    df = df.sort_values("Strike").reset_index(drop=True)
    logging.debug(f"[get_oi_data] DataFrame sorted. Shape: {df.shape}")
    return df


def fetch_historical_daily_candles(symbol_name: str = "^NSEI", days_back: int = 90) -> pd.DataFrame:
    logging.info(f"[fetch_historical_daily_candles] Called with {symbol_name}, days_back={days_back}")
    end_dt = dt.datetime.now()
    start_dt = end_dt - dt.timedelta(days=days_back)
    logging.debug(f"[fetch_historical_daily_candles] start_dt={start_dt}, end_dt={end_dt}")

    try:
        yf_symbol = symbol_name if symbol_name.startswith("^") else "^NSEI"
        logging.debug(f"[fetch_historical_daily_candles] Using yf_symbol: {yf_symbol}")
        df = yf.download(
            yf_symbol,
            start=start_dt.strftime("%Y-%m-%d"),
            end=end_dt.strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=False
        )
        logging.debug(f"[fetch_historical_daily_candles] yfinance DataFrame shape: {None if df is None else df.shape}")
        if df is None or df.empty:
            logging.warning("[fetch_historical_daily_candles] No data returned from yfinance")
            return pd.DataFrame()
        df = df.reset_index()
        logging.debug(f"[fetch_historical_daily_candles] After reset_index: {df.head(2)}")
        df = df.rename(columns={
            "Date": "time", "Open": "open", "High": "high",
            "Low": "low", "Close": "close", "Volume": "volume"
        })
        logging.debug(f"[fetch_historical_daily_candles] Columns renamed: {df.columns.tolist()}")
        df = df[["time", "open", "high", "low", "close", "volume"]]
        logging.debug(f"[fetch_historical_daily_candles] Columns filtered: {df.columns.tolist()}")
        df["time"] = pd.to_datetime(df["time"])
        logging.debug(f"[fetch_historical_daily_candles] Datetime converted: {df.dtypes}")
        df = df.sort_values("time").reset_index(drop=True)
        logging.debug(f"[fetch_historical_daily_candles] DataFrame sorted. Shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"[fetch_historical_daily_candles] Exception: {e}", exc_info=True)
        return pd.DataFrame()


# --- NEW/CORRECTED HELPER FUNCTION FOR INTRADAY DATA AND INDICATORS ---
def fetch_intraday_data_and_indicators(
        symbol_name: str = "^NSEI",
        period: str = "7d",  # Period for yfinance, e.g., "7d" for 7 days
        interval: str = "5m"  # Interval for yfinance, e.g., "5m" for 5 minutes
) -> dict:
    """
    Fetches intraday data and calculates technical indicators using pandas_ta.
    Returns a dictionary of calculated indicator values.
    Handles potential yfinance data issues by returning default zero values.
    """
    logging.info(
        f"[fetch_intraday_data_and_indicators] Fetching {interval} data for {symbol_name} for period {period}.")
    try:
        yf_symbol = symbol_name if symbol_name.startswith("^") else "^NSEI"

        df = yf.download(
            yf_symbol,
            period=period,
            interval=interval,
            auto_adjust=False,
            prepost=False
        )

        if df is None or df.empty:
            logging.warning(
                f"[fetch_intraday_data_and_indicators] No {interval} data returned from yfinance for {yf_symbol}. Returning default indicators.")
            return {
                "atr": 0.0, "ma_15m": 0.0, "ma_50_15m": 0.0,
                "rsi": 0.0, "macd_line": 0.0, "macd_signal": 0.0, "macd_hist": 0.0,
                "stoch_k": 0.0, "stoch_d": 0.0, "trend_direction": "Neutral",
                "closes": [], "highs": [], "lows": [],
                "rsi_history": []  # Ensure this is also defaulted
            }

        # --- Robust Column Handling for yfinance Output ---
        logging.debug(
            f"[fetch_intraday_data_and_indicators] Raw df.columns before any flattening/rename: {df.columns.tolist()}")

        # 1. If DataFrame has a MultiIndex column, flatten it using col[0]
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in
                          df.columns]
            logging.debug(
                f"[fetch_intraday_data_and_indicators] Columns flattened from MultiIndex. Intermediate columns: {df.columns.tolist()}")

        # 2. Reset index to make 'Date' or 'Datetime' a regular column, named 'time'
        df = df.reset_index(names=['time'])

        # 3. Rename standard yfinance columns to a consistent lowercase format
        df = df.rename(columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "Adj Close": "adj close"
        })
        logging.debug(
            f"[fetch_intraday_data_and_indicators] Columns renamed to standard lowercase. Processed columns: {df.columns.tolist()}")

        # 4. Ensure 'close' column exists; use 'adj close' as a fallback if 'close' is missing
        if 'adj close' in df.columns and 'close' not in df.columns:
            df['close'] = df['adj close']

        # 5. Filter for only the required OHLCV and time columns, creating a fresh copy
        required_cols_for_ta = ['time', 'open', 'high', 'low', 'close', 'volume']

        if not all(col in df.columns for col in required_cols_for_ta):
            logging.error(
                f"[fetch_intraday_data_and_indicators] Critical: Missing required OHLCV columns after all transformations: {required_cols_for_ta}. Actual columns: {df.columns.tolist()}. Returning default indicators.")
            return {
                "atr": 0.0, "ma_15m": 0.0, "ma_50_15m": 0.0, "rsi": 0.0, "macd_line": 0.0,
                "macd_signal": 0.0, "macd_hist": 0.0, "stoch_k": 0.0, "stoch_d": 0.0,
                "trend_direction": "Neutral", "closes": [], "highs": [], "lows": [],
                "rsi_history": []  # Ensure this is also defaulted
            }

        df_processed = df[required_cols_for_ta].copy()

        # Convert 'time' column to datetime and set as index for pandas_ta
        df_processed['time'] = pd.to_datetime(df_processed['time'])
        df_processed = df_processed.set_index('time').sort_index()

        # --- Debugging before TA ---
        logging.debug(f"[fetch_intraday_data_and_indicators] df_processed.shape BEFORE TA: {df_processed.shape}")
        if df_processed.empty:
            logging.warning(
                "[fetch_intraday_data_and_indicators] df_processed is empty before TA calculations. Returning default indicators.")
            return {
                "atr": 0.0, "ma_15m": 0.0, "ma_50_15m": 0.0, "rsi": 0.0, "macd_line": 0.0,
                "macd_signal": 0.0, "macd_hist": 0.0, "stoch_k": 0.0, "stoch_d": 0.0,
                "trend_direction": "Neutral", "closes": [], "highs": [], "lows": [],
                "rsi_history": []  # Ensure this is also defaulted
            }

        # Ensure enough data points for calculations
        min_bars_for_atr = 14
        min_bars_for_sma50 = 50

        # Calculate ATR
        if len(df_processed) >= min_bars_for_atr:
            df_processed.ta.atr(append=True)  # This will create 'ATRr_14'
            logging.debug(
                f"[fetch_intraday_data_and_indicators] After ATR calculation, Count of NaNs in ATRr_14: {df_processed['ATRr_14'].isnull().sum()}")
        else:
            df_processed['ATRr_14'] = pd.NA  # Assign pandas Not Available type
            logging.warning(
                f"[fetch_intraday_data_and_indicators] Not enough data ({len(df_processed)} bars) for ATR calculation (min {min_bars_for_atr} needed). ATRr_14 will be NaN.")

        # Calculate SMAs
        if len(df_processed) >= min_bars_for_sma50:
            df_processed.ta.sma(close='close', length=3, append=True)  # SMA_3
            df_processed.ta.sma(close='close', length=50, append=True)  # SMA_50
        else:
            df_processed['SMA_3'] = pd.NA
            df_processed['SMA_50'] = pd.NA
            logging.warning(
                f"[fetch_intraday_data_and_indicators] Not enough data ({len(df_processed)} bars) for SMA_50 calculation (min {min_bars_for_sma50} needed). SMAs will be NaN.")

        # Calculate RSI, MACD, Stochastic
        df_processed.ta.rsi(append=True)  # RSI_14
        df_processed.ta.macd(append=True)  # MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
        df_processed.ta.stoch(append=True)  # STOCHk_14_3_3, STOCHd_14_3_3

        # --- Debugging before final dropna ---
        logging.debug(
            f"[fetch_intraday_data_and_indicators] df_processed.shape BEFORE final dropna: {df_processed.shape}")
        if 'ATRr_14' in df_processed.columns:
            logging.debug(
                f"[fetch_intraday_data_and_indicators] df_processed['ATRr_14'] head before final dropna:\n{df_processed['ATRr_14'].head().to_string()}")
            logging.debug(
                f"[fetch_intraday_data_and_indicators] df_processed['ATRr_14'] tail before final dropna:\n{df_processed['ATRr_14'].tail().to_string()}")
            logging.debug(
                f"[fetch_intraday_data_and_indicators] Count of NaNs in ATRr_14 before final dropna: {df_processed['ATRr_14'].isnull().sum()}")

        # Drop any rows with NaN values that result from indicator calculations
        df_processed = df_processed.dropna()

        # --- Debugging after final dropna ---
        logging.debug(
            f"[fetch_intraday_data_and_indicators] df_processed.shape AFTER final dropna: {df_processed.shape}")
        if not df_processed.empty:
            logging.debug(
                f"[fetch_intraday_data_and_indicators] df_processed.tail() AFTER final dropna:\n{df_processed.tail().to_string()}")
            if 'ATRr_14' in df_processed.columns and not df_processed['ATRr_14'].empty:
                logging.debug(
                    f"[fetch_intraday_data_and_indicators] ATRr_14 value AFTER dropna: {df_processed['ATRr_14'].iloc[-1].item()}")

        if df_processed.empty:
            logging.warning(
                f"[fetch_intraday_data_and_indicators] DataFrame became empty after dropping NaNs for {yf_symbol}. Returning default indicators.")
            return {
                "atr": 0.0, "ma_15m": 0.0, "ma_50_15m": 0.0,
                "rsi": 0.0, "macd_line": 0.0, "macd_signal": 0.0, "macd_hist": 0.0,
                "stoch_k": 0.0, "stoch_d": 0.0, "trend_direction": "Neutral",
                "closes": [], "highs": [], "lows": [],
                "rsi_history": []
            }

        # Determine trend direction from 50-period MA
        trend_direction = "Neutral"
        if 'SMA_50' in df_processed.columns and not df_processed['SMA_50'].empty and pd.notna(
                df_processed['SMA_50'].iloc[-1]):
            if df_processed['close'].iloc[-1] > df_processed['SMA_50'].iloc[-1]:
                trend_direction = "Bullish"
            elif df_processed['close'].iloc[-1] < df_processed['SMA_50'].iloc[-1]:
                trend_direction = "Bearish"
        else:
            logging.warning(
                "[fetch_intraday_data_and_indicators] 'SMA_50' column not found or empty for trend direction. Remains Neutral.")

        # Extract latest values (ensure they are scalars using .item() and defensive checks)
        latest_values = {
            "atr": df_processed['ATRr_14'].iloc[-1].item() if 'ATRr_14' in df_processed.columns and not df_processed[
                'ATRr_14'].empty and pd.notna(df_processed['ATRr_14'].iloc[-1]) else 0.0,
            "ma_15m": df_processed['SMA_3'].iloc[-1].item() if 'SMA_3' in df_processed.columns and not df_processed[
                'SMA_3'].empty and pd.notna(df_processed['SMA_3'].iloc[-1]) else 0.0,
            "ma_50_15m": df_processed['SMA_50'].iloc[-1].item() if 'SMA_50' in df_processed.columns and not
            df_processed['SMA_50'].empty and pd.notna(df_processed['SMA_50'].iloc[-1]) else 0.0,
            "rsi": df_processed['RSI_14'].iloc[-1].item() if 'RSI_14' in df_processed.columns and not df_processed[
                'RSI_14'].empty and pd.notna(df_processed['RSI_14'].iloc[-1]) else 0.0,
            "macd_line": df_processed['MACD_12_26_9'].iloc[-1].item() if 'MACD_12_26_9' in df_processed.columns and not
            df_processed['MACD_12_26_9'].empty and pd.notna(df_processed['MACD_12_26_9'].iloc[-1]) else 0.0,
            "macd_signal": df_processed['MACDs_12_26_9'].iloc[
                -1].item() if 'MACDs_12_26_9' in df_processed.columns and not df_processed[
                'MACDs_12_26_9'].empty and pd.notna(df_processed['MACDs_12_26_9'].iloc[-1]) else 0.0,
            "macd_hist": df_processed['MACDh_12_26_9'].iloc[
                -1].item() if 'MACDh_12_26_9' in df_processed.columns and not df_processed[
                'MACDh_12_26_9'].empty and pd.notna(df_processed['MACDh_12_26_9'].iloc[-1]) else 0.0,
            "stoch_k": df_processed['STOCHk_14_3_3'].iloc[-1].item() if 'STOCHk_14_3_3' in df_processed.columns and not
            df_processed['STOCHk_14_3_3'].empty and pd.notna(df_processed['STOCHk_14_3_3'].iloc[-1]) else 0.0,
            "stoch_d": df_processed['STOCHd_14_3_3'].iloc[-1].item() if 'STOCHd_14_3_3' in df_processed.columns and not
            df_processed['STOCHd_14_3_3'].empty and pd.notna(df_processed['STOCHd_14_3_3'].iloc[-1]) else 0.0,
            "trend_direction": trend_direction,
            "closes": df_processed['close'].to_list(),
            "highs": df_processed['high'].to_list(),
            "lows": df_processed['low'].to_list(),
            "rsi_history": df_processed['RSI_14'].to_list() if 'RSI_14' in df_processed.columns and not df_processed[
                'RSI_14'].empty else []  # Ensure this line is present
        }
        logging.info(f"[fetch_intraday_data_and_indicators] Successfully calculated indicators: {latest_values}")
        return latest_values

    except Exception as e:
        logging.error(f"[fetch_intraday_data_and_indicators] Error fetching or calculating indicators: {e}",
                      exc_info=True)
        return {
            "atr": 0.0, "ma_15m": 0.0, "ma_50_15m": 0.0,
            "rsi": 0.0, "macd_line": 0.0, "macd_signal": 0.0, "macd_hist": 0.0,
            "stoch_k": 0.0, "stoch_d": 0.0, "trend_direction": "Neutral",
            "closes": [], "highs": [], "lows": [],
            "rsi_history": []  # Ensure this is also defaulted in case of error
        }


def filter_strikes_near_spot(df: pd.DataFrame, spot: float, window: int = 5, step: int = 50) -> Union[
    tuple[pd.DataFrame, int], tuple[pd.DataFrame, int]]:
    logging.info(
        f"[filter_strikes_near_spot] Entry | df.shape: {df.shape}, spot: {spot}, window: {window}, step: {step}")
    if df.empty:
        logging.info("[filter_strikes_near_spot] Input DataFrame is empty. Returning empty and 0.")
        return df, 0
    atm_strike_candidates = df.Strike.values
    logging.info(f"[filter_strikes_near_spot] ATM strike candidates: {atm_strike_candidates}")
    if not atm_strike_candidates.size:
        logging.warning("[filter_strikes_near_spot] No strike prices found. Returning empty and 0.")
        return df, 0
    atm = int(round(spot / step) * step)
    logging.info(f"[filter_strikes_near_spot] Calculated ATM: {atm}")
    if atm not in atm_strike_candidates:
        logging.info(f"[filter_strikes_near_spot] ATM {atm} not in candidates, finding nearest.")
        atm = min(atm_strike_candidates, key=lambda x: abs(x - spot))
        logging.info(f"[filter_strikes_near_spot] Nearest ATM: {atm}")
    low, high = atm - window * step, atm + window * step
    logging.info(f"[filter_strikes_near_spot] Filtering between {low} and {high}")
    result_df = df[df.Strike.between(low, high)].copy()
    logging.info(f"[filter_strikes_near_spot] Result df.shape: {result_df.shape}")
    return result_df, int(atm)


def calculate_pcr(df: pd.DataFrame) -> float:
    logging.info(f"[calculate_pcr] Called. df.shape: {df.shape}")
    try:
        ce_sum = df.CE_OI.sum()
        pe_sum = df.PE_OI.sum()
        logging.info(f"[calculate_pcr] CE_OI.sum: {ce_sum}, PE_OI.sum: {pe_sum}")
        if ce_sum > 0:
            pcr = round(pe_sum / ce_sum, 2)
        else:
            pcr = 0.0
        logging.info(f"[calculate_pcr] PCR: {pcr}")
        return pcr
    except Exception as e:
        logging.error(f"[calculate_pcr] Exception: {e}", exc_info=True)
        return 0.0


def is_market_open_ist() -> bool:
    ist = zoneinfo.ZoneInfo("Asia/Kolkata")
    now = dt.datetime.now(ist)
    logging.info(f"[is_market_open_ist] Now IST: {now}")
    if now.weekday() >= 5:
        logging.info(f"[is_market_open_ist] Weekend")
        return False
    open_status = dt.time(9, 15) <= now.time() <= dt.time(15, 30)
    logging.info(f"[is_market_open_ist] Market open: {open_status}")
    return open_status


def calculate_option_skew(df: pd.DataFrame, spot: float) -> tuple[float, float, float, int]:
    logging.info(f"[calculate_option_skew] df.shape: {df.shape}, spot: {spot}")
    if df.empty:
        logging.info(f"[calculate_option_skew] DataFrame empty, returning zeros.")
        return 0.0, 0.0, 0.0, 0
    idx = (df.Strike - spot).abs().idxmin()
    logging.info(f"[calculate_option_skew] ATM idx: {idx}")
    r = df.loc[idx]
    ce_ltp = r.CE_LTP if 'CE_LTP' in r else 0.0
    pe_ltp = r.PE_LTP if 'PE_LTP' in r else 0.0
    strike = r.Strike if 'Strike' in r else 0
    skew_val = round(ce_ltp - pe_ltp, 2)
    logging.info(f"[calculate_option_skew] ce_ltp: {ce_ltp}, pe_ltp: {pe_ltp}, strike: {strike}, skew: {skew_val}")
    return skew_val, ce_ltp, pe_ltp, int(strike)


def calculate_max_pain(df: pd.DataFrame) -> tuple[int, float]:
    logging.info(f"[calculate_max_pain] df.shape: {df.shape}")
    if df.empty:
        logging.info("[calculate_max_pain] DataFrame empty")
        return 0, 0.0
    strikes = df.Strike.values
    logging.info(f"[calculate_max_pain] strikes: {strikes}")
    if not strikes.size:
        logging.info("[calculate_max_pain] strikes.size==0")
        return 0, 0.0
    try:
        losses = [((s - strikes).clip(0) * df.PE_OI).sum() + ((strikes - s).clip(0) * df.CE_OI).sum() for s in strikes]
        logging.info(f"[calculate_max_pain] losses array: {losses}")
        if not losses:
            return 0, 0.0
        mi = int(pd.Series(losses).idxmin())
        logging.info(f"[calculate_max_pain] Max pain index: {mi}")
        return int(strikes[mi]), losses[mi]
    except Exception as e:
        logging.error(f"[calculate_max_pain] Exception: {e}", exc_info=True)
        return 0, 0.0


def summarize_sentiment(df: pd.DataFrame) -> tuple[str, int, int]:
    logging.info(f"[summarize_sentiment] df.shape: {df.shape}")
    if df.empty:
        logging.info("[summarize_sentiment] DataFrame empty")
        return "⚖️ Neutral", 0, 0
    cnt = df.Bias.value_counts()
    b, r = cnt.get("Bullish", 0), cnt.get("Bearish", 0)
    lbl = "📈 Bullish" if b > r else ("📉 Bearish" if r > b else "⚖️ Neutral")
    logging.info(f"[summarize_sentiment] b: {b}, r: {r}, lbl: {lbl}")
    return lbl, b, r


def days_to_expiry(exp: str) -> int:
    for fmt in ("%Y-%m-%d", "%d-%b-%Y", "%Y-%m-%dT%H:%M:%S.000Z"):
        try:
            return max((dt.datetime.strptime(exp.split('T')[0], fmt).date() - dt.date.today()).days, 0)
        except ValueError:
            pass
    logging.warning(f"Could not parse expiry date format: {exp}")
    return 0


def theta_call_per_year(S: float, K: float, r: float, sig: float, T: float) -> float:
    if T <= 0 or sig <= 0 or S <= 0 or K <= 0: return 0.0
    try:
        d1 = (math.log(S / K) + (r + 0.5 * sig * sig) * T) / (sig * math.sqrt(T))
        npdf = math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi)
        return - (S * npdf * sig) / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * (
                0.5 * (1 + math.erf((d1 - sig * math.sqrt(T)) / math.sqrt(2))))
    except (ValueError, ZeroDivisionError) as e:
        logging.error(f"Error in theta_call_per_year calculation: {e}. S={S}, K={K}, r={r}, sig={sig}, T={T}")
        return 0.0


def breakdown_confidence_extended(
        pcr: float, skew: float, mp: int, spot: float, bcnt: int, rcnt: int,
        prev_close: float, today_high: float, today_low: float, ma_val: float,
        vol_bias: int, skip_price: bool, aggressive: bool = False
) -> tuple[int, list]:
    score, steps = 0, []
    if aggressive:
        if pcr > 1.05:
            score += 1;
            steps.append("PCR_Bullish_Aggressive")
        elif pcr < 0.95:
            score -= 1;
            steps.append("PCR_Bearish_Aggressive")
    else:
        if pcr > 1.10:
            score += 1;
            steps.append("PCR_Bullish_Conservative")
        elif pcr < 0.80:
            score -= 1;
            steps.append("PCR_Bearish_Conservative")
    if spot < mp:
        score += 1;
        steps.append("MaxPain_Bullish")
    elif spot > mp:
        score -= 1;
        steps.append("MaxPain_Bearish")
    if bcnt > rcnt:
        score += 1;
        steps.append("StrikeCount_Bullish")
    elif rcnt > bcnt:
        score -= 1;
        steps.append("StrikeCount_Bearish")
    oi_only = score
    if skip_price: return oi_only, steps
    if spot > prev_close:
        score += 1;
        steps.append("Price_PrevClose_Bullish")
    elif spot < prev_close:
        score -= 1;
        steps.append("Price_PrevClose_Bearish")
    if spot >= today_high and today_high != spot:
        score += 1;
        steps.append("Price_NewHigh_Bullish")
    elif spot <= today_low and today_low != spot:
        score -= 1;
        steps.append("Price_NewLow_Bearish")
    if spot > ma_val:
        score += 1;
        steps.append("Price_MA_Bullish")
    elif spot < ma_val:
        score -= 1;
        steps.append("Price_MA_Bearish")
    if vol_bias > 0:
        score += 1;
        steps.append("VolumeBias_Bullish")
    elif vol_bias < 0:
        score -= 1;
        steps.append("VolumeBias_Bearish")
    return score, steps


def final_suggestion_extended(score: int, skip_price: bool) -> tuple[str, int]:
    max_possible_score = 4 if skip_price else 7
    conf = int((abs(score) / max_possible_score) * 100) if max_possible_score > 0 else 0
    if score >= (2 if skip_price else 4):
        return "BUY CALL", conf
    elif score <= (-2 if skip_price else -4):
        return "BUY PUT", conf
    else:
        return "Stay Neutral", conf


def detect_divergence(closes: list, indicator_values: list, lookback_bars: int = 5) -> int:
    """
    Detects bullish or bearish divergence based on recent price and indicator trends.
    A very simplified approach checking the last `lookback_bars`.
    Returns:
        +1 for bullish divergence (price lower low, indicator higher low)
        -1 for bearish divergence (price higher high, indicator lower high)
        0 for no clear divergence
    Note: This will likely return 0 if market is closed or data is sparse/flat.
    """
    if len(closes) < lookback_bars or len(indicator_values) < lookback_bars:
        logging.debug(
            f"[detect_divergence] Not enough data for divergence: closes={len(closes)}, indicator_values={len(indicator_values)}, lookback_bars={lookback_bars}")
        return 0  # Not enough data for divergence

    # Get recent segments for comparison
    recent_closes = closes[-lookback_bars:]
    recent_indicator = indicator_values[-lookback_bars:]
    logging.debug(f"[detect_divergence] Recent closes: {recent_closes}")
    logging.debug(f"[detect_divergence] Recent indicator: {recent_indicator}")

    # Check for Bearish Divergence: Price making higher highs, Indicator making lower highs
    if recent_closes[-1] > recent_closes[0] and recent_indicator[-1] < recent_indicator[0]:
        price_trend_up = all(recent_closes[i] < recent_closes[i + 1] for i in range(lookback_bars - 1))
        indicator_trend_down = all(recent_indicator[i] > recent_indicator[i + 1] for i in range(lookback_bars - 1))
        if price_trend_up and indicator_trend_down:
            logging.debug("[detect_divergence] Bearish Divergence Detected: Price HH, Indicator LH")
            return -1  # Bearish Divergence

    # Check for Bullish Divergence: Price making lower lows, Indicator making higher lows
    if recent_closes[-1] < recent_closes[0] and recent_indicator[-1] > recent_indicator[0]:
        price_trend_down = all(recent_closes[i] > recent_closes[i + 1] for i in range(lookback_bars - 1))  # Corrected
        indicator_trend_up = all(
            recent_indicator[i] < recent_indicator[i + 1] for i in range(lookback_bars - 1))  # Corrected
        if price_trend_down and indicator_trend_up:
            logging.debug("[detect_divergence] Bullish Divergence Detected: Price LL, Indicator HL")
            return 1  # Bullish Divergence

    logging.debug("[detect_divergence] No clear divergence detected.")
    return 0


def get_live_premium(df: pd.DataFrame, atm_strike: int, signal_type: str) -> Union[float, None]:
    logging.info(
        f"[get_live_premium] Entry | df.shape: {df.shape}, atm_strike: {atm_strike}, signal_type: {signal_type}")
    if df.empty:
        logging.info("[get_live_premium] DataFrame is empty, returning None.")
        return None
    try:
        logging.info(f"[get_live_premium] Checking signal_type: {signal_type}")
        if signal_type == "BUY CALL":
            logging.info("[get_live_premium] Signal is BUY CALL.")
            val = df.loc[df["Strike"] == atm_strike, "CE_LTP"]
            logging.debug(f"[get_live_premium] CE_LTP Series for ATM: {val}")
        elif signal_type == "BUY PUT":
            logging.info("[get_live_premium] Signal is BUY PUT.")
            val = df.loc[df["Strike"] == atm_strike, "PE_LTP"]
            logging.debug(f"[get_live_premium] PE_LTP Series for ATM: {val}")
        else:
            logging.info(f"[get_live_premium] Unknown signal_type: {signal_type}. Returning None.")
            return None
        if val.empty:
            logging.info(f"[get_live_premium] No premium found for atm_strike={atm_strike}. Returning None.")
            return None
        premium_val = val.iloc[0]
        logging.info(f"[get_live_premium] Returning premium: {premium_val}")
        return premium_val
    except Exception as e:
        logging.error(f"[get_live_premium] Exception: {e}", exc_info=True)
        return None


def get_dashboard_data(
        df: pd.DataFrame, spot: float, real_history: list,
        atm: Union[int, None] = None, ce_ltp: Union[float, None] = None, pe_ltp: Union[float, None] = None,
        ups_front_key: Union[str, None] = None,
        adaptive_atr_threshold: float = FIXED_ATR_LOW_THRESHOLD,
        atr_val: float = 0.0, ma_15m: float = 0.0, trend_filter_ma_50: float = 0.0,
        trend_direction: str = "Neutral", vol_bias: int = 0, prev_close: float = 0.0,
        rsi_val: float = 0.0, macd_vals: tuple[float, float, float] = (0.0, 0.0, 0.0),
        stoch_vals: tuple[float, float] = (0.0, 0.0),
        daily_ma_val: float = 0.0, volume_strength_score: int = 0, divergence_score: int = 0,
        # NEW: Raw score components are now passed as parameters
        oi_score_raw: int = 0, price_score_raw: int = 0, momentum_score_raw: int = 0, additional_score_raw: int = 0,
        active_trade_status: dict = None
) -> dict:
    now_ist = dt.datetime.now(zoneinfo.ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S IST")
    expiry = get_nearest_expiry(SYMBOL) or "N/A"
    pcr = calculate_pcr(df)
    skew_val, current_ce_ltp, current_pe_ltp, current_atm = calculate_option_skew(df, spot)
    if atm is None: atm = current_atm
    if ce_ltp is None: ce_ltp = current_ce_ltp
    if pe_ltp is None: pe_ltp = current_pe_ltp
    mp, mp_loss = calculate_max_pain(df)
    label_oi, bc, rc = summarize_sentiment(df)

    # _skip_price_for_display logic (based on ATR being meaningful)
    _skip_price_for_display = (atr_val == 0.0)

    # generate_final_suggestion now returns individual scores
    suggestion, final_score_overall, oi_score, price_score_gen, momentum_score_gen, additional_score_gen = generate_final_suggestion(
        pcr, skew_val, spot, mp, atr_val, ma_15m, adaptive_atr_threshold,
        rsi_val, macd_vals, stoch_vals,
        daily_ma_val, volume_strength_score, divergence_score
    )

    trend_score = 0
    if trend_direction == "Bullish":
        trend_score = 1
    elif trend_direction == "Bearish":
        trend_score = -1

    final_score_with_trend = final_score_overall + trend_score

    sig_real, conf_real = final_suggestion_extended(final_score_with_trend, _skip_price_for_display)

    sentiment_market = "Bullish" if final_score_with_trend > 0 else (
        "Bearish" if final_score_with_trend < 0 else "Neutral")

    premium_at_signal = current_ce_ltp if sig_real == "BUY CALL" else (pe_ltp if sig_real == "BUY PUT" else 0.0)

    # RETURN DICTIONARY: Ensure all parameters are explicitly included as keys
    return {
        "timestamp": now_ist,
        "spot_price": spot,
        "expiry": expiry,
        "atr": atr_val,
        "atr_threshold": adaptive_atr_threshold,
        "ma_15m": ma_15m,
        "ma_50_15m": trend_filter_ma_50,
        "trend_direction": trend_direction,
        "daily_ma": daily_ma_val,
        "vol_strength_score": volume_strength_score,
        "divergence_score": divergence_score,
        "pcr": pcr,
        "skew": skew_val,
        "ce_ltp": ce_ltp,
        "pe_ltp": pe_ltp,
        "max_pain": mp,
        "max_pain_loss": mp_loss,
        "oi_bias_label": label_oi,
        "oi_bull_strikes": bc,
        "oi_bear_strikes": rc,
        "rsi": rsi_val,
        "macd_line": macd_vals[0],
        "macd_signal": macd_vals[1],
        "macd_hist": macd_vals[2],
        "stoch_k": stoch_vals[0],
        "stoch_d": stoch_vals[1],
        "trade_confidence": conf_real,
        "market_sentiment": sentiment_market,
        "final_suggestion": sig_real,
        "final_score": final_score_with_trend,
        # Raw score components from parameters
        "oi_score_raw": oi_score_raw,
        "price_score_raw": price_score_raw,
        "momentum_score_raw": momentum_score_raw,
        "additional_score_raw": additional_score_raw,
        "premium_at_signal": premium_at_signal,
        "active_trade": active_trade_status,
        "prev_close": prev_close
    }