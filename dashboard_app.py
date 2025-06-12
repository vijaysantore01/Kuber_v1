import streamlit as st  # st MUST be imported first for set_page_config
import time
import pandas_ta as ta
import pandas as pd
import logging
import sys
import os
from typing import Union
from pandas.api.types import is_scalar

# --- Streamlit UI Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(layout="wide", page_title="NIFTY Bot Dashboard")

# --- Laxmi Kuber Mantra (Added at the very top) ---
st.markdown(
    "<h1 style='text-align: center; color: #DAA520; font-family: \"Georgia\", serif; text-shadow: 1px 1px 2px rgba(0,0,0,0.2);'>ॐ श्रीं ह्रीं क्लीं श्रीं क्लीं वित्तेश्वराय नमः॥</h1>",
    unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: #555555; font-size: 14px; margin-top: -10px;'>Om Shreem Hreem Kleem Shreem Kleem Vitteshvaraya Namah॥</p>",
    unsafe_allow_html=True)
st.markdown("---")  # Add a separator below the mantra

# --- Initial Logging Setup ---
if 'logging_setup_complete' not in st.session_state:
    try:
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)  # Set to logging.DEBUG in Kuber_smart_12_purense_1.py for detailed bot logs
        for handler in root_logger.handlers[:]:
            if isinstance(handler, logging.StreamHandler):
                root_logger.removeHandler(handler)
        st.session_state.logging_setup_complete = True
    except Exception as e:
        print(f"WARNING: Could not configure Streamlit logging handlers due to: {e}", file=sys.stderr)

# Import necessary functions/variables from your main bot script
try:
    # Set dummy API keys for SmartConnect if not already set, for import purposes.
    if "SMARTAPI_API_KEY" not in os.environ:
        os.environ["SMARTAPI_API_KEY"] = "DUMMY_KEY_FOR_IMPORT"
    if "ANGEL_USER_ID" not in os.environ:
        os.environ["ANGEL_USER_ID"] = "DUMMY"
    if "ANGEL_PASSWORD" not in os.environ:
        os.environ["ANGEL_PASSWORD"] = "DUMMY"
    if "ANGEL_TOTP_SEED" not in os.environ:
        os.environ["ANGEL_TOTP_SEED"] = "DUMMYDUMMYDUMMYDUMMY"

    if not os.path.exists("upstox_v2_token.txt"):
        with open("upstox_v2_token.txt", "w") as f:
            f.write("DUMMY_UPSTOX_TOKEN")

    from Kuber_nifty_version1 import (
        get_spot_price, get_nearest_expiry, get_oi_data, filter_strikes_near_spot,
        calculate_option_skew, calculate_pcr, calculate_max_pain, get_live_premium,
        final_suggestion_extended, get_dashboard_data,
        active_trade, real_signal_history,
        FIXED_ATR_LOW_THRESHOLD, SYMBOL, ATR_PERCENTILE,
        is_market_open_ist,
        generate_final_suggestion,
        fetch_historical_daily_candles,
        fetch_intraday_data_and_indicators,
        detect_divergence
    )

    # Initialize st.session_state variables robustly
    if 'active_trade' not in st.session_state:
        st.session_state.active_trade = None
    if 'real_signal_history' not in st.session_state:
        st.session_state.real_signal_history = []

    NIFTY_SPOT_UPSTOX_KEY_FOR_APP = None
    UPS_FRONT_KEY_FOR_APP = None

except ImportError as e:
    st.error(
        f"Error importing bot logic: {e}. Please ensure 'Kuber_smart_12_purense_1.py' is in the same directory and all its dependencies are installed (e.g., pandas_ta, yfinance, nsepython, twilio).")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred during initial Streamlit setup: {e}")
    st.stop()

# --- Streamlit UI Controls ---
REFRESH_INTERVAL_SECONDS = st.sidebar.slider("Auto-Refresh Interval (seconds)", 5, 60, 15)


# Function to run the bot logic and update the dashboard
def update_dashboard():
    try:
        logging.info("[update_dashboard] Started dashboard update.")

        # Initialize ALL dashboard-relevant variables to their defaults
        pcr = 0.0
        skew_val = 0.0
        ce_ltp = 0.0
        pe_ltp = 0.0
        mp = 0
        atm_strike = 0

        atr_val = 0.0
        ma_15m = 0.0
        adaptive_atr_threshold = FIXED_ATR_LOW_THRESHOLD
        trend_direction = "Neutral"
        trend_filter_ma_50 = 0.0
        main_rsi_val = 0.0
        main_macd_vals = (0.0, 0.0, 0.0)
        main_stoch_vals = (0.0, 0.0)

        daily_ma_val = 0.0
        prev_close = 0.0
        today_high = 0.0
        today_low = 0.0

        volume_strength_score = 0
        divergence_score = 0

        # Individual score components, initialized to 0
        oi_score_raw_calc = 0
        price_score_raw_calc = 0
        momentum_score_raw_calc = 0
        additional_score_raw_calc = 0

        logging.info("[update_dashboard] Checking market open...")
        if not is_market_open_ist():
            st.info("Market is currently closed. Dashboard will update when market opens or on refresh.")
            logging.info("[update_dashboard] Market closed, waiting to rerun.")

        logging.info("[update_dashboard] Fetching spot price...")
        spot = get_spot_price(SYMBOL)
        if spot == 0.0:
            st.warning("Could not fetch spot price for dashboard. Retrying...")
            logging.warning("[update_dashboard] Could not fetch spot price, rerunning.")
            time.sleep(REFRESH_INTERVAL_SECONDS)
            st.rerun()
            return

        logging.info("[update_dashboard] Fetching expiry date...")
        expiry = get_nearest_expiry(SYMBOL)
        if not expiry:
            st.warning("Could not fetch expiry date for dashboard. Retrying...")
            logging.warning("[update_dashboard] Could not fetch expiry, rerunning.")
            time.sleep(REFRESH_INTERVAL_SECONDS)
            st.rerun()
            return

        logging.info("[update_dashboard] Fetching OI data...")
        df_oi = get_oi_data(SYMBOL, expiry)
        df_filt, atm_strike = filter_strikes_near_spot(df_oi, spot)
        logging.info(f"[update_dashboard] Filtered OI DataFrame shape: {df_filt.shape}, ATM Strike: {atm_strike}")

        if df_filt.empty:
            st.warning("Filtered OI DataFrame is empty for dashboard. Displaying defaults.")
            logging.warning("[update_dashboard] Filtered OI DataFrame is empty.")
        else:
            logging.info("[update_dashboard] Calculating OI metrics...")
            skew_val, ce_ltp, pe_ltp, _ = calculate_option_skew(df_filt, spot)
            logging.info(f"[update_dashboard] Skew: {skew_val}, CE_LTP: {ce_ltp}, PE_LTP: {pe_ltp}")
            pcr = calculate_pcr(df_filt)
            logging.info(f"[update_dashboard] PCR: {pcr}")
            mp, _ = calculate_max_pain(df_filt)
            logging.info(f"[update_dashboard] Max Pain: {mp}")

        # Daily MA Trend Filter (from yfinance daily candles)
        daily_bars_symbol = "^NSEI"
        logging.info("[update_dashboard] Fetching daily bars for MA calculation...")
        df_daily_bars = fetch_historical_daily_candles(daily_bars_symbol, days_back=90)
        logging.info(f"[update_dashboard] daily bars shape: {df_daily_bars.shape}")

        if not df_daily_bars.empty and len(df_daily_bars) >= 50:
            try:
                logging.info("[update_dashboard] Attempting to extract 'close' column for daily MA...")
                close_col = None
                if 'close' in df_daily_bars.columns:
                    close_col = df_daily_bars['close']
                else:
                    for col in df_daily_bars.columns:
                        if isinstance(col, tuple) and 'close' in col:
                            close_col = df_daily_bars[col]
                            break

                if close_col is None:
                    raise KeyError("No 'close' column found in daily bars dataframe.")

                daily_ma_val = close_col.iloc[-50:].mean().item()
                logging.info(f"[update_dashboard] Calculated daily_ma_val: {daily_ma_val}")

            except Exception as e:
                logging.error(f"[update_dashboard] Error computing daily_ma_val: {e}", exc_info=True)
                daily_ma_val = 0.0
        else:
            logging.warning(
                f"Insufficient daily bars for 50-day MA calculation for {daily_bars_symbol}. Daily MA will be 0.0.")
            daily_ma_val = 0.0

        # Calculate prev_close, today_high, today_low from df_daily_bars
        if not df_daily_bars.empty:
            if len(df_daily_bars) >= 2:
                prev_close = df_daily_bars['close'].iloc[-2].item()
            else:
                prev_close = 0.0
            today_high = spot
            today_low = spot
        else:
            prev_close = 0.0
            today_high = 0.0
            today_low = 0.0

        # Fetch intraday data and calculate technical indicators
        logging.info("[update_dashboard] Fetching intraday data and calculating technical indicators...")
        intraday_indicators = fetch_intraday_data_and_indicators(SYMBOL)

        # Assign calculated indicator values from the new helper function
        atr_val = intraday_indicators['atr']
        ma_15m = intraday_indicators['ma_15m']
        trend_filter_ma_50 = intraday_indicators['ma_50_15m']
        trend_direction = intraday_indicators['trend_direction']
        main_rsi_val = intraday_indicators['rsi']
        main_macd_vals = (
        intraday_indicators['macd_line'], intraday_indicators['macd_signal'], intraday_indicators['macd_hist'])
        main_stoch_vals = (intraday_indicators['stoch_k'], intraday_indicators['stoch_d'])

        # Calculate Divergence Score
        divergence_score = detect_divergence(intraday_indicators['closes'], intraday_indicators['rsi_history'])

        # Implement volume_strength_score calculation here if you have a function for it
        # volume_strength_score = calculate_volume_strength(intraday_indicators['volumes']) # Example

        # Generate final suggestion and get individual score components
        suggestion, final_score_overall, oi_score_raw_calc, price_score_raw_calc, momentum_score_raw_calc, additional_score_raw_calc = generate_final_suggestion(
            pcr, skew_val, spot, mp, atr_val, ma_15m, adaptive_atr_threshold,
            main_rsi_val, main_macd_vals, main_stoch_vals,
            daily_ma_val, volume_strength_score, divergence_score
        )

        # Get dashboard data using the consolidated function
        logging.info("[update_dashboard] Calling get_dashboard_data...")
        dashboard_data = get_dashboard_data(
            df_filt, spot, st.session_state.real_signal_history,
            atm=atm_strike, ce_ltp=ce_ltp, pe_ltp=pe_ltp,
            ups_front_key=UPS_FRONT_KEY_FOR_APP,
            adaptive_atr_threshold=FIXED_ATR_LOW_THRESHOLD,
            atr_val=atr_val, ma_15m=ma_15m, trend_filter_ma_50=trend_filter_ma_50,
            trend_direction=trend_direction, vol_bias=volume_strength_score,
            prev_close=prev_close,
            rsi_val=main_rsi_val, macd_vals=main_macd_vals, stoch_vals=main_stoch_vals,
            daily_ma_val=daily_ma_val, volume_strength_score=volume_strength_score,
            divergence_score=divergence_score,
            # Pass the calculated raw scores to get_dashboard_data
            oi_score_raw=oi_score_raw_calc,
            price_score_raw=price_score_raw_calc,
            momentum_score_raw=momentum_score_raw_calc,
            additional_score_raw=additional_score_raw_calc,
            active_trade_status=st.session_state.active_trade
        )
        logging.info(f"[update_dashboard] dashboard_data: {dashboard_data}")

        # --- Streamlit UI Layout (Single Page View) ---

        # Row 1: Spot Price, Final Suggestion, Overall Confidence
        col_main_spot, col_main_suggestion, col_main_confidence = st.columns([1, 2, 1])
        with col_main_spot:
            st.metric(label="Current Spot Price", value=f"₹{dashboard_data['spot_price']:.2f}")

        with col_main_suggestion:
            suggestion_color = 'green' if 'BUY CALL' in dashboard_data['final_suggestion'] else \
                'red' if 'BUY PUT' in dashboard_data['final_suggestion'] else 'orange'
            suggestion_icon = "📈" if 'BUY CALL' in dashboard_data['final_suggestion'] else \
                "📉" if 'BUY PUT' in dashboard_data['final_suggestion'] else "⚖️"
            st.markdown(
                f"<div style='background-color:#F0F2F6; padding: 10px; border-radius: 5px; text-align: center; font-family: \"Verdana\", Arial, sans-serif;'>"
                f"<h3 style='color:{suggestion_color}; font-size:28px; margin-bottom: 0px;'>{suggestion_icon} {dashboard_data['final_suggestion']}</h3>"
                f"<p style='color:grey; font-size:15px; margin-top: 5px;'>Overall Market Sentiment: <b>{dashboard_data['market_sentiment']}</b></p>"
                f"</div>", unsafe_allow_html=True
            )

        with col_main_confidence:
            st.metric(label="Overall Confidence", value=f"{dashboard_data['trade_confidence']}%")

        st.markdown("---")  # Separator

        # Row 2: Active Trade Status & PnL
        col_trade_status, col_pnl_display = st.columns([2, 1])
        if dashboard_data['active_trade']:
            current_premium = get_live_premium(df_filt, dashboard_data['active_trade']['atm'],
                                               dashboard_data['active_trade']['signal'])
            pnl_val = 0.0
            pnl_delta_str = ""
            entry_premium = dashboard_data['active_trade']['entry_premium']
            if not is_scalar(entry_premium):
                try:
                    entry_premium = entry_premium.item()
                except Exception:
                    entry_premium = None

            if (
                    current_premium is not None and
                    entry_premium is not None and
                    entry_premium != 0
            ):
                pnl_val = current_premium - entry_premium
                pnl_delta_str = f"{pnl_val / entry_premium:.2%}"

            with col_trade_status:
                st.markdown(
                    f"**Active Trade:** <span style='font-size:18px;'>{dashboard_data['active_trade']['signal']} @ {dashboard_data['active_trade']['atm']}</span> (Entry: ₹{entry_premium:.2f})",
                    unsafe_allow_html=True
                )
            with col_pnl_display:
                pnl_color = "green" if pnl_val >= 0 else "red"
                st.markdown(
                    f"<div style='text-align: right;'>"
                    f"<p style='font-size:14px; color:grey; margin-bottom: -5px;'>Live PnL</p>"
                    f"<h3 style='color:{pnl_color}; margin-top: 0px;'>₹{pnl_val:.2f} ({pnl_delta_str})</h3>"
                    f"</div>", unsafe_allow_html=True
                )
        else:
            with col_trade_status:
                st.markdown("<h3>Active Trade: <span style='color:orange;'>None</span></h3>", unsafe_allow_html=True)
            with col_pnl_display:
                st.empty()  # Clear PnL metrics if no active trade

        st.markdown("---")  # Separator

        # Row 3: Score Breakdown (Interactive)
        st.subheader("Decision Score Breakdown")
        col_oi_score, col_price_score, col_momentum_score, col_additional_score = st.columns(4)

        def get_score_color(score):
            if score > 0: return "green"
            if score < 0: return "red"
            return "orange"

        with col_oi_score:
            score_color = get_score_color(dashboard_data['oi_score_raw'])
            st.markdown(
                f"**OI Score:** <span style='color:{score_color}; font-size:18px;'>{dashboard_data['oi_score_raw']}</span>",
                unsafe_allow_html=True)
            st.markdown(f"<small><i>PCR, Skew, Max Pain</i></small>", unsafe_allow_html=True)

        with col_price_score:
            score_color = get_score_color(dashboard_data['price_score_raw'])
            st.markdown(
                f"**Price Score:** <span style='color:{score_color}; font-size:18px;'>{dashboard_data['price_score_raw']}</span>",
                unsafe_allow_html=True)
            st.markdown(f"<small><i>15m MA Comparison</i></small>", unsafe_allow_html=True)

        with col_momentum_score:
            score_color = get_score_color(dashboard_data['momentum_score_raw'])
            st.markdown(
                f"**Momentum Score:** <span style='color:{score_color}; font-size:18px;'>{dashboard_data['momentum_score_raw']}</span>",
                unsafe_allow_html=True)
            st.markdown(f"<small><i>RSI, MACD, Stoch</i></small>", unsafe_allow_html=True)

        with col_additional_score:
            score_color = get_score_color(dashboard_data['additional_score_raw'])
            st.markdown(
                f"**Additional Score:** <span style='color:{score_color}; font-size:18px;'>{dashboard_data['additional_score_raw']}</span>",
                unsafe_allow_html=True)
            st.markdown(f"<small><i>Daily MA, Vol, Div</i></small>", unsafe_allow_html=True)

        st.markdown("---")  # Separator

        # Row 4: Detailed Metrics Tables (in expanders for cleanliness)
        with st.expander("📊 Detailed Market Metrics"):
            col_detail_oi, col_detail_price, col_detail_indicators = st.columns(3)

            with col_detail_oi:
                oi_data_display = {
                    "Metric": ["PCR", "Skew", "Max Pain", "OI Bias Label", "Bullish Strikes", "Bearish Strikes",
                               "CE LTP", "PE LTP"],
                    "Value": [
                        f"{dashboard_data['pcr']:.2f}",
                        f"{dashboard_data['skew']:.2f}",
                        f"{dashboard_data['max_pain']}",
                        dashboard_data['oi_bias_label'],
                        f"{dashboard_data['oi_bull_strikes']}",
                        f"{dashboard_data['oi_bear_strikes']}",
                        f"₹{dashboard_data['ce_ltp']:.2f}",
                        f"₹{dashboard_data['pe_ltp']:.2f}"
                    ]
                }
                st.subheader("Open Interest")
                st.table(pd.DataFrame(oi_data_display))

            with col_detail_price:
                price_data_display = {
                    "Metric": ["ATR (5m)", "ATR Threshold", "15m MA", "50-period 5m MA", "Trend Direction", "Daily MA",
                               "Vol Strength", "Prev Close"],
                    "Value": [
                        f"{dashboard_data['atr']:.2f}",
                        f"{dashboard_data['atr_threshold']:.2f}",
                        f"{dashboard_data['ma_15m']:.2f}",
                        f"{dashboard_data['ma_50_15m']:.2f}",
                        dashboard_data['trend_direction'],
                        f"{dashboard_data['daily_ma']:.2f}",
                        f"{dashboard_data['vol_strength_score']}",
                        f"₹{dashboard_data['prev_close']:.2f}"
                    ]
                }
                st.subheader("Price Metrics")
                st.table(pd.DataFrame(price_data_display))

            with col_detail_indicators:
                indicator_data_display = {
                    "Metric": ["RSI", "MACD Line", "MACD Signal", "MACD Hist", "Stochastic %K", "Stochastic %D",
                               "Divergence Score"],
                    "Value": [
                        f"{dashboard_data['rsi']:.2f}",
                        f"{dashboard_data['macd_line']:.2f}",
                        f"{dashboard_data['macd_signal']:.2f}",
                        f"{dashboard_data['macd_hist']:.2f}",
                        f"{dashboard_data['stoch_k']:.2f}",
                        f"{dashboard_data['stoch_d']:.2f}",
                        f"{dashboard_data['divergence_score']}"
                    ]
                }
                st.subheader("Technical Indicators")
                st.table(pd.DataFrame(indicator_data_display))

        st.markdown("---")  # Separator

        # Row 5: OI Chart
        st.subheader("Option Chain OI Visuals")
        if not df_filt.empty:
            chart_data = df_filt[['Strike', 'CE_OI', 'PE_OI']].set_index('Strike')
            st.bar_chart(chart_data)
        else:
            st.info("No filtered OI data to display chart.")

        st.markdown("---")  # Separator

        # Row 6: Trade History
        st.subheader("Trade History")
        if st.session_state.real_signal_history:
            history_df = pd.DataFrame(st.session_state.real_signal_history)
            st.dataframe(history_df)
        else:
            st.info("No trade history available yet.")

        logging.info("[update_dashboard] Dashboard update finished successfully.")

    except Exception as e:
        st.error(f"An unexpected error occurred during dashboard update: {e}")
        logging.error(f"[update_dashboard] Exception: {e}", exc_info=True)
        # Clear UI elements on error to prevent displaying stale/incorrect data
        st.empty()  # Clears all dynamic content on the main page
        st.stop()  # Stop further execution to prevent infinite loops on error

    finally:
        time.sleep(REFRESH_INTERVAL_SECONDS)  # Use the user-controlled refresh interval
        st.rerun()


# Call the update function to run the dashboard
update_dashboard()