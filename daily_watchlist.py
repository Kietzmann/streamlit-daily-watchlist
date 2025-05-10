# -*- coding: utf-8 -*-
import asyncio
import copy
import logging
import math
import multiprocessing
import os
import platform
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import ccxt.async_support as ccxt  # Keep for main process tasks like fetch_tickers
import numpy as np
import pandas as pd
import streamlit as st

# --- Import the updated worker function ---
# Ensure worker_tasks.py is in the same directory or Python path
from worker_tasks import process_symbol_batch

VOLUME_ALPHA = 0.3  # Increased for faster response (adjustable via sidebar if desired)
MIN_VOLUME_THRESHOLD = 1e-6  # Minimum volume to consider valid (prevents zero-volume skew)

# =============================================================================
# Configuration & Logging Setup (Main Process)
# =============================================================================
log_file_main = "watchlist_main.log"
file_handler_main = logging.FileHandler(log_file_main, mode="w")  # Overwrite log each run
stream_handler_main = logging.StreamHandler()

# Setup logging BEFORE getting the logger instance if not already configured
logging.basicConfig(
    level=logging.INFO,  # Change to logging.DEBUG for more detail
    format="%(asctime)s - %(levelname)s - [%(processName)s:%(threadName)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[file_handler_main, stream_handler_main],  # Log to file and console
    force=True,  # Use force=True in case basicConfig was called elsewhere (e.g., imported module)
)
logger = logging.getLogger(__name__)  # Now get the logger instance
logger.info("Watchlist main logger initialized.")

# Configure Streamlit page (handle potential rerun error)
try:
    st.set_page_config(
        page_title="Kraken USDT Dashboard - Short Term", page_icon="⏱️", layout="wide", initial_sidebar_state="expanded"
    )
    logger.info("Streamlit page configured.")
except st.errors.StreamlitAPIException as e:
    if "set_page_config() has already been called" in str(e):
        logger.debug("Streamlit set_page_config already called, ignoring.")
        pass  # Ignore if called multiple times during dev/reruns
    else:
        logger.error(f"Streamlit configuration error: {e}", exc_info=True)
        raise  # Raise other set_page_config errors

# --- Import only the ewmac function from ewmac.py ---
# Ensure ewmac.py is in the same directory or Python path
# DO NOT import robust_vol_calc if you don't want to use it
try:
    from ewmac import ewmac

    logger = logging.getLogger(__name__)  # Get logger after basicConfig potentially called below
    logger.info("Successfully imported 'ewmac' function from ewmac.py")
except ImportError as e:
    # Fallback or error handling if ewmac.py is missing
    logging.error("Could not import 'ewmac' from ewmac.py. MR Wings calculation will fail.", exc_info=True)
    st.error("Error: ewmac.py not found. Cannot calculate MR Wings. Please ensure the file exists.")

    # Define a dummy function to avoid NameErrors later, although calculations will be wrong
    def ewmac(price, vol, Lfast, Lslow):
        return pd.Series(np.zeros(len(price)), index=price.index) if isinstance(price, pd.Series) else pd.Series(dtype=float)



# =============================================================================
# Constants and Global Variables
# =============================================================================
CLIP_LOWER = -100
CLIP_UPPER = 100

# --- Supported Exchanges ---
SUPPORTED_EXCHANGES = {
    "kraken": "Kraken",
    "okx": "OKX",
    "binance": "Binance",
    "bybit": "Bybit",
    "coinbase": "Coinbase"
}

# --- Indicator Weights ---
CHANGE_1M_WEIGHT = 0.15
CHANGE_5M_WEIGHT = 0.30
CHANGE_15M_WEIGHT = 0.25  # Adjusted slightly to make room for MR Wings
BREAKOUT_15M_WEIGHT = 0.20  # Adjusted slightly
MR_WINGS_WEIGHT = 0.10  # Example weight for Mean Reversion (adjust as needed)
SHORT_TERM_VOL_ADJ_WEIGHT = 0.8

# --- Breakout Parameters ---
BREAKOUT_LOOKBACK = 24  # Periods (15m)
BREAKOUT_SMOOTH = 6  # Periods (15m)

# --- MR Wings Parameters ---
MR_LFAST = 4  # Lookback for mr_wings internal fast EWMA (15m periods)
# VOL_LOOKBACK_MR = 35 # Lookback for volatility (INFO ONLY: not used directly as we use existing 5m vol)

STABLECOINS = {  # Keep this list relevant
    "USDC",
    "DAI",
    "BUSD",
    "TUSD",
    "PAX",
    "GUSD",
    "USDK",
    "UST",
    "SUSD",
    "FRAX",
    "LUSD",
    "MIM",
    "USDQ",
    "TBTC",
    "WBTC",
    "EUL",
    "EUR",
    "EURT",
    "USDS",
    "USTS",
    "USTC",
    "USDR",
    "PYUSD",
    "EURR",
    "GBP",
    "AUD",
    "EURQ",
    "T",
    "USDG",
    "WAXL",
    "IDEX",
    "FIS",
    "CSM",
    "MV",
    "POWR",
    "ATLAS",
    "XCN",
    "BOBA",
    "OXY",
    "BNC",
    "POLIS",
    "AIR",
    "C98",
    "BODEN",
    "HDX",
    "MSOL",
    "REP",
    "ANLOG",
}
VOLUME_ALPHA = 0.2  # Smoothing factor for volume delta
# DEFAULT_NUM_CORES = 8  # Adjust based on your system
CPU_COUNT = multiprocessing.cpu_count()
DEFAULT_NUM_THREADS = min(32, CPU_COUNT * 4)
# DEFAULT_NUM_THREADS = 8
EXCHANGE_ID = "kraken"
CCXT_TIMEOUT = 30000  # 30 seconds for CCXT requests


# =============================================================================
# Indicator Calculation Functions
# =============================================================================

# NOTE: Removing robust_vol_calc, ewmac_forecast*, ewmac_calc_vol as they rely on robust_vol_calc
#       We keep ewmac (imported), mr_wings (defined below), breakout (defined below).


# --- MR Wings Definition (Uses imported ewmac) ---
def mr_wings(price, vol, Lfast=4):
    """
    Calculates a mean reversion signal based on EWMA divergence.
    Interprets lookbacks as number of periods.
    Requires APPROXIMATE price unit volatility for the 'vol' parameter if not using robust_vol_calc.

    :param price: The price series (consistent intervals)
    :type price: pd.Series
    :param vol: The period price unit volatility series (APPROXIMATED if needed)
    :type vol: pd.Series
    :param Lfast: Lookback span for fast EWMA in **periods**
    :type Lfast: int
    :returns: pd.Series -- Mean reversion forecast signal
    """
    func_logger = logging.getLogger(__name__ + ".mr_wings")  # Optional specific logger
    if not isinstance(price, pd.Series) or not isinstance(vol, pd.Series):
        func_logger.warning("mr_wings received invalid input types.")
        return pd.Series(np.zeros(len(price)), index=price.index) if isinstance(price, pd.Series) else pd.Series(dtype=float)
    if price.empty or vol.empty:
        func_logger.debug("mr_wings received empty price or vol series.")
        return pd.Series(np.zeros(len(price)), index=price.index) if isinstance(price, pd.Series) else pd.Series(dtype=float)

    Lslow = Lfast * 4  # Standard slow period multiple for mr_wings
    func_logger.debug(f"Calculating internal ewmac for mr_wings: Lfast={Lfast}, Lslow={Lslow}")

    # Use the imported ewmac function
    ewmac_signal = ewmac(price, vol, Lfast, Lslow)

    # Check if ewmac calculation was successful (ewmac returns zeros on error/bad input)
    if ewmac_signal.eq(0).all():
        func_logger.warning("Internal EWMA signal for mr_wings is all zeros. Cannot calculate standard deviation.")
        return pd.Series(np.zeros(len(price)), index=price.index)

    # Need sufficient non-zero points for meaningful std dev
    min_periods_std = max(3, int(Lslow / 2))
    non_zero_ewmac = ewmac_signal[ewmac_signal != 0]
    if len(non_zero_ewmac) < min_periods_std:
        func_logger.warning(
            f"Not enough non-zero EWMA signal points ({len(non_zero_ewmac)}) for mr_wings std dev (min={min_periods_std}). Returning zeros."
        )
        return pd.Series(np.zeros(len(price)), index=price.index)

    # Use expanding standard deviation for robustness
    ewmac_std = non_zero_ewmac.expanding(min_periods=min_periods_std).std()
    # Reindex to match original signal and fill gaps (especially at the start)
    ewmac_std = ewmac_std.reindex(ewmac_signal.index).fillna(method="ffill").fillna(1e-9)  # Fill NaNs, ensure non-zero

    func_logger.debug(f"Calculated ewmac_std. Tail:\n{ewmac_std.tail()}")

    # Define threshold (e.g., 3 standard deviations) - adjust multiplier as needed
    threshold = ewmac_std * 2.0
    mr_signal = ewmac_signal.copy()

    # Ensure alignment before comparison (should already be aligned, but good practice)
    mr_signal, threshold = mr_signal.align(threshold, join="left", fill_value=0)

    # Apply threshold: set signal to 0 if within threshold band
    mr_signal[mr_signal.abs() <= threshold.abs()] = 0.0  # Use abs() comparison for safety

    # Mean reversion signal is the opposite sign of the divergence
    mr_signal = -mr_signal

    func_logger.debug(f"Calculated final mr_signal. Tail:\n{mr_signal.tail()}")
    return mr_signal.fillna(0.0)  # Fill any remaining NaNs


# --- Breakout Function (Keep as is) ---
def breakout(price, lookback_periods=10, smooth_periods=None):
    """
    Calculates a breakout forecast signal. Interprets lookbacks as periods.
    (Function definition remains the same as provided in your code)
    """
    if not isinstance(price, pd.Series) or price.empty:
        return pd.Series(dtype=float)
    min_data_needed = max(2, int(lookback_periods / 2))
    if len(price) < min_data_needed:
        logger.debug(f"Breakout calc needs {min_data_needed} periods, got {len(price)}. Returning zeros.")
        return pd.Series([0.0] * len(price), index=price.index)
    if smooth_periods is None:
        smooth_periods = max(int(lookback_periods / 4.0), 1)
    if smooth_periods >= lookback_periods:
        smooth_periods = max(1, int(lookback_periods / 2))
        logger.debug(f"Breakout smooth_periods adjusted to {smooth_periods} as it was >= lookback_periods {lookback_periods}")
    min_periods_roll = max(1, int(np.ceil(lookback_periods / 2.0)))
    roll_max = price.rolling(lookback_periods, min_periods=min_periods_roll).max()
    roll_min = price.rolling(lookback_periods, min_periods=min_periods_roll).min()
    roll_range = (roll_max - roll_min).replace(0, np.nan).fillna(method="ffill").fillna(1e-9)
    roll_mean = (roll_max + roll_min) / 2.0
    output = 40.0 * ((price - roll_mean) / roll_range)
    min_periods_smooth = max(1, int(np.ceil(smooth_periods / 2.0)))
    smoothed_output = output.ewm(span=smooth_periods, min_periods=min_periods_smooth).mean()
    smoothed_output = smoothed_output.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return smoothed_output


# =============================================================================
# User Authentication Section (Keep as is)
# =============================================================================
# def load_credentials(usernames_file="usernames.txt", passwords_file="passwords.txt"):
#     # (Function definition remains the same as provided in your code)
#     credentials_dict = {}
#     try:
#         if not os.path.exists(usernames_file):
#             logger.warning(f"{usernames_file} not found. Creating empty file.")
#             open(usernames_file, "w").close()
#         if not os.path.exists(passwords_file):
#             logger.warning(f"{passwords_file} not found. Creating empty file.")
#             open(passwords_file, "w").close()
#         with open(usernames_file, "r") as f_users, open(passwords_file, "r") as f_pass:
#             usernames = [line.strip() for line in f_users if line.strip()]
#             passwords = [line.strip() for line in f_pass if line.strip()]
#         if not usernames or not passwords:
#             logger.warning("Credentials files are empty or only one is populated.")
#         elif len(usernames) != len(passwords):
#             logger.warning("Username and password file lengths differ. Using the minimum matching pairs.")
#             min_len = min(len(usernames), len(passwords))
#             credentials_dict = dict(zip(usernames[:min_len], passwords[:min_len]))
#         else:
#             credentials_dict = dict(zip(usernames, passwords))
#         if not credentials_dict:
#             logger.warning("No valid credentials loaded after processing files.")
#         else:
#             logger.info(f"Loaded {len(credentials_dict)} credential pairs.")
#         return credentials_dict
#     except FileNotFoundError:
#         logger.error("Credentials file not found. Check permissions.")
#         st.error("Credential file access error.")
#         return {}
#     except Exception as e:
#         st.error(f"Error loading credentials: {e}")
#         logger.error(f"Error loading credentials: {e}", exc_info=True)
#         return {}
#
#
# credentials = load_credentials()
# if "authenticated" not in st.session_state:
#     st.session_state["authenticated"] = False
# if "username" not in st.session_state:
#     st.session_state["username"] = None
#
# if not st.session_state.get("authenticated", False):
#     st.sidebar.title("Login")
#     username_input = st.sidebar.text_input("Username", key="login_username_input")
#     password_input = st.sidebar.text_input("Password", type="password", key="login_password_input")
#     login_button = st.sidebar.button("Login", key="login_submit_button")
#     if login_button:
#         entered_pw = credentials.get(username_input)
#         if entered_pw and entered_pw == password_input:
#             st.session_state["authenticated"] = True
#             st.session_state["username"] = username_input
#             logger.info(f"User '{username_input}' logged in successfully.")
#             st.rerun()
#         else:
#             st.sidebar.error("Invalid username or password.")
#             logger.warning(f"Failed login attempt for username: '{username_input}'")
#     if not st.session_state.get("authenticated", False):
#         st.warning("Please log in using the sidebar.")
#         st.stop()

# =============================================================================
# Main Application Logic
# =============================================================================
st.title(f"📈 Kraken USDT Dashboard (Short-Term Focus)")
st.write(f"Welcome, **{st.session_state.get('username', 'User')}**!")
st.markdown("---")
dashboard_placeholder = st.empty()


# =============================================================================
# Synchronous Helper Functions
# =============================================================================


# --- Keep calculate_change_n_bars ---
def calculate_change_n_bars(ohlcv, n_bars=1):
    """Calculates the percentage change between the close n bars ago and the latest close."""
    # (Function definition remains the same as provided in your code)
    if not isinstance(ohlcv, list) or len(ohlcv) < n_bars + 1:
        return 0.0
    try:
        old_close = ohlcv[-(n_bars + 1)][4]
        recent_close = ohlcv[-1][4]
    except (IndexError, TypeError) as e:
        return 0.0
    if old_close is None or recent_close is None or old_close == 0:
        return 0.0
    try:
        old_close = float(old_close)
        recent_close = float(recent_close)
        change = ((recent_close / old_close) - 1) * 100
        return change if np.isfinite(change) else 0.0
    except (ValueError, TypeError) as e:
        return 0.0


# --- Keep calculate_volatility_n_bars (This calculates % volatility, used for 5m vol) ---
def calculate_volatility_n_bars(ohlcv, n_bars=12, vol_type="range"):
    """Calculates PERCENTAGE volatility over the last n bars (using High-Low Range or StdDev)."""
    # (Function definition remains the same as provided in your code)
    if not isinstance(ohlcv, list) or len(ohlcv) < n_bars:
        return 0.0
    try:
        recent_bars = ohlcv[-n_bars:]
    except IndexError:
        return 0.0
    try:
        if vol_type == "range":
            highs = [float(b[2]) for b in recent_bars if b and len(b) > 2 and b[2] is not None]
            lows = [float(b[3]) for b in recent_bars if b and len(b) > 3 and b[3] is not None]
            closes = [float(b[4]) for b in recent_bars if b and len(b) > 4 and b[4] is not None]
            if not highs or not lows or not closes:
                return 0.0
            period_high = max(highs)
            period_low = min(lows)
            last_close = closes[-1]
            if last_close == 0:
                return 0.0
            volatility = abs(period_high - period_low) / last_close * 100
            return volatility if not np.isnan(volatility) else 0.0
        else:  # Default to standard deviation percentage
            closes = [float(b[4]) for b in recent_bars if b and len(b) > 4 and b[4] is not None]
            if len(closes) < 2:
                return 0.0
            std_dev = np.std(closes)
            mean_close = np.mean(closes)
            if mean_close == 0:
                return 0.0
            volatility = (std_dev / mean_close) * 100
            return volatility if not np.isnan(volatility) else 0.0
    except (ValueError, TypeError, IndexError) as e:
        logger.debug(f"Error calculating short-term volatility ({n_bars} bars, type={vol_type}): {e}")
        return 0.0


# --- Updated Volume Delta Functions ---
def update_volume_delta(symbol, current_volume, last_volume_dict, avg_vol_delta_dict, price=None):
    """
    Calculates the exponentially smoothed volume delta based on CANDLE volume.
    Optionally converts volume to quote units if price is provided.

    :param symbol: Trading pair (e.g., "BTC/USDT")
    :param current_volume: Latest 5m candle volume (base asset)
    :param last_volume_dict: Dict storing previous volumes
    :param avg_vol_delta_dict: Dict storing smoothed deltas
    :param price: Current price to convert to quote units (optional)
    :return: Smoothed volume delta
    """
    func_logger = logging.getLogger(__name__ + ".update_volume_delta")

    # Validate inputs
    if not isinstance(current_volume, (int, float)) or not np.isfinite(current_volume):
        func_logger.debug(f"{symbol}: Invalid current_volume '{current_volume}', returning 0.")
        return 0.0
    if current_volume < MIN_VOLUME_THRESHOLD:
        func_logger.debug(f"{symbol}: Current volume {current_volume:.6f} below threshold, returning previous delta.")
        return avg_vol_delta_dict.get(symbol, 0.0)  # Preserve previous delta

    # Convert to quote volume if price is provided (for consistency with vol_scale)
    if price is not None and isinstance(price, (int, float)) and np.isfinite(price) and price > 0:
        current_volume = current_volume * price
        last_vol = last_volume_dict.get(symbol, 0.0)  # Default to 0 for first run
        func_logger.debug(f"{symbol}: Converted to quote volume: {current_volume:.2f} (price={price:.4f})")
    else:
        last_vol = last_volume_dict.get(symbol, 0.0)  # Default to 0 for first run

    # Calculate raw delta
    raw_delta = current_volume - last_vol

    # Smooth the delta
    prev_avg = avg_vol_delta_dict.get(symbol, 0.0)
    avg_delta = VOLUME_ALPHA * raw_delta + (1 - VOLUME_ALPHA) * prev_avg

    # Update dictionaries
    avg_vol_delta_dict[symbol] = avg_delta
    last_volume_dict[symbol] = current_volume

    func_logger.debug(
        f"{symbol}: CurrVol={current_volume:.2f}, LastVol={last_vol:.2f}, "
        f"RawDelta={raw_delta:.2f}, PrevAvg={prev_avg:.2f}, NewAvg={avg_delta:.2f}"
    )

    return avg_delta


def process_volume_delta(tickers, last_volume_dict, avg_vol_delta_dict):
    """
    Adds 'vol_delta' key to each ticker dictionary using the latest candle volume.
    Uses quote volume for consistency if price is available.
    """
    logger.info("Calculating volume delta using latest 5m candle volume...")
    processed_count = 0
    zero_volume_count = 0
    symbols_missing_candle_vol = 0

    for ticker in tickers:
        symbol = ticker.get("symbol")
        if not symbol:
            ticker["vol_delta"] = 0.0
            logger.debug(f"Missing symbol in ticker, setting vol_delta to 0.")
            continue

        current_volume = ticker.get("latest_candle_vol_5m")
        price = ticker.get("price")  # Use current price for quote volume conversion

        if current_volume is None:
            ticker["vol_delta"] = avg_vol_delta_dict.get(symbol, 0.0)  # Preserve previous delta
            symbols_missing_candle_vol += 1
            logger.debug(f"{symbol}: Missing latest_candle_vol_5m, preserving previous delta.")
            continue

        try:
            current_volume = float(current_volume)
            if not np.isfinite(current_volume):
                raise ValueError("Non-finite volume")
        except (ValueError, TypeError) as e:
            logger.warning(f"{symbol}: Invalid latest_candle_vol_5m '{current_volume}', preserving previous delta. Error: {e}")
            ticker["vol_delta"] = avg_vol_delta_dict.get(symbol, 0.0)
            symbols_missing_candle_vol += 1
            continue

        if current_volume < MIN_VOLUME_THRESHOLD:
            zero_volume_count += 1
            ticker["vol_delta"] = avg_vol_delta_dict.get(symbol, 0.0)  # Preserve previous delta
            logger.debug(f"{symbol}: Volume {current_volume:.6f} below threshold, preserving previous delta.")
            continue

        ticker["vol_delta"] = update_volume_delta(symbol, current_volume, last_volume_dict, avg_vol_delta_dict, price=price)
        processed_count += 1

    logger.info(
        f"Volume delta calc complete. Processed:{processed_count}, "
        f"ZeroVol:{zero_volume_count}, MissingVol:{symbols_missing_candle_vol}"
    )


# --- Keep Calibration and Normalization Functions ---
def calibrate_ranking_beta(tickers, perc_field="change_15m"):
    """Dynamically calibrates the ranking exponent beta based on score distribution."""
    # (Function definition remains the same as provided in your code)
    scores = [
        t.get(perc_field) for t in tickers if t and perc_field in t and t[perc_field] is not None and not pd.isna(t[perc_field])
    ]
    if len(scores) < 5:
        return 0.25
    scores_sorted = sorted([s for s in scores if isinstance(s, (int, float))], reverse=True)
    if not scores_sorted:
        return 0.25
    diffs = np.diff(scores_sorted)
    positive_diffs = np.abs(diffs[diffs < -1e-9])
    if positive_diffs.size == 0 or np.all(positive_diffs < 1e-6):
        return 0.25
    median_diff = np.median(positive_diffs)
    if median_diff < 1e-6:
        return 0.25
    ideal_beta = 1.0 / median_diff
    calibrated_beta = np.clip(ideal_beta, 0.1, 10.0)
    logger.info(f"Beta Calibration ({perc_field}): MedDiff={median_diff:.3f}, Ideal={ideal_beta:.3f}, Calib={calibrated_beta:.3f}")
    return calibrated_beta


def calibrate_drive(tickers, ranking_beta, drive_candidates, perc_field="change_15m"):
    """Selects a drive value (placeholder - currently just returns first candidate)."""
    # (Function definition remains the same as provided in your code)
    selected_drive = drive_candidates[0] if drive_candidates else 0.0
    logger.info(f"Drive Calibration: Selected Drive = {selected_drive}")
    return selected_drive


def normalize_indicator_values(tickers, field_name, target_min=-1, target_max=1):
    """Normalizes a specific indicator field across all tickers to a target range."""
    # (Function definition remains the same as provided in your code)
    if field_name is None:
        return
    values = [
        t.get(field_name)
        for t in tickers
        if t and field_name in t and t[field_name] is not None and pd.notna(t[field_name]) and np.isfinite(t[field_name])
    ]
    norm_field_name = f"normalized_{field_name}"
    center_val = (target_max + target_min) / 2.0
    if not values:
        logger.debug(f"Normalization ({field_name}): No valid values.")
        [t.update({norm_field_name: center_val}) for t in tickers if t]
        return
    try:
        min_val, max_val = min(values), max(values)
    except Exception as e:
        logger.warning(f"Normalization ({field_name}): Min/Max failed: {e}")
        return
    range_val = max_val - min_val
    target_range = target_max - target_min
    if range_val < 1e-9:
        logger.debug(f"Normalization ({field_name}): Range near zero.")
        [t.update({norm_field_name: center_val}) for t in tickers if t]
    else:
        for ticker in tickers:
            if not ticker:
                continue
            original_value = ticker.get(field_name)
            if original_value is not None and pd.notna(original_value) and np.isfinite(original_value):
                try:
                    scaled_01 = (original_value - min_val) / range_val
                    ticker[norm_field_name] = scaled_01 * target_range + target_min
                except ZeroDivisionError:
                    ticker[norm_field_name] = center_val
                except Exception as e:
                    logger.warning(f"Normalization error for {field_name} in {ticker.get('symbol')}: {e}")
                    ticker[norm_field_name] = center_val
            else:
                ticker[norm_field_name] = center_val
    logger.debug(f"Normalization ({field_name}): Completed.")


# --- Scoring Function (MODIFIED to include MR Wings) ---
def compute_scores(tickers, ranking_beta, drive, vol_scale, base_rank_field="change_15m"):
    """Computes final scores based on short-term indicators including breakout AND MR Wings."""
    valid_tickers = [
        t
        for t in tickers
        if t
        and base_rank_field in t
        and t.get(base_rank_field) is not None
        and pd.notna(t[base_rank_field])
        and np.isfinite(t[base_rank_field])
        and "volatility_5m" in t
        and pd.notna(t.get("volatility_5m"))
        and np.isfinite(t.get("volatility_5m"))  # Ensure 5m vol is valid for adjustment
        and "breakout_15m" in t
        and pd.notna(t.get("breakout_15m"))
        and np.isfinite(t.get("breakout_15m"))  # Ensure breakout is valid
        and "mr_wings_15m" in t
        and pd.notna(t.get("mr_wings_15m"))
        and np.isfinite(t.get("mr_wings_15m"))  # Ensure MR Wings is valid
    ]
    if not valid_tickers:
        logger.warning(
            "Scoring: No valid tickers found after filtering for scoring (need base field, 5m vol, breakout, MR Wings)."
        )
        return []
    try:
        sorted_tickers = sorted(valid_tickers, key=lambda x: x.get(base_rank_field, -float("inf")), reverse=True)
    except Exception as e:
        logger.error(f"Scoring: Error sorting tickers on '{base_rank_field}': {e}. Proceeding unsorted.", exc_info=True)
        sorted_tickers = valid_tickers  # Fallback

    N = len(sorted_tickers)
    if N == 0:
        return []

    for idx, ticker in enumerate(sorted_tickers):
        rank = idx + 1
        ticker["rank"] = rank
        symbol = ticker.get("symbol", "N/A")
        try:  # Calculate Weight/Boost
            max_exponent = ranking_beta * (N - 1) if N > 1 else 0
            raw_exponent = ranking_beta * (N - rank)
            target_exponent_scale = 5.0
            normalized_exponent = (raw_exponent / max_exponent * target_exponent_scale) if max_exponent != 0 else 0
            weight = math.exp(normalized_exponent)
            ticker["weight"] = weight
        except Exception as e:
            logger.warning(f"Scoring Error (Weight Calc) for {symbol}: {e}. Defaulting weight to 1.0")
            ticker["weight"] = 1.0
            weight = 1.0
        boost = weight * drive
        ticker["boost"] = boost

        try:  # Calculate Vol Factor
            vol_delta = ticker.get("vol_delta", 0.0)
            if not isinstance(vol_delta, (int, float)) or not np.isfinite(vol_delta):
                vol_delta = 0.0
            if not isinstance(vol_scale, (int, float)) or vol_scale <= 0 or not np.isfinite(vol_scale):
                logger.warning(f"Invalid vol_scale ({vol_scale}), defaulting vol_factor to 0 for {symbol}")
                vol_factor = 0.0
            else:
                vol_factor = np.clip(vol_delta / vol_scale, -2.0, 2.0)
            ticker["vol_factor"] = vol_factor
        except Exception as e:
            logger.warning(f"Scoring Error (Vol Facftor) for {symbol}: {e}. Defaulting vol_factor to 0.")
            ticker["vol_factor"] = 0.0
            vol_factor = 0.0

        try:  # Calculate Base Score Adjusted
            base_perf = ticker.get(base_rank_field, 0.0)
            ticker["base_perf"] = base_perf
            base_score = base_perf * (1 + boost * (1 + vol_factor))
            ticker["base_score_adjusted"] = base_score
        except Exception as e:
            logger.warning(f"Scoring Error (Base Score Adj) for {symbol}: {e}. Defaulting to 0.")
            ticker["base_score_adjusted"] = 0.0
            base_score = 0.0

        try:  # Calculate Indicator Adjustment
            # Get normalized values, defaulting to 0 if not present or invalid
            norm_1m = ticker.get("normalized_change_1m", 0.0) if pd.notna(ticker.get("normalized_change_1m")) else 0.0
            norm_5m = ticker.get("normalized_change_5m", 0.0) if pd.notna(ticker.get("normalized_change_5m")) else 0.0
            norm_15m = ticker.get("normalized_change_15m", 0.0) if pd.notna(ticker.get("normalized_change_15m")) else 0.0
            norm_breakout = ticker.get("normalized_breakout_15m", 0.0) if pd.notna(ticker.get("normalized_breakout_15m")) else 0.0
            norm_mr_wings = (
                ticker.get("normalized_mr_wings_15m", 0.0) if pd.notna(ticker.get("normalized_mr_wings_15m")) else 0.0
            )  # GET NORMALIZED MR WINGS

            adj = (
                1
                + CHANGE_1M_WEIGHT * norm_1m
                + CHANGE_5M_WEIGHT * norm_5m
                + CHANGE_15M_WEIGHT * norm_15m
                + BREAKOUT_15M_WEIGHT * norm_breakout
                + MR_WINGS_WEIGHT * norm_mr_wings  # ADD MR WINGS TO ADJUSTMENT
            )
            adj = max(0.1, adj)
            ticker["indicator_adjustment"] = adj
        except Exception as e:
            logger.warning(f"Scoring Error (Indicator Adj) for {symbol}: {e}. Defaulting adj to 1.")
            ticker["indicator_adjustment"] = 1.0
            adj = 1.0

        try:  # Calculate Score Pre Volatility Adjustment
            score_pre_vol = base_score * adj
            ticker["score_pre_vol"] = score_pre_vol
        except Exception as e:
            logger.warning(f"Scoring Error (Pre Vol) for {symbol}: {e}. Defaulting to 0.")
            ticker["score_pre_vol"] = 0.0
            score_pre_vol = 0.0

        try:  # Calculate Final Score with Volatility Divisor
            vol_short = ticker.get("volatility_5m", 1.0)  # Use 5m volatility calculated earlier
            if not isinstance(vol_short, (int, float)) or not np.isfinite(vol_short) or vol_short < 0:
                vol_short = 1.0
            ticker["volatility_short"] = vol_short
            vol_short_adjusted = vol_short + 1e-9
            volatility_divisor = 1 + (vol_short_adjusted / 100.0 * SHORT_TERM_VOL_ADJ_WEIGHT)
            volatility_divisor = max(0.1, volatility_divisor)  # Prevent division by very small number
            final_score = score_pre_vol / volatility_divisor
            ticker["score"] = final_score if np.isfinite(final_score) else 0.0
        except Exception as e:
            logger.warning(f"Scoring Error (Final Score) for {symbol}: {e}. Defaulting score to 0.")
            ticker["score"] = 0.0

        # logger.debug( # Less verbose logging
        #     f"Scoring {symbol:<10} (Rnk {rank:>3})|Base={base_perf:>6.2f}|W={weight:.2f}|Bst={boost:.2f}|VolF={vol_factor:.2f}|"
        #     f"Adj={adj:.2f}(NrmMR={norm_mr_wings:.2f})|PreVol={score_pre_vol:>8.2f}|Vol5m={vol_short:.2f}%|Final={ticker['score']:>8.2f}"
        # )

    final_sorted_tickers = sorted(sorted_tickers, key=lambda x: x.get("score", -float("inf")), reverse=True)
    return final_sorted_tickers


# --- Table Creation Function (MODIFIED to include MR Wings) ---
def create_table(tickers, base_rank_field, base_rank_label):
    """Creates DataFrame showing score and relevant SHORT-TERM indicators, including Breakout AND MR Wings."""
    if not tickers:
        logger.warning("create_table: Received empty ticker list.")
        return pd.DataFrame()
    # data = {
    #     "Symbol": [t.get("symbol", "N/A") for t in tickers],
    #     "Price": [round(float(price), 4) if (price := t.get("price")) is not None else None for t in tickers],
    #     "Score": [round(float(score), 2) if (score := t.get("score")) is not None else None for t in tickers],
    #     base_rank_label: [
    #         round(float(base_rank), 2) if (base_rank := t.get(base_rank_field)) is not None else None for t in tickers
    #     ],
    #     "1m %": [round(float(ch1m), 2) if (ch1m := t.get("change_1m")) is not None else None for t in tickers],
    #     "5m %": [round(float(ch5m), 2) if (ch5m := t.get("change_5m")) is not None else None for t in tickers],
    #     "15m %": [round(float(ch15m), 2) if (ch15m := t.get("change_15m")) is not None else None for t in tickers],
    #     f"Brkout ({BREAKOUT_LOOKBACK}p)": [
    #         round(float(brk), 2) if (brk := t.get("breakout_15m")) is not None else None for t in tickers
    #     ],
    #     f"MR Wings ({MR_LFAST})": [
    #         round(float(wings), 2) if (wings := t.get("mr_wings_15m")) is not None else None for t in tickers
    #     ],
    #     "Vol (5m)%": [
    #         round(float(vol5m), 2) if (vol5m := t.get("volatility_5m")) is not None else None for t in tickers
    #     ],
    #     "Quote Vol (24h)": [
    #         round(float(qvol), 0) if (qvol := t.get("quoteVolume")) is not None else None for t in tickers
    #     ],
    #     "Vol Delta (Smooth)": [
    #         round(float(vd), 2) if (vd := t.get("vol_delta")) is not None else None for t in tickers
    #     ],
    # }

    data = {
        "Symbol": [t.get("symbol", "N/A") for t in tickers],
        "Price": [price if (price := t.get("price")) is not None else None for t in tickers],
        "Score": [score if (score := t.get("score")) is not None else None for t in tickers],
        base_rank_label: [
            base_rank if (base_rank := t.get(base_rank_field)) is not None else None for t in tickers
        ],
        "1m %": [ch1m if (ch1m := t.get("change_1m")) is not None else None for t in tickers],
        "5m %": [ch5m if (ch5m := t.get("change_5m")) is not None else None for t in tickers],
        "15m %": [ch15m if (ch15m := t.get("change_15m")) is not None else None for t in tickers],
        f"Brkout ({BREAKOUT_LOOKBACK}p)": [
            brk if (brk := t.get("breakout_15m")) is not None else None for t in tickers
        ],
        f"MR Wings ({MR_LFAST})": [
            wings if (wings := t.get("mr_wings_15m")) is not None else None for t in tickers
        ],
        "Vol (5m)%": [
            vol5m if (vol5m := t.get("volatility_5m")) is not None else None for t in tickers
        ],
        "Quote Vol (24h)": [
            qvol if (qvol := t.get("quoteVolume")) is not None else None for t in tickers
        ],
        "Vol Delta (Smooth)": [
            vd if (vd := t.get("vol_delta")) is not None else None for t in tickers
        ],
    }

    try:
        df = pd.DataFrame(data)
        df["Score_numeric"] = pd.to_numeric(df["Score"], errors="coerce").fillna(-float("inf"))
        df = df.sort_values(by="Score_numeric", ascending=False).reset_index(drop=True)
        df = df.drop(columns=["Score_numeric"])
        df.index = df.index + 1  # Start rank at 1
        df.index.name = "Rank"
        return df
    except Exception as e:
        logger.error(f"Error creating pandas DataFrame in create_table: {e}", exc_info=True)
        return pd.DataFrame({"Error": [f"Failed to create table: {e}"]})


# =============================================================================
# Async Data Fetching Orchestration (Keep as is - calls worker correctly)
# =============================================================================
async def fetch_tickers_usdt(exchange):
    """Fetches and filters tickers for USDT quote from the specified exchange."""
    # (Function definition remains the same as provided in your code)
    logger.info(f"Fetching USDT tickers from {exchange.id}...")
    min_quote_vol = 10000
    tickers_usdt = {}
    try:
        all_tickers = await exchange.fetch_tickers()
        logger.info(f"Fetched {len(all_tickers)} tickers initially from {exchange.id}.")
    except ccxt.RateLimitExceeded as e:
        logger.error(f"fetch_tickers failed: Rate Limit Exceeded. {e}", exc_info=True)
        st.error(f"Rate limit hit. Try again later. ({e})")
        return {}
    except (ccxt.NetworkError, ccxt.ExchangeError, asyncio.TimeoutError) as e:
        logger.error(f"fetch_tickers failed: Network/Exchange error. {e}", exc_info=True)
        st.error(f"Could not fetch tickers. Check connection/status. ({e})")
        return {}
    except Exception as e:
        logger.error(f"fetch_tickers failed: Unexpected error. {e}", exc_info=True)
        st.error(f"Unexpected error fetching tickers: {e}")
        return {}
    excluded = {"stable": 0, "quote": 0, "lowvol": 0, "invalid": 0}
    if not all_tickers:
        logger.warning("fetch_tickers returned no data.")
        return {}
    for symbol, ticker in all_tickers.items():
        if not symbol or "/" not in symbol or not ticker or not isinstance(ticker, dict):
            excluded["invalid"] += 1
            continue
        last_price = ticker.get("last")
        quote_volume = ticker.get("quoteVolume")
        if last_price is None or last_price <= 0:
            excluded["invalid"] += 1
            continue
        if quote_volume is None:
            quote_volume = 0.0
        try:
            base, quote = symbol.split("/")
        except ValueError:
            excluded["invalid"] += 1
            continue
        if quote.upper() in ["USDT", "USD"]:
            if base.upper() in STABLECOINS:
                excluded["stable"] += 1
                continue
            try:
                if float(quote_volume) < min_quote_vol:
                    excluded["lowvol"] += 1
                    continue
            except (ValueError, TypeError):
                excluded["invalid"] += 1
                continue
            tickers_usdt[symbol] = ticker
        else:
            excluded["quote"] += 1
    logger.info(
        f"Filtered to {len(tickers_usdt)} USDT/USD tickers (MinVol:${min_quote_vol:,.0f}). Excluded: {excluded['stable']} stable, {excluded['quote']} quote, {excluded['lowvol']} lowvol, {excluded['invalid']} invalid."
    )
    return tickers_usdt


def fetch_and_process_data_parallel(exchange_id, symbols, ccxt_timeout, num_threads):
    """Orchestrates parallel fetching using worker processes."""
    # (Function definition remains the same as provided in previous CORRECTED answers,
    #  it correctly passes 5 args including placeholders to the worker)
    if not symbols:
        logger.warning("fetch_and_process_data_parallel called with no symbols.")
        return []
    mp_context_name = "spawn" if platform.system() != "Linux" else "fork"
    logger.info(f"Attempting MP context: '{mp_context_name}'")
    try:
        mp_context = multiprocessing.get_context(mp_context_name)
        logger.info(f"Using MP context: '{mp_context_name}'.")
    except ValueError:
        logger.warning(f"MP context '{mp_context_name}' failed, using default: '{multiprocessing.get_start_method()}'.")
        mp_context = multiprocessing
    # if nu <= 0:
    #     num_cores = max(1, os.cpu_count() // 2)
    #     logger.warning(f"Invalid num_cores, defaulting to {num_cores}")
    chunk_size = max(1, math.ceil(len(symbols) / num_threads))
    symbol_chunks = [symbols[i : i + chunk_size] for i in range(0, len(symbols), chunk_size)]
    actual_num_chunks = len(symbol_chunks)
    logger.info(f"Split {len(symbols)} symbols into {actual_num_chunks} chunks (~{chunk_size} each).")
    PLACEHOLDER_TIMEFRAME = "15m"
    PLACEHOLDER_LIMIT = 100  # Needed for worker function signature
    all_results = []
    # with ProcessPoolExecutor(max_workers=num_cores, mp_context=mp_context) as executor:
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        logger.info(f"Submitting {actual_num_chunks} fetch tasks to ThreadPoolExecutor...")
        futures = {}
        for chunk in symbol_chunks:
            future = executor.submit(
                process_symbol_batch, exchange_id, chunk, PLACEHOLDER_TIMEFRAME, PLACEHOLDER_LIMIT, ccxt_timeout
            )
            futures[future] = chunk
        for future in as_completed(futures):
            chunk = futures[future]
            try:
                chunk_results = future.result()
                if isinstance(chunk_results, list):
                    all_results.extend(chunk_results)
                else:
                    logger.error(f"Worker returned unexpected type: {type(chunk_results)} for chunk {chunk[:3]}...")
                    all_results.extend([{"symbol": s, "error": f"Worker invalid type: {type(chunk_results)}"} for s in chunk])
            except Exception as e:
                logger.error(f"Worker future raised exception for chunk {chunk[:3]}...: {e}", exc_info=True)
                all_results.extend([{"symbol": s, "error": f"Future Exec Error: {e}"} for s in chunk])
    logger.info(f"Collected {len(all_results)} total entries from workers.")
    successful_count = sum(1 for res in all_results if isinstance(res, dict) and "symbol" in res and not res.get("error"))
    error_count = len(all_results) - successful_count
    logger.info(f"Parallel Fetch Summary: Success={successful_count}, Errors/Failed={error_count}")
    if error_count > 0:
        logger.warning(f"Sample fetch errors: {[res for res in all_results if isinstance(res, dict) and res.get('error')][:5]}")
    return all_results


# =============================================================================
# Ticker Processing Function (MODIFIED to calculate MR Wings with approximation)
# =============================================================================
def process_ticker_data(symbol, ticker_data, worker_results_map):
    """
    Merges ticker info with worker OHLCV results.
    Calculates short-term indicators including breakout AND MR Wings (using approx vol).
    """
    logger.debug(f"Processing ticker data for {symbol}...")
    ticker = copy.deepcopy(ticker_data)  # Start fresh
    ticker["symbol"] = symbol
    # Ensure basic fields are floats
    ticker["last"] = float(ticker.get("last", 0.0) or 0.0)
    ticker["quoteVolume"] = float(ticker.get("quoteVolume", 0.0) or 0.0)
    ticker["percentage_api"] = float(ticker.get("percentage", 0.0) or 0.0)

    worker_data = worker_results_map.get(symbol)

    # --- Initialize all fields we expect to calculate ---
    default_fields = {
        "ohlcv_1m": None,
        "ohlcv_5m": None,
        "ohlcv_15m": None,
        "price": ticker["last"],  # Default to API price
        "fetch_error": None,
        "change_1m": 0.0,
        "change_5m": 0.0,
        "change_15m": 0.0,
        "volatility_5m": 0.0,  # Calculated from 5m OHLCV
        "breakout_15m": 0.0,
        "mr_wings_15m": 0.0,  # Add default for MR Wings
        "latest_candle_vol_5m": None,  # Extracted from 5m OHLCV
    }
    # Use update to add defaults without overwriting existing keys from API ticker
    ticker.update({k: v for k, v in default_fields.items() if k not in ticker})

    # --- Process Data from Worker ---
    if worker_data and isinstance(worker_data, dict) and not worker_data.get("error"):
        # Update ticker with all non-errored data from worker
        for key, value in worker_data.items():
            # Only update if the key exists in defaults (avoid adding random keys)
            # and the worker provided a non-None value
            if key in default_fields and value is not None:
                ticker[key] = value

        # Prioritize price from worker if available
        worker_price = worker_data.get("price")
        if worker_price is not None:
            try:
                ticker["price"] = float(worker_price)
                ticker["last"] = ticker["price"]
            except (ValueError, TypeError):
                logger.warning(f"{symbol}: Invalid price '{worker_price}' from worker, using API price {ticker['last']}.")
                ticker["price"] = ticker["last"]
        else:
            ticker["price"] = ticker["last"]  # Fallback if worker price missing

        # --- Calculate Indicators based on fetched OHLCV ---
        ohlcv_1m = ticker.get("ohlcv_1m")
        if ohlcv_1m:
            ticker["change_1m"] = calculate_change_n_bars(ohlcv_1m, n_bars=1)

        ohlcv_5m = ticker.get("ohlcv_5m")
        if ohlcv_5m:
            ticker["change_5m"] = calculate_change_n_bars(ohlcv_5m, n_bars=1)
            # Calculate 5m PERCENTAGE volatility (used for scoring adjustment later)
            ticker["volatility_5m"] = calculate_volatility_n_bars(ohlcv_5m, n_bars=12, vol_type="range")
            # Extract latest 5m candle volume (for vol delta)
            if isinstance(ohlcv_5m, list) and len(ohlcv_5m) > 0:
                try:
                    latest_volume = ohlcv_5m[-1][5]
                    ticker["latest_candle_vol_5m"] = float(latest_volume) if latest_volume is not None else 0.0
                except (IndexError, TypeError, ValueError) as e:
                    logger.warning(f"{symbol}: Error extracting volume from 5m OHLCV: {e}. Data: {ohlcv_5m[-1:]}")
                    ticker["latest_candle_vol_5m"] = 0.0
            else:
                ticker["latest_candle_vol_5m"] = 0.0

        ohlcv_15m = ticker.get("ohlcv_15m")
        if ohlcv_15m:
            ticker["change_15m"] = calculate_change_n_bars(ohlcv_15m, n_bars=1)

            # --- Calculate 15m Indicators (Breakout & MR Wings) ---
            price_series_15m = None
            closes_15m = []
            # Ensure we have enough data and extract valid close prices
            if isinstance(ohlcv_15m, list) and len(ohlcv_15m) > 1:
                try:
                    closes_15m = [float(c[4]) for c in ohlcv_15m if c and len(c) > 4 and c[4] is not None]
                    if closes_15m:
                        price_series_15m = pd.Series(closes_15m)  # Timestamps not critical for these indicators
                except (ValueError, TypeError, IndexError) as e:
                    logger.warning(f"{symbol}: Error extracting 15m close prices: {e}")

            # Calculate 15m Breakout
            min_len_for_breakout = max(2, int(BREAKOUT_LOOKBACK / 2))
            if price_series_15m is not None and len(price_series_15m) >= min_len_for_breakout:
                try:
                    breakout_series = breakout(
                        price_series_15m, lookback_periods=BREAKOUT_LOOKBACK, smooth_periods=BREAKOUT_SMOOTH
                    )
                    if not breakout_series.empty:
                        ticker["breakout_15m"] = breakout_series.iloc[-1]
                    # else: logger.debug(f"{symbol}: Breakout calc returned empty series.") # Reduce noise
                except Exception as e:
                    logger.error(f"{symbol}: Error calculating 15m breakout: {e}", exc_info=True)
            # else: logger.debug(f"{symbol}: Insufficient 15m data for breakout ({len(closes_15m)} closes).") # Reduce noise

            # Calculate 15m MR Wings (using approximation for volatility)
            min_len_for_mr = MR_LFAST * 4 + 1  # Need enough data for internal EWMAs
            if price_series_15m is not None and len(price_series_15m) >= min_len_for_mr:
                # Check if we have the PRE-CALCULATED 5m percentage volatility
                vol_5m_percent = ticker.get("volatility_5m")
                if vol_5m_percent is not None and vol_5m_percent > 0:
                    try:
                        # --- APPROXIMATION of 15m Price Unit Volatility ---
                        # Convert 5m % volatility to an approximate 15m price unit volatility series
                        # This assumes the recent % vol applies across the 15m price series
                        approx_vol_15m_pu_series = (vol_5m_percent / 100.0) * price_series_15m
                        # Ensure it's safe (positive, non-zero)
                        approx_vol_15m_pu_series = approx_vol_15m_pu_series.clip(lower=1e-9).fillna(1e-9)
                        logger.debug(
                            f"{symbol}: Approximated 15m PU Vol using 5m %Vol ({vol_5m_percent:.2f}%). Approx Vol tail: {approx_vol_15m_pu_series.iloc[-3:].to_list()}"
                        )

                        # Calculate MR Wings using the price series and the APPROXIMATED volatility series
                        mr_wings_series = mr_wings(price_series_15m, approx_vol_15m_pu_series, Lfast=MR_LFAST)

                        if not mr_wings_series.empty:
                            ticker["mr_wings_15m"] = mr_wings_series.iloc[-1]
                            # logger.debug(f"{symbol}: Calculated 15m MR Wings ({MR_LFAST}): {ticker['mr_wings_15m']:.3f}") # Reduce noise
                        # else: logger.debug(f"{symbol}: MR Wings calc returned empty series.") # Reduce noise

                    except Exception as e:
                        logger.error(f"{symbol}: Error calculating 15m MR Wings with approx vol: {e}", exc_info=True)
                else:
                    logger.debug(f"{symbol}: Cannot calculate MR Wings, missing valid volatility_5m ({vol_5m_percent}).")
            # else: logger.debug(f"{symbol}: Insufficient 15m data for MR Wings ({len(closes_15m)} closes).") # Reduce noise
            # --- End 15m MR Wings Calculation ---

    # --- Handle cases where worker failed or returned error ---
    elif worker_data and worker_data.get("error"):
        ticker["fetch_error"] = worker_data.get("error")
        logger.warning(f"Worker reported fetch error for {symbol}: {ticker['fetch_error']}")
        # Ensure calculated fields are defaulted on error
        ticker["latest_candle_vol_5m"] = 0.0
        ticker["breakout_15m"] = 0.0
        ticker["mr_wings_15m"] = 0.0
        ticker["volatility_5m"] = 0.0
        ticker["change_1m"] = 0.0
        ticker["change_5m"] = 0.0
        ticker["change_15m"] = 0.0
    else:  # Case where worker didn't return data for this symbol at all
        ticker["fetch_error"] = "Missing worker result"
        logger.warning(f"No result found in worker_results_map for symbol {symbol}.")
        # Ensure calculated fields are defaulted
        ticker["latest_candle_vol_5m"] = 0.0
        ticker["breakout_15m"] = 0.0
        ticker["mr_wings_15m"] = 0.0
        ticker["volatility_5m"] = 0.0
        ticker["change_1m"] = 0.0
        ticker["change_5m"] = 0.0
        ticker["change_15m"] = 0.0

    return ticker


# =============================================================================
# Main Asynchronous Orchestration Function
# =============================================================================
async def async_dashboard_main(selected_exchanges, num_cores):
    """Main async function for the SHORT-TERM focused dashboard with multiple exchanges."""
    global dashboard_placeholder
    start_run_time = time.time()
    all_results = []

    with dashboard_placeholder.container():
        st.info(f"🚀 Initializing exchanges & fetching data (Short-Term Focus)...")
        st.markdown("---")
        progress_bar = st.progress(0)
        status_text = st.empty()

        for exchange_id in selected_exchanges:
            exchange = None
            try:
                # --- 1. Initialize Exchange ---
                status_text.info(f"Connecting to {SUPPORTED_EXCHANGES[exchange_id]}...")
                try:
                    exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True, "timeout": CCXT_TIMEOUT})
                    logger.info(f"Initialized {SUPPORTED_EXCHANGES[exchange_id]} exchange instance.")
                    progress_bar.progress(5)
                except (ccxt.AuthenticationError, ccxt.ExchangeError, Exception) as e:
                    logger.error(f"Fatal: Failed to initialize {SUPPORTED_EXCHANGES[exchange_id]} exchange: {e}", exc_info=True)
                    status_text.error(f"❌ Failed to connect to {SUPPORTED_EXCHANGES[exchange_id]}: {e}")
                    if exchange:
                        await exchange.close()
                    continue

                # --- 2. Fetch Filtered Tickers ---
                status_text.info(f"Fetching and filtering USDT/USD tickers from {SUPPORTED_EXCHANGES[exchange_id]}...")
                tickers_dict = await fetch_tickers_usdt(exchange)
                if not tickers_dict:
                    status_text.error(f"❌ No valid USDT/USD tickers found on {SUPPORTED_EXCHANGES[exchange_id]} matching criteria.")
                    if exchange:
                        await exchange.close()
                    continue
                symbols = list(tickers_dict.keys())
                status_text.info(f"✅ Found {len(symbols)} USDT/USD symbols matching criteria on {SUPPORTED_EXCHANGES[exchange_id]}.")
                progress_bar.progress(10)

                # --- 3. Fetch OHLCV in Parallel (Workers) ---
                status_text.info(f"Fetching short-term OHLCV from {SUPPORTED_EXCHANGES[exchange_id]} ({len(symbols)} symbols, {num_cores} cores)...")
                ohlcv_fetch_start = time.time()
                worker_results = fetch_and_process_data_parallel(exchange_id, symbols, CCXT_TIMEOUT, num_cores)
                logger.info(f"Parallel SHORT TF OHLCV fetch complete for {SUPPORTED_EXCHANGES[exchange_id]} ({time.time() - ohlcv_fetch_start:.2f}s).")
                progress_bar.progress(40)
                worker_results_map = {res["symbol"]: res for res in worker_results if isinstance(res, dict) and "symbol" in res}
                successful_worker_count = sum(
                    1 for res in worker_results if isinstance(res, dict) and "symbol" in res and not res.get("error")
                )
                total_worker_results = len(worker_results)
                failed_worker_count = total_worker_results - successful_worker_count

                # Handle complete fetch failure
                if successful_worker_count == 0 and total_worker_results > 0:
                    msg = f"❌ **Critical:** Failed to fetch OHLCV data for *any* symbol on {SUPPORTED_EXCHANGES[exchange_id]} ({failed_worker_count}/{total_worker_results} failures)."
                    logger.error(msg + " Check worker logs for details.")
                    status_text.error(msg + " Displaying only basic ticker info.")
                    continue

                # Warn about partial failures
                elif failed_worker_count > 0:
                    warn_msg = f"⚠️ Fetched OHLCV successfully for {successful_worker_count}/{len(symbols)} symbols on {SUPPORTED_EXCHANGES[exchange_id]} ({failed_worker_count} failures). Analysis proceeding."
                    logger.warning(warn_msg)
                    status_text.warning(warn_msg)

                # --- 4. Process Ticker Data ---
                status_text.info(f"Processing data & calculating indicators for {SUPPORTED_EXCHANGES[exchange_id]}...")
                processing_start = time.time()
                tickers_list = []
                total_symbols_to_process = len(tickers_dict)
                processed_count = 0
                error_processing_count = 0

                for i, (symbol, ticker_data) in enumerate(tickers_dict.items()):
                    try:
                        processed_ticker = process_ticker_data(symbol, ticker_data, worker_results_map)
                        # Add exchange info to the ticker
                        processed_ticker["exchange"] = SUPPORTED_EXCHANGES[exchange_id]
                        tickers_list.append(processed_ticker)
                        if processed_ticker.get("fetch_error"):
                            error_processing_count += 1
                        else:
                            processed_count += 1
                    except Exception as e:
                        logger.error(f"Error processing data for {symbol} in main loop: {e}", exc_info=True)
                        error_processing_count += 1
                    progress_bar.progress(40 + int((i + 1) / total_symbols_to_process * 30))

                logger.info(
                    f"Processed {processed_count} symbols successfully on {SUPPORTED_EXCHANGES[exchange_id]}, {error_processing_count} with errors/missing data ({time.time() - processing_start:.2f}s)."
                )

                # Add processed tickers to all results
                all_results.extend(tickers_list)

            except Exception as e:
                logger.error(f"Error processing {SUPPORTED_EXCHANGES[exchange_id]}: {e}", exc_info=True)
                status_text.error(f"❌ Error processing {SUPPORTED_EXCHANGES[exchange_id]}: {e}")
            finally:
                if exchange:
                    try:
                        await exchange.close()
                        logger.info(f"Closed {SUPPORTED_EXCHANGES[exchange_id]} exchange connection.")
                    except Exception as e:
                        logger.error(f"Error closing {SUPPORTED_EXCHANGES[exchange_id]} connection: {e}")

        # --- Process all results together ---
        if all_results:
            # --- 5. Calculate Volume Delta ---
            status_text.info("Processing volume delta...")
            progress_bar.progress(75)
            if "last_volume_dict" not in st.session_state:
                st.session_state["last_volume_dict"] = {}
                logger.info("Initialized 'last_volume_dict' in session state.")
            if "avg_vol_delta_dict" not in st.session_state:
                st.session_state["avg_vol_delta_dict"] = {}
                logger.info("Initialized 'avg_vol_delta_dict' in session state.")
            process_volume_delta(all_results, st.session_state["last_volume_dict"], st.session_state["avg_vol_delta_dict"])
            volumes = [
                t.get("quoteVolume") for t in all_results if t and t.get("quoteVolume") is not None and t.get("quoteVolume") > 0
            ]
            vol_scale = float(np.median(volumes)) if volumes else 1.0
            vol_scale = max(vol_scale, 1.0)
            logger.info(f"Volume Scale (Median Quote Vol) calculated: {vol_scale:.2f}")

            # --- 6. Normalize Indicators ---
            status_text.info("Normalizing indicators...")
            progress_bar.progress(80)
            normalize_indicator_values(all_results, "change_1m")
            normalize_indicator_values(all_results, "change_5m")
            normalize_indicator_values(all_results, "change_15m")
            normalize_indicator_values(all_results, "breakout_15m")
            normalize_indicator_values(all_results, "mr_wings_15m")
            normalize_indicator_values(all_results, "volatility_5m")
            normalize_indicator_values(all_results, "vol_delta")

            # --- 7. Scoring ---
            status_text.info("Calculating final scores...")
            progress_bar.progress(85)
            scoring_start = time.time()
            drive_candidates = [0.5 * i for i in range(1, 11)]
            base_rank_field = "change_15m"
            base_rank_label = "15m %"
            beta = calibrate_ranking_beta(all_results, perc_field=base_rank_field)
            drive = calibrate_drive(all_results, beta, drive_candidates, perc_field=base_rank_field)
            final_tickers = compute_scores(copy.deepcopy(all_results), beta, drive, vol_scale, base_rank_field=base_rank_field)
            logger.info(f"Scoring complete ({time.time() - scoring_start:.2f}s). Found {len(final_tickers)} scored tickers.")
            progress_bar.progress(95)

            # --- 8. Display Results ---
            status_text.info("📊 Preparing display...")
            st.subheader(f"Multi-Exchange USDT/USD Rankings (Short-Term Focus, Base: {base_rank_label})")
            if final_tickers:
                results_df = create_table(final_tickers, base_rank_field, base_rank_label)
                st.dataframe(results_df, use_container_width=True)
                logger.info("Top 5 Results:\n" + results_df.head().to_string())
            else:
                st.warning("No tickers available to display after scoring.")
            st.markdown("---")
            progress_bar.progress(100)
            execution_time = time.time() - start_run_time
            status_text.success(
                f"✅ Dashboard updated successfully ({execution_time:.2f}s). Displaying {len(final_tickers)} symbols across {len(selected_exchanges)} exchanges."
            )
            st.caption(f"Last update: {pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M:%S %Z}")
            logger.info(f"--- Dashboard Update Complete ({execution_time:.2f}s) ---")
        else:
            status_text.error("❌ No data available from any selected exchange.")
            progress_bar.progress(100)

# =============================================================================
# Streamlit UI Control and Async Runner
# =============================================================================
def run_async_dashboard(selected_exchanges):
    """Manages the asyncio event loop and runs the main dashboard function."""
    logger.info(f"run_async_dashboard called with exchanges: {selected_exchanges}")
    try:
        try:
            loop = asyncio.get_event_loop_policy().get_event_loop()
            logger.debug("Got existing event loop.")
        except RuntimeError as ex:
            if "There is no current event loop" in str(ex):
                logger.info("No current event loop, creating new one.")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            else:
                logger.error(f"RuntimeError getting event loop: {ex}", exc_info=True)
                raise
        num_threads_to_use = DEFAULT_NUM_THREADS
        if not loop.is_running():
            logger.info("Event loop not running. Running until complete.")
            loop.run_until_complete(async_dashboard_main(selected_exchanges, num_threads_to_use))
        else:
            logger.info("Event loop running. Creating task.")
            loop.create_task(async_dashboard_main(selected_exchanges, num_threads_to_use))
    except Exception as e:
        logger.error(f"Error in run_async_dashboard: {e}", exc_info=True)
        st.error(f"Critical error running dashboard logic: {e}")

# --- Sidebar Controls ---
st.sidebar.markdown("---")
st.sidebar.header("Dashboard Controls")

# Initialize session state for tracking calculations
if "calculations_running" not in st.session_state:
    st.session_state.calculations_running = False
if "last_selected_exchanges" not in st.session_state:
    st.session_state.last_selected_exchanges = []

# Add exchange selection
selected_exchanges = st.sidebar.multiselect(
    "Select Exchanges",
    options=list(SUPPORTED_EXCHANGES.keys()),
    format_func=lambda x: SUPPORTED_EXCHANGES[x],
    default=["kraken"],
    help="Select one or more exchanges to fetch data from"
)

# Check if exchanges have changed
if selected_exchanges != st.session_state.last_selected_exchanges:
    if st.session_state.calculations_running:
        st.sidebar.warning("⚠️ Exchange selection changed. Please click 'Start Calculations' to update.")
    st.session_state.last_selected_exchanges = selected_exchanges
    st.session_state.calculations_running = False

st.sidebar.info(f"Using up to {DEFAULT_NUM_THREADS} threads for fetching.")

# Add start button
if st.sidebar.button("🚀 Start Calculations", key="start_calculations_button", help="Click to start fetching and analyzing data from selected exchanges."):
    st.sidebar.info("Starting calculations...")
    logger.info(f"Start calculations button clicked for exchanges: {selected_exchanges}")
    st.session_state.calculations_running = True
    st.session_state.pop("last_volume_dict", None)
    st.session_state.pop("avg_vol_delta_dict", None)
    logger.info("Cleared volume tracking state for refresh.")
    run_async_dashboard(selected_exchanges)

# =============================================================================
# Script Entry Point Guard
# =============================================================================
if __name__ == "__main__":
    logger.info("Script execution started (__name__ == '__main__').")
    # --- Set Multiprocessing Start Method (Important for cross-platform compatibility) ---
    try:
        current_method = multiprocessing.get_start_method(allow_none=True)
        logger.info(f"Current MP start method: {current_method}")
        target_method = (
            "spawn" if platform.system() != "Linux" else "fork"
        )  # 'spawn' preferred for safety/consistency, 'fork' often faster on Linux
        if current_method != target_method:
            try:
                multiprocessing.set_start_method(target_method, force=True)
                logger.info(f"Set MP start method to '{target_method}'.")
            except (ValueError, RuntimeError) as e:
                logger.warning(f"Could not force MP start method to '{target_method}': {e}. Using default: '{current_method}'.")
    except Exception as e:
        logger.error(f"Failed to set multiprocessing start method: {e}", exc_info=True)
        st.error(f"Error setting up multiprocessing: {e}. App might not function correctly.")

    # Display initial message if no calculations have been run
    if not st.session_state.calculations_running:
        st.info("👋 Welcome! Please select exchanges and click 'Start Calculations' to begin.")
        st.stop()  # Stop execution until user clicks start
