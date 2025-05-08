# ewmac.py
# -*- coding: utf-8 -*-
"""
Functions related to calculating Exponential Weighted Moving Average Crossover (EWMAC) forecasts.
"""
import pandas as pd
import numpy as np
import logging # Optional: Add logging if desired within these functions

# Use the main logger setup in watchlist if run together, or configure basic if run standalone
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


def robust_vol_calc(price_series, lookback_days):
    """
    Calculates the price unit volatility of a price series.

    This version uses rolling standard deviation of price CHANGES.
    Handles edge cases like insufficient data.

    :param price_series: pd.Series of prices
    :param lookback_days: int, lookback period for volatility calculation
    :return: pd.Series of price unit volatility (aligned with price_series index)
    """
    if not isinstance(price_series, pd.Series):
        logger.error("robust_vol_calc: Input must be a pandas Series.")
        return pd.Series(dtype=float) # Return empty series on type error
    if price_series.empty:
        logger.debug("robust_vol_calc: Input Series is empty.")
        return pd.Series(dtype=float)

    logger.debug(f"robust_vol_calc: Calculating vol with lookback {lookback_days} on series length {len(price_series)}")

    # Calculate price differences (more robust to price level than pct_change for unit vol)
    price_diff = price_series.diff()

    # Calculate rolling standard deviation of price differences
    # Ensure min_periods handles the start of the series gracefully
    min_periods_vol = max(2, int(lookback_days * 0.5)) # Need at least 2 points for std dev

    if len(price_diff.dropna()) < min_periods_vol:
         logger.warning(f"robust_vol_calc: Not enough non-NaN price differences ({len(price_diff.dropna())}) for min_periods ({min_periods_vol}). Returning zeros.")
         # Return zeros matching the original index
         return pd.Series(np.zeros(len(price_series)), index=price_series.index)

    try:
        vol_price_unit = price_diff.rolling(window=lookback_days, min_periods=min_periods_vol).std()

        # Fill NaNs at the beginning with a forward fill, then backfill, then small value
        vol_price_unit = vol_price_unit.fillna(method='ffill').fillna(method='bfill').fillna(1e-9)

        # Ensure volatility is positive, add epsilon for safety
        vol_price_unit = vol_price_unit.abs() + 1e-12 # Use abs() before adding epsilon

        logger.debug(f"robust_vol_calc: Calculation complete. Vol tail:\n{vol_price_unit.tail()}")
        return vol_price_unit

    except Exception as e:
         logger.error(f"robust_vol_calc: Error during rolling std calculation: {e}", exc_info=True)
         # Return zeros on unexpected error
         return pd.Series(np.zeros(len(price_series)), index=price_series.index)


def ewmac(price, vol, Lfast, Lslow):
    """
    Calculate the ewmac trading rule forecast, given a price, price unit volatility
    and EWMA speeds Lfast and Lslow.

    Assumes that 'price' and vol have the same frequency (e.g., daily, hourly).

    :param price: The price series (pd.Series)
    :param vol: The price unit volatility (pd.Series aligned to price index)
    :param Lfast: Lookback period for fast EWMA (in number of periods)
    :param Lslow: Lookback period for slow EWMA (in number of periods)
    :returns: pd.Series -- unscaled, uncapped forecast
    """
    func_logger = logging.getLogger(__name__ + '.ewmac') # Use specific logger
    func_logger.debug(f"ewmac called: Lfast={Lfast}, Lslow={Lslow}, Price len={len(price)}, Vol len={len(vol)}")

    # --- Input Validation ---
    if not isinstance(price, pd.Series): func_logger.error("Input 'price' must be a pd.Series."); return pd.Series(dtype=float)
    if not isinstance(vol, pd.Series): func_logger.error("Input 'vol' must be a pd.Series."); return pd.Series(dtype=float)
    if price.empty or vol.empty: func_logger.debug("Price or Vol series is empty."); return pd.Series(dtype=float)

    # --- Alignment & Safety ---
    # Ensure volatility is aligned with price index, fill gaps, ensure positivity
    try:
        vol_aligned = vol.reindex(price.index, method='ffill').fillna(method='bfill')
        if vol_aligned.isnull().any():
             func_logger.warning("Volatility series contains NaNs after ffill/bfill. Filling remaining with 1e-9.")
             vol_aligned.fillna(1e-9, inplace=True)
        # Ensure non-negative and add epsilon
        vol_safe = vol_aligned.abs() + 1e-12
    except Exception as e_align:
        func_logger.error(f"Error aligning/cleaning volatility: {e_align}", exc_info=True)
        return pd.Series(np.zeros(len(price)), index=price.index) # Return zeros if alignment fails

    # --- Calculation ---
    try:
        # Determine min_periods dynamically, ensuring it's at least 1
        min_p_fast = max(1, int(Lfast * 0.5))
        min_p_slow = max(1, int(Lslow * 0.5))

        func_logger.debug(f"Calculating EWMAs: Lfast={Lfast}, Lslow={Lslow}, min_p_fast={min_p_fast}, min_p_slow={min_p_slow}")
        fast_ewma = price.ewm(span=Lfast, min_periods=min_p_fast, adjust=True).mean()
        slow_ewma = price.ewm(span=Lslow, min_periods=min_p_slow, adjust=True).mean()
        raw_ewmac = fast_ewma - slow_ewma

        func_logger.debug(f"Raw EWMAC calculated. Dividing by safe volatility. Vol tail:\n{vol_safe.tail()}")
        scaled_ewmac = raw_ewmac / vol_safe

        # Handle potential infinities resulting from division
        scaled_ewmac.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Fill any remaining NaNs (e.g., from initial periods) with 0
        scaled_ewmac.fillna(0.0, inplace=True)

        func_logger.debug("EWMAC calculation successful.")
        return scaled_ewmac

    except Exception as e_calc:
        func_logger.error(f"Error during EWMAC calculation: {e_calc}", exc_info=True)
        return pd.Series(np.zeros(len(price)), index=price.index) # Return zeros on error

# --- Optional example/helper functions from original file (can be removed if not needed) ---

def ewmac_calc_vol(price, Lfast, Lslow, vol_days=35):
    """ Uses robust_vol_calc internally. """
    func_logger = logging.getLogger(__name__ + '.ewmac_calc_vol')
    if not isinstance(price, pd.Series): func_logger.error("Input 'price' must be pd.Series."); return pd.Series(dtype=float)
    if price.empty: return pd.Series(dtype=float)
    func_logger.debug(f"Calculating vol with lookback {vol_days} then ewmac Lfast={Lfast}, Lslow={Lslow}")
    vol = robust_vol_calc(price, vol_days)
    forecast = ewmac(price, vol, Lfast, Lslow)
    return forecast

def ewmac_forecast_with_defaults(price, Lfast=32, Lslow=128):
    """ Example function calling ewmac_calc_vol. """
    return ewmac_calc_vol(price, Lfast=Lfast, Lslow=Lslow)

def ewmac_forecast_with_defaults_no_vol(price, vol, Lfast=16, Lslow=32):
    """ Example function calling ewmac directly. """
    return ewmac(price, vol, Lfast=Lfast, Lslow=Lslow)