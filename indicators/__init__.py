"""Technical indicators for swing and intraday trading.

``compute_all(df)``       — swing-trading indicators (EMA, RSI14, MACD, ATR, Bollinger, breakout).
``compute_intraday(df)``  — intraday indicators (VWAP, SuperTrend, RSI7, fast MACD, ORB).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .trend import add_trend_indicators
from .momentum import add_momentum_indicators
from .volatility import add_volatility_indicators
from .breakout import add_breakout_indicators


def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """Swing-trading indicators. Requires at least 30 bars."""
    if df.empty or len(df) < 30:
        return df
    df = add_trend_indicators(df)
    df = add_momentum_indicators(df)
    df = add_volatility_indicators(df)
    df = _add_volume_indicators(df)
    df = add_breakout_indicators(df)
    return df


def _add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add vol_avg20 and vol_ratio for swing scoring."""
    df = df.copy()
    vol = df["Volume"].replace(0, np.nan)
    df["vol_avg20"] = vol.rolling(20).mean()
    df["vol_ratio"] = vol / df["vol_avg20"].replace(0, np.finfo(float).eps)
    return df


# ---------------------------------------------------------------------------
# Elementary helpers (used by compute_intraday)
# ---------------------------------------------------------------------------


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 7) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series, fast: int, slow: int, signal: int):
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 7) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def _vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    typical_price = (high + low + close) / 3
    cum_tp_vol = (typical_price * volume).cumsum()
    cum_vol = volume.cumsum().replace(0, np.nan)
    return cum_tp_vol / cum_vol


def _supertrend(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 7, multiplier: float = 3.0):
    atr = _atr(high, low, close, period)
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    st_direction = pd.Series(1, index=close.index, dtype=int)
    supertrend = pd.Series(np.nan, index=close.index)

    for i in range(1, len(close)):
        if close.iloc[i] > upper_band.iloc[i - 1]:
            st_direction.iloc[i] = 1
        elif close.iloc[i] < lower_band.iloc[i - 1]:
            st_direction.iloc[i] = -1
        else:
            st_direction.iloc[i] = st_direction.iloc[i - 1]

        if st_direction.iloc[i] == 1:
            lower_band.iloc[i] = max(lower_band.iloc[i], lower_band.iloc[i - 1]) if st_direction.iloc[i - 1] == 1 else lower_band.iloc[i]
            supertrend.iloc[i] = lower_band.iloc[i]
        else:
            upper_band.iloc[i] = min(upper_band.iloc[i], upper_band.iloc[i - 1]) if st_direction.iloc[i - 1] == -1 else upper_band.iloc[i]
            supertrend.iloc[i] = upper_band.iloc[i]

    return supertrend, st_direction


def _opening_range(high: pd.Series, low: pd.Series, close: pd.Series, orb_bars: int = 3):
    """Opening range = high/low of first *orb_bars* bars of the session."""
    if len(high) < orb_bars:
        orb_high = high.max()
        orb_low = low.min()
    else:
        orb_high = high.iloc[:orb_bars].max()
        orb_low = low.iloc[:orb_bars].min()

    above_orb = (close > orb_high).astype(int)
    below_orb = (close < orb_low).astype(int)
    return orb_high, orb_low, above_orb, below_orb


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def compute_intraday(df: pd.DataFrame, interval_minutes: int = 15) -> pd.DataFrame:
    """Add all intraday indicator columns to *df* (in-place copy returned).

    Expected input columns: Open, High, Low, Close, Volume.
    """
    df = df.copy()

    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    volume = df["Volume"].astype(float)

    # RSI(7)
    df["rsi7"] = _rsi(close, period=7)

    # MACD(5, 13, 3) — fast intraday settings
    _, _, macd_hist = _macd(close, fast=5, slow=13, signal=3)
    df["macd_fast_hist"] = macd_hist

    # ATR(7)
    df["atr7"] = _atr(high, low, close, period=7)

    # VWAP
    df["vwap"] = _vwap(high, low, close, volume)
    df["vwap_pct"] = (close - df["vwap"]) / df["vwap"].replace(0, np.nan) * 100

    # SuperTrend(7, 3)
    st_line, st_dir = _supertrend(high, low, close, period=7, multiplier=3.0)
    df["supertrend"] = st_line
    df["st_direction"] = st_dir

    # Opening Range Breakout (first N bars based on interval)
    orb_bars = max(1, 30 // interval_minutes)  # ~first 30 minutes
    orb_h, orb_l, above, below = _opening_range(high, low, close, orb_bars)
    df["orb_high"] = orb_h
    df["orb_low"] = orb_l
    df["above_orb"] = above
    df["below_orb"] = below

    # Volume ratio vs 20-bar SMA
    vol_sma = volume.rolling(window=20, min_periods=1).mean()
    df["vol_ratio"] = volume / vol_sma.replace(0, np.nan)

    return df
