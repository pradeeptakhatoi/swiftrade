"""RSI and MACD momentum scoring for swing trading."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
    return 100 - (100 / (1 + rs))


def _macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def add_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["Close"]

    df["rsi"] = _rsi(close)

    macd_line, signal_line, histogram = _macd(close)
    df["macd_line"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"] = histogram

    return df


def _rsi_score(rsi: float) -> float:
    """Map RSI to a 0-100 score. Peak at RSI ~60."""
    if np.isnan(rsi):
        return 0.0
    if rsi < 30:
        return 65.0
    if rsi < 50:
        return 65 + (rsi - 30) * 1.25           # 65 -> 90
    if rsi < 65:
        return 90 + (rsi - 50) * (10 / 15)      # 90 -> 100
    if rsi < 70:
        return 100 - (rsi - 65) * 3             # 100 -> 85
    return max(0.0, 85 - (rsi - 70) * 4.25)     # 85 -> 0


def score(df: pd.DataFrame) -> float:
    """Momentum score 0-100.

    60% RSI(14) bell curve favouring 55-65.
    40% MACD histogram direction and sign.
    """
    if len(df) < 26:
        return 0.0

    last = df.iloc[-1]
    rsi_val = last.get("rsi", np.nan)
    hist = last.get("macd_hist", 0.0)
    prev_hist = df["macd_hist"].iloc[-2] if len(df) > 1 else 0.0

    rsi_s = _rsi_score(float(rsi_val))

    if hist > 0 and hist >= prev_hist:
        macd_s = 100.0
    elif hist > 0:
        macd_s = 70.0
    elif hist > prev_hist:
        macd_s = 45.0
    else:
        macd_s = 10.0

    return rsi_s * 0.6 + macd_s * 0.4
