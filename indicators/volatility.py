"""ATR and Bollinger Band volatility scoring for swing trading."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


def _bollinger(
    close: pd.Series, period: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    sma = close.rolling(period).mean()
    std = close.rolling(period).std(ddof=0)
    upper = sma + num_std * std
    lower = sma - num_std * std
    band_width = (upper - lower).replace(0, np.finfo(float).eps)
    pct_b = (close - lower) / band_width
    return upper, sma, lower, pct_b


def add_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["Close"]

    df["atr"] = _atr(df)
    df["atr_pct"] = df["atr"] / close * 100

    upper, bb_mid, lower, pct_b = _bollinger(close)
    df["bb_upper"] = upper
    df["bb_mid"] = bb_mid
    df["bb_lower"] = lower
    df["pct_b"] = pct_b

    return df


def score(df: pd.DataFrame) -> float:
    """Volatility/risk score 0-100.

    %B component (60%): where price sits in the Bollinger Band.
    ATR% component (40%): rewards tradeable volatility (1-3% sweet spot).
    """
    if len(df) < 20:
        return 0.0

    last = df.iloc[-1]
    pct_b = float(last.get("pct_b", 0.5))
    atr_pct = float(last.get("atr_pct", 2.0))

    if pct_b > 1.0:
        bb_s = 90.0
    elif pct_b > 0.5:
        bb_s = 50 + (pct_b - 0.5) * 80
    elif pct_b > 0.2:
        bb_s = (pct_b - 0.2) / 0.3 * 50
    else:
        bb_s = 0.0

    if 1.0 <= atr_pct <= 3.0:
        atr_s = 100.0
    elif atr_pct < 1.0:
        atr_s = atr_pct * 100
    else:
        atr_s = max(0.0, 100 - (atr_pct - 3.0) * 20)

    return bb_s * 0.6 + atr_s * 0.4
