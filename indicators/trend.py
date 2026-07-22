"""EMA crossover and 200-EMA trend scoring for swing trading."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def add_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["Close"]

    df["ema20"] = _ema(close, 20)
    df["ema50"] = _ema(close, 50)
    df["ema200"] = _ema(close, 200)

    df["ema_cross_bullish"] = (df["ema20"] > df["ema50"]).astype(int)
    # True on the bar where the 20 EMA crossed above the 50 EMA
    df["ema_just_crossed"] = (
        (df["ema20"] > df["ema50"]) & (df["ema20"].shift(1) <= df["ema50"].shift(1))
    ).astype(int)

    df["above_200ema"] = (close > df["ema200"]).astype(int)
    df["pct_from_200ema"] = (close - df["ema200"]) / df["ema200"] * 100
    df["pct_from_50ema"] = (close - df["ema50"]) / df["ema50"] * 100

    return df


def score(df: pd.DataFrame) -> float:
    """Trend score 0-100.

    - 50 pts: 20 EMA above 50 EMA (bullish intermediate trend)
    - 50 pts: price above 200 EMA (bullish long-term trend)
    - +10 bonus for fresh 20/50 crossover within last 5 bars
    - Penalty when price >15% above 200 EMA (too extended)
    """
    if len(df) < 200:
        return 0.0

    last = df.iloc[-1]
    pts = 0.0

    # Intermediate trend
    if last["ema_cross_bullish"]:
        pts += 50
        recent_cross = df["ema_just_crossed"].iloc[-5:].any()
        if recent_cross:
            pts += 10

    # Long-term trend + extension penalty
    if last["above_200ema"]:
        pct = last["pct_from_200ema"]
        if pct <= 15:
            pts += 50
        else:
            pts += max(0.0, 50 - (pct - 15) * 2)

    return min(100.0, pts)
