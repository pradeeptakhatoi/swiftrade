"""52-week high proximity and 20-day breakout scoring for swing trading."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_breakout_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["Close"]
    high = df["High"]

    # Rolling 252-day high (~52 weeks) and distance from it
    high_252 = high.rolling(252, min_periods=50).max()
    df["high_52wk"] = high_252
    df["pct_from_52wk_high"] = (close - high_252) / high_252 * 100  # always <= 0

    # Recent 20-day high breakout: close > highest high of prior 20 bars
    prior_high_20 = high.shift(1).rolling(20).max()
    df["broke_20d_high"] = (close > prior_high_20).astype(int)

    return df


def score(df: pd.DataFrame) -> float:
    """Breakout score 0-100.

    Proximity to 52-week high (80 pts):
      Within 2% -> 80, within 10% -> ~64, >30% -> 0.
    Recent 20-day high breakout bonus (20 pts):
      Within last 3 bars -> 20, within 7 bars -> 10.
    """
    if len(df) < 50:
        return 0.0

    last = df.iloc[-1]
    pct_from_52wk = float(last.get("pct_from_52wk_high", -50))

    proximity_pts = max(0.0, 80 + pct_from_52wk * 4)

    recent_3 = df["broke_20d_high"].iloc[-3:].any()
    recent_7 = df["broke_20d_high"].iloc[-7:].any()
    if recent_3:
        breakout_pts = 20.0
    elif recent_7:
        breakout_pts = 10.0
    else:
        breakout_pts = 0.0

    return min(100.0, proximity_pts + breakout_pts)
