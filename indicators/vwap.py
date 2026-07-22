"""VWAP (Volume-Weighted Average Price) scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd


def score(df: pd.DataFrame) -> float:
    """Return a 0-100 score based on price position relative to VWAP.

    * Price well above VWAP  → bullish momentum  → high score
    * Price near VWAP        → neutral            → mid score
    * Price below VWAP       → bearish            → low score
    """
    if "vwap_pct" not in df.columns or df.empty:
        return 50.0

    vwap_pct = float(df["vwap_pct"].iloc[-1])
    if np.isnan(vwap_pct):
        return 50.0

    # Score mapping:  -2% → 0,  0% → 50,  +2% → 100
    s = 50.0 + vwap_pct * 25.0
    return max(0.0, min(100.0, s))
