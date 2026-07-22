"""SuperTrend direction scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd


def score(df: pd.DataFrame) -> float:
    """Return a 0-100 score based on SuperTrend direction and recency.

    * Bullish direction (1)  with recent flip → 100
    * Bullish direction (1)  sustained        →  80
    * Bearish direction (-1) sustained        →  20
    * Bearish direction (-1) with recent flip →   0
    """
    if "st_direction" not in df.columns or df.empty:
        return 50.0

    direction = int(df["st_direction"].iloc[-1])

    # Check for a recent direction change (last 3 bars)
    if len(df) >= 3:
        recent = df["st_direction"].iloc[-3:]
        flipped = recent.nunique() > 1
    else:
        flipped = False

    if direction == 1:
        return 100.0 if flipped else 80.0
    else:
        return 0.0 if flipped else 20.0
