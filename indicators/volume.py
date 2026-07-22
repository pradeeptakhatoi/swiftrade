"""Volume scoring based on relative volume (vol_ratio)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def score(df: pd.DataFrame) -> float:
    """Intraday volume score 0-100 (current vs 20-bar SMA).

    * vol_ratio >= 2.0  → very high activity → 100
    * vol_ratio ~1.0    → average            →  50
    * vol_ratio <= 0.5  → low activity       →  10
    """
    if "vol_ratio" not in df.columns or df.empty:
        return 50.0

    vr = float(df["vol_ratio"].iloc[-1])
    if np.isnan(vr):
        return 50.0

    if vr <= 0.5:
        return 10.0
    elif vr <= 1.0:
        return 10.0 + (vr - 0.5) * 80.0
    elif vr <= 2.0:
        return 50.0 + (vr - 1.0) * 50.0
    else:
        return 100.0


def swing_score(df: pd.DataFrame) -> float:
    """Swing volume score 0-100.

    Maps vol_ratio linearly: 0.5x avg → 0, 2x avg → 100 (capped).
    Volume surge confirms smart money participation.
    """
    if len(df) < 20:
        return 0.0

    vr = float(df["vol_ratio"].iloc[-1])
    if np.isnan(vr):
        return 0.0

    return min(100.0, max(0.0, (vr - 0.5) * (100 / 1.5)))
