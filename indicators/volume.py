"""Volume scoring based on relative volume (vol_ratio)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def score(df: pd.DataFrame) -> float:
    """Return a 0-100 score based on volume ratio (current vs 20-bar SMA).

    * vol_ratio >= 2.0  → very high activity → 100
    * vol_ratio ~1.0    → average            →  50
    * vol_ratio <= 0.5  → low activity       →  10
    """
    if "vol_ratio" not in df.columns or df.empty:
        return 50.0

    vr = float(df["vol_ratio"].iloc[-1])
    if np.isnan(vr):
        return 50.0

    # Linear mapping:  0.5 → 10,  1.0 → 50,  2.0 → 100
    if vr <= 0.5:
        return 10.0
    elif vr <= 1.0:
        return 10.0 + (vr - 0.5) * 80.0   # 10 → 50
    elif vr <= 2.0:
        return 50.0 + (vr - 1.0) * 50.0   # 50 → 100
    else:
        return 100.0
