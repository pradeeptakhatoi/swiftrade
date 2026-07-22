"""Opening Range Breakout (ORB) scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd


def score(df: pd.DataFrame) -> float:
    """Return a 0-100 score based on ORB status.

    * Price above opening-range high → bullish breakout → high score
    * Price inside the range         → neutral          → mid score
    * Price below opening-range low  → bearish breakdown→ low score
    """
    if "above_orb" not in df.columns or df.empty:
        return 50.0

    above = int(df["above_orb"].iloc[-1])
    below = int(df.get("below_orb", pd.Series([0])).iloc[-1])

    if above:
        # Sustained breakout (last 3 bars all above) is stronger
        if len(df) >= 3 and df["above_orb"].iloc[-3:].sum() == 3:
            return 100.0
        return 85.0
    elif below:
        if len(df) >= 3 and df.get("below_orb") is not None and df["below_orb"].iloc[-3:].sum() == 3:
            return 0.0
        return 15.0
    else:
        return 50.0
