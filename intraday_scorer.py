"""
Intraday scoring engine.

Uses faster indicator settings than the swing scorer:
  RSI period 7  (vs 14 for swing)
  MACD 5/13/3   (vs 12/26/9 for swing)
  ATR period 7  (vs 14 for swing)

Trade setup: ATR-based, with a 1× multiplier (tighter than swing's 1.5×)
and a 2.5:1 target (slightly more aggressive than the 2:1 minimum).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.universe import ticker_display_name
from indicators import compute_intraday
from indicators.vwap import score as vwap_score
from indicators.supertrend import score as supertrend_score
from indicators.orb import score as orb_score
from indicators.volume import score as volume_score

MIN_BARS = 20  # much lower than swing — intraday has fewer bars per session


def _momentum_score_intraday(df: pd.DataFrame) -> float:
    """RSI(7) + MACD(5,13,3) momentum score."""
    if "rsi7" not in df.columns or len(df) < 5:
        return 50.0

    last = df.iloc[-1]
    rsi = float(last.get("rsi7", 50))
    hist = float(last.get("macd_fast_hist", 0))
    prev_hist = float(df["macd_fast_hist"].iloc[-2]) if len(df) > 1 else 0.0

    # RSI bell curve — intraday peak at ~60
    if np.isnan(rsi):
        rsi_s = 50.0
    elif rsi < 30:
        rsi_s = 60.0
    elif rsi < 50:
        rsi_s = 60.0 + (rsi - 30) * 1.5    # 60 → 90
    elif rsi < 65:
        rsi_s = 90.0 + (rsi - 50) * (10 / 15)  # 90 → 100
    elif rsi < 70:
        rsi_s = 100.0 - (rsi - 65) * 4         # 100 → 80
    else:
        rsi_s = max(0.0, 80.0 - (rsi - 70) * 5)

    # MACD grade
    if hist > 0 and hist >= prev_hist:
        macd_s = 100.0
    elif hist > 0:
        macd_s = 70.0
    elif hist > prev_hist:
        macd_s = 40.0
    else:
        macd_s = 10.0

    return rsi_s * 0.55 + macd_s * 0.45


def _composite(sub_scores: dict[str, float], weights: dict[str, float]) -> float:
    total_w = sum(weights.values())
    if total_w == 0:
        return 0.0
    return sum(sub_scores[k] * weights.get(k, 0) for k in sub_scores) / total_w


def compute_intraday_trade_setup(df: pd.DataFrame, atr_multiplier: float = 1.0) -> dict:
    """
    Intraday trade setup (tighter than swing):
      Stop  = entry − 1× ATR(7)
      T1    = entry + 1.5× risk
      T2    = entry + 2.5× risk
    """
    last = df.iloc[-1]
    atr_val = float(df["atr7"].dropna().iloc[-1]) if "atr7" in df.columns else float(last["Close"]) * 0.005
    entry = float(df["Close"].dropna().iloc[-1])
    risk = atr_multiplier * atr_val

    stop_loss = entry - risk
    target1 = entry + 1.5 * risk
    target2 = entry + 2.5 * risk

    return {
        "entry": entry,
        "stop_loss": stop_loss,
        "target1": target1,
        "target2": target2,
        "rr1": round(1.5, 1),
        "rr2": round(2.5, 1),
        "atr": atr_val,
    }


def score_single_intraday(
    ticker: str,
    df: pd.DataFrame,
    weights: dict[str, float],
    params: dict,
) -> dict | None:
    if df.empty or len(df) < MIN_BARS:
        return None

    interval_minutes = int(params.get("interval_minutes", 15))
    df = compute_intraday(df, interval_minutes=interval_minutes)

    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last

    sub_scores = {
        "vwap": vwap_score(df),
        "supertrend": supertrend_score(df),
        "momentum": _momentum_score_intraday(df),
        "volume": volume_score(df),
        "orb": orb_score(df),
    }

    composite = _composite(sub_scores, weights)
    setup = compute_intraday_trade_setup(df, params.get("atr_multiplier", 1.0))
    change_pct = (float(last["Close"]) - float(prev["Close"])) / max(float(prev["Close"]), 1e-9) * 100

    return {
        "ticker": ticker,
        "name": ticker_display_name(ticker),
        "price": round(float(last["Close"]), 2),
        "change_pct": round(change_pct, 2),
        "score": round(composite, 1),
        "vwap_score": round(sub_scores["vwap"], 1),
        "supertrend_score": round(sub_scores["supertrend"], 1),
        "momentum_score": round(sub_scores["momentum"], 1),
        "volume_score": round(sub_scores["volume"], 1),
        "orb_score": round(sub_scores["orb"], 1),
        # Key indicator values for display
        "rsi7": round(float(last.get("rsi7", np.nan)), 1),
        "vwap": round(float(last.get("vwap", np.nan)), 2),
        "vwap_pct": round(float(last.get("vwap_pct", np.nan)), 2),
        "st_direction": int(last.get("st_direction", 0)),
        "above_orb": int(last.get("above_orb", 0)),
        "vol_ratio": round(float(last.get("vol_ratio", np.nan)), 2),
        **{k: round(v, 2) if isinstance(v, float) else v for k, v in setup.items()},
    }


def score_universe_intraday(
    data: dict[str, pd.DataFrame],
    weights: dict[str, float],
    params: dict,
    progress_cb=None,
) -> list[dict]:
    results: list[dict] = []
    for ticker, df in data.items():
        try:
            row = score_single_intraday(ticker, df, weights, params)
            if row is not None:
                results.append(row)
        except Exception:
            pass
        if progress_cb:
            progress_cb()
    results.sort(key=lambda r: r["score"], reverse=True)
    return results
