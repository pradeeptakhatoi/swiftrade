"""
Swing scoring engine.

Uses standard indicator settings:
  RSI period 14
  MACD 12/26/9
  ATR period 14

Trade setup: ATR-based, with a 1.5x multiplier and 2:1 / 3:1 targets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.universe import ticker_display_name
from indicators import compute_all
from indicators import trend, momentum, volatility, breakout
from indicators.volume import swing_score as volume_score

MIN_BARS = 220  # 200-EMA needs at least 200 bars


def _composite(sub_scores: dict[str, float], weights: dict[str, float]) -> float:
    total_w = sum(weights.values())
    if total_w == 0:
        return 0.0
    return sum(sub_scores[k] * weights.get(k, 0) for k in sub_scores) / total_w


def compute_trade_setup(df: pd.DataFrame, atr_multiplier: float = 1.5) -> dict:
    """Swing trade setup.

    Entry:    current close
    Stop:     entry - atr_multiplier x ATR(14)
    Target 1: entry + 2 x risk   (2:1 R:R)
    Target 2: entry + 3 x risk   (3:1 R:R)
    """
    last = df.iloc[-1]
    atr_val = float(last.get("atr", last["Close"] * 0.02))

    entry = float(last["Close"])
    risk = atr_multiplier * atr_val
    stop_loss = entry - risk
    target1 = entry + 2 * risk
    target2 = entry + 3 * risk

    return {
        "entry": entry,
        "stop_loss": stop_loss,
        "target1": target1,
        "target2": target2,
        "rr1": round((target1 - entry) / max(risk, 0.01), 2),
        "rr2": round((target2 - entry) / max(risk, 0.01), 2),
        "atr": atr_val,
    }


def score_single(
    ticker: str,
    df: pd.DataFrame,
    weights: dict[str, float],
    params: dict,
) -> dict | None:
    """Score one ticker for swing trading. Returns None if insufficient data."""
    if df.empty or len(df) < MIN_BARS:
        return None

    df = compute_all(df)
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last

    sub_scores = {
        "trend": trend.score(df),
        "momentum": momentum.score(df),
        "volume": volume_score(df),
        "breakout": breakout.score(df),
        "volatility": volatility.score(df),
    }

    composite = _composite(sub_scores, weights)
    setup = compute_trade_setup(df, params.get("atr_multiplier", 1.5))
    change_pct = (float(last["Close"]) - float(prev["Close"])) / max(float(prev["Close"]), 1e-9) * 100

    return {
        "ticker": ticker,
        "name": ticker_display_name(ticker),
        "price": round(float(last["Close"]), 2),
        "change_pct": round(change_pct, 2),
        "score": round(composite, 1),
        "trend_score": round(sub_scores["trend"], 1),
        "momentum_score": round(sub_scores["momentum"], 1),
        "volume_score": round(sub_scores["volume"], 1),
        "breakout_score": round(sub_scores["breakout"], 1),
        "volatility_score": round(sub_scores["volatility"], 1),
        # Key indicator values
        "rsi": round(float(last.get("rsi", np.nan)), 1),
        "macd_hist": round(float(last.get("macd_hist", np.nan)), 4),
        "vol_ratio": round(float(last.get("vol_ratio", np.nan)), 2),
        "pct_from_52wk": round(float(last.get("pct_from_52wk_high", np.nan)), 1),
        "atr_pct": round(float(last.get("atr_pct", np.nan)), 2),
        "pct_b": round(float(last.get("pct_b", np.nan)), 2),
        **{k: round(v, 2) if isinstance(v, float) else v for k, v in setup.items()},
    }


def score_universe(
    data: dict[str, pd.DataFrame],
    weights: dict[str, float],
    params: dict,
    progress_cb=None,
) -> list[dict]:
    """Score all tickers, return list sorted descending by composite score."""
    results: list[dict] = []
    for ticker, df in data.items():
        try:
            row = score_single(ticker, df, weights, params)
            if row is not None:
                results.append(row)
        except Exception:
            pass
        if progress_cb:
            progress_cb()
    results.sort(key=lambda r: r["score"], reverse=True)
    return results
