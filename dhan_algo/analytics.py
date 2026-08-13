"""Performance analytics over a set of closed (round-trip) trades.

Pure functions that take a :class:`pandas.DataFrame` of closed trades and return
summary metrics, an equity curve, an R-multiple distribution, and per-group
breakdowns. They operate on the same columns the trade simulations already
produce (``net_pnl``, ``gross_pnl``, ``cost``, ``r_multiple``, ``entry_time``,
``exit_time``, ``exit_reason``, ``symbol``, ``strategy``), so the identical code
analyses a live simulation result and a persisted performance log.

Everything here is read-only and side-effect free; nothing touches the broker,
the network, or disk.
"""

from __future__ import annotations

import pandas as pd

# Columns the analytics rely on. Missing optional columns are tolerated.
_REQUIRED = ("net_pnl",)


def _empty_metrics() -> dict[str, float]:
    return {
        "trades": 0,
        "wins": 0,
        "losses": 0,
        "win_rate": 0.0,
        "gross_pnl": 0.0,
        "total_costs": 0.0,
        "net_pnl": 0.0,
        "expectancy": 0.0,
        "avg_r": 0.0,
        "avg_win": 0.0,
        "avg_loss": 0.0,
        "profit_factor": 0.0,
        "max_drawdown": 0.0,
    }


def _ordered(df: pd.DataFrame) -> pd.DataFrame:
    """Return trades ordered by exit time (falling back to entry time)."""
    for col in ("exit_time", "entry_time"):
        if col in df.columns:
            return df.assign(_k=pd.to_datetime(df[col], errors="coerce")).sort_values(
                "_k", kind="stable"
            ).drop(columns="_k")
    return df


def compute_metrics(df: pd.DataFrame) -> dict[str, float]:
    """Summary performance statistics for a set of closed trades.

    Mirrors the definitions used by the backtest result objects so numbers are
    consistent across the app: win rate, expectancy (avg net P&L per trade),
    profit factor (gross profit / gross loss), average R-multiple, and the
    max peak-to-trough drawdown of the cumulative net-P&L curve.
    """
    if df is None or df.empty or "net_pnl" not in df.columns:
        return _empty_metrics()

    net = pd.to_numeric(df["net_pnl"], errors="coerce").fillna(0.0)
    trades = int(len(net))
    wins_mask = net > 0
    loss_mask = net < 0
    wins = int(wins_mask.sum())
    losses = int(loss_mask.sum())

    gross_profit = float(net[wins_mask].sum())
    gross_loss = float(-net[loss_mask].sum())
    if gross_loss == 0:
        profit_factor = float("inf") if gross_profit > 0 else 0.0
    else:
        profit_factor = gross_profit / gross_loss

    gross_pnl = (
        float(pd.to_numeric(df["gross_pnl"], errors="coerce").fillna(0.0).sum())
        if "gross_pnl" in df.columns
        else float(net.sum())
    )
    total_costs = (
        float(pd.to_numeric(df["cost"], errors="coerce").fillna(0.0).sum())
        if "cost" in df.columns
        else 0.0
    )
    avg_r = (
        float(pd.to_numeric(df["r_multiple"], errors="coerce").fillna(0.0).mean())
        if "r_multiple" in df.columns
        else 0.0
    )

    curve = _ordered(df)
    running = pd.to_numeric(curve["net_pnl"], errors="coerce").fillna(0.0).cumsum()
    peak = running.cummax()
    max_drawdown = float((peak - running).max()) if trades else 0.0

    return {
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "win_rate": 100.0 * wins / trades if trades else 0.0,
        "gross_pnl": gross_pnl,
        "total_costs": total_costs,
        "net_pnl": float(net.sum()),
        "expectancy": float(net.sum() / trades) if trades else 0.0,
        "avg_r": avg_r,
        "avg_win": gross_profit / wins if wins else 0.0,
        "avg_loss": gross_loss / losses if losses else 0.0,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
    }


def equity_curve(df: pd.DataFrame) -> pd.DataFrame:
    """Cumulative net P&L after each trade, in chronological (exit) order.

    Returns a DataFrame with a 1-based ``trade`` index and an ``equity`` column
    suitable for ``st.line_chart``.
    """
    if df is None or df.empty or "net_pnl" not in df.columns:
        return pd.DataFrame({"trade": [], "equity": []})
    ordered = _ordered(df)
    running = pd.to_numeric(ordered["net_pnl"], errors="coerce").fillna(0.0).cumsum()
    return pd.DataFrame(
        {"trade": range(1, len(running) + 1), "equity": running.to_numpy()}
    )


def r_distribution(df: pd.DataFrame, bin_width: float = 0.5) -> pd.DataFrame:
    """Histogram of realised R-multiples bucketed by *bin_width*.

    Returns a DataFrame with a ``bucket`` label column and a ``count`` column.
    Empty when there is no ``r_multiple`` data.
    """
    if (
        df is None
        or df.empty
        or "r_multiple" not in df.columns
        or bin_width <= 0
    ):
        return pd.DataFrame({"bucket": [], "count": []})

    r = pd.to_numeric(df["r_multiple"], errors="coerce").dropna()
    if r.empty:
        return pd.DataFrame({"bucket": [], "count": []})

    # Snap each R to the floor of its bin so buckets are stable, ordered labels.
    import math

    lo = math.floor(float(r.min()) / bin_width) * bin_width
    hi = math.floor(float(r.max()) / bin_width) * bin_width
    edges: list[float] = []
    e = lo
    while e <= hi + 1e-9:
        edges.append(round(e, 6))
        e += bin_width

    counts: list[int] = []
    labels: list[str] = []
    for edge in edges:
        in_bin = ((r >= edge) & (r < edge + bin_width)).sum()
        labels.append(f"[{edge:g}, {edge + bin_width:g})")
        counts.append(int(in_bin))
    return pd.DataFrame({"bucket": labels, "count": counts})


def breakdown(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Per-group performance table.

    *by* is a column name (e.g. ``symbol``, ``strategy``, ``exit_reason``) or the
    synthetic key ``hour`` (hour-of-day parsed from ``entry_time``). Each row
    carries trade count, net P&L, win rate, and average R for that group,
    sorted by net P&L descending.
    """
    if df is None or df.empty or "net_pnl" not in df.columns:
        return pd.DataFrame(
            {"group": [], "trades": [], "net_pnl": [], "win_rate": [], "avg_r": []}
        )

    work = df.copy()
    if by == "hour":
        work["group"] = (
            pd.to_datetime(work.get("entry_time"), errors="coerce").dt.hour
        )
    elif by in work.columns:
        work["group"] = work[by]
    else:
        return pd.DataFrame(
            {"group": [], "trades": [], "net_pnl": [], "win_rate": [], "avg_r": []}
        )

    work["net_pnl"] = pd.to_numeric(work["net_pnl"], errors="coerce").fillna(0.0)
    work["_win"] = work["net_pnl"] > 0
    if "r_multiple" in work.columns:
        work["_r"] = pd.to_numeric(work["r_multiple"], errors="coerce").fillna(0.0)
    else:
        work["_r"] = 0.0

    rows: list[dict[str, object]] = []
    for key, grp in work.groupby("group", dropna=False):
        n = int(len(grp))
        rows.append(
            {
                "group": key,
                "trades": n,
                "net_pnl": round(float(grp["net_pnl"].sum()), 2),
                "win_rate": round(100.0 * int(grp["_win"].sum()) / n, 1) if n else 0.0,
                "avg_r": round(float(grp["_r"].mean()), 2) if n else 0.0,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("net_pnl", ascending=False, kind="stable").reset_index(
            drop=True
        )
    return out
