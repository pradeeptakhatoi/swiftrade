"""Intraday trade-simulation backtest.

Replays intraday bars through the same signal logic used by the live intraday
scanner and simulates the resulting trades so you can measure *net-of-cost*
expectancy — not just a static score.

Design (long-only, matching the scanner's long-biased signals):

* The simulation runs **per trading day**. ``indicators.compute_intraday``
  builds VWAP/ORB as intra-session state (cumulative VWAP, opening-range from
  the first bars), so each session is scored independently — exactly what the
  live scanner sees on a single day. This also avoids look-ahead: at bar *i*
  the score is computed only from bars ``0..i`` of that day.
* On each bar, while flat, the intraday composite score is computed. If it is
  at or above ``score_threshold`` a long is entered at that bar's close
  (the signal bar). Stop and target come from the scanner's ATR-based setup.
* While in a position, each subsequent bar is checked for a stop or target
  touch (stop assumed first if both are touched in one bar). Any position
  still open on the last bar of the session is squared off at that bar's close
  (intraday end-of-day exit).
* ``TradingCosts`` (from :mod:`dhan_algo.backtest`) applies adverse slippage to
  fills and charges brokerage/taxes/fees on both legs, so reported P&L is net
  of realistic friction. With ``costs=None`` the run is frictionless.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

import pandas as pd

from dhan_algo.backtest import TradingCosts
from exit_rules import ExitConfig, finalize_long, open_long, step_long
from indicators import compute_intraday
from indicators.orb import score as orb_score
from indicators.supertrend import score as supertrend_score
from indicators.volume import score as volume_score
from indicators.vwap import score as vwap_score
from intraday_scorer import (
    MIN_BARS,
    _composite,
    _momentum_score_intraday,
    compute_intraday_trade_setup,
)

DEFAULT_WEIGHTS = {
    "vwap": 1.0,
    "supertrend": 1.0,
    "momentum": 1.5,
    "volume": 0.5,
    "orb": 1.0,
}


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class IntradayTrade:
    symbol: str
    entry_time: str
    exit_time: str
    qty: int
    entry_price: float
    exit_price: float
    stop: float
    target: float
    exit_reason: str  # "stop" | "target" | "eod"
    gross_pnl: float
    cost: float
    net_pnl: float
    r_multiple: float


@dataclass
class IntradayBacktestResult:
    trades: list[IntradayTrade] = field(default_factory=list)

    # --- aggregate stats -------------------------------------------------
    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def wins(self) -> int:
        return sum(1 for t in self.trades if t.net_pnl > 0)

    @property
    def losses(self) -> int:
        return sum(1 for t in self.trades if t.net_pnl < 0)

    @property
    def win_rate(self) -> float:
        return 100.0 * self.wins / self.total_trades if self.total_trades else 0.0

    @property
    def gross_pnl(self) -> float:
        return sum(t.gross_pnl for t in self.trades)

    @property
    def total_costs(self) -> float:
        return sum(t.cost for t in self.trades)

    @property
    def net_pnl(self) -> float:
        return sum(t.net_pnl for t in self.trades)

    @property
    def avg_r(self) -> float:
        return sum(t.r_multiple for t in self.trades) / self.total_trades if self.total_trades else 0.0

    @property
    def expectancy(self) -> float:
        """Average net P&L per trade."""
        return self.net_pnl / self.total_trades if self.total_trades else 0.0

    @property
    def profit_factor(self) -> float:
        """Gross profit / gross loss (net of costs). ``inf`` if no losers."""
        gains = sum(t.net_pnl for t in self.trades if t.net_pnl > 0)
        loss = sum(-t.net_pnl for t in self.trades if t.net_pnl < 0)
        if loss == 0:
            return float("inf") if gains > 0 else 0.0
        return gains / loss

    @property
    def equity_curve(self) -> list[float]:
        """Cumulative net P&L after each trade (entry-time ordered)."""
        curve: list[float] = []
        running = 0.0
        for t in self.trades:
            running += t.net_pnl
            curve.append(running)
        return curve

    @property
    def max_drawdown(self) -> float:
        """Largest peak-to-trough drop of the equity curve (>= 0)."""
        peak = 0.0
        max_dd = 0.0
        running = 0.0
        for t in self.trades:
            running += t.net_pnl
            peak = max(peak, running)
            max_dd = max(max_dd, peak - running)
        return max_dd

    def summary(self) -> str:
        pf = self.profit_factor
        pf_str = "inf" if pf == float("inf") else f"{pf:.2f}"
        lines = [
            "=== Intraday Simulation Summary ===",
            f"Trades       : {self.total_trades}",
            f"Wins / Losses: {self.wins} / {self.losses}",
            f"Win rate     : {self.win_rate:.1f}%",
            f"Gross P&L    : {self.gross_pnl:,.2f}",
            f"Costs        : {self.total_costs:,.2f}",
            f"Net P&L      : {self.net_pnl:,.2f}",
            f"Avg R        : {self.avg_r:.2f}",
            f"Expectancy   : {self.expectancy:,.2f} / trade",
            f"Profit factor: {pf_str}",
            f"Max drawdown : {self.max_drawdown:,.2f}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------


def _session_dates(df: pd.DataFrame) -> pd.Series | None:
    """Return a per-row session-date series, or ``None`` if not derivable."""
    for col in ("Datetime", "Date", "timestamp"):
        if col in df.columns:
            ts = pd.to_datetime(df[col], errors="coerce")
            if ts.notna().any():
                return ts.dt.date
    if isinstance(df.index, pd.DatetimeIndex):
        return pd.Series(df.index.date, index=df.index)
    return None


def _row_time(row: pd.Series, idx: int) -> str:
    for col in ("Datetime", "Date", "timestamp"):
        if col in row.index and pd.notna(row.get(col)):
            return str(row[col])
    return str(idx)


def _iter_sessions(df: pd.DataFrame) -> Iterator[pd.DataFrame]:
    """Yield one DataFrame per trading session, oldest first."""
    dates = _session_dates(df)
    if dates is None:
        yield df.reset_index(drop=True)
        return
    work = df.copy()
    work["_session"] = list(dates)
    for _, group in work.groupby("_session", sort=True):
        yield group.drop(columns=["_session"]).reset_index(drop=True)


def _score_prefix(df_slice: pd.DataFrame, weights: dict[str, float], interval_minutes: int) -> float | None:
    """Composite intraday score for the last bar of *df_slice*.

    Recomputes indicators on the prefix so nothing after the current bar leaks
    into the score (no look-ahead). Returns ``None`` if there is not enough data.
    """
    if len(df_slice) < MIN_BARS:
        return None
    computed = compute_intraday(df_slice, interval_minutes=interval_minutes)
    sub_scores = {
        "vwap": vwap_score(computed),
        "supertrend": supertrend_score(computed),
        "momentum": _momentum_score_intraday(computed),
        "volume": volume_score(computed),
        "orb": orb_score(computed),
    }
    return _composite(sub_scores, weights)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def simulate_intraday(
    symbol: str,
    df: pd.DataFrame,
    weights: dict[str, float] | None = None,
    params: dict[str, Any] | None = None,
    *,
    score_threshold: float = 70.0,
    qty: int = 1,
    costs: TradingCosts | None = None,
    product: str = "INTRA",
    allow_reentry: bool = True,
    exit_config: ExitConfig | None = None,
) -> IntradayBacktestResult:
    """Simulate long-only intraday trades for one symbol.

    *df* holds intraday OHLCV bars (columns ``Open, High, Low, Close, Volume``)
    plus a datetime column (``Datetime``/``Date``) or index used to split days.

    *exit_config* enables optional exit-management rules (break-even, trailing
    stop, partial profit, time stop) applied within each session; when omitted
    only the plain ATR stop/target and end-of-day square-off apply.
    """
    weights = weights or dict(DEFAULT_WEIGHTS)
    params = params or {}
    interval_minutes = int(params.get("interval_minutes", 15))
    atr_multiplier = float(params.get("atr_multiplier", 1.0))
    cfg = exit_config or ExitConfig()

    result = IntradayBacktestResult()

    for session in _iter_sessions(df):
        n = len(session)
        if n < MIN_BARS + 1:
            continue

        highs = session["High"].astype(float).tolist()
        lows = session["Low"].astype(float).tolist()
        closes = session["Close"].astype(float).tolist()
        last_idx = n - 1

        pos: dict[str, Any] | None = None
        traded_today = False

        for i in range(n):
            # 1) Manage an open position on this bar.
            if pos is not None:
                closed, reason = step_long(
                    pos, highs[i], lows[i], closes[i], i,
                    is_last=(i == last_idx), cfg=cfg, end_reason="eod",
                )
                if closed:
                    result.trades.append(
                        _close_trade(
                            symbol, pos, reason,
                            _row_time(session.iloc[i], i), qty, costs, product,
                        )
                    )
                    pos = None

            # 2) Consider a new entry (flat, not the last bar, warmup met).
            if pos is None and i < last_idx and (i + 1) >= MIN_BARS:
                if allow_reentry or not traded_today:
                    score = _score_prefix(session.iloc[: i + 1], weights, interval_minutes)
                    if score is not None and score >= score_threshold:
                        setup = compute_intraday_trade_setup(
                            compute_intraday(session.iloc[: i + 1], interval_minutes=interval_minutes),
                            atr_multiplier,
                        )
                        entry_raw = setup["entry"]
                        stop = setup["stop_loss"]
                        target = setup["target1"]
                        if stop < entry_raw:  # valid long risk
                            entry_fill = (
                                costs.slippage_price("BUY", entry_raw) if costs else entry_raw
                            )
                            pos = open_long(
                                entry_fill=entry_fill,
                                stop=stop,
                                target=target,
                                entry_idx=i,
                                entry_time=_row_time(session.iloc[i], i),
                                qty=qty,
                                entry_charge=(
                                    costs.total("BUY", entry_fill, qty, product) if costs else 0.0
                                ),
                            )
                            traded_today = True

    return result


def _close_trade(
    symbol: str,
    pos: dict[str, Any],
    reason: str,
    exit_time: str,
    qty: int,
    costs: TradingCosts | None,
    product: str,
) -> IntradayTrade:
    pnl = finalize_long(pos, costs, product)
    return IntradayTrade(
        symbol=symbol,
        entry_time=pos["entry_time"],
        exit_time=exit_time,
        qty=qty,
        entry_price=round(pos["entry_fill"], 2),
        exit_price=round(pnl["avg_exit"], 2),
        stop=round(pos["initial_stop"], 2),
        target=round(pos["target"], 2),
        exit_reason=reason,
        gross_pnl=round(pnl["gross"], 2),
        cost=round(pnl["cost"], 2),
        net_pnl=round(pnl["net"], 2),
        r_multiple=round(pnl["r_multiple"], 2),
    )


def simulate_intraday_universe(
    data: dict[str, pd.DataFrame],
    weights: dict[str, float] | None = None,
    params: dict[str, Any] | None = None,
    *,
    score_threshold: float = 70.0,
    qty: int = 1,
    costs: TradingCosts | None = None,
    product: str = "INTRA",
    allow_reentry: bool = True,
    exit_config: ExitConfig | None = None,
    progress_cb=None,
) -> IntradayBacktestResult:
    """Run :func:`simulate_intraday` over many symbols and merge the trades.

    Trades are ordered by entry time so the aggregate equity curve and drawdown
    reflect the true chronological sequence.
    """
    merged = IntradayBacktestResult()
    for symbol, df in data.items():
        try:
            res = simulate_intraday(
                symbol, df, weights, params,
                score_threshold=score_threshold, qty=qty,
                costs=costs, product=product, allow_reentry=allow_reentry,
                exit_config=exit_config,
            )
            merged.trades.extend(res.trades)
        except Exception:
            pass
        if progress_cb:
            progress_cb()
    merged.trades.sort(key=lambda t: t.entry_time)
    return merged
