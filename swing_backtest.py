"""Swing trade-simulation backtest.

Replays daily bars through the same signal logic used by the live swing
scanner and simulates the resulting trades so you can measure *net-of-cost*
expectancy — not just a static score.

Design (long-only, matching the scanner's long-biased signals):

* The simulation walks the daily series bar by bar. At each bar, while flat,
  the swing composite score is recomputed on the prefix ``0..i`` only (no
  look-ahead) and, if it is at or above ``score_threshold``, a long is entered
  at that bar's close. Stop and targets come from the scanner's ATR-based
  setup (``swing_scorer.compute_trade_setup``).
* While in a position, each subsequent daily bar is checked for a stop or
  target touch (stop assumed first if both are touched in one bar). Unlike the
  intraday sim, swing trades are **held across days**; an optional
  ``max_hold_bars`` caps the holding period. Any position still open on the
  last available bar is closed at that bar's close.
* ``TradingCosts`` (from :mod:`dhan_algo.backtest`) applies adverse slippage to
  fills and charges brokerage/taxes/fees on both legs, so reported P&L is net
  of realistic friction. With ``costs=None`` the run is frictionless. Swing
  trades default to the ``CNC`` (delivery) product.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import pandas as pd

from dhan_algo.backtest import TradingCosts
from exit_rules import ExitConfig, finalize_long, open_long, step_long
from indicators import compute_all
from indicators import breakout, momentum, trend, volatility
from indicators.volume import swing_score as volume_score
from swing_scorer import MIN_BARS, _composite, compute_trade_setup

DEFAULT_WEIGHTS = {
    "trend": 1.0,
    "momentum": 1.0,
    "volume": 0.8,
    "breakout": 0.8,
    "volatility": 0.5,
}


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class SwingTrade:
    symbol: str
    entry_time: str
    exit_time: str
    qty: int
    entry_price: float
    exit_price: float
    stop: float
    target: float
    exit_reason: str  # "stop" | "target" | "time" | "end"
    bars_held: int
    gross_pnl: float
    cost: float
    net_pnl: float
    r_multiple: float


@dataclass
class SwingBacktestResult:
    trades: list[SwingTrade] = field(default_factory=list)

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
    def avg_bars_held(self) -> float:
        return sum(t.bars_held for t in self.trades) / self.total_trades if self.total_trades else 0.0

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
            "=== Swing Simulation Summary ===",
            f"Trades       : {self.total_trades}",
            f"Wins / Losses: {self.wins} / {self.losses}",
            f"Win rate     : {self.win_rate:.1f}%",
            f"Gross P&L    : {self.gross_pnl:,.2f}",
            f"Costs        : {self.total_costs:,.2f}",
            f"Net P&L      : {self.net_pnl:,.2f}",
            f"Avg R        : {self.avg_r:.2f}",
            f"Avg bars held: {self.avg_bars_held:.1f}",
            f"Expectancy   : {self.expectancy:,.2f} / trade",
            f"Profit factor: {pf_str}",
            f"Max drawdown : {self.max_drawdown:,.2f}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row_time(row: pd.Series, idx: int) -> str:
    for col in ("Datetime", "Date", "timestamp"):
        if col in row.index and pd.notna(row.get(col)):
            return str(row[col])
    return str(idx)


def _eval_prefix(
    df_slice: pd.DataFrame, weights: dict[str, float]
) -> tuple[float, pd.DataFrame] | None:
    """Composite swing score for the last bar of *df_slice* plus computed df.

    Recomputes indicators on the prefix so nothing after the current bar leaks
    into the score (no look-ahead). Returns ``None`` if there is not enough
    history.
    """
    if len(df_slice) < MIN_BARS:
        return None
    computed = compute_all(df_slice)
    sub_scores = {
        "trend": trend.score(computed),
        "momentum": momentum.score(computed),
        "volume": volume_score(computed),
        "breakout": breakout.score(computed),
        "volatility": volatility.score(computed),
    }
    return _composite(sub_scores, weights), computed


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def simulate_swing(
    symbol: str,
    df: pd.DataFrame,
    weights: dict[str, float] | None = None,
    params: dict[str, Any] | None = None,
    *,
    score_threshold: float = 60.0,
    qty: int = 1,
    costs: TradingCosts | None = None,
    product: str = "CNC",
    max_hold_bars: int | None = None,
    allow_reentry: bool = True,
    exit_config: ExitConfig | None = None,
) -> SwingBacktestResult:
    """Simulate long-only swing trades for one symbol on daily bars.

    *df* holds daily OHLCV bars (columns ``Open, High, Low, Close, Volume``)
    plus a ``Date``/``Datetime`` column or a datetime index for timestamps.

    *exit_config* enables optional exit-management rules (break-even, trailing
    stop, partial profit, time stop). When omitted, only the plain ATR
    stop/target and the ``max_hold_bars`` time cap apply.
    """
    weights = weights or dict(DEFAULT_WEIGHTS)
    params = params or {}
    atr_multiplier = float(params.get("atr_multiplier", 1.5))

    cfg = exit_config or ExitConfig()
    if cfg.max_hold_bars is None and max_hold_bars is not None:
        cfg = replace(cfg, max_hold_bars=max_hold_bars)

    result = SwingBacktestResult()
    df = df.reset_index() if not isinstance(df.index, pd.RangeIndex) else df.copy()
    df = df.reset_index(drop=True)
    n = len(df)
    if n < MIN_BARS + 1:
        return result

    highs = df["High"].astype(float).tolist()
    lows = df["Low"].astype(float).tolist()
    closes = df["Close"].astype(float).tolist()
    last_idx = n - 1

    pos: dict[str, Any] | None = None
    traded = False

    for i in range(n):
        # 1) Manage an open position on this bar.
        if pos is not None:
            closed, reason = step_long(
                pos, highs[i], lows[i], closes[i], i,
                is_last=(i == last_idx), cfg=cfg, end_reason="end",
            )
            if closed:
                result.trades.append(
                    _close_trade(
                        symbol, pos, reason, _row_time(df.iloc[i], i), i,
                        qty, costs, product,
                    )
                )
                pos = None

        # 2) Consider a new entry (flat, not the last bar, warmup met).
        if pos is None and i < last_idx and (i + 1) >= MIN_BARS:
            if allow_reentry or not traded:
                evaluated = _eval_prefix(df.iloc[: i + 1], weights)
                if evaluated is not None:
                    score, computed = evaluated
                    if score >= score_threshold:
                        setup = compute_trade_setup(computed, atr_multiplier)
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
                                entry_time=_row_time(df.iloc[i], i),
                                qty=qty,
                                entry_charge=(
                                    costs.total("BUY", entry_fill, qty, product) if costs else 0.0
                                ),
                            )
                            traded = True

    return result


def _close_trade(
    symbol: str,
    pos: dict[str, Any],
    reason: str,
    exit_time: str,
    exit_idx: int,
    qty: int,
    costs: TradingCosts | None,
    product: str,
) -> SwingTrade:
    pnl = finalize_long(pos, costs, product)
    return SwingTrade(
        symbol=symbol,
        entry_time=pos["entry_time"],
        exit_time=exit_time,
        qty=qty,
        entry_price=round(pos["entry_fill"], 2),
        exit_price=round(pnl["avg_exit"], 2),
        stop=round(pos["initial_stop"], 2),
        target=round(pos["target"], 2),
        exit_reason=reason,
        bars_held=exit_idx - pos["entry_idx"],
        gross_pnl=round(pnl["gross"], 2),
        cost=round(pnl["cost"], 2),
        net_pnl=round(pnl["net"], 2),
        r_multiple=round(pnl["r_multiple"], 2),
    )


def simulate_swing_universe(
    data: dict[str, pd.DataFrame],
    weights: dict[str, float] | None = None,
    params: dict[str, Any] | None = None,
    *,
    score_threshold: float = 60.0,
    qty: int = 1,
    costs: TradingCosts | None = None,
    product: str = "CNC",
    max_hold_bars: int | None = None,
    allow_reentry: bool = True,
    exit_config: ExitConfig | None = None,
    progress_cb=None,
) -> SwingBacktestResult:
    """Run :func:`simulate_swing` over many symbols and merge the trades.

    Trades are ordered by entry time so the aggregate equity curve and drawdown
    reflect the true chronological sequence.
    """
    merged = SwingBacktestResult()
    for symbol, df in data.items():
        try:
            res = simulate_swing(
                symbol, df, weights, params,
                score_threshold=score_threshold, qty=qty,
                costs=costs, product=product,
                max_hold_bars=max_hold_bars, allow_reentry=allow_reentry,
                exit_config=exit_config,
            )
            merged.trades.extend(res.trades)
        except Exception:
            pass
        if progress_cb:
            progress_cb()
    merged.trades.sort(key=lambda t: t.entry_time)
    return merged
