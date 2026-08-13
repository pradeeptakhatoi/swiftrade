"""Tests for the intraday trade-simulation backtest."""

from __future__ import annotations

import pandas as pd

from dhan_algo.backtest import TradingCosts
from intraday_backtest import (
    IntradayBacktestResult,
    simulate_intraday,
    simulate_intraday_universe,
)


def _bars(closes: list[float], date: str = "2024-01-02") -> pd.DataFrame:
    """Build one session of 1-minute OHLCV bars from a close-price path."""
    rows = []
    start = pd.Timestamp(f"{date} 09:15:00")
    for i, c in enumerate(closes):
        o = closes[i - 1] if i > 0 else c
        rows.append({
            "Datetime": start + pd.Timedelta(minutes=i),
            "Open": o,
            "High": max(o, c) + 0.5,
            "Low": min(o, c) - 0.5,
            "Close": c,
            "Volume": 1000,
        })
    return pd.DataFrame(rows)


_PARAMS = {"interval_minutes": 1, "atr_multiplier": 1.0}


class TestSimulateIntraday:
    def test_rising_path_produces_winning_target_exits(self):
        closes = [100 + i * 0.5 for i in range(30)]
        res = simulate_intraday(
            "SYM", _bars(closes), params=_PARAMS, score_threshold=0.0,
        )
        assert res.total_trades > 0
        # A steadily rising market should hit targets, not stops.
        assert any(t.exit_reason == "target" for t in res.trades)
        assert res.gross_pnl > 0
        assert all(t.gross_pnl >= 0 for t in res.trades)

    def test_falling_path_produces_losing_stop_exits(self):
        closes = [100 - i * 0.5 for i in range(30)]
        res = simulate_intraday(
            "SYM", _bars(closes), params=_PARAMS, score_threshold=0.0,
        )
        assert res.total_trades > 0
        assert any(t.exit_reason == "stop" for t in res.trades)
        assert res.gross_pnl < 0

    def test_eod_square_off_when_levels_not_hit(self):
        # Huge ATR multiplier pushes stop/target far away, so a gently rising
        # path never touches either -> the position is squared off at EOD.
        closes = [100 + i * 0.1 for i in range(30)]
        res = simulate_intraday(
            "SYM", _bars(closes),
            params={"interval_minutes": 1, "atr_multiplier": 50.0},
            score_threshold=0.0, allow_reentry=False,
        )
        assert res.total_trades == 1
        assert res.trades[0].exit_reason == "eod"

    def test_no_signal_no_trades(self):
        closes = [100 + i * 0.5 for i in range(30)]
        # Impossibly high threshold -> never enters.
        res = simulate_intraday(
            "SYM", _bars(closes), params=_PARAMS, score_threshold=1000.0,
        )
        assert res.total_trades == 0
        assert res.net_pnl == 0.0

    def test_single_trade_when_reentry_disabled(self):
        closes = [100 - i * 0.5 for i in range(30)]
        res = simulate_intraday(
            "SYM", _bars(closes), params=_PARAMS, score_threshold=0.0,
            allow_reentry=False,
        )
        assert res.total_trades == 1

    def test_entry_price_is_a_bar_close_without_costs(self):
        closes = [round(100 + i * 0.5, 2) for i in range(30)]
        res = simulate_intraday(
            "SYM", _bars(closes), params=_PARAMS, score_threshold=0.0,
        )
        assert res.total_trades > 0
        # Frictionless: the entry fill equals the signal bar's close.
        close_set = set(closes)
        assert res.trades[0].entry_price in close_set

    def test_warmup_short_session_skipped(self):
        closes = [100 + i for i in range(10)]  # < MIN_BARS + 1
        res = simulate_intraday("SYM", _bars(closes), params=_PARAMS, score_threshold=0.0)
        assert res.total_trades == 0


class TestCosts:
    def test_costs_none_is_frictionless(self):
        closes = [100 + i * 0.5 for i in range(30)]
        res = simulate_intraday(
            "SYM", _bars(closes), params=_PARAMS, score_threshold=0.0, costs=None,
        )
        assert res.total_costs == 0.0
        assert res.net_pnl == res.gross_pnl
        assert all(t.cost == 0.0 for t in res.trades)

    def test_costs_reduce_net_pnl(self):
        closes = [100 + i * 0.5 for i in range(30)]
        gross_run = simulate_intraday(
            "SYM", _bars(closes), params=_PARAMS, score_threshold=0.0, costs=None,
        )
        cost_run = simulate_intraday(
            "SYM", _bars(closes), params=_PARAMS, score_threshold=0.0,
            costs=TradingCosts(), qty=10,
        )
        assert cost_run.total_costs > 0.0
        assert cost_run.net_pnl < cost_run.gross_pnl
        # Per-trade costs sum to the reported total.
        assert round(sum(t.cost for t in cost_run.trades), 2) == round(cost_run.total_costs, 2)
        assert gross_run.total_costs == 0.0


class TestResultStats:
    def test_equity_curve_and_summary(self):
        closes = [100 + i * 0.5 for i in range(30)]
        res = simulate_intraday(
            "SYM", _bars(closes), params=_PARAMS, score_threshold=0.0,
        )
        assert len(res.equity_curve) == res.total_trades
        assert res.max_drawdown >= 0.0
        summary = res.summary()
        assert "Intraday Simulation Summary" in summary
        assert "Win rate" in summary

    def test_win_rate_and_profit_factor_bounds(self):
        closes = [100 + i * 0.5 for i in range(30)]
        res = simulate_intraday(
            "SYM", _bars(closes), params=_PARAMS, score_threshold=0.0,
        )
        assert 0.0 <= res.win_rate <= 100.0
        assert res.profit_factor >= 0.0


class TestUniverse:
    def test_universe_merges_and_orders_trades(self):
        data = {
            "A": _bars([100 + i * 0.5 for i in range(30)], date="2024-01-02"),
            "B": _bars([200 - i * 0.5 for i in range(30)], date="2024-01-02"),
        }
        res = simulate_intraday_universe(data, params=_PARAMS, score_threshold=0.0)
        assert isinstance(res, IntradayBacktestResult)
        assert res.total_trades > 0
        # Trades ordered by entry time.
        times = [t.entry_time for t in res.trades]
        assert times == sorted(times)
