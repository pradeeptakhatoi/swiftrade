"""Tests for the swing trade-simulation backtest."""

from __future__ import annotations

import pandas as pd

from dhan_algo.backtest import TradingCosts
from exit_rules import ExitConfig
from swing_backtest import (
    SwingBacktestResult,
    simulate_swing,
    simulate_swing_universe,
)
from swing_scorer import MIN_BARS


def _bars(closes: list[float], start: str = "2022-01-03") -> pd.DataFrame:
    """Build daily OHLCV bars from a close-price path (business days)."""
    rows = []
    dates = pd.bdate_range(start=start, periods=len(closes))
    for i, c in enumerate(closes):
        o = closes[i - 1] if i > 0 else c
        rows.append({
            "Date": dates[i],
            "Open": o,
            "High": max(o, c) + 0.5,
            "Low": min(o, c) - 0.5,
            "Close": c,
            "Volume": 100000,
        })
    return pd.DataFrame(rows)


# Enough history to clear the 200-EMA warmup plus room to trade.
_N = MIN_BARS + 40
_PARAMS = {"atr_multiplier": 1.0}


class TestSimulateSwing:
    def test_rising_path_produces_winning_target_exits(self):
        closes = [100 + i * 0.5 for i in range(_N)]
        res = simulate_swing("SYM", _bars(closes), params=_PARAMS, score_threshold=0.0)
        assert res.total_trades > 0
        assert any(t.exit_reason == "target" for t in res.trades)
        assert res.gross_pnl > 0
        assert all(t.gross_pnl >= 0 for t in res.trades)

    def test_falling_path_produces_losing_stop_exits(self):
        # Rise for exactly the warmup window so the (only) entry lands on the
        # first eligible bar, then fall hard so that position is stopped out.
        rise = [100 + i * 0.5 for i in range(MIN_BARS)]
        fall = [rise[-1] - (i + 1) * 8.0 for i in range(10)]
        res = simulate_swing(
            "SYM", _bars(rise + fall), params=_PARAMS, score_threshold=0.0,
            allow_reentry=False,
        )
        assert res.total_trades == 1
        assert res.trades[0].exit_reason == "stop"
        assert res.trades[0].gross_pnl < 0

    def test_end_of_data_square_off_when_levels_not_hit(self):
        # Huge ATR multiplier pushes stop/target far away, so a gently rising
        # path never touches either -> the open position closes on the last bar.
        closes = [100 + i * 0.1 for i in range(_N)]
        res = simulate_swing(
            "SYM", _bars(closes), params={"atr_multiplier": 100.0},
            score_threshold=0.0, allow_reentry=False,
        )
        assert res.total_trades == 1
        assert res.trades[0].exit_reason == "end"

    def test_max_hold_bars_forces_time_exit(self):
        # Wide levels + short max hold -> the position is closed on the time cap.
        closes = [100 + i * 0.1 for i in range(_N)]
        res = simulate_swing(
            "SYM", _bars(closes), params={"atr_multiplier": 100.0},
            score_threshold=0.0, allow_reentry=False, max_hold_bars=3,
        )
        assert res.total_trades == 1
        assert res.trades[0].exit_reason == "time"
        assert res.trades[0].bars_held == 3

    def test_no_signal_no_trades(self):
        closes = [100 + i * 0.5 for i in range(_N)]
        res = simulate_swing(
            "SYM", _bars(closes), params=_PARAMS, score_threshold=1000.0,
        )
        assert res.total_trades == 0
        assert res.net_pnl == 0.0

    def test_single_trade_when_reentry_disabled(self):
        closes = [100 + i * 0.5 for i in range(_N)]
        res = simulate_swing(
            "SYM", _bars(closes), params=_PARAMS, score_threshold=0.0,
            allow_reentry=False,
        )
        assert res.total_trades == 1

    def test_warmup_short_series_skipped(self):
        closes = [100 + i for i in range(MIN_BARS - 5)]  # < MIN_BARS + 1
        res = simulate_swing("SYM", _bars(closes), params=_PARAMS, score_threshold=0.0)
        assert res.total_trades == 0


class TestCosts:
    def test_costs_none_is_frictionless(self):
        closes = [100 + i * 0.5 for i in range(_N)]
        res = simulate_swing(
            "SYM", _bars(closes), params=_PARAMS, score_threshold=0.0, costs=None,
        )
        assert res.total_costs == 0.0
        assert res.net_pnl == res.gross_pnl
        assert all(t.cost == 0.0 for t in res.trades)

    def test_costs_reduce_net_pnl(self):
        closes = [100 + i * 0.5 for i in range(_N)]
        cost_run = simulate_swing(
            "SYM", _bars(closes), params=_PARAMS, score_threshold=0.0,
            costs=TradingCosts(), qty=10,
        )
        assert cost_run.total_costs > 0.0
        assert cost_run.net_pnl < cost_run.gross_pnl
        assert round(sum(t.cost for t in cost_run.trades), 2) == round(
            cost_run.total_costs, 2
        )


class TestResultStats:
    def test_equity_curve_and_summary(self):
        closes = [100 + i * 0.5 for i in range(_N)]
        res = simulate_swing(
            "SYM", _bars(closes), params=_PARAMS, score_threshold=0.0,
        )
        assert len(res.equity_curve) == res.total_trades
        assert res.max_drawdown >= 0.0
        summary = res.summary()
        assert "Swing Simulation Summary" in summary
        assert "Win rate" in summary

    def test_win_rate_and_profit_factor_bounds(self):
        closes = [100 + i * 0.5 for i in range(_N)]
        res = simulate_swing(
            "SYM", _bars(closes), params=_PARAMS, score_threshold=0.0,
        )
        assert 0.0 <= res.win_rate <= 100.0
        assert res.profit_factor >= 0.0
        assert res.avg_bars_held >= 0.0


class TestExitConfig:
    def test_exit_config_max_hold_matches_param(self):
        closes = [100 + i * 0.1 for i in range(_N)]
        via_param = simulate_swing(
            "SYM", _bars(closes), params={"atr_multiplier": 100.0},
            score_threshold=0.0, allow_reentry=False, max_hold_bars=3,
        )
        via_cfg = simulate_swing(
            "SYM", _bars(closes), params={"atr_multiplier": 100.0},
            score_threshold=0.0, allow_reentry=False,
            exit_config=ExitConfig(max_hold_bars=3),
        )
        assert via_param.trades[0].exit_reason == "time"
        assert via_cfg.trades[0].exit_reason == "time"
        assert via_param.trades[0].bars_held == via_cfg.trades[0].bars_held == 3

    def test_trailing_stop_changes_exit(self):
        # Rise then pull back: a tight trailing stop should exit on "stop"
        # rather than riding to the end of data.
        rise = [100 + i * 0.5 for i in range(MIN_BARS)]
        tail = [rise[-1] + 5, rise[-1] + 6, rise[-1] - 5, rise[-1] - 6]
        res = simulate_swing(
            "SYM", _bars(rise + tail), params={"atr_multiplier": 3.0},
            score_threshold=0.0, allow_reentry=False,
            exit_config=ExitConfig(trail_r=0.5),
        )
        assert res.total_trades == 1
        assert res.trades[0].exit_reason in {"stop", "target"}


class TestUniverse:
    def test_universe_merges_and_orders_trades(self):
        data = {
            "A": _bars([100 + i * 0.5 for i in range(_N)]),
            "B": _bars([50 + i * 0.4 for i in range(_N)]),
        }
        res = simulate_swing_universe(data, params=_PARAMS, score_threshold=0.0)
        assert isinstance(res, SwingBacktestResult)
        assert res.total_trades > 0
        times = [t.entry_time for t in res.trades]
        assert times == sorted(times)
