"""Tests for the performance-analytics pure functions."""

from __future__ import annotations

import math

import pandas as pd

from dhan_algo import analytics


def _df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# A small, deterministic closed-trade set: 2 winners, 1 loser.
_TRADES = [
    {
        "strategy": "swing", "symbol": "A",
        "entry_time": "2022-01-03 09:20", "exit_time": "2022-01-04 10:00",
        "exit_reason": "target", "qty": 10,
        "gross_pnl": 220.0, "cost": 20.0, "net_pnl": 200.0, "r_multiple": 2.0,
    },
    {
        "strategy": "swing", "symbol": "B",
        "entry_time": "2022-01-05 10:20", "exit_time": "2022-01-06 11:00",
        "exit_reason": "stop", "qty": 10,
        "gross_pnl": -90.0, "cost": 10.0, "net_pnl": -100.0, "r_multiple": -1.0,
    },
    {
        "strategy": "intraday", "symbol": "A",
        "entry_time": "2022-01-07 14:20", "exit_time": "2022-01-07 15:00",
        "exit_reason": "target", "qty": 10,
        "gross_pnl": 110.0, "cost": 10.0, "net_pnl": 100.0, "r_multiple": 1.0,
    },
]


class TestComputeMetrics:
    def test_empty_frame(self):
        m = analytics.compute_metrics(pd.DataFrame())
        assert m["trades"] == 0
        assert m["net_pnl"] == 0.0
        assert m["profit_factor"] == 0.0

    def test_core_stats(self):
        m = analytics.compute_metrics(_df(_TRADES))
        assert m["trades"] == 3
        assert m["wins"] == 2
        assert m["losses"] == 1
        assert round(m["win_rate"], 1) == 66.7
        assert m["net_pnl"] == 200.0
        assert m["gross_pnl"] == 240.0
        assert m["total_costs"] == 40.0
        # profit factor = (200 + 100) / 100
        assert m["profit_factor"] == 3.0
        assert round(m["expectancy"], 2) == round(200.0 / 3, 2)
        assert round(m["avg_r"], 3) == round((2.0 - 1.0 + 1.0) / 3, 3)
        assert m["avg_win"] == 150.0
        assert m["avg_loss"] == 100.0

    def test_profit_factor_infinite_with_no_losers(self):
        winners = [t for t in _TRADES if t["net_pnl"] > 0]
        m = analytics.compute_metrics(_df(winners))
        assert m["profit_factor"] == float("inf")

    def test_max_drawdown_uses_chronological_order(self):
        # Loser sits between winners chronologically -> a 100 trough.
        m = analytics.compute_metrics(_df(_TRADES))
        # equity path: +200, +100, +200 -> peak 200 then 100 -> dd 100
        assert m["max_drawdown"] == 100.0


class TestEquityCurve:
    def test_cumulative_and_ordered(self):
        curve = analytics.equity_curve(_df(_TRADES))
        assert list(curve["trade"]) == [1, 2, 3]
        # ordered by exit_time: 200, 100, 200
        assert list(curve["equity"]) == [200.0, 100.0, 200.0]

    def test_empty(self):
        assert analytics.equity_curve(pd.DataFrame()).empty


class TestRDistribution:
    def test_buckets_counts(self):
        dist = analytics.r_distribution(_df(_TRADES), bin_width=1.0)
        total = dist["count"].sum()
        assert total == 3
        # buckets span -1.0 .. 2.0
        assert not dist.empty

    def test_empty_without_r_column(self):
        df = _df([{"net_pnl": 5.0}])
        assert analytics.r_distribution(df).empty


class TestBreakdown:
    def test_by_symbol(self):
        table = analytics.breakdown(_df(_TRADES), "symbol")
        groups = dict(zip(table["group"], table["net_pnl"]))
        assert groups["A"] == 300.0
        assert groups["B"] == -100.0
        # sorted by net_pnl desc -> A first
        assert table.iloc[0]["group"] == "A"

    def test_by_strategy_win_rate(self):
        table = analytics.breakdown(_df(_TRADES), "strategy")
        swing = table[table["group"] == "swing"].iloc[0]
        assert swing["trades"] == 2
        assert swing["win_rate"] == 50.0

    def test_by_hour(self):
        table = analytics.breakdown(_df(_TRADES), "hour")
        hours = set(table["group"])
        assert {9, 10, 14} <= hours

    def test_unknown_column_returns_empty(self):
        assert analytics.breakdown(_df(_TRADES), "nope").empty
