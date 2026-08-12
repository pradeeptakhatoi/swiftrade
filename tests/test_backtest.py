"""Tests for the backtesting harness."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from dhan_algo.backtest import (
    BacktestResult,
    ReplayTicker,
    TradingCosts,
    load_csv,
    run_backtest,
)
from dhan_algo.strategy import (
    MultiStrategy,
    Order,
    SmaDemo,
    SmaDemoMulti,
    Strategy,
    Ticker,
)


# ---------------------------------------------------------------------------
# ReplayTicker
# ---------------------------------------------------------------------------


class TestReplayTicker:
    def test_get_ltp_returns_none_initially(self):
        ticker = ReplayTicker()
        assert ticker.get_ltp("2885") is None

    def test_set_and_get(self):
        ticker = ReplayTicker()
        ticker.set_price("2885", 1500.0)
        assert ticker.get_ltp("2885") == 1500.0

    def test_watchlist(self):
        ticker = ReplayTicker()
        ticker.set_price("A", 10.0)
        ticker.set_price("B", 20.0)
        wl = ticker.watchlist
        assert wl == {"A": 10.0, "B": 20.0}

    def test_implements_ticker_protocol(self):
        ticker = ReplayTicker()
        assert isinstance(ticker, Ticker)


# ---------------------------------------------------------------------------
# load_csv
# ---------------------------------------------------------------------------


class TestLoadCsv:
    def test_loads_single_symbol(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        csv_file.write_text(
            "timestamp,security_id,open,high,low,close,volume\n"
            "2024-01-01,2885,100,105,99,102,1000\n"
            "2024-01-02,2885,102,108,101,107,1200\n"
        )
        bars = load_csv(csv_file)
        assert "2885" in bars
        assert len(bars["2885"]) == 2
        assert bars["2885"][0]["close"] == 102.0
        assert bars["2885"][1]["open"] == 102.0

    def test_loads_multi_symbol(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        csv_file.write_text(
            "timestamp,security_id,open,high,low,close,volume\n"
            "2024-01-01,A,10,15,9,12,100\n"
            "2024-01-01,B,20,25,19,22,200\n"
            "2024-01-02,A,12,16,11,14,150\n"
        )
        bars = load_csv(csv_file)
        assert len(bars) == 2
        assert len(bars["A"]) == 2
        assert len(bars["B"]) == 1


# ---------------------------------------------------------------------------
# run_backtest
# ---------------------------------------------------------------------------


def _make_bars(security_id: str, prices: list[float]) -> list[dict]:
    """Helper to build bar dicts from a list of close prices."""
    return [
        {
            "timestamp": f"2024-01-{i + 1:02d}",
            "security_id": security_id,
            "open": p,
            "high": p + 1,
            "low": p - 1,
            "close": p,
            "volume": 100,
        }
        for i, p in enumerate(prices)
    ]


class TestRunBacktest:
    def test_buy_then_sell_profit(self, test_settings):
        """Buy low, sell high -> positive P&L."""

        class BuySellStrategy(MultiStrategy):
            def __init__(self):
                self._count = 0

            def on_tick(self, ticker: Ticker, security_id: str, segment) -> list[Order]:
                self._count += 1
                if self._count == 1:
                    return [Order(side="BUY", qty=1, security_id=security_id)]
                elif self._count == 2:
                    return [Order(side="SELL", qty=1, security_id=security_id)]
                return []

        bars = {"SYM": _make_bars("SYM", [100.0, 110.0])}
        result = run_backtest(BuySellStrategy(), bars, settings=test_settings)

        assert result.total_trades == 2
        assert result.buy_count == 1
        assert result.sell_count == 1
        assert result.pnl == 10.0

    def test_buy_then_sell_loss(self, test_settings):
        """Buy high, sell low -> negative P&L."""

        class BuySellLoss(MultiStrategy):
            def __init__(self):
                self._count = 0

            def on_tick(self, ticker: Ticker, security_id: str, segment) -> list[Order]:
                self._count += 1
                if self._count == 1:
                    return [Order(side="BUY", qty=1, security_id=security_id)]
                elif self._count == 2:
                    return [Order(side="SELL", qty=1, security_id=security_id)]
                return []

        bars = {"SYM": _make_bars("SYM", [110.0, 100.0])}
        result = run_backtest(BuySellLoss(), bars, settings=test_settings)

        assert result.pnl == -10.0

    def test_risk_guard_blocks_qty(self, test_settings):
        """Orders exceeding max_qty should be skipped."""
        test_settings.max_qty = 5

        class BigOrder(MultiStrategy):
            def on_tick(self, ticker: Ticker, security_id: str, segment) -> list[Order]:
                return [Order(side="BUY", qty=100, security_id=security_id)]

        bars = {"SYM": _make_bars("SYM", [100.0])}
        result = run_backtest(BigOrder(), bars, settings=test_settings)
        assert result.total_trades == 0

    def test_risk_guard_blocks_notional(self, test_settings):
        """Orders exceeding max_order_value should be skipped."""
        test_settings.max_order_value = 100

        class ExpensiveOrder(MultiStrategy):
            def on_tick(self, ticker: Ticker, security_id: str, segment) -> list[Order]:
                return [Order(side="BUY", qty=10, security_id=security_id)]

        bars = {"SYM": _make_bars("SYM", [1000.0])}
        result = run_backtest(ExpensiveOrder(), bars, settings=test_settings)
        assert result.total_trades == 0

    def test_multi_symbol(self, test_settings):
        """Strategy receives ticks for multiple symbols."""

        class CountTicks(MultiStrategy):
            def __init__(self):
                self.ticks: dict[str, int] = {}

            def on_tick(self, ticker: Ticker, security_id: str, segment) -> list[Order]:
                self.ticks[security_id] = self.ticks.get(security_id, 0) + 1
                return []

        bars = {
            "A": _make_bars("A", [10.0, 11.0, 12.0]),
            "B": _make_bars("B", [20.0, 21.0]),
        }
        strat = CountTicks()
        run_backtest(strat, bars, settings=test_settings)
        assert strat.ticks["A"] == 3
        assert strat.ticks["B"] == 2

    def test_old_strategy_compat(self, test_settings):
        """A legacy Strategy (SmaDemo) works via the adapter + ltp patch."""
        # 20 prices with a golden cross pattern
        prices = list(range(100, 120))  # steadily rising
        bars = {"2885": _make_bars("2885", prices)}
        strategy = SmaDemo(short_period=5, long_period=20, qty=1)

        result = run_backtest(strategy, bars, settings=test_settings)
        # SmaDemo needs exactly long_period prices before it can signal,
        # so we should at least have no errors and get a result back
        assert isinstance(result, BacktestResult)
        assert result.total_trades >= 0

    def test_summary_format(self, test_settings):
        result = BacktestResult(
            pnl=150.0, unrealized_pnl=50.0,
            total_trades=4, buy_count=2, sell_count=2,
        )
        summary = result.summary()
        assert "Backtest Summary" in summary
        assert "150.00" in summary
        assert "50.00" in summary
        assert "200.00" in summary  # net

    def test_sma_demo_multi_backtest(self, test_settings):
        """SmaDemoMulti works through the backtest harness."""
        # Rising then falling prices to generate a golden cross then death cross
        prices_a = [100, 99, 98, 97, 96, 95, 94, 93, 92, 91,
                     90, 89, 88, 87, 86, 85, 84, 83, 82, 81,
                     90, 95, 100, 105, 110]
        prices_b = [50, 49, 48, 47, 46, 45, 44, 43, 42, 41,
                     40, 39, 38, 37, 36, 35, 34, 33, 32, 31,
                     40, 45, 50, 55, 60]
        bars = {
            "A": _make_bars("A", prices_a),
            "B": _make_bars("B", prices_b),
        }
        strategy = SmaDemoMulti(short_period=5, long_period=20, qty=1)
        result = run_backtest(strategy, bars, settings=test_settings)
        assert isinstance(result, BacktestResult)


# ---------------------------------------------------------------------------
# TradingCosts
# ---------------------------------------------------------------------------


class _BuySell(MultiStrategy):
    """One BUY on the first tick, one SELL on the second."""

    def __init__(self):
        self._count = 0

    def on_tick(self, ticker: Ticker, security_id: str, segment) -> list[Order]:
        self._count += 1
        if self._count == 1:
            return [Order(side="BUY", qty=1, security_id=security_id)]
        if self._count == 2:
            return [Order(side="SELL", qty=1, security_id=security_id)]
        return []


class TestTradingCosts:
    def test_slippage_adverse_direction(self):
        costs = TradingCosts(slippage_pct=0.001)
        assert costs.slippage_price("BUY", 100.0) == pytest.approx(100.1)
        assert costs.slippage_price("SELL", 100.0) == pytest.approx(99.9)

    def test_brokerage_capped_at_flat(self):
        costs = TradingCosts()
        # Small turnover -> percentage brokerage (0.03%).
        assert costs.brokerage(1000.0) == pytest.approx(0.3)
        # Huge turnover -> flat cap of Rs 20.
        assert costs.brokerage(10_000_000.0) == pytest.approx(20.0)

    def test_intraday_stt_sell_side_only(self):
        costs = TradingCosts()
        buy = costs.breakdown("BUY", 100.0, 10, product="INTRA")
        sell = costs.breakdown("SELL", 100.0, 10, product="INTRA")
        assert buy["stt"] == 0.0
        assert sell["stt"] == pytest.approx(0.00025 * 1000)

    def test_delivery_stt_both_sides(self):
        costs = TradingCosts()
        buy = costs.breakdown("BUY", 100.0, 10, product="CNC")
        sell = costs.breakdown("SELL", 100.0, 10, product="CNC")
        assert buy["stt"] == pytest.approx(0.001 * 1000)
        assert sell["stt"] == pytest.approx(0.001 * 1000)

    def test_stamp_duty_buy_side_only(self):
        costs = TradingCosts()
        assert costs.breakdown("BUY", 100.0, 10, product="INTRA")["stamp_duty"] > 0
        assert costs.breakdown("SELL", 100.0, 10, product="INTRA")["stamp_duty"] == 0.0

    def test_breakdown_total_is_sum_of_components(self):
        costs = TradingCosts()
        b = costs.breakdown("SELL", 110.0, 10, product="INTRA")
        parts = b["brokerage"] + b["stt"] + b["exchange"] + b["sebi"] + b["stamp_duty"] + b["gst"]
        assert b["total"] == pytest.approx(parts)
        assert costs.total("SELL", 110.0, 10, "INTRA") == pytest.approx(b["total"])

    def test_gst_on_brokerage_exchange_sebi(self):
        costs = TradingCosts()
        b = costs.breakdown("BUY", 100.0, 10, product="INTRA")
        expected_gst = 0.18 * (b["brokerage"] + b["exchange"] + b["sebi"])
        assert b["gst"] == pytest.approx(expected_gst)


class TestRunBacktestWithCosts:
    def test_costs_none_is_frictionless(self, test_settings):
        bars = {"SYM": _make_bars("SYM", [100.0, 110.0])}
        result = run_backtest(_BuySell(), bars, settings=test_settings, costs=None)
        assert result.total_costs == 0.0
        assert result.pnl == 10.0
        assert result.net_pnl == 10.0
        assert all(f.cost == 0.0 for f in result.fills)

    def test_costs_reduce_net_pnl(self, test_settings):
        bars = {"SYM": _make_bars("SYM", [100.0, 110.0])}
        result = run_backtest(
            _BuySell(), bars, settings=test_settings, costs=TradingCosts()
        )
        assert result.total_costs > 0.0
        assert result.net_pnl < result.pnl + result.unrealized_pnl
        # Per-fill cost is recorded and sums to the total.
        assert sum(f.cost for f in result.fills) == pytest.approx(result.total_costs)

    def test_slippage_worsens_realized_pnl(self, test_settings):
        """Slippage-only model buys higher and sells lower than close."""
        bars = {"SYM": _make_bars("SYM", [100.0, 110.0])}
        costs = TradingCosts(
            slippage_pct=0.001, brokerage_flat=0.0, brokerage_pct=0.0,
            stt_delivery=0.0, stt_intraday_sell=0.0, exchange_txn_pct=0.0,
            sebi_pct=0.0, stamp_duty_delivery=0.0, stamp_duty_intraday=0.0,
            gst_pct=0.0,
        )
        result = run_backtest(_BuySell(), bars, settings=test_settings, costs=costs)
        # Buy fills at 100.1, sell fills at 109.89 -> realized 9.79 < 10.0.
        assert result.total_costs == 0.0  # charges zeroed; only slippage
        assert result.pnl == pytest.approx(9.79)
        buy_fill = next(f for f in result.fills if f.side == "BUY")
        sell_fill = next(f for f in result.fills if f.side == "SELL")
        assert buy_fill.price == pytest.approx(100.1)
        assert sell_fill.price == pytest.approx(109.89)

    def test_cost_product_override(self, test_settings):
        """cost_product overrides the order's product for the charge model."""
        bars = {"SYM": _make_bars("SYM", [100.0, 110.0])}
        intra = run_backtest(
            _BuySell(), bars, settings=test_settings,
            costs=TradingCosts(), cost_product="INTRA",
        )
        cnc = run_backtest(
            _BuySell(), bars, settings=test_settings,
            costs=TradingCosts(), cost_product="CNC",
        )
        # Delivery STT (both sides, 0.1%) dwarfs intraday sell-only STT.
        assert cnc.total_costs > intra.total_costs

    def test_summary_includes_costs_line(self, test_settings):
        result = BacktestResult(
            pnl=150.0, unrealized_pnl=50.0, total_costs=25.0,
            total_trades=4, buy_count=2, sell_count=2,
        )
        summary = result.summary()
        assert "Costs" in summary
        assert "25.00" in summary
        assert "175.00" in summary  # net = 150 + 50 - 25
