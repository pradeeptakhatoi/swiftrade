"""Tests for strategy framework extensions."""

from unittest.mock import MagicMock, patch

from dhan_algo.strategy import (
    MultiStrategy,
    OrbBreakoutStrategy,
    Order,
    PollingTicker,
    SmaDemo,
    SmaDemoMulti,
    Strategy,
    Ticker,
    _StrategyAdapter,
    run_multi_strategy_loop,
)


# ---------------------------------------------------------------------------
# Order backwards compatibility
# ---------------------------------------------------------------------------


class TestOrderCompat:
    def test_order_without_security_id(self):
        """Existing code creating Order(side=..., qty=...) still works."""
        o = Order(side="BUY", qty=5)
        assert o.side == "BUY"
        assert o.qty == 5
        assert o.security_id == ""

    def test_order_with_security_id(self):
        o = Order(side="SELL", qty=10, security_id="2885")
        assert o.security_id == "2885"

    def test_order_keyword_only(self):
        o = Order(side="BUY", qty=1, order_type="LIMIT", product="CNC", price=100.0)
        assert o.order_type == "LIMIT"
        assert o.security_id == ""


# ---------------------------------------------------------------------------
# PollingTicker
# ---------------------------------------------------------------------------


class TestPollingTicker:
    def test_get_ltp_returns_none_before_refresh(self, mock_client):
        ticker = PollingTicker(mock_client, ["2885"])
        assert ticker.get_ltp("2885") is None

    @patch("dhan_algo.strategy.ltp", return_value=1500.0)
    def test_refresh_all_populates_prices(self, mock_ltp, mock_client):
        ticker = PollingTicker(mock_client, ["2885", "3456"])
        mock_ltp.side_effect = [1500.0, 2200.0]
        ticker.refresh_all()
        assert ticker.get_ltp("2885") == 1500.0
        assert ticker.get_ltp("3456") == 2200.0

    @patch("dhan_algo.strategy.ltp", return_value=1500.0)
    def test_watchlist_returns_copy(self, mock_ltp, mock_client):
        ticker = PollingTicker(mock_client, ["2885"])
        ticker.refresh_all()
        wl = ticker.watchlist
        assert isinstance(wl, dict)
        assert wl["2885"] == 1500.0
        # Mutating the returned dict should not affect internal state
        wl["2885"] = 0
        assert ticker.get_ltp("2885") == 1500.0

    def test_implements_ticker_protocol(self, mock_client):
        ticker = PollingTicker(mock_client, [])
        assert isinstance(ticker, Ticker)


# ---------------------------------------------------------------------------
# _StrategyAdapter
# ---------------------------------------------------------------------------


class TestStrategyAdapter:
    def test_wraps_strategy_evaluate(self, mock_client):
        strategy = SmaDemo(short_period=2, long_period=3, qty=1)
        adapter = _StrategyAdapter(strategy, mock_client)

        # Build a simple ticker that returns prices
        ticker = MagicMock()
        ticker.get_ltp.return_value = 100.0

        # Patch ltp so SmaDemo.evaluate works
        with patch("dhan_algo.strategy.ltp", return_value=100.0):
            orders = adapter.on_tick(ticker, "2885", None)
        # Still collecting prices, no signal
        assert orders == []

    def test_adapter_fills_security_id(self, mock_client):
        """When the wrapped strategy returns an Order, security_id is filled."""
        strategy = MagicMock(spec=Strategy)
        strategy.evaluate.return_value = Order(side="BUY", qty=1)
        adapter = _StrategyAdapter(strategy, mock_client)

        ticker = MagicMock()
        orders = adapter.on_tick(ticker, "2885", None)
        assert len(orders) == 1
        assert orders[0].security_id == "2885"

    def test_adapter_returns_empty_on_none(self, mock_client):
        strategy = MagicMock(spec=Strategy)
        strategy.evaluate.return_value = None
        adapter = _StrategyAdapter(strategy, mock_client)

        ticker = MagicMock()
        orders = adapter.on_tick(ticker, "2885", None)
        assert orders == []


# ---------------------------------------------------------------------------
# SmaDemoMulti
# ---------------------------------------------------------------------------


class TestSmaDemoMulti:
    def test_independent_per_symbol(self):
        multi = SmaDemoMulti(short_period=2, long_period=3, qty=1)
        ticker = MagicMock()

        # Feed 3 prices to symbol A (enough for long_period=3)
        for p in [100.0, 101.0, 102.0]:
            ticker.get_ltp.return_value = p
            multi.on_tick(ticker, "A", None)

        # Symbol B should still be collecting
        ticker.get_ltp.return_value = 200.0
        result = multi.on_tick(ticker, "B", None)
        assert result == []

    def test_golden_cross_signal(self):
        multi = SmaDemoMulti(short_period=2, long_period=3, qty=5)
        ticker = MagicMock()

        # Create a death cross first (short < long), then golden cross
        # Prices: 100, 90, 80 -> short(80,90)=85, long=90 -> short < long
        # Then: 100, 90, 80, 120 -> short(80,120)=100, long(90,80,120)=96.67 -> short > long = golden cross
        prices = [100.0, 90.0, 80.0, 120.0]
        orders = []
        for p in prices:
            ticker.get_ltp.return_value = p
            orders.extend(multi.on_tick(ticker, "SYM", None))

        buy_orders = [o for o in orders if o.side == "BUY"]
        assert len(buy_orders) == 1
        assert buy_orders[0].security_id == "SYM"
        assert buy_orders[0].qty == 5

    def test_returns_none_for_missing_price(self):
        multi = SmaDemoMulti()
        ticker = MagicMock()
        ticker.get_ltp.return_value = None
        result = multi.on_tick(ticker, "X", None)
        assert result == []


# ---------------------------------------------------------------------------
# run_multi_strategy_loop
# ---------------------------------------------------------------------------


class TestRunMultiStrategyLoop:
    @patch("dhan_algo.strategy.time.sleep", side_effect=KeyboardInterrupt)
    @patch("dhan_algo.strategy.place")
    @patch("dhan_algo.strategy.ltp", return_value=100.0)
    def test_loop_calls_on_tick_and_exits(self, mock_ltp, mock_place, mock_sleep, mock_client, test_settings):
        strategy = MagicMock(spec=MultiStrategy)
        strategy.on_tick.return_value = []

        run_multi_strategy_loop(
            strategy, mock_client, ["2885", "3456"],
            settings=test_settings,
        )

        # on_start should be called once
        strategy.on_start.assert_called_once()
        # on_tick should be called for each symbol
        assert strategy.on_tick.call_count == 2
        # on_stop should be called on exit
        strategy.on_stop.assert_called_once()

    @patch("dhan_algo.strategy.time.sleep", side_effect=KeyboardInterrupt)
    @patch("dhan_algo.strategy.place")
    @patch("dhan_algo.strategy.ltp", return_value=100.0)
    def test_loop_routes_orders(self, mock_ltp, mock_place, mock_sleep, mock_client, test_settings):
        strategy = MagicMock(spec=MultiStrategy)
        strategy.on_tick.return_value = [Order(side="BUY", qty=1, security_id="2885")]

        run_multi_strategy_loop(
            strategy, mock_client, ["2885"],
            settings=test_settings,
        )

        mock_place.assert_called_once()
        call_kwargs = mock_place.call_args
        assert call_kwargs[0][1] == "2885"  # security_id


# ---------------------------------------------------------------------------
# OrbBreakoutStrategy
# ---------------------------------------------------------------------------


def _make_orb_strategy(**kwargs) -> OrbBreakoutStrategy:
    """Create an OrbBreakoutStrategy with pre-loaded state (skip yfinance)."""
    defaults = dict(
        qty=10, interval_minutes=15, atr_multiplier=1.0,
        min_volume_ratio=1.0, risk_per_trade=0,
        symbol_map={"2885": "RELIANCE"},
    )
    defaults.update(kwargs)
    strategy = OrbBreakoutStrategy(**defaults)
    return strategy


def _inject_state(strategy: OrbBreakoutStrategy, security_id: str, **overrides):
    """Inject per-symbol state to avoid needing yfinance in tests."""
    state = {
        "symbol": "RELIANCE",
        "orb_high": 2850.0,
        "orb_low": 2800.0,
        "atr": 20.0,
        "st_direction": 1,
        "vol_ratio": 1.5,
        "long_traded": False,
        "short_traded": False,
    }
    state.update(overrides)
    strategy._state[security_id] = state


class TestOrbBreakoutStrategy:
    def test_long_breakout_signal(self):
        """Price above orb_high with bullish ST + volume -> BUY order."""
        strategy = _make_orb_strategy()
        _inject_state(strategy, "2885", st_direction=1, vol_ratio=1.5)

        ticker = MagicMock()
        ticker.get_ltp.return_value = 2855.0  # above orb_high=2850

        orders = strategy.on_tick(ticker, "2885", None)
        assert len(orders) == 1
        assert orders[0].side == "BUY"
        assert orders[0].security_id == "2885"
        assert orders[0].qty == 10

    def test_short_breakout_signal(self):
        """Price below orb_low with bearish ST + volume -> SELL order."""
        strategy = _make_orb_strategy()
        _inject_state(strategy, "2885", st_direction=-1, vol_ratio=1.5)

        ticker = MagicMock()
        ticker.get_ltp.return_value = 2795.0  # below orb_low=2800

        orders = strategy.on_tick(ticker, "2885", None)
        assert len(orders) == 1
        assert orders[0].side == "SELL"

    def test_no_signal_inside_range(self):
        """Price inside ORB range -> no orders."""
        strategy = _make_orb_strategy()
        _inject_state(strategy, "2885")

        ticker = MagicMock()
        ticker.get_ltp.return_value = 2825.0  # between 2800 and 2850

        orders = strategy.on_tick(ticker, "2885", None)
        assert orders == []

    def test_no_duplicate_long_trade(self):
        """Second breakout in same direction -> no order."""
        strategy = _make_orb_strategy()
        _inject_state(strategy, "2885", st_direction=1, vol_ratio=1.5)

        ticker = MagicMock()
        ticker.get_ltp.return_value = 2855.0

        orders1 = strategy.on_tick(ticker, "2885", None)
        assert len(orders1) == 1

        # Second tick still above ORB -> no new order
        orders2 = strategy.on_tick(ticker, "2885", None)
        assert orders2 == []

    def test_no_duplicate_short_trade(self):
        """Second short breakout -> no order."""
        strategy = _make_orb_strategy()
        _inject_state(strategy, "2885", st_direction=-1, vol_ratio=1.5)

        ticker = MagicMock()
        ticker.get_ltp.return_value = 2795.0

        orders1 = strategy.on_tick(ticker, "2885", None)
        assert len(orders1) == 1
        orders2 = strategy.on_tick(ticker, "2885", None)
        assert orders2 == []

    def test_volume_filter_blocks(self):
        """Low volume -> no order even on breakout."""
        strategy = _make_orb_strategy(min_volume_ratio=1.5)
        _inject_state(strategy, "2885", st_direction=1, vol_ratio=0.8)  # below min

        ticker = MagicMock()
        ticker.get_ltp.return_value = 2855.0  # above orb_high

        orders = strategy.on_tick(ticker, "2885", None)
        assert orders == []

    def test_supertrend_filter_blocks_long(self):
        """Bearish SuperTrend blocks long breakout."""
        strategy = _make_orb_strategy()
        _inject_state(strategy, "2885", st_direction=-1, vol_ratio=1.5)

        ticker = MagicMock()
        ticker.get_ltp.return_value = 2855.0  # above orb_high but ST bearish

        orders = strategy.on_tick(ticker, "2885", None)
        assert orders == []

    def test_supertrend_filter_blocks_short(self):
        """Bullish SuperTrend blocks short breakout."""
        strategy = _make_orb_strategy()
        _inject_state(strategy, "2885", st_direction=1, vol_ratio=1.5)

        ticker = MagicMock()
        ticker.get_ltp.return_value = 2795.0  # below orb_low but ST bullish

        orders = strategy.on_tick(ticker, "2885", None)
        assert orders == []

    def test_order_has_sl_and_target(self):
        """Verify stop_loss and target are set on the Order."""
        strategy = _make_orb_strategy(atr_multiplier=1.0)
        _inject_state(strategy, "2885", atr=20.0, st_direction=1, vol_ratio=1.5)

        ticker = MagicMock()
        ticker.get_ltp.return_value = 2860.0

        orders = strategy.on_tick(ticker, "2885", None)
        assert len(orders) == 1
        order = orders[0]
        # SL = entry - 1.0 * ATR = 2860 - 20 = 2840
        assert order.stop_loss == 2840.0
        # Target = entry + 2 * risk = 2860 + 40 = 2900
        assert order.target == 2900.0
        assert order.price == 2860.0

    def test_short_order_has_correct_sl_target(self):
        """Verify SL and target for short orders."""
        strategy = _make_orb_strategy(atr_multiplier=1.0)
        _inject_state(strategy, "2885", atr=20.0, st_direction=-1, vol_ratio=1.5)

        ticker = MagicMock()
        ticker.get_ltp.return_value = 2790.0

        orders = strategy.on_tick(ticker, "2885", None)
        assert len(orders) == 1
        order = orders[0]
        # SL = entry + 1.0 * ATR = 2790 + 20 = 2810
        assert order.stop_loss == 2810.0
        # Target = entry - 2 * risk = 2790 - 40 = 2750
        assert order.target == 2750.0

    def test_position_sizing_with_risk_per_trade(self, test_settings):
        """When risk_per_trade is set, qty is calculated from ATR."""
        strategy = _make_orb_strategy(
            qty=1, risk_per_trade=500, atr_multiplier=1.0,
            settings=test_settings,
        )
        # Low price so the risk-based sizing (not max_order_value) is the
        # binding constraint: 50 * 810 = 40_500 < max_order_value (50_000).
        _inject_state(
            strategy, "2885",
            orb_high=800.0, orb_low=750.0, atr=10.0,
            st_direction=1, vol_ratio=1.5,
        )

        ticker = MagicMock()
        ticker.get_ltp.return_value = 810.0  # above orb_high

        orders = strategy.on_tick(ticker, "2885", None)
        assert len(orders) == 1
        # risk = 1.0 * 10 = 10, qty = 500 / 10 = 50
        assert orders[0].qty == 50

    def test_no_signal_without_state(self):
        """Symbol without initialised state returns no orders."""
        strategy = _make_orb_strategy()
        # Don't inject state for "9999"

        ticker = MagicMock()
        ticker.get_ltp.return_value = 100.0

        orders = strategy.on_tick(ticker, "9999", None)
        assert orders == []

    def test_no_signal_when_price_is_none(self):
        """None LTP returns no orders."""
        strategy = _make_orb_strategy()
        _inject_state(strategy, "2885")

        ticker = MagicMock()
        ticker.get_ltp.return_value = None

        orders = strategy.on_tick(ticker, "2885", None)
        assert orders == []

    def test_both_directions_can_trade(self):
        """Long and short can both fire independently."""
        strategy = _make_orb_strategy()
        _inject_state(strategy, "2885", st_direction=1, vol_ratio=1.5)

        ticker = MagicMock()

        # Long breakout
        ticker.get_ltp.return_value = 2855.0
        long_orders = strategy.on_tick(ticker, "2885", None)
        assert len(long_orders) == 1 and long_orders[0].side == "BUY"

        # Switch SuperTrend to bearish for short
        strategy._state["2885"]["st_direction"] = -1

        # Short breakout
        ticker.get_ltp.return_value = 2795.0
        short_orders = strategy.on_tick(ticker, "2885", None)
        assert len(short_orders) == 1 and short_orders[0].side == "SELL"
