"""Tests for position sizing, bracket orders, and SL+target placement."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from dhan_algo.orders import calculate_position_size, place_bracket, place_with_sl_target


# ---------------------------------------------------------------------------
# calculate_position_size
# ---------------------------------------------------------------------------


class TestCalculatePositionSize:
    def test_basic_calculation(self):
        # risk 500, entry=100, SL=95 -> risk/share=5, qty=100
        qty = calculate_position_size(100.0, 95.0, 500.0, max_qty=200, max_order_value=100_000)
        assert qty == 100

    def test_respects_max_qty(self):
        qty = calculate_position_size(100.0, 95.0, 5000.0, max_qty=50, max_order_value=100_000)
        assert qty == 50

    def test_respects_max_order_value(self):
        qty = calculate_position_size(1000.0, 990.0, 5000.0, max_qty=1000, max_order_value=10_000)
        assert qty == 10  # 10_000 / 1000

    def test_zero_risk_per_share(self):
        qty = calculate_position_size(100.0, 100.0, 500.0, max_qty=50, max_order_value=50_000)
        assert qty == 0

    def test_negative_entry(self):
        qty = calculate_position_size(0.0, 95.0, 500.0, max_qty=50, max_order_value=50_000)
        assert qty == 0

    def test_floors_to_integer(self):
        # 500 / 3 = 166.67 -> 166
        qty = calculate_position_size(100.0, 97.0, 500.0, max_qty=500, max_order_value=100_000)
        assert qty == 166

    def test_sell_side_uses_absolute_risk(self):
        # SELL: entry=95, SL=100 -> abs(95-100)=5
        qty = calculate_position_size(95.0, 100.0, 500.0, max_qty=200, max_order_value=100_000)
        assert qty == 100

    def test_tiny_risk_per_share(self):
        # risk/share < 0.01 -> 0
        qty = calculate_position_size(100.0, 99.995, 500.0, max_qty=200, max_order_value=100_000)
        assert qty == 0

    def test_both_limits_applied(self):
        # max_qty would allow 50, max_order_value would allow 25 -> pick smaller
        qty = calculate_position_size(2000.0, 1990.0, 5000.0, max_qty=50, max_order_value=50_000)
        assert qty == 25  # 50_000 / 2000 = 25 < 50


# ---------------------------------------------------------------------------
# place_bracket
# ---------------------------------------------------------------------------


class TestPlaceBracket:
    @patch("dhan_algo.orders.journal_record")
    @patch("dhan_algo.orders.check_order", return_value=None)
    def test_dry_run_returns_plan(self, mock_check, mock_journal, mock_client, test_settings):
        test_settings.dhan_live = False
        result = place_bracket(
            mock_client, "2885",
            side="BUY", qty=10,
            entry_price=100.0, stop_loss_price=95.0, target_price=110.0,
            settings=test_settings,
        )
        assert result is not None
        assert result["status"] == "dry_run"
        assert "SUPER_ORDER" in result["plan"]
        assert "SL=95.00" in result["plan"]
        assert "target=110.00" in result["plan"]
        mock_journal.assert_called_once()

    @patch("dhan_algo.orders.journal_record")
    @patch("dhan_algo.orders.check_order", return_value="exceeded MAX_QTY")
    def test_blocked_by_risk(self, mock_check, mock_journal, mock_client, test_settings):
        result = place_bracket(
            mock_client, "2885",
            side="BUY", qty=100,
            entry_price=100.0, stop_loss_price=95.0, target_price=110.0,
            settings=test_settings,
        )
        assert result is None
        mock_journal.assert_called_once()
        assert mock_journal.call_args[1]["status"] == "blocked"

    @patch("dhan_algo.orders.journal_record")
    @patch("dhan_algo.orders.check_order", return_value=None)
    def test_live_calls_place_super_order(self, mock_check, mock_journal, mock_client, test_settings):
        test_settings.dhan_live = True
        mock_client.place_super_order.return_value = {"status": "success", "data": {"orderId": "123"}}
        result = place_bracket(
            mock_client, "2885",
            side="BUY", qty=10,
            entry_price=100.0, stop_loss_price=95.0, target_price=110.0,
            settings=test_settings,
        )
        mock_client.place_super_order.assert_called_once()
        kwargs = mock_client.place_super_order.call_args[1]
        assert kwargs["stopLossPrice"] == 95.0
        assert kwargs["targetPrice"] == 110.0
        assert kwargs["quantity"] == 10
        assert result["status"] == "success"


# ---------------------------------------------------------------------------
# place_with_sl_target
# ---------------------------------------------------------------------------


class TestPlaceWithSlTarget:
    @patch("dhan_algo.orders.place")
    def test_places_three_orders(self, mock_place, mock_client, test_settings):
        mock_place.return_value = {"status": "dry_run", "plan": "test"}
        result = place_with_sl_target(
            mock_client, "2885",
            side="BUY", qty=10,
            entry_price=100.0, stop_loss_price=95.0, target_price=110.0,
            settings=test_settings,
        )
        assert mock_place.call_count == 3
        assert "entry" in result
        assert "stop_loss" in result
        assert "target" in result

    @patch("dhan_algo.orders.place")
    def test_exit_side_is_opposite(self, mock_place, mock_client, test_settings):
        mock_place.return_value = {"status": "dry_run", "plan": "test"}
        place_with_sl_target(
            mock_client, "2885",
            side="BUY", qty=10,
            entry_price=100.0, stop_loss_price=95.0, target_price=110.0,
            settings=test_settings,
        )
        calls = mock_place.call_args_list
        assert calls[0][1]["side"] == "BUY"    # entry
        assert calls[1][1]["side"] == "SELL"   # SL
        assert calls[2][1]["side"] == "SELL"   # target

    @patch("dhan_algo.orders.place")
    def test_sl_order_uses_trigger_price(self, mock_place, mock_client, test_settings):
        mock_place.return_value = {"status": "dry_run", "plan": "test"}
        place_with_sl_target(
            mock_client, "2885",
            side="BUY", qty=10,
            entry_price=100.0, stop_loss_price=95.0, target_price=110.0,
            settings=test_settings,
        )
        sl_call = mock_place.call_args_list[1][1]
        assert sl_call["order_type"] == "SL"
        assert sl_call["trigger_price"] == 95.0

    @patch("dhan_algo.orders.place")
    def test_sell_side_reverses_exits(self, mock_place, mock_client, test_settings):
        mock_place.return_value = {"status": "dry_run", "plan": "test"}
        place_with_sl_target(
            mock_client, "2885",
            side="SELL", qty=5,
            entry_price=100.0, stop_loss_price=105.0, target_price=90.0,
            settings=test_settings,
        )
        calls = mock_place.call_args_list
        assert calls[0][1]["side"] == "SELL"   # entry
        assert calls[1][1]["side"] == "BUY"    # SL
        assert calls[2][1]["side"] == "BUY"    # target
