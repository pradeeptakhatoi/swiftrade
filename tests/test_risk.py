"""Tests for risk guards and daily-loss cap."""

from unittest.mock import patch

from dhan_algo.risk import (
    check_order,
    consecutive_losses,
    open_positions_count,
)


def test_qty_guard_blocks(mock_client, test_settings):
    test_settings.max_qty = 10
    result = check_order(
        qty=20, price=100.0, security_id="2885",
        client=mock_client, settings=test_settings,
    )
    assert result is not None
    assert "MAX_QTY" in result


def test_qty_guard_passes(mock_client, test_settings):
    mock_client.get_positions.return_value = {"status": "success", "data": []}
    test_settings.max_qty = 50
    result = check_order(
        qty=10, price=100.0, security_id="2885",
        client=mock_client, settings=test_settings,
    )
    assert result is None


def test_notional_guard_blocks(mock_client, test_settings):
    test_settings.max_order_value = 5_000
    result = check_order(
        qty=10, price=1000.0, security_id="2885",
        client=mock_client, settings=test_settings,
    )
    assert result is not None
    assert "MAX_ORDER_VALUE" in result


def test_notional_guard_passes(mock_client, test_settings):
    mock_client.get_positions.return_value = {"status": "success", "data": []}
    test_settings.max_order_value = 50_000
    result = check_order(
        qty=5, price=100.0, security_id="2885",
        client=mock_client, settings=test_settings,
    )
    assert result is None


def test_daily_loss_cap_blocks(mock_client, test_settings):
    test_settings.max_daily_loss = 5_000
    mock_client.get_positions.return_value = {
        "status": "success",
        "data": [
            {"realizedProfit": -3000},
            {"realizedProfit": -2500},
        ],
    }
    result = check_order(
        qty=1, price=100.0, security_id="2885",
        client=mock_client, settings=test_settings,
    )
    assert result is not None
    assert "MAX_DAILY_LOSS" in result


def test_daily_loss_cap_passes(mock_client, test_settings):
    test_settings.max_daily_loss = 10_000
    mock_client.get_positions.return_value = {
        "status": "success",
        "data": [
            {"realizedProfit": -1000},
            {"realizedProfit": 500},
        ],
    }
    result = check_order(
        qty=1, price=100.0, security_id="2885",
        client=mock_client, settings=test_settings,
    )
    assert result is None


def _positions(*rows):
    return {"status": "success", "data": list(rows)}


class TestMaxOpenPositions:
    def test_blocks_new_symbol_at_cap(self, mock_client, test_settings):
        test_settings.max_open_positions = 2
        mock_client.get_positions.return_value = _positions(
            {"securityId": "111", "netQty": 5, "realizedProfit": 0},
            {"securityId": "222", "netQty": 3, "realizedProfit": 0},
        )
        result = check_order(
            qty=1, price=100.0, security_id="333",
            client=mock_client, settings=test_settings,
        )
        assert result is not None
        assert "MAX_OPEN_POSITIONS" in result

    def test_allows_adding_to_existing_position(self, mock_client, test_settings):
        test_settings.max_open_positions = 2
        mock_client.get_positions.return_value = _positions(
            {"securityId": "111", "netQty": 5, "realizedProfit": 0},
            {"securityId": "222", "netQty": 3, "realizedProfit": 0},
        )
        result = check_order(
            qty=1, price=100.0, security_id="111",
            client=mock_client, settings=test_settings,
        )
        assert result is None

    def test_closing_side_not_blocked(self, mock_client, test_settings):
        test_settings.max_open_positions = 2
        mock_client.get_positions.return_value = _positions(
            {"securityId": "111", "netQty": 5, "realizedProfit": 0},
            {"securityId": "222", "netQty": 3, "realizedProfit": 0},
        )
        result = check_order(
            qty=1, price=100.0, security_id="333",
            client=mock_client, settings=test_settings, side="SELL",
        )
        assert result is None

    def test_disabled_when_zero(self, mock_client, test_settings):
        test_settings.max_open_positions = 0
        mock_client.get_positions.return_value = _positions(
            {"securityId": "111", "netQty": 5, "realizedProfit": 0},
            {"securityId": "222", "netQty": 3, "realizedProfit": 0},
        )
        result = check_order(
            qty=1, price=100.0, security_id="333",
            client=mock_client, settings=test_settings,
        )
        assert result is None

    def test_open_positions_count_helper(self, mock_client):
        mock_client.get_positions.return_value = _positions(
            {"securityId": "111", "netQty": 5},
            {"securityId": "222", "netQty": 0},
            {"securityId": "333", "netQty": -2},
        )
        assert open_positions_count(mock_client) == 2


class TestConsecutiveLosses:
    def test_blocks_after_loss_streak(self, mock_client, test_settings):
        test_settings.max_consecutive_losses = 3
        mock_client.get_positions.return_value = _positions(
            {"securityId": "1", "netQty": 0, "realizedProfit": 100},
            {"securityId": "2", "netQty": 0, "realizedProfit": -50},
            {"securityId": "3", "netQty": 0, "realizedProfit": -50},
            {"securityId": "4", "netQty": 0, "realizedProfit": -50},
        )
        result = check_order(
            qty=1, price=100.0, security_id="9",
            client=mock_client, settings=test_settings,
        )
        assert result is not None
        assert "MAX_CONSECUTIVE_LOSSES" in result

    def test_streak_reset_by_a_win(self, mock_client, test_settings):
        test_settings.max_consecutive_losses = 2
        # Latest closed trade is a win -> streak is 0.
        mock_client.get_positions.return_value = _positions(
            {"securityId": "1", "netQty": 0, "realizedProfit": -50},
            {"securityId": "2", "netQty": 0, "realizedProfit": -50},
            {"securityId": "3", "netQty": 0, "realizedProfit": 100},
        )
        result = check_order(
            qty=1, price=100.0, security_id="9",
            client=mock_client, settings=test_settings,
        )
        assert result is None

    def test_open_positions_do_not_count_as_losses(self, mock_client, test_settings):
        test_settings.max_consecutive_losses = 1
        # Negative unrealised on an *open* position must not trip the streak.
        mock_client.get_positions.return_value = _positions(
            {"securityId": "1", "netQty": 5, "realizedProfit": -500},
        )
        result = check_order(
            qty=1, price=100.0, security_id="9",
            client=mock_client, settings=test_settings,
        )
        assert result is None

    def test_consecutive_losses_helper(self, mock_client):
        mock_client.get_positions.return_value = _positions(
            {"securityId": "1", "netQty": 0, "realizedProfit": 10},
            {"securityId": "2", "netQty": 0, "realizedProfit": -5},
            {"securityId": "3", "netQty": 0, "realizedProfit": -5},
        )
        assert consecutive_losses(mock_client) == 2


class TestExitBypass:
    def test_exit_skips_qty_guard(self, mock_client, test_settings):
        test_settings.max_qty = 10
        result = check_order(
            qty=100, price=100.0, security_id="2885",
            client=mock_client, settings=test_settings, is_exit=True,
        )
        assert result is None

    def test_exit_skips_daily_loss_halt(self, mock_client, test_settings):
        test_settings.max_daily_loss = 5_000
        mock_client.get_positions.return_value = _positions(
            {"securityId": "1", "netQty": 5, "realizedProfit": -9000},
        )
        result = check_order(
            qty=1, price=100.0, security_id="1",
            client=mock_client, settings=test_settings, side="SELL", is_exit=True,
        )
        assert result is None

    def test_exit_does_not_query_positions(self, mock_client, test_settings):
        # An exit short-circuits before any broker call.
        check_order(
            qty=1, price=100.0, security_id="1",
            client=mock_client, settings=test_settings, is_exit=True,
        )
        mock_client.get_positions.assert_not_called()
