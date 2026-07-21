"""Tests for risk guards and daily-loss cap."""

from unittest.mock import patch

from dhan_algo.risk import check_order


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
