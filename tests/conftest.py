"""Shared test fixtures."""

from unittest.mock import MagicMock

import pytest

from dhan_algo.config import Settings


@pytest.fixture()
def test_settings() -> Settings:
    return Settings(
        dhan_client_id="1000000001",
        dhan_access_token="test-token",
        dhan_live=False,
        max_qty=50,
        max_order_value=50_000,
        max_daily_loss=10_000,
        strategy_interval=60,
        log_level="DEBUG",
        journal_path="test_trades.csv",
    )


@pytest.fixture()
def mock_client() -> MagicMock:
    client = MagicMock()
    client.NSE = "NSE_EQ"
    client.BUY = "BUY"
    client.SELL = "SELL"
    client.MARKET = "MARKET"
    client.LIMIT = "LIMIT"
    client.INTRA = "INTRADAY"
    client.CNC = "CNC"
    return client
