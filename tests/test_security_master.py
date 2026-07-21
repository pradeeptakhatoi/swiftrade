"""Tests for security ID resolution."""

from unittest.mock import MagicMock

import pandas as pd

from dhan_algo.security_master import KNOWN_IDS, resolve_security_id


def test_known_id_lookup(mock_client):
    """Known symbols resolve without hitting the SDK."""
    result = resolve_security_id(mock_client, "RELIANCE")
    assert result == "2885"
    mock_client.fetch_security_list.assert_not_called()


def test_scrip_master_lookup(mock_client):
    """Unknown symbol is resolved from the scrip master DataFrame."""
    df = pd.DataFrame({
        "SEM_SMST_SECURITY_ID": [99999],
        "SEM_TRADING_SYMBOL": ["TESTCO"],
        "SEM_EXCH_INSTRUMENT_TYPE": ["ES"],
        "SEM_EXM_EXCH_ID": ["NSE"],
    })
    mock_client.fetch_security_list.return_value = df
    result = resolve_security_id(mock_client, "TESTCO", segment_hint="NSE")
    assert result == "99999"


def test_unknown_symbol_returns_none(mock_client):
    """Symbol not in master returns None."""
    df = pd.DataFrame({
        "SEM_SMST_SECURITY_ID": [11111],
        "SEM_TRADING_SYMBOL": ["OTHERCO"],
        "SEM_EXM_EXCH_ID": ["NSE"],
    })
    mock_client.fetch_security_list.return_value = df
    result = resolve_security_id(mock_client, "NOSUCHSYMBOL")
    assert result is None
