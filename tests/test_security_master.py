"""Tests for security ID resolution across the full NSE equity universe."""

from unittest.mock import patch

import pandas as pd
import pytest

from dhan_algo.config import Settings
from dhan_algo.security_master import (
    KNOWN_IDS,
    reset_cache,
    resolve_security_id,
    resolve_security_ids,
)


@pytest.fixture(autouse=True)
def _clear_master_cache():
    """Each test starts with an empty scrip-master cache."""
    reset_cache()
    yield
    reset_cache()


def _master(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _nse_eq(symbol: str, sid, *, inst="ES") -> dict:
    return {
        "SEM_EXM_EXCH_ID": "NSE",
        "SEM_SMST_SECURITY_ID": sid,
        "SEM_TRADING_SYMBOL": symbol,
        "SEM_EXCH_INSTRUMENT_TYPE": inst,
    }


def test_known_id_lookup(mock_client):
    """Known symbols resolve without hitting the SDK."""
    result = resolve_security_id(mock_client, "RELIANCE")
    assert result == "2885"
    mock_client.fetch_security_list.assert_not_called()


def test_scrip_master_lookup(mock_client):
    """Unknown symbol is resolved from the scrip master DataFrame."""
    mock_client.fetch_security_list.return_value = _master([_nse_eq("TESTCO", 99999)])
    assert resolve_security_id(mock_client, "TESTCO", segment_hint="NSE") == "99999"


def test_unknown_symbol_returns_none(mock_client):
    """Symbol not in master returns None."""
    mock_client.fetch_security_list.return_value = _master([_nse_eq("OTHERCO", 11111)])
    assert resolve_security_id(mock_client, "NOSUCHSYMBOL") is None


def test_full_universe_many_symbols(mock_client):
    """Any symbol in the master resolves, not just the hardcoded five."""
    rows = [_nse_eq(f"SYM{i}", 1000 + i) for i in range(500)]
    mock_client.fetch_security_list.return_value = _master(rows)
    assert resolve_security_id(mock_client, "SYM0") == "1000"
    assert resolve_security_id(mock_client, "SYM499") == "1499"


def test_case_insensitive_and_trimmed(mock_client):
    mock_client.fetch_security_list.return_value = _master([_nse_eq("TATAMOTORS", 3456)])
    assert resolve_security_id(mock_client, "  tatamotors ") == "3456"


def test_equity_row_preferred_over_derivative(mock_client):
    """A symbol present as both equity and option resolves to the equity id."""
    mock_client.fetch_security_list.return_value = _master([
        _nse_eq("DUAL", 222, inst="OP"),   # option — must be excluded
        _nse_eq("DUAL", 111, inst="ES"),   # cash equity — the wanted id
    ])
    assert resolve_security_id(mock_client, "DUAL") == "111"


def test_float_security_id_is_normalised(mock_client):
    """Ids read as floats (e.g. from CSV) drop the trailing .0."""
    mock_client.fetch_security_list.return_value = _master([_nse_eq("FLOATER", 4200.0)])
    assert resolve_security_id(mock_client, "FLOATER") == "4200"


def test_master_downloaded_only_once(mock_client):
    """The scrip master is fetched once and cached for subsequent lookups."""
    mock_client.fetch_security_list.return_value = _master([
        _nse_eq("AAA", 1), _nse_eq("BBB", 2),
    ])
    resolve_security_id(mock_client, "AAA")
    resolve_security_id(mock_client, "BBB")
    assert mock_client.fetch_security_list.call_count == 1


def test_batch_resolution(mock_client):
    mock_client.fetch_security_list.return_value = _master([
        _nse_eq("AAA", 1), _nse_eq("BBB", 2),
    ])
    out = resolve_security_ids(mock_client, ["AAA", "RELIANCE", "BBB", "NOPE"])
    assert out == {"AAA": "1", "RELIANCE": "2885", "BBB": "2", "NOPE": None}
    assert mock_client.fetch_security_list.call_count == 1


def test_local_csv_fallback(tmp_path, mock_client):
    """When the download fails, resolution falls back to a local CSV."""
    csv = tmp_path / "scrip.csv"
    _master([_nse_eq("LOCALCO", 7777)]).to_csv(csv, index=False)
    mock_client.fetch_security_list.side_effect = RuntimeError("network down")

    settings = Settings(security_master_path=str(csv))
    with patch("dhan_algo.security_master.get_settings", return_value=settings):
        assert resolve_security_id(mock_client, "LOCALCO") == "7777"


def test_no_client_no_csv_falls_back_to_known_ids():
    """With neither a client nor a local CSV, only KNOWN_IDS resolve."""
    settings = Settings(security_master_path="does-not-exist.csv")
    with patch("dhan_algo.security_master.get_settings", return_value=settings):
        assert resolve_security_id(None, "RELIANCE") == "2885"
        assert resolve_security_id(None, "OBSCURECO") is None
