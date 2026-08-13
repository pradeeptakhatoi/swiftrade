"""Tests for the trade journal CSV writer."""

import csv
import os
from unittest.mock import patch

from dhan_algo.journal import (
    _CLOSED_FIELDNAMES,
    _FIELDNAMES,
    closed_trades_path,
    record,
    record_closed_trade,
)


def test_creates_csv_with_headers(tmp_path, test_settings):
    """First call creates the file with a header row."""
    path = str(tmp_path / "journal.csv")
    test_settings.journal_path = path
    with patch("dhan_algo.journal.get_settings", return_value=test_settings):
        record(
            security_id="2885",
            side="BUY",
            qty=10,
            order_type="MARKET",
            product="INTRA",
            price=2500.0,
            notional=25000.0,
            status="placed",
            detail="test order",
        )
    assert os.path.exists(path)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        assert list(reader.fieldnames) == _FIELDNAMES
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["security_id"] == "2885"
        assert rows[0]["status"] == "placed"


def test_appends_multiple_rows(tmp_path, test_settings):
    """Multiple calls append rows without duplicating headers."""
    path = str(tmp_path / "journal.csv")
    test_settings.journal_path = path
    with patch("dhan_algo.journal.get_settings", return_value=test_settings):
        for status in ("placed", "blocked", "dry_run"):
            record(
                security_id="2885",
                side="BUY",
                qty=1,
                order_type="MARKET",
                product="INTRA",
                price=100.0,
                notional=100.0,
                status=status,
            )
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 3
        assert [r["status"] for r in rows] == ["placed", "blocked", "dry_run"]


class TestClosedTradesPath:
    def test_derives_from_base(self):
        assert closed_trades_path("trades.csv") == "trades.closed.csv"

    def test_handles_directories(self):
        assert closed_trades_path(os.path.join("logs", "j.csv")) == os.path.join(
            "logs", "j.closed.csv"
        )

    def test_no_extension_defaults_csv(self):
        assert closed_trades_path("trades") == "trades.closed.csv"

    def test_uses_settings_when_omitted(self, test_settings):
        test_settings.journal_path = "mine.csv"
        with patch("dhan_algo.journal.get_settings", return_value=test_settings):
            assert closed_trades_path() == "mine.closed.csv"


class TestRecordClosedTrade:
    def _write(self, path):
        record_closed_trade(
            strategy="swing",
            symbol="RELIANCE",
            entry_time="2022-01-03",
            exit_time="2022-01-10",
            qty=10,
            entry_price=2500.0,
            exit_price=2600.0,
            exit_reason="target",
            gross_pnl=1000.0,
            cost=40.0,
            net_pnl=960.0,
            r_multiple=2.0,
            path=path,
        )

    def test_creates_with_header(self, tmp_path):
        path = str(tmp_path / "trades.closed.csv")
        self._write(path)
        assert os.path.exists(path)
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            assert list(reader.fieldnames) == _CLOSED_FIELDNAMES
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["symbol"] == "RELIANCE"
        assert rows[0]["net_pnl"] == "960.0"
        assert rows[0]["exit_reason"] == "target"

    def test_appends_without_duplicate_header(self, tmp_path):
        path = str(tmp_path / "trades.closed.csv")
        self._write(path)
        self._write(path)
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2

    def test_defaults_to_derived_path(self, tmp_path, test_settings):
        test_settings.journal_path = str(tmp_path / "trades.csv")
        with patch("dhan_algo.journal.get_settings", return_value=test_settings):
            record_closed_trade(
                strategy="intraday", symbol="TCS",
                entry_time="t0", exit_time="t1", qty=1,
                entry_price=100.0, exit_price=101.0, exit_reason="eod",
                gross_pnl=1.0, cost=0.1, net_pnl=0.9, r_multiple=0.5,
            )
        expected = str(tmp_path / "trades.closed.csv")
        assert os.path.exists(expected)
