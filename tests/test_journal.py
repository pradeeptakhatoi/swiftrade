"""Tests for the trade journal CSV writer."""

import csv
import os
from unittest.mock import patch

from dhan_algo.journal import _FIELDNAMES, record


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
