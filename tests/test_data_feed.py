"""Tests for the unified market-data fetch (Dhan-first, Yahoo fallback)."""

from datetime import date

import pandas as pd

from dhan_algo.data_feed import (
    DHAN,
    YAHOO,
    DhanUnavailable,
    _fetch_reason,
    bars_to_frame,
    fetch_universe,
    period_to_dates,
)


# --- period_to_dates --------------------------------------------------------

class TestPeriodToDates:
    def test_years(self):
        frm, to = period_to_dates("1y", today=date(2024, 1, 10))
        assert to == "2024-01-10"
        assert frm == "2023-01-10"

    def test_days(self):
        frm, to = period_to_dates("5d", today=date(2024, 1, 10))
        assert frm == "2024-01-05"

    def test_months(self):
        frm, _ = period_to_dates("6mo", today=date(2024, 6, 30))
        assert frm == "2024-01-02"  # 180 days back

    def test_unknown_defaults_to_one_year(self):
        frm, to = period_to_dates("garbage", today=date(2024, 1, 10))
        assert frm == "2023-01-10"  # 365 days back, same as "1y"


# --- bars_to_frame ----------------------------------------------------------

class TestBarsToFrame:
    def test_converts_lowercase_bars_to_titlecase(self):
        bars = [
            {"timestamp": "t1", "open": 1, "high": 2, "low": 0.5,
             "close": 1.5, "volume": 100},
            {"timestamp": "t2", "open": 1.5, "high": 3, "low": 1,
             "close": 2.5, "volume": 200},
        ]
        df = bars_to_frame(bars)
        assert list(df.columns) == ["Date", "Open", "High", "Low", "Close", "Volume"]
        assert len(df) == 2
        assert df.iloc[1]["Close"] == 2.5

    def test_empty_bars_returns_empty_frame(self):
        assert bars_to_frame([]).empty


# --- helpers for fetch_universe ---------------------------------------------

def _frame(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": range(n),
            "Open": [1.0] * n, "High": [1.0] * n, "Low": [1.0] * n,
            "Close": [1.0] * n, "Volume": [1] * n,
        }
    )


def _fake_yahoo(frames: dict[str, pd.DataFrame]):
    def _fetch(symbols, *, period, interval="1d"):
        return {s: frames[s] for s in symbols if s in frames}
    return _fetch


# --- fetch_universe ---------------------------------------------------------

class TestFetchUniverse:
    def test_yahoo_source_uses_batch(self):
        yf = _fake_yahoo({"AAA": _frame(10), "BBB": _frame(10)})
        data, report = fetch_universe(
            ["AAA", "BBB"], source=YAHOO, min_bars=5, yahoo_fetch=yf,
        )
        assert set(data) == {"AAA", "BBB"}
        assert report.source_requested == YAHOO
        assert not report.fell_back

    def test_min_bars_filters_short_frames(self):
        yf = _fake_yahoo({"AAA": _frame(10), "BBB": _frame(3)})
        data, report = fetch_universe(
            ["AAA", "BBB"], source=YAHOO, min_bars=5, yahoo_fetch=yf,
        )
        assert set(data) == {"AAA"}
        assert report.skipped == ["BBB"]

    def test_dhan_source_uses_dhan_fetch(self):
        def dhan(client, sym, **kw):
            return _frame(250)
        data, report = fetch_universe(
            ["AAA"], source=DHAN, min_bars=220, client=object(),
            dhan_fetch=dhan, pace=0,
        )
        assert set(data) == {"AAA"}
        assert report.ok == ["AAA"]
        assert not report.fell_back

    def test_dhan_failure_falls_back_to_yahoo(self):
        def dhan(client, sym, **kw):
            return None  # simulate unresolved / empty
        yf = _fake_yahoo({"AAA": _frame(250)})
        data, report = fetch_universe(
            ["AAA"], source=DHAN, min_bars=220, client=object(),
            dhan_fetch=dhan, yahoo_fetch=yf, pace=0,
        )
        assert set(data) == {"AAA"}
        assert report.dhan_failed == ["AAA"]
        assert report.fell_back == ["AAA"]

    def test_dhan_exception_is_isolated_and_recovers(self):
        def dhan(client, sym, **kw):
            if sym == "BAD":
                raise RuntimeError("api down")
            return _frame(250)
        yf = _fake_yahoo({"BAD": _frame(250)})
        data, report = fetch_universe(
            ["GOOD", "BAD"], source=DHAN, min_bars=220, client=object(),
            dhan_fetch=dhan, yahoo_fetch=yf, pace=0,
        )
        assert set(data) == {"GOOD", "BAD"}
        assert report.dhan_failed == ["BAD"]
        assert report.fell_back == ["BAD"]

    def test_dhan_without_client_degrades_to_yahoo(self):
        yf = _fake_yahoo({"AAA": _frame(250)})
        called = {"dhan": False}

        def dhan(*a, **k):
            called["dhan"] = True
            return _frame(250)

        data, report = fetch_universe(
            ["AAA"], source=DHAN, min_bars=220, client=None,
            dhan_fetch=dhan, yahoo_fetch=yf,
        )
        assert set(data) == {"AAA"}
        assert called["dhan"] is False  # never attempted without a client
        assert report.fell_back == ["AAA"]

    def test_no_fallback_leaves_failed_symbols_out(self):
        def dhan(client, sym, **kw):
            return None
        yf = _fake_yahoo({"AAA": _frame(250)})
        data, report = fetch_universe(
            ["AAA"], source=DHAN, min_bars=220, client=object(),
            dhan_fetch=dhan, yahoo_fetch=yf, fallback=False, pace=0,
        )
        assert data == {}
        assert report.skipped == ["AAA"]
        assert report.fell_back == []

    def test_dhan_minute_passes_interval_minutes(self):
        seen = {}

        def dhan(client, sym, *, interval, from_date, to_date, interval_minutes):
            seen["interval"] = interval
            seen["interval_minutes"] = interval_minutes
            return _frame(30)

        fetch_universe(
            ["AAA"], source=DHAN, interval="minute", period="5d",
            min_bars=20, client=object(), dhan_fetch=dhan,
            interval_minutes=15, pace=0,
        )
        assert seen["interval"] == "minute"
        assert seen["interval_minutes"] == 15


# --- diagnostic reasons -----------------------------------------------------

class TestFetchReason:
    def test_token_error(self):
        assert _fetch_reason(RuntimeError("Invalid token")) == "Dhan token invalid or expired"

    def test_subscription_error(self):
        exc = DhanUnavailable("no candles returned (Data API not subscribed or no history)")
        assert _fetch_reason(exc) == "Data API not subscribed"

    def test_rate_limit_error(self):
        assert _fetch_reason(RuntimeError("429 too many requests")) == "Dhan rate limit hit"

    def test_scrip_master_error(self):
        exc = DhanUnavailable("symbol not found in scrip master")
        assert _fetch_reason(exc) == "symbol not found in scrip master"

    def test_unknown_error_passes_through(self):
        assert _fetch_reason(RuntimeError("weird failure")) == "weird failure"


class TestFetchReportReasons:
    def test_reason_recorded_on_exception(self):
        def dhan(client, sym, **kw):
            raise DhanUnavailable("symbol not found in scrip master")
        yf = _fake_yahoo({"AAA": _frame(250)})
        _, report = fetch_universe(
            ["AAA"], source=DHAN, min_bars=220, client=object(),
            dhan_fetch=dhan, yahoo_fetch=yf, pace=0,
        )
        assert report.reasons["AAA"] == "symbol not found in scrip master"
        assert report.dominant_reason() == "symbol not found in scrip master"

    def test_no_client_reason_is_not_logged_in(self):
        yf = _fake_yahoo({"AAA": _frame(250)})
        _, report = fetch_universe(
            ["AAA"], source=DHAN, min_bars=220, client=None, yahoo_fetch=yf,
        )
        assert report.reasons["AAA"] == "not logged in to Dhan"
        assert report.dominant_reason() == "not logged in to Dhan"

    def test_dominant_reason_picks_most_common(self):
        def dhan(client, sym, **kw):
            if sym == "SOLO":
                raise RuntimeError("429 too many requests")
            raise DhanUnavailable("no candles returned (Data API not subscribed or no history)")
        yf = _fake_yahoo({s: _frame(250) for s in ("A", "B", "SOLO")})
        _, report = fetch_universe(
            ["A", "B", "SOLO"], source=DHAN, min_bars=220, client=object(),
            dhan_fetch=dhan, yahoo_fetch=yf, pace=0,
        )
        assert report.dominant_reason() == "Data API not subscribed"

    def test_empty_reasons_dominant_is_blank(self):
        yf = _fake_yahoo({"AAA": _frame(250)})
        _, report = fetch_universe(
            ["AAA"], source=YAHOO, min_bars=5, yahoo_fetch=yf,
        )
        assert report.dominant_reason() == ""
