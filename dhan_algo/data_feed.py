"""Unified market-data fetch for the scanners.

Provides one entry point, :func:`fetch_universe`, that returns a
``dict[symbol -> OHLCV DataFrame]`` in the shape the scorers expect
(title-case ``Open/High/Low/Close/Volume`` columns).

Sources:

* **Dhan API** — one REST call per instrument via
  :func:`dhan_algo.backtest.fetch_historical`. Real-time/authoritative,
  requires a subscription and a resolvable security_id.
* **Yahoo Finance** — one batched download for the whole basket. Free but
  ~15-minute delayed.

Policy: try the requested source first; when Dhan is requested and a symbol
fails (no client, unresolved id, API error, empty data), fall back to Yahoo
for the affected symbols. All external calls are injectable so the module is
unit-testable without network access.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import date, timedelta

import pandas as pd

from dhan_algo.backtest import fetch_historical
from dhan_algo.security_master import resolve_security_id

logger = logging.getLogger(__name__)

DHAN = "Dhan API"
YAHOO = "Yahoo Finance"

_OHLCV = ("Open", "High", "Low", "Close", "Volume")
_PERIOD_DAYS = {"d": 1, "wk": 7, "mo": 30, "y": 365}


class DhanUnavailable(Exception):
    """A single-symbol Dhan fetch failed, carrying a human-readable reason."""


def _fetch_reason(exc: Exception) -> str:
    """Classify a Dhan fetch failure into a short, user-facing reason.

    Groups the many low-level error strings into a handful of actionable
    buckets (auth, subscription, rate limit) so the UI can tell the user
    *why* it fell back rather than dumping a raw traceback.
    """
    msg = str(exc).lower()
    if any(k in msg for k in ("token", "unauthor", "invalid client", "dh-901", "expired")):
        return "Dhan token invalid or expired"
    if any(k in msg for k in ("not subscribed", "subscription", "entitle", "not authorized for")):
        return "Data API not subscribed"
    if any(k in msg for k in ("rate", "too many", "429")):
        return "Dhan rate limit hit"
    if "scrip master" in msg or "not found" in msg:
        return "symbol not found in scrip master"
    if "no candles" in msg:
        return "no candles returned (Data API not subscribed or no history)"
    return str(exc) or "unknown Dhan error"


def period_to_dates(period: str, *, today: date | None = None) -> tuple[str, str]:
    """Convert a yfinance-style *period* (``"1y"``, ``"6mo"``, ``"5d"``) to
    ``(from_date, to_date)`` ISO date strings. Unknown formats default to 1 year.
    """
    today = today or date.today()
    m = re.fullmatch(r"\s*(\d+)\s*(d|wk|mo|y)\s*", period.lower())
    days = _PERIOD_DAYS[m.group(2)] * int(m.group(1)) if m else 365
    return (today - timedelta(days=days)).isoformat(), today.isoformat()


def bars_to_frame(bars: list[dict]) -> pd.DataFrame:
    """Convert Dhan bar dicts (lowercase ohlcv) to a title-case OHLCV frame."""
    if not bars:
        return pd.DataFrame()
    rows = [
        {
            "Date": str(b.get("timestamp", "")),
            "Open": float(b.get("open", 0)),
            "High": float(b.get("high", 0)),
            "Low": float(b.get("low", 0)),
            "Close": float(b.get("close", 0)),
            "Volume": int(b.get("volume", 0)),
        }
        for b in bars
    ]
    return pd.DataFrame(rows)


def fetch_dhan_frame(
    client,
    symbol: str,
    *,
    interval: str,
    from_date: str,
    to_date: str,
    interval_minutes: int = 15,
    resolve=resolve_security_id,
    fetch=fetch_historical,
) -> pd.DataFrame:
    """Fetch a single symbol's OHLCV frame from Dhan.

    Raises :class:`DhanUnavailable` with a human-readable reason when the
    symbol cannot be resolved or no candles come back, so callers can report
    *why* a fetch failed rather than silently falling back.
    """
    sid = resolve(client, symbol)
    if sid is None:
        raise DhanUnavailable("symbol not found in scrip master")
    if interval == "minute":
        bars = fetch(
            client, sid, from_date=from_date, to_date=to_date,
            interval="minute", interval_minutes=interval_minutes,
        )
    else:
        bars = fetch(
            client, sid, from_date=from_date, to_date=to_date, interval="day",
        )
    frame = bars_to_frame(bars)
    if frame.empty:
        raise DhanUnavailable(
            "no candles returned (Data API not subscribed or no history)"
        )
    return frame


def fetch_yahoo_frames(
    symbols: list[str],
    *,
    period: str,
    interval: str = "1d",
    downloader=None,
) -> dict[str, pd.DataFrame]:
    """Batch-download the basket from Yahoo and return per-symbol OHLCV frames."""
    if downloader is None:
        import yfinance as yf

        downloader = yf.download

    out: dict[str, pd.DataFrame] = {}
    tickers_yf = [f"{s}.NS" for s in symbols]
    df_all = downloader(
        tickers_yf, period=period, interval=interval,
        auto_adjust=True, group_by="ticker", threads=True,
    )
    if df_all is None or getattr(df_all, "empty", True):
        return out

    multi = isinstance(df_all.columns, pd.MultiIndex)
    for sym in symbols:
        tkr = f"{sym}.NS"
        try:
            df_bars = df_all[tkr] if multi else df_all
            df_bars = df_bars.dropna(how="all").reset_index()
            for col in _OHLCV:
                if col not in df_bars.columns and col.lower() in df_bars.columns:
                    df_bars.rename(columns={col.lower(): col}, inplace=True)
            out[sym] = df_bars
        except Exception:  # noqa: BLE001 - skip any malformed ticker slice
            pass
    return out


@dataclass
class FetchReport:
    """Diagnostics for a universe fetch, surfaced in the UI."""

    source_requested: str = ""
    ok: list[str] = field(default_factory=list)
    dhan_failed: list[str] = field(default_factory=list)
    fell_back: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    reasons: dict[str, str] = field(default_factory=dict)

    def dominant_reason(self) -> str:
        """Return the most common per-symbol failure reason, or ``""``.

        Lets the UI summarise *why* Dhan was unavailable in one phrase
        instead of listing a distinct reason for every symbol.
        """
        if not self.reasons:
            return ""
        counts: dict[str, int] = {}
        for reason in self.reasons.values():
            counts[reason] = counts.get(reason, 0) + 1
        return max(counts, key=counts.get)


def _yahoo_interval(interval: str, interval_minutes: int) -> str:
    return f"{interval_minutes}m" if interval == "minute" else "1d"


def fetch_universe(
    symbols: list[str],
    *,
    source: str,
    interval: str = "day",
    period: str = "1y",
    min_bars: int = 1,
    client=None,
    fallback: bool = True,
    pace: float = 0.0,
    interval_minutes: int = 15,
    today: date | None = None,
    dhan_fetch=fetch_dhan_frame,
    yahoo_fetch=fetch_yahoo_frames,
    sleep=time.sleep,
) -> tuple[dict[str, pd.DataFrame], FetchReport]:
    """Fetch OHLCV frames for *symbols* from *source*, Dhan-first with fallback.

    Returns ``(data, report)`` where *data* maps symbol -> OHLCV DataFrame,
    filtered to those with at least *min_bars* rows.
    """
    report = FetchReport(source_requested=source)
    frames: dict[str, pd.DataFrame] = {}
    yf_interval = _yahoo_interval(interval, interval_minutes)

    want_dhan = source == DHAN and client is not None
    if source == DHAN and client is None:
        # Requested Dhan but not logged in — degrade to Yahoo for everything.
        report.fell_back = list(symbols)
        for sym in symbols:
            report.reasons[sym] = "not logged in to Dhan"

    if want_dhan:
        from_date, to_date = period_to_dates(period, today=today)
        for sym in symbols:
            try:
                frame = dhan_fetch(
                    client, sym, interval=interval,
                    from_date=from_date, to_date=to_date,
                    interval_minutes=interval_minutes,
                )
            except Exception as exc:  # noqa: BLE001 - isolate per-symbol failures
                logger.warning("Dhan fetch failed for %s: %s", sym, exc)
                report.reasons[sym] = _fetch_reason(exc)
                frame = None
            if frame is not None and not frame.empty:
                frames[sym] = frame
            else:
                report.dhan_failed.append(sym)
                report.reasons.setdefault(sym, "no data returned")
            if pace:
                sleep(pace)

        if fallback and report.dhan_failed:
            recovered = yahoo_fetch(
                report.dhan_failed, period=period, interval=yf_interval,
            )
            for sym, frame in recovered.items():
                if frame is not None and not frame.empty:
                    frames[sym] = frame
                    report.fell_back.append(sym)
    else:
        frames = yahoo_fetch(symbols, period=period, interval=yf_interval)

    # Enforce the minimum-bar requirement uniformly.
    data: dict[str, pd.DataFrame] = {}
    for sym, frame in frames.items():
        if frame is not None and len(frame) >= min_bars:
            data[sym] = frame
    report.ok = list(data.keys())
    # Anything requested but not usable (unresolved, empty, or too short).
    report.skipped = [s for s in symbols if s not in data]
    return data, report
