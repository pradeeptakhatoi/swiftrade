"""Market data helpers (LTP) and WebSocket ticker."""

from __future__ import annotations

import logging
import threading
from typing import Callable

from dhanhq import dhanhq, MarketFeed

from dhan_algo.client import ok

logger = logging.getLogger(__name__)

_SEGMENT_MAP: dict[str, int] = {
    "NSE_EQ": MarketFeed.NSE,
    "NSE": MarketFeed.NSE,
    "BSE": MarketFeed.BSE,
    "MCX": MarketFeed.MCX,
}


def ltp(client: dhanhq, security_id: str, segment: str | None = None) -> float | None:
    """Last traded price for one instrument. Defaults to NSE equity."""
    segment = segment or client.NSE
    resp = client.ticker_data(securities={segment: [int(security_id)]})
    if not ok(resp):
        logger.error("LTP fetch failed: %s", resp)
        return None
    # Response nests as data -> data -> {segment} -> {security_id} -> last_price
    try:
        node = resp["data"]["data"][segment][str(security_id)]
        return float(node.get("last_price"))
    except (KeyError, TypeError, ValueError):
        logger.error("Unexpected LTP payload: %s", resp.get("data"))
        return None


class WebSocketTicker:
    """Wraps ``MarketFeed`` to implement the ``Ticker`` protocol."""

    def __init__(
        self,
        client: dhanhq,
        security_ids: list[str],
        segment: str | None = None,
        on_tick_callback: Callable[[str, str | None], None] | None = None,
    ):
        self._client = client
        self._segment = segment or "NSE_EQ"
        self._callback = on_tick_callback
        self._lock = threading.Lock()
        self._prices: dict[str, float] = {}
        self._feed: MarketFeed | None = None

        exchange_code = _SEGMENT_MAP.get(self._segment, MarketFeed.NSE)
        self._instruments = [(exchange_code, str(sid), MarketFeed.Ticker) for sid in security_ids]

    def get_ltp(self, security_id: str, segment: str | None = None) -> float | None:
        with self._lock:
            return self._prices.get(security_id)

    @property
    def watchlist(self) -> dict[str, float]:
        with self._lock:
            return dict(self._prices)

    def _on_ticks(self, tick_data: dict) -> None:
        sid = str(tick_data.get("security_id", ""))
        price = tick_data.get("LTP")
        if not sid or price is None:
            return
        with self._lock:
            self._prices[sid] = float(price)
        if self._callback:
            self._callback(sid, self._segment)

    def start(self) -> None:
        """Connect and start receiving ticks in a background thread."""
        self._feed = MarketFeed(
            self._client,
            instruments=self._instruments,
            on_ticks=self._on_ticks,
        )
        self._feed.run_forever()
        logger.info("WebSocket feed started for %d instruments.", len(self._instruments))

    def stop(self) -> None:
        if self._feed is not None:
            self._feed.close()
            logger.info("WebSocket feed stopped.")

    def subscribe(self, security_id: str, segment: str | None = None) -> None:
        exchange_code = _SEGMENT_MAP.get(segment or self._segment, MarketFeed.NSE)
        instruments = [(exchange_code, str(security_id), MarketFeed.Ticker)]
        if self._feed is not None:
            self._feed.subscribe(instruments)

    def unsubscribe(self, security_id: str, segment: str | None = None) -> None:
        exchange_code = _SEGMENT_MAP.get(segment or self._segment, MarketFeed.NSE)
        instruments = [(exchange_code, str(security_id), MarketFeed.Ticker)]
        if self._feed is not None:
            self._feed.unsubscribe(instruments)
