"""Market data helpers (LTP)."""

from __future__ import annotations

import logging

from dhanhq import dhanhq

from dhan_algo.client import ok

logger = logging.getLogger(__name__)


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
