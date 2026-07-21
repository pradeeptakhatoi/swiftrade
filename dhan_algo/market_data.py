"""Market data helpers (LTP)."""

from __future__ import annotations

from dhanhq import dhanhq

from dhan_algo.client import ok


def ltp(client: dhanhq, security_id: str, segment: str | None = None) -> float | None:
    """Last traded price for one instrument. Defaults to NSE equity."""
    segment = segment or client.NSE
    resp = client.ticker_data(securities={segment: [int(security_id)]})
    if not ok(resp):
        print("LTP fetch failed:", resp)
        return None
    # Response nests as data -> data -> {segment} -> {security_id} -> last_price
    try:
        node = resp["data"]["data"][segment][str(security_id)]
        return float(node.get("last_price"))
    except (KeyError, TypeError, ValueError):
        print("Unexpected LTP payload:", resp.get("data"))
        return None
