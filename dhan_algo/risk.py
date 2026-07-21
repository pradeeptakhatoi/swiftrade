"""Risk guards: per-order limits and daily-loss cap."""

from __future__ import annotations

import logging

from dhanhq import dhanhq

from dhan_algo.client import ok
from dhan_algo.config import Settings, get_settings
from dhan_algo.market_data import ltp

logger = logging.getLogger(__name__)


def _realized_pnl(client: dhanhq) -> float:
    """Sum realizedProfit across all positions."""
    resp = client.get_positions()
    if not ok(resp):
        return 0.0
    positions = resp.get("data") or []
    return sum(float(p.get("realizedProfit", 0)) for p in positions)


def check_order(
    qty: int,
    price: float,
    security_id: str,
    client: dhanhq,
    settings: Settings | None = None,
) -> str | None:
    """Return an error message if the order should be blocked, else None."""
    settings = settings or get_settings()

    if qty > settings.max_qty:
        return f"qty {qty} exceeds MAX_QTY {settings.max_qty}"

    ref_price = price if price > 0 else (ltp(client, security_id) or 0)
    notional = ref_price * qty
    if notional > settings.max_order_value:
        return (
            f"order value ~{notional:.0f} exceeds "
            f"MAX_ORDER_VALUE {settings.max_order_value}"
        )

    daily_loss = _realized_pnl(client)
    if daily_loss < 0 and abs(daily_loss) >= settings.max_daily_loss:
        return (
            f"daily realized loss {daily_loss:.0f} has breached "
            f"MAX_DAILY_LOSS {settings.max_daily_loss}"
        )

    return None


def kill_switch(client: dhanhq) -> dict:
    """Activate the Dhan kill switch — disables trading for the rest of the day."""
    resp = client.kill_switch("activate")
    logger.info("Kill switch activated: %s", resp)
    return resp
