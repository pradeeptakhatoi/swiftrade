"""Order placement with risk guards and dry-run support."""

from __future__ import annotations

import logging

from dhanhq import dhanhq

from dhan_algo.client import ok
from dhan_algo.config import Settings, get_settings
from dhan_algo.journal import record as journal_record
from dhan_algo.market_data import ltp
from dhan_algo.risk import check_order

logger = logging.getLogger(__name__)


def place(
    client: dhanhq,
    security_id: str,
    side: str,
    qty: int,
    order_type: str = "MARKET",
    product: str = "INTRA",
    price: float = 0.0,
    segment: str | None = None,
    settings: Settings | None = None,
):
    """Place an order with risk guards. Honors dry-run mode."""
    settings = settings or get_settings()
    d_seg = segment or client.NSE
    txn = client.BUY if side.upper() == "BUY" else client.SELL
    otype = client.MARKET if order_type.upper() == "MARKET" else client.LIMIT
    prod = client.INTRA if product.upper() in ("INTRA", "INTRADAY") else client.CNC

    # ---- risk guards ----
    ref_price = price if price > 0 else (ltp(client, security_id, d_seg) or 0)
    block_reason = check_order(qty, ref_price, security_id, client, settings)
    if block_reason:
        logger.warning("BLOCKED: %s", block_reason)
        journal_record(
            security_id=security_id,
            side=side.upper(),
            qty=qty,
            order_type=order_type,
            product=product,
            price=ref_price,
            notional=ref_price * qty,
            status="blocked",
            detail=block_reason,
        )
        return None

    notional = ref_price * qty
    plan = (
        f"{side.upper()} {qty} of security {security_id} "
        f"[{order_type}/{product}] ~{notional:.0f} INR"
    )

    if not settings.dhan_live:
        logger.info("[DRY_RUN] would place: %s", plan)
        journal_record(
            security_id=security_id,
            side=side.upper(),
            qty=qty,
            order_type=order_type,
            product=product,
            price=ref_price,
            notional=notional,
            status="dry_run",
            detail=plan,
        )
        return {"status": "dry_run", "plan": plan}

    resp = client.place_order(
        security_id=str(security_id),
        exchange_segment=d_seg,
        transaction_type=txn,
        quantity=int(qty),
        order_type=otype,
        product_type=prod,
        price=float(price),
    )
    if ok(resp):
        logger.info("PLACED: %s", resp)
        journal_record(
            security_id=security_id,
            side=side.upper(),
            qty=qty,
            order_type=order_type,
            product=product,
            price=ref_price,
            notional=notional,
            status="placed",
            detail=str(resp),
        )
    else:
        logger.warning("ORDER FAILED: %s", resp)
        journal_record(
            security_id=security_id,
            side=side.upper(),
            qty=qty,
            order_type=order_type,
            product=product,
            price=ref_price,
            notional=notional,
            status="failed",
            detail=str(resp),
        )
    return resp


def show_positions(client: dhanhq) -> None:
    resp = client.get_positions()
    logger.info("Positions: %s", resp.get("data") if ok(resp) else resp)


def show_orders(client: dhanhq) -> None:
    resp = client.get_order_list()
    if ok(resp):
        logger.info("Orders: %s", resp.get("data"))
    else:
        logger.error("Orders query failed: %s", resp)
