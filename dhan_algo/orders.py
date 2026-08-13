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
    trigger_price: float = 0.0,
    segment: str | None = None,
    settings: Settings | None = None,
    is_exit: bool = False,
):
    """Place an order with risk guards. Honors dry-run mode.

    ``is_exit=True`` marks a risk-reducing exit so the trading-halt guards are
    skipped and the position can always be closed.
    """
    settings = settings or get_settings()
    d_seg = segment or client.NSE
    txn = client.BUY if side.upper() == "BUY" else client.SELL

    ot = order_type.upper()
    if ot == "MARKET":
        otype = client.MARKET
    elif ot in ("SL", "STOP_LOSS"):
        otype = client.SL
    elif ot in ("SL-M", "SL_MARKET"):
        otype = client.SLM
    else:
        otype = client.LIMIT

    prod = client.INTRA if product.upper() in ("INTRA", "INTRADAY") else client.CNC

    # ---- risk guards ----
    ref_price = price if price > 0 else (ltp(client, security_id, d_seg) or 0)
    block_reason = check_order(
        qty, ref_price, security_id, client, settings, side=side, is_exit=is_exit
    )
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

    order_kwargs = dict(
        security_id=str(security_id),
        exchange_segment=d_seg,
        transaction_type=txn,
        quantity=int(qty),
        order_type=otype,
        product_type=prod,
        price=float(price),
    )
    if trigger_price > 0:
        order_kwargs["trigger_price"] = float(trigger_price)

    resp = client.place_order(**order_kwargs)
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


def calculate_position_size(
    entry_price: float,
    stop_loss_price: float,
    risk_amount: float,
    max_qty: int,
    max_order_value: float,
    *,
    capital: float = 0.0,
    max_position_pct: float = 0.0,
) -> int:
    """Calculate quantity based on risk amount and stop-loss distance.

    qty = floor(risk_amount / abs(entry - stop_loss)),
    clamped to *max_qty* and *max_order_value*.

    When both *capital* and *max_position_pct* are positive, the position is
    additionally capped so its notional never exceeds that percentage of
    capital (capital-aware sizing).

    Returns 0 if inputs are invalid.
    """
    risk_per_share = abs(entry_price - stop_loss_price)
    if risk_per_share < 0.01 or entry_price <= 0:
        return 0
    qty = int(risk_amount / risk_per_share)
    qty = min(qty, max_qty)
    max_qty_by_value = int(max_order_value / entry_price)
    qty = min(qty, max_qty_by_value)
    if capital > 0 and max_position_pct > 0:
        cap_value = capital * (max_position_pct / 100.0)
        qty = min(qty, int(cap_value / entry_price))
    return max(qty, 0)


def place_bracket(
    client: dhanhq,
    security_id: str,
    side: str,
    qty: int,
    entry_price: float,
    stop_loss_price: float,
    target_price: float,
    trailing_jump: float = 0.0,
    segment: str | None = None,
    settings: Settings | None = None,
) -> dict | None:
    """Place a bracket order using Dhan Super Order API.

    Sends entry + stop-loss + target as a single atomic order.
    Returns the API response dict, or ``None`` if blocked by risk guards.
    """
    settings = settings or get_settings()
    d_seg = segment or client.NSE
    txn = client.BUY if side.upper() == "BUY" else client.SELL

    ref_price = entry_price if entry_price > 0 else (ltp(client, security_id, d_seg) or 0)
    block_reason = check_order(qty, ref_price, security_id, client, settings, side=side)
    if block_reason:
        logger.warning("BLOCKED bracket: %s", block_reason)
        journal_record(
            security_id=security_id, side=side.upper(), qty=qty,
            order_type="SUPER_ORDER", product="INTRA",
            price=ref_price, notional=ref_price * qty,
            status="blocked", detail=block_reason,
        )
        return None

    notional = ref_price * qty
    plan = (
        f"{side.upper()} {qty} of security {security_id} "
        f"[SUPER_ORDER/INTRA] entry={entry_price:.2f} "
        f"SL={stop_loss_price:.2f} target={target_price:.2f} ~{notional:.0f} INR"
    )

    if not settings.dhan_live:
        logger.info("[DRY_RUN] would place bracket: %s", plan)
        journal_record(
            security_id=security_id, side=side.upper(), qty=qty,
            order_type="SUPER_ORDER", product="INTRA",
            price=ref_price, notional=notional,
            status="dry_run", detail=plan,
        )
        return {"status": "dry_run", "plan": plan}

    resp = client.place_super_order(
        security_id=str(security_id),
        exchange_segment=d_seg,
        transaction_type=txn,
        quantity=int(qty),
        order_type=client.LIMIT,
        product_type=client.INTRA,
        price=float(entry_price),
        targetPrice=float(target_price),
        stopLossPrice=float(stop_loss_price),
        trailingJump=float(trailing_jump),
    )

    status = "placed" if ok(resp) else "failed"
    log_fn = logger.info if status == "placed" else logger.warning
    log_fn("BRACKET %s: %s", status.upper(), resp)
    journal_record(
        security_id=security_id, side=side.upper(), qty=qty,
        order_type="SUPER_ORDER", product="INTRA",
        price=ref_price, notional=notional,
        status=status, detail=str(resp),
    )
    return resp


def place_with_sl_target(
    client: dhanhq,
    security_id: str,
    side: str,
    qty: int,
    entry_price: float,
    stop_loss_price: float,
    target_price: float,
    order_type: str = "LIMIT",
    product: str = "CNC",
    segment: str | None = None,
    settings: Settings | None = None,
) -> dict:
    """Place entry + separate SL + target orders (for CNC/swing trades).

    Returns a dict with ``entry``, ``stop_loss``, and ``target`` responses.
    """
    exit_side = "SELL" if side.upper() == "BUY" else "BUY"

    entry_resp = place(
        client, security_id,
        side=side, qty=qty, order_type=order_type, product=product,
        price=entry_price, segment=segment, settings=settings,
    )

    sl_resp = place(
        client, security_id,
        side=exit_side, qty=qty, order_type="SL", product=product,
        price=stop_loss_price, trigger_price=stop_loss_price,
        segment=segment, settings=settings,
    )

    target_resp = place(
        client, security_id,
        side=exit_side, qty=qty, order_type="LIMIT", product=product,
        price=target_price, segment=segment, settings=settings,
    )

    return {"entry": entry_resp, "stop_loss": sl_resp, "target": target_resp}


def show_positions(client: dhanhq) -> None:
    resp = client.get_positions()
    logger.info("Positions: %s", resp.get("data") if ok(resp) else resp)


def show_orders(client: dhanhq) -> None:
    resp = client.get_order_list()
    if ok(resp):
        logger.info("Orders: %s", resp.get("data"))
    else:
        logger.error("Orders query failed: %s", resp)
