"""Risk guards: per-order limits, daily-loss cap, position & loss-streak halts."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from dhanhq import dhanhq

from dhan_algo.client import ok
from dhan_algo.config import Settings, get_settings
from dhan_algo.market_data import ltp

logger = logging.getLogger(__name__)


def _positions(client: dhanhq) -> list[dict]:
    """Return the raw positions list, or ``[]`` if the query fails."""
    resp = client.get_positions()
    if not ok(resp):
        return []
    return resp.get("data") or []


def _net_qty(p: dict) -> int:
    """Signed net quantity of a position (0 means flat/closed)."""
    try:
        return int(float(p.get("netQty", p.get("net_qty", 0)) or 0))
    except (TypeError, ValueError):
        return 0


def _security_id(p: dict) -> str:
    return str(p.get("securityId", p.get("security_id", "")))


@dataclass
class PositionSnapshot:
    """One-shot view of account positions used by the risk guards."""

    realized_pnl: float
    open_count: int
    open_ids: set[str]
    loss_streak: int


def _snapshot(client: dhanhq) -> PositionSnapshot:
    """Compute realized P&L, open-position count, and the trailing loss streak.

    The loss streak is the number of *closed* positions (net qty 0) at the tail
    of the broker's positions list whose ``realizedProfit`` is negative; the
    first non-loss breaks the run.
    """
    positions = _positions(client)
    realized = sum(float(p.get("realizedProfit", 0) or 0) for p in positions)
    open_ids = {_security_id(p) for p in positions if _net_qty(p) != 0}
    closed = [p for p in positions if _net_qty(p) == 0]

    streak = 0
    for p in reversed(closed):
        if float(p.get("realizedProfit", 0) or 0) < 0:
            streak += 1
        else:
            break

    return PositionSnapshot(
        realized_pnl=realized,
        open_count=len(open_ids),
        open_ids=open_ids,
        loss_streak=streak,
    )


def _realized_pnl(client: dhanhq) -> float:
    """Sum realizedProfit across all positions."""
    return _snapshot(client).realized_pnl


def open_positions_count(client: dhanhq) -> int:
    """Number of distinct securities with a non-zero net position."""
    return _snapshot(client).open_count


def consecutive_losses(client: dhanhq) -> int:
    """Trailing run of losing closed positions (see :func:`_snapshot`)."""
    return _snapshot(client).loss_streak


def check_order(
    qty: int,
    price: float,
    security_id: str,
    client: dhanhq,
    settings: Settings | None = None,
    *,
    side: str = "BUY",
) -> str | None:
    """Return an error message if the order should be blocked, else None.

    The per-order size guards (MAX_QTY / MAX_ORDER_VALUE) and the MAX_DAILY_LOSS
    halt apply to every order. The concurrent-position cap and consecutive-loss
    cooldown apply only to *opening* orders (``side="BUY"``) so that closing an
    existing position is never blocked.
    """
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

    snap = _snapshot(client)

    if snap.realized_pnl < 0 and abs(snap.realized_pnl) >= settings.max_daily_loss:
        return (
            f"daily realized loss {snap.realized_pnl:.0f} has breached "
            f"MAX_DAILY_LOSS {settings.max_daily_loss}"
        )

    if side.upper() == "BUY":
        max_losses = getattr(settings, "max_consecutive_losses", 0)
        if max_losses and snap.loss_streak >= max_losses:
            return (
                f"{snap.loss_streak} consecutive losses reached "
                f"MAX_CONSECUTIVE_LOSSES {max_losses}; trading paused"
            )

        max_open = getattr(settings, "max_open_positions", 0)
        if (
            max_open
            and str(security_id) not in snap.open_ids
            and snap.open_count >= max_open
        ):
            return (
                f"open positions {snap.open_count} at "
                f"MAX_OPEN_POSITIONS {max_open}"
            )

    return None


def kill_switch(client: dhanhq) -> dict:
    """Activate the Dhan kill switch — disables trading for the rest of the day."""
    resp = client.kill_switch("activate")
    logger.info("Kill switch activated: %s", resp)
    return resp
