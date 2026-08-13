"""Trade journal -- append order events to a CSV file."""

from __future__ import annotations

import csv
import logging
import os
from datetime import datetime, timezone

from dhan_algo.config import get_settings

logger = logging.getLogger(__name__)

_FIELDNAMES = [
    "timestamp",
    "security_id",
    "side",
    "qty",
    "order_type",
    "product",
    "price",
    "notional",
    "status",
    "detail",
]

_CLOSED_FIELDNAMES = [
    "timestamp",
    "strategy",
    "symbol",
    "entry_time",
    "exit_time",
    "qty",
    "entry_price",
    "exit_price",
    "exit_reason",
    "gross_pnl",
    "cost",
    "net_pnl",
    "r_multiple",
]


def _ensure_header(path: str, fieldnames: list[str] | None = None) -> None:
    """Create the CSV file with a header row if it does not exist."""
    if os.path.exists(path):
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames or _FIELDNAMES)
        writer.writeheader()
    logger.debug("Created journal file: %s", path)


def closed_trades_path(path: str | None = None) -> str:
    """Derive the round-trip (closed-trade) log path from the journal path.

    ``trades.csv`` -> ``trades.closed.csv``. When *path* is omitted, the
    configured ``journal_path`` is used as the base.
    """
    base = path if path is not None else get_settings().journal_path
    root, ext = os.path.splitext(base)
    ext = ext or ".csv"
    return f"{root}.closed{ext}"


def record(
    security_id: str,
    side: str,
    qty: int,
    order_type: str,
    product: str,
    price: float,
    notional: float,
    status: str,
    detail: str = "",
) -> None:
    """Append one row to the trade journal CSV."""
    settings = get_settings()
    path = settings.journal_path
    _ensure_header(path)
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "security_id": security_id,
        "side": side,
        "qty": qty,
        "order_type": order_type,
        "product": product,
        "price": price,
        "notional": notional,
        "status": status,
        "detail": detail,
    }
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
        writer.writerow(row)
    logger.debug("Journal entry: %s %s %d @ %.2f [%s]", side, security_id, qty, price, status)


def record_closed_trade(
    strategy: str,
    symbol: str,
    entry_time: str,
    exit_time: str,
    qty: int,
    entry_price: float,
    exit_price: float,
    exit_reason: str,
    gross_pnl: float,
    cost: float,
    net_pnl: float,
    r_multiple: float,
    path: str | None = None,
) -> None:
    """Append one round-trip trade to the closed-trade performance log.

    Unlike :func:`record` (which logs order *attempts*), this captures a
    completed entry-to-exit trade with realised P&L so the performance page can
    compute equity curves, expectancy, and R-distributions.
    """
    target = path if path is not None else closed_trades_path()
    _ensure_header(target, _CLOSED_FIELDNAMES)
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "strategy": strategy,
        "symbol": symbol,
        "entry_time": entry_time,
        "exit_time": exit_time,
        "qty": qty,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "gross_pnl": gross_pnl,
        "cost": cost,
        "net_pnl": net_pnl,
        "r_multiple": r_multiple,
    }
    with open(target, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CLOSED_FIELDNAMES)
        writer.writerow(row)
    logger.debug(
        "Closed trade: %s %s %d net=%.2f (%s)",
        strategy, symbol, qty, net_pnl, exit_reason,
    )
