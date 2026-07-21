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


def _ensure_header(path: str) -> None:
    """Create the CSV file with a header row if it does not exist."""
    if os.path.exists(path):
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
        writer.writeheader()
    logger.debug("Created journal file: %s", path)


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
