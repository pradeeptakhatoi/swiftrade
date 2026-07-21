"""DhanHQ client construction and response helpers."""

from __future__ import annotations

import sys

from dhanhq import DhanContext, dhanhq

from dhan_algo.auth import ensure_token
from dhan_algo.config import Settings, get_settings


def get_client(settings: Settings | None = None) -> dhanhq:
    """Build and return a dhanhq client, resolving the token automatically."""
    settings = settings or get_settings()
    if not settings.dhan_client_id:
        sys.exit("Set DHAN_CLIENT_ID environment variable first.")
    token = ensure_token(settings)
    return dhanhq(DhanContext(settings.dhan_client_id, token))


def ok(resp: dict) -> bool:
    """DhanHQ returns {'status': 'success'|'failure', 'data'/'remarks': ...}."""
    return isinstance(resp, dict) and resp.get("status") == "success"


def show_funds(client: dhanhq) -> None:
    resp = client.get_fund_limits()
    if ok(resp):
        data = resp.get("data", {})
        avail = data.get("availabelBalance", data)  # note: Dhan's key spelling
        print(f"Funds / available balance: {avail}")
    else:
        print("Could not fetch funds:", resp)
