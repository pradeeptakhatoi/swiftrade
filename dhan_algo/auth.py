"""TOTP-based token generation for DhanHQ.

Where to get the TOTP secret:
    web.dhan.co > Profile > DhanHQ Trading APIs > Optional Settings > Enable TOTP.
"""

from __future__ import annotations

import sys

import pyotp
from dhanhq import DhanLogin

from dhan_algo.config import Settings


def generate_access_token(client_id: str, pin: str, totp_secret: str) -> str:
    """Compute a TOTP code and exchange it for a 24-hour access token."""
    totp_code = pyotp.TOTP(totp_secret).now()
    dl = DhanLogin(client_id)
    token = dl.generate_token(pin, totp_code)
    return token


def ensure_token(settings: Settings) -> str:
    """Return an access token — either the one already set, or generate via TOTP.

    Exits with a clear error if neither path is available.
    """
    if settings.dhan_access_token:
        return settings.dhan_access_token

    if settings.dhan_pin and settings.dhan_totp_secret:
        return generate_access_token(
            settings.dhan_client_id,
            settings.dhan_pin,
            settings.dhan_totp_secret,
        )

    sys.exit(
        "No access token available. Either:\n"
        "  1. Set DHAN_ACCESS_TOKEN (24h token from web.dhan.co), or\n"
        "  2. Set DHAN_PIN + DHAN_TOTP_SECRET for automatic TOTP login."
    )
