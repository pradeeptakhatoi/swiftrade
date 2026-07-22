"""Security-ID resolution from symbol names."""

from __future__ import annotations

import logging

from dhanhq import dhanhq

logger = logging.getLogger(__name__)

KNOWN_IDS = {
    "RELIANCE": "2885",
    "TCS": "11536",
    "INFY": "1594",
    "HDFCBANK": "1333",
    "SBIN": "3045",
}

# Module-level cache for the security master DataFrame
_security_master_cache: dict[str, object] = {}


def _get_security_master(client: dhanhq):
    """Fetch and cache the compact security master DataFrame."""
    cache_key = "compact"
    if cache_key not in _security_master_cache:
        df = client.fetch_security_list(cache_key)
        if df is not None:
            _security_master_cache[cache_key] = df
        return df
    return _security_master_cache[cache_key]


def resolve_security_id(
    client: dhanhq, symbol: str, segment_hint: str = "NSE"
) -> str | None:
    """Look up a security_id by trading symbol from the scrip master.

    Downloads the compact master once (needs pandas, pulled in by dhanhq).
    Column names are matched flexibly so this survives minor schema changes.
    """
    symbol = symbol.upper().strip()
    if symbol in KNOWN_IDS:
        return KNOWN_IDS[symbol]

    df = _get_security_master(client)
    if df is None:
        logger.error("Security master returned None — cannot resolve %s", symbol)
        return None
    cols = list(df.columns)

    def find(substr: str) -> str | None:
        return next((c for c in cols if substr in c.upper()), None)

    sid_col = find("SECURITY_ID")
    sym_col = (
        next((c for c in cols if c.upper() == "SEM_TRADING_SYMBOL"), None)
        or find("SYMBOL")
    )
    exch_col = find("EXCH_ID") or find("SEGMENT")
    if not (sid_col and sym_col):
        logger.error("Could not identify columns in scrip master: %s", cols)
        return None

    m = df[df[sym_col].astype(str).str.upper() == symbol]
    if exch_col is not None:
        m = m[m[exch_col].astype(str).str.upper().str.contains(segment_hint, na=False)]
    if m.empty:
        logger.error("No match for %s in %s", symbol, segment_hint)
        return None
    return str(m.iloc[0][sid_col])
