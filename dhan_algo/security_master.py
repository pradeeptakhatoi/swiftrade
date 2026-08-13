"""Security-ID resolution from symbol names.

Resolves any symbol in the full NSE cash-equity universe (~2,900 scrips), not
just a hardcoded handful. On first use the DhanHQ scrip master is downloaded
(or read from a local CSV fallback), filtered to NSE equity shares, and turned
into an in-memory ``symbol -> security_id`` map for O(1) lookups.
"""

from __future__ import annotations

import logging
import os

import pandas as pd
from dhanhq import dhanhq

from dhan_algo.config import get_settings

logger = logging.getLogger(__name__)

# Fast-path seeds. Also used as a last-resort fallback when the full scrip
# master cannot be loaded (e.g. no client and no local CSV).
KNOWN_IDS = {
    "RELIANCE": "2885",
    "TCS": "11536",
    "INFY": "1594",
    "HDFCBANK": "1333",
    "SBIN": "3045",
}

# Instrument category for NSE cash equity shares in the scrip master.
_EQUITY_INSTRUMENT = "ES"

# Caches (module-level; cleared via reset_cache()).
_raw_master: dict[str, pd.DataFrame] = {}
_symbol_maps: dict[str, dict[str, str]] = {}


def reset_cache() -> None:
    """Drop the cached scrip master and derived symbol maps."""
    _raw_master.clear()
    _symbol_maps.clear()


def _find(cols: list[str], substr: str) -> str | None:
    return next((c for c in cols if substr in c.upper()), None)


def _load_local_master(path: str) -> pd.DataFrame | None:
    """Read a locally cached scrip-master CSV, if present."""
    if path and os.path.exists(path):
        try:
            return pd.read_csv(path, low_memory=False)
        except Exception as exc:  # noqa: BLE001 - corrupt/locked file is non-fatal
            logger.warning("Failed reading local scrip master %s: %s", path, exc)
    return None


def _get_master_df(client: dhanhq | None) -> pd.DataFrame | None:
    """Fetch (and cache) the scrip master: Dhan download first, local CSV fallback."""
    if "df" in _raw_master:
        return _raw_master["df"]

    df = None
    if client is not None:
        try:
            df = client.fetch_security_list("compact")
        except Exception as exc:  # noqa: BLE001 - network/SDK failure -> fall back
            logger.warning("fetch_security_list failed: %s", exc)
            df = None
    if df is None:
        df = _load_local_master(get_settings().security_master_path)

    if df is not None:
        _raw_master["df"] = df
    return df


def _build_symbol_map(df: pd.DataFrame, segment_hint: str) -> dict[str, str]:
    """Build ``{SYMBOL -> security_id}`` for equity scrips on *segment_hint*.

    Column names are matched flexibly so this survives minor schema changes.
    Rows are filtered to the requested exchange and (when the column exists)
    to equity instruments only, so a symbol never resolves to a derivative.
    """
    cols = list(df.columns)
    sid_col = _find(cols, "SECURITY_ID")
    sym_col = (
        next((c for c in cols if c.upper() == "SEM_TRADING_SYMBOL"), None)
        or _find(cols, "TRADING_SYMBOL")
        or _find(cols, "SYMBOL")
    )
    exch_col = _find(cols, "EXCH_ID") or _find(cols, "SEGMENT")
    inst_col = _find(cols, "INSTRUMENT_TYPE")
    if not (sid_col and sym_col):
        logger.error("Could not identify columns in scrip master: %s", cols)
        return {}

    m = df
    if exch_col is not None:
        m = m[m[exch_col].astype(str).str.upper() == segment_hint.upper()]
    if inst_col is not None:
        m = m[m[inst_col].astype(str).str.upper() == _EQUITY_INSTRUMENT]

    symbols = m[sym_col].astype(str).str.upper().str.strip()
    sids = m[sid_col].astype(str).str.replace(r"\.0$", "", regex=True)
    # zip preserves order; a later duplicate symbol wins (rare for cash equity).
    return dict(zip(symbols, sids))


def _get_symbol_map(client: dhanhq | None, segment_hint: str) -> dict[str, str]:
    key = segment_hint.upper()
    if key not in _symbol_maps:
        df = _get_master_df(client)
        _symbol_maps[key] = _build_symbol_map(df, segment_hint) if df is not None else {}
    return _symbol_maps[key]


def resolve_security_id(
    client: dhanhq | None, symbol: str, segment_hint: str = "NSE"
) -> str | None:
    """Look up a security_id by trading symbol from the scrip master.

    Returns the id string, or ``None`` if the symbol cannot be resolved.
    """
    symbol = symbol.upper().strip()
    if symbol in KNOWN_IDS:
        return KNOWN_IDS[symbol]

    mapping = _get_symbol_map(client, segment_hint)
    if not mapping:
        logger.error("Security master unavailable — cannot resolve %s", symbol)
        return KNOWN_IDS.get(symbol)

    sid = mapping.get(symbol)
    if sid is None:
        logger.error("No match for %s in %s", symbol, segment_hint)
    return sid


def resolve_security_ids(
    client: dhanhq | None, symbols: list[str], segment_hint: str = "NSE"
) -> dict[str, str | None]:
    """Resolve many symbols at once, reusing a single cached scrip-master map."""
    mapping = _get_symbol_map(client, segment_hint)
    out: dict[str, str | None] = {}
    for s in symbols:
        u = s.upper().strip()
        out[s] = KNOWN_IDS.get(u) or (mapping.get(u) if mapping else None)
    return out
