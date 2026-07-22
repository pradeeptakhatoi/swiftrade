"""Backtesting harness — replay historical bars through a strategy."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from dhan_algo.config import Settings, get_settings
from dhan_algo.strategy import (
    MultiStrategy,
    Order,
    Strategy,
    _StrategyAdapter,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ReplayTicker — deterministic price feed for backtesting
# ---------------------------------------------------------------------------


class ReplayTicker:
    """In-memory ticker driven by ``set_price()`` — no network calls."""

    def __init__(self) -> None:
        self._prices: dict[str, float] = {}

    def get_ltp(self, security_id: str, segment: str | None = None) -> float | None:
        return self._prices.get(security_id)

    @property
    def watchlist(self) -> dict[str, float]:
        return dict(self._prices)

    def set_price(self, security_id: str, price: float) -> None:
        self._prices[security_id] = price


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class SimulatedFill:
    timestamp: str
    security_id: str
    side: str
    qty: int
    price: float
    notional: float


@dataclass
class BacktestResult:
    fills: list[SimulatedFill] = field(default_factory=list)
    positions: dict[str, int] = field(default_factory=dict)
    pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_trades: int = 0
    buy_count: int = 0
    sell_count: int = 0

    def summary(self) -> str:
        lines = [
            "=== Backtest Summary ===",
            f"Total trades : {self.total_trades}",
            f"Buys         : {self.buy_count}",
            f"Sells        : {self.sell_count}",
            f"Realized P&L : {self.pnl:,.2f}",
            f"Unrealized   : {self.unrealized_pnl:,.2f}",
            f"Net P&L      : {self.pnl + self.unrealized_pnl:,.2f}",
            f"Positions    : {self.positions}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


def load_csv(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Read a CSV with columns: timestamp, security_id, open, high, low, close, volume.

    Returns ``{security_id: [bar_dict, ...]}``.
    """
    result: dict[str, list[dict[str, Any]]] = {}
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            sid = row["security_id"]
            bar = {
                "timestamp": row["timestamp"],
                "security_id": sid,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": int(row.get("volume", 0)),
            }
            result.setdefault(sid, []).append(bar)
    return result


def fetch_historical(
    client: Any,
    security_id: str,
    from_date: str,
    to_date: str,
    interval: str = "day",
    segment: str = "NSE_EQ",
    exchange_token: str = "",
) -> list[dict[str, Any]]:
    """Fetch OHLC bars via the DhanHQ ``HistoricalData`` API.

    ``interval`` may be ``"day"`` or ``"minute"``.
    """
    from dhanhq import HistoricalData

    hd = HistoricalData(client)
    if interval == "minute":
        resp = hd.intraday_minute_data(
            security_id=security_id,
            exchange_segment=segment,
            instrument_type="EQUITY",
            from_date=from_date,
            to_date=to_date,
        )
    else:
        resp = hd.historical_daily_data(
            security_id=security_id,
            exchange_segment=segment,
            instrument_type="EQUITY",
            from_date=from_date,
            to_date=to_date,
        )

    bars: list[dict[str, Any]] = []
    raw = resp.get("data", []) if isinstance(resp, dict) else []
    for row in raw:
        bars.append(
            {
                "timestamp": str(row.get("timestamp", row.get("start_Time", ""))),
                "security_id": security_id,
                "open": float(row.get("open", 0)),
                "high": float(row.get("high", 0)),
                "low": float(row.get("low", 0)),
                "close": float(row.get("close", 0)),
                "volume": int(row.get("volume", 0)),
            }
        )
    return bars


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------


def run_backtest(
    strategy: MultiStrategy | Strategy,
    bars: dict[str, list[dict[str, Any]]],
    settings: Settings | None = None,
) -> BacktestResult:
    """Replay historical bars through *strategy* and return a ``BacktestResult``.

    If *strategy* is a legacy ``Strategy``, it is wrapped automatically via
    ``_StrategyAdapter`` and ``ltp()`` is monkey-patched to read from the
    replay ticker so that ``SmaDemo.evaluate()`` works unchanged.
    """
    settings = settings or get_settings()
    ticker = ReplayTicker()
    result = BacktestResult()
    positions: dict[str, int] = {}
    cost_basis: dict[str, float] = {}

    # Wrap legacy Strategy
    mock_client: Any = MagicMock()
    mock_client.NSE = "NSE_EQ"
    if isinstance(strategy, Strategy):
        multi = _StrategyAdapter(strategy, mock_client)
    else:
        multi = strategy

    security_ids = list(bars.keys())
    multi.on_start(ticker, security_ids)

    # Interleave bars across symbols sorted by timestamp
    all_bars: list[dict[str, Any]] = []
    for sid_bars in bars.values():
        all_bars.extend(sid_bars)
    all_bars.sort(key=lambda b: b["timestamp"])

    # Monkey-patch ltp so that SmaDemo.evaluate() reads from the ReplayTicker
    import dhan_algo.market_data as _md

    original_ltp = _md.ltp

    def _patched_ltp(_client: Any, security_id: str, segment: str | None = None) -> float | None:
        return ticker.get_ltp(security_id, segment)

    _md.ltp = _patched_ltp
    # Also patch in strategy module since it imports ltp at module level
    import dhan_algo.strategy as _strat

    original_strat_ltp = _strat.ltp
    _strat.ltp = _patched_ltp

    try:
        for bar in all_bars:
            sid = bar["security_id"]
            close = bar["close"]
            timestamp = bar["timestamp"]

            ticker.set_price(sid, close)
            orders = multi.on_tick(ticker, sid, None)

            for order in orders:
                order_sid = order.security_id or sid
                notional = close * order.qty

                # Risk guards
                if order.qty > settings.max_qty:
                    logger.debug("Backtest: blocked qty %d > max %d", order.qty, settings.max_qty)
                    continue
                if notional > settings.max_order_value:
                    logger.debug("Backtest: blocked notional %.0f > max %.0f", notional, settings.max_order_value)
                    continue

                fill = SimulatedFill(
                    timestamp=timestamp,
                    security_id=order_sid,
                    side=order.side,
                    qty=order.qty,
                    price=close,
                    notional=notional,
                )
                result.fills.append(fill)
                result.total_trades += 1

                if order.side == "BUY":
                    result.buy_count += 1
                    prev_pos = positions.get(order_sid, 0)
                    prev_cost = cost_basis.get(order_sid, 0.0)
                    positions[order_sid] = prev_pos + order.qty
                    cost_basis[order_sid] = prev_cost + notional
                else:
                    result.sell_count += 1
                    prev_pos = positions.get(order_sid, 0)
                    prev_cost = cost_basis.get(order_sid, 0.0)
                    if prev_pos > 0:
                        avg_cost = prev_cost / prev_pos
                        realized = (close - avg_cost) * min(order.qty, prev_pos)
                        result.pnl += realized
                        sold_qty = min(order.qty, prev_pos)
                        positions[order_sid] = prev_pos - sold_qty
                        cost_basis[order_sid] = prev_cost - (avg_cost * sold_qty)
                    else:
                        # Short selling / no position
                        positions[order_sid] = prev_pos - order.qty
                        cost_basis[order_sid] = prev_cost - notional

        # Calculate unrealized P&L
        for sid, pos in positions.items():
            if pos != 0:
                current = ticker.get_ltp(sid) or 0.0
                avg = cost_basis.get(sid, 0.0) / pos if pos != 0 else 0.0
                result.unrealized_pnl += (current - avg) * pos

        result.positions = {k: v for k, v in positions.items() if v != 0}
    finally:
        _md.ltp = original_ltp
        _strat.ltp = original_strat_ltp
        multi.on_stop()

    return result
