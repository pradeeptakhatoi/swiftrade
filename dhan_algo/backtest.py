"""Backtesting harness — replay historical bars through a strategy."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
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
# Trading costs — realistic Indian NSE equity friction model
# ---------------------------------------------------------------------------


def _is_intraday(product: str) -> bool:
    return product.upper() in ("INTRA", "INTRADAY", "MIS")


@dataclass
class TradingCosts:
    """Realistic transaction-cost and slippage model for NSE equity.

    Defaults approximate a discount broker on NSE cash equity (rates current
    as of 2024/2025). All ``*_pct`` values are fractions of turnover (price ×
    qty), applied per executed order. Costs are intentionally on the
    conservative side — a backtest that ignores friction is fiction.

    Charge components (per order):
      - Brokerage: ``min(brokerage_flat, brokerage_pct × turnover)``.
      - STT: intraday charges the sell side only; delivery charges both sides.
      - Exchange transaction charge, SEBI turnover fee: both sides.
      - Stamp duty: buy side only (higher for delivery).
      - GST: applied on (brokerage + exchange charge + SEBI fee).

    Slippage is applied adversely to the fill price: buys fill higher, sells
    fill lower, by ``slippage_pct`` of price.
    """

    # Adverse slippage as a fraction of price (5 bps = 0.0005).
    slippage_pct: float = 0.0005

    # Brokerage (per executed order).
    brokerage_flat: float = 20.0
    brokerage_pct: float = 0.0003  # 0.03% of turnover

    # Securities Transaction Tax.
    stt_delivery: float = 0.001        # 0.1% both sides
    stt_intraday_sell: float = 0.00025  # 0.025% sell side only

    # Regulatory / exchange charges (both sides).
    exchange_txn_pct: float = 0.0000297  # NSE ~0.00297%
    sebi_pct: float = 0.000001           # Rs 10 per crore

    # Stamp duty (buy side only).
    stamp_duty_delivery: float = 0.00015  # 0.015%
    stamp_duty_intraday: float = 0.00003  # 0.003%

    # Goods & Services Tax on brokerage + exchange + SEBI charges.
    gst_pct: float = 0.18

    def slippage_price(self, side: str, price: float) -> float:
        """Return the fill price after adverse slippage."""
        adj = price * self.slippage_pct
        return price + adj if side.upper() == "BUY" else price - adj

    def brokerage(self, turnover: float) -> float:
        return min(self.brokerage_flat, self.brokerage_pct * turnover)

    def breakdown(
        self, side: str, price: float, qty: int, product: str = "INTRA"
    ) -> dict[str, float]:
        """Itemised charges for a single executed order."""
        side_u = side.upper()
        turnover = price * qty
        intraday = _is_intraday(product)

        brokerage = self.brokerage(turnover)
        if intraday:
            stt = self.stt_intraday_sell * turnover if side_u == "SELL" else 0.0
        else:
            stt = self.stt_delivery * turnover
        exchange = self.exchange_txn_pct * turnover
        sebi = self.sebi_pct * turnover
        if side_u == "BUY":
            stamp_duty = (
                self.stamp_duty_intraday if intraday else self.stamp_duty_delivery
            ) * turnover
        else:
            stamp_duty = 0.0
        gst = self.gst_pct * (brokerage + exchange + sebi)

        total = brokerage + stt + exchange + sebi + stamp_duty + gst
        return {
            "brokerage": brokerage,
            "stt": stt,
            "exchange": exchange,
            "sebi": sebi,
            "stamp_duty": stamp_duty,
            "gst": gst,
            "total": total,
        }

    def total(
        self, side: str, price: float, qty: int, product: str = "INTRA"
    ) -> float:
        """Total charges for a single executed order."""
        return self.breakdown(side, price, qty, product)["total"]


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
    cost: float = 0.0


@dataclass
class BacktestResult:
    fills: list[SimulatedFill] = field(default_factory=list)
    positions: dict[str, int] = field(default_factory=dict)
    pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_trades: int = 0
    buy_count: int = 0
    sell_count: int = 0
    total_costs: float = 0.0

    @property
    def net_pnl(self) -> float:
        """P&L after transaction costs (slippage is already in fill prices)."""
        return self.pnl + self.unrealized_pnl - self.total_costs

    def summary(self) -> str:
        lines = [
            "=== Backtest Summary ===",
            f"Total trades : {self.total_trades}",
            f"Buys         : {self.buy_count}",
            f"Sells        : {self.sell_count}",
            f"Realized P&L : {self.pnl:,.2f}",
            f"Unrealized   : {self.unrealized_pnl:,.2f}",
            f"Costs        : {self.total_costs:,.2f}",
            f"Net P&L      : {self.net_pnl:,.2f}",
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
    interval_minutes: int | None = None,
) -> list[dict[str, Any]]:
    """Fetch OHLC bars via the DhanHQ client's historical-data API.

    ``interval`` may be ``"day"`` or ``"minute"``. For minute data,
    *interval_minutes* (e.g. 1, 5, 15) selects the candle size. The methods
    live directly on the dhanhq client in the v2 SDK.
    """
    if interval == "minute":
        resp = client.intraday_minute_data(
            security_id=security_id,
            exchange_segment=segment,
            instrument_type="EQUITY",
            from_date=from_date,
            to_date=to_date,
            interval=interval_minutes or 1,
        )
    else:
        resp = client.historical_daily_data(
            security_id=security_id,
            exchange_segment=segment,
            instrument_type="EQUITY",
            from_date=from_date,
            to_date=to_date,
        )

    data = resp.get("data", {}) if isinstance(resp, dict) else {}
    return _bars_from_response(data, security_id)


def _epoch_to_iso(ts: Any) -> str:
    """Convert a Dhan epoch-seconds timestamp to an ISO string; pass through
    anything non-numeric unchanged."""
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()
    except (TypeError, ValueError, OSError):
        return str(ts)


def _bars_from_response(data: Any, security_id: str) -> list[dict[str, Any]]:
    """Normalise a DhanHQ historical response into a list of bar dicts.

    The v2 Data API returns *columnar* data (``{"open": [...], "high": [...],
    ...}``); older shapes return a list of row dicts. Both are handled.
    """
    bars: list[dict[str, Any]] = []
    if isinstance(data, dict):
        closes = data.get("close", [])
        opens = data.get("open", [])
        highs = data.get("high", [])
        lows = data.get("low", [])
        volumes = data.get("volume", [])
        stamps = data.get("timestamp", data.get("start_Time", []))
        for i in range(len(closes)):
            bars.append(
                {
                    "timestamp": _epoch_to_iso(stamps[i]) if i < len(stamps) else "",
                    "security_id": security_id,
                    "open": float(opens[i]) if i < len(opens) else 0.0,
                    "high": float(highs[i]) if i < len(highs) else 0.0,
                    "low": float(lows[i]) if i < len(lows) else 0.0,
                    "close": float(closes[i]),
                    "volume": int(volumes[i]) if i < len(volumes) else 0,
                }
            )
    elif isinstance(data, list):
        for row in data:
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
    costs: TradingCosts | None = None,
    cost_product: str | None = None,
) -> BacktestResult:
    """Replay historical bars through *strategy* and return a ``BacktestResult``.

    If *strategy* is a legacy ``Strategy``, it is wrapped automatically via
    ``_StrategyAdapter`` and ``ltp()`` is monkey-patched to read from the
    replay ticker so that ``SmaDemo.evaluate()`` works unchanged.

    If *costs* is provided, fills are priced with adverse slippage and each
    executed order is charged the itemised transaction costs (brokerage, STT,
    exchange/SEBI fees, stamp duty, GST). When *costs* is ``None`` the backtest
    is frictionless: fills occur at the bar close with zero cost.

    *cost_product* overrides the product (``"INTRA"``/``"CNC"``) used for the
    cost model; when ``None`` the order's own ``product`` is used.
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

                # Risk guards use the bar close (unslipped reference price).
                ref_notional = close * order.qty
                if order.qty > settings.max_qty:
                    logger.debug("Backtest: blocked qty %d > max %d", order.qty, settings.max_qty)
                    continue
                if ref_notional > settings.max_order_value:
                    logger.debug("Backtest: blocked notional %.0f > max %.0f", ref_notional, settings.max_order_value)
                    continue

                # Apply adverse slippage to the fill price, then charge costs.
                if costs is not None:
                    fill_price = costs.slippage_price(order.side, close)
                    product = cost_product or getattr(order, "product", "INTRA")
                    order_cost = costs.total(order.side, fill_price, order.qty, product)
                else:
                    fill_price = close
                    order_cost = 0.0
                notional = fill_price * order.qty

                fill = SimulatedFill(
                    timestamp=timestamp,
                    security_id=order_sid,
                    side=order.side,
                    qty=order.qty,
                    price=fill_price,
                    notional=notional,
                    cost=order_cost,
                )
                result.fills.append(fill)
                result.total_trades += 1
                result.total_costs += order_cost

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
                        realized = (fill_price - avg_cost) * min(order.qty, prev_pos)
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
