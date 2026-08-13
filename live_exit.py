"""Live exit management — apply the backtest exit rules to real positions.

The swing/intraday backtests manage an open long bar by bar with
:func:`exit_rules.step_long` (hard stop, break‑even, trailing stop, partial
profit, time stop). This module reuses that *exact* engine for live trading: a
:class:`LiveExitManager` tracks registered positions and, on each incoming
price, emits the SELL actions that should be sent to the broker.

The live feed provides a last price, not full OHLC bars, so every price update
is treated as a degenerate bar (``high == low == close == price``). That is
price‑level exact for stop/target/break‑even/trailing decisions; the only thing
lost versus true bars is intra‑poll extremes, which is inherent to polling.

Design notes:

* Each fill *leg* that :func:`exit_rules.step_long` appends maps 1:1 to a market
  SELL: a partial adds a non‑closing action, while a stop/target/time leg closes
  the position. :class:`LiveExitManager` diffs the leg list before and after each
  step to surface exactly the new legs as :class:`ExitAction` objects.
* Recorded leg prices are the theoretical rule levels (as in the backtest). Live
  market fills will differ slightly; this is acceptable for an exit‑trigger tool
  and keeps reported P&L consistent with the simulations.
* Long‑only, matching the scanners.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from dhan_algo.backtest import TradingCosts
from exit_rules import ExitConfig, finalize_long, open_long, step_long


@dataclass
class ExitAction:
    """A SELL the caller should place in response to a price update."""

    security_id: str
    symbol: str
    qty: int
    reason: str  # "stop" | "partial" | "target" | "time" | "eod"
    price: float
    closed: bool  # True when this action closes the position
    side: str = "SELL"


@dataclass
class TrackedPosition:
    security_id: str
    symbol: str
    product: str
    cfg: ExitConfig
    pos: dict[str, Any]
    bar_idx: int = 0


class LiveExitManager:
    """Track live long positions and manage their exits with ``exit_rules``."""

    def __init__(self, costs: TradingCosts | None = None) -> None:
        self.costs = costs
        self.positions: dict[str, TrackedPosition] = {}

    # -- lifecycle ---------------------------------------------------------

    def register(
        self,
        security_id: str,
        symbol: str,
        *,
        entry: float,
        stop: float,
        target: float,
        qty: int,
        cfg: ExitConfig | None = None,
        product: str = "INTRA",
    ) -> TrackedPosition:
        """Start managing a long position. Raises on invalid inputs."""
        security_id = str(security_id)
        if qty <= 0:
            raise ValueError(f"qty must be positive, got {qty}")
        if stop >= entry:
            raise ValueError(
                f"stop {stop} must be below entry {entry} for a long position"
            )
        pos = open_long(
            entry_fill=float(entry),
            stop=float(stop),
            target=float(target),
            entry_idx=0,
            entry_time=datetime.now(timezone.utc).isoformat(),
            qty=int(qty),
            entry_charge=0.0,
        )
        tp = TrackedPosition(
            security_id=security_id,
            symbol=symbol,
            product=product,
            cfg=cfg or ExitConfig(),
            pos=pos,
        )
        self.positions[security_id] = tp
        return tp

    def drop(self, security_id: str) -> None:
        """Stop tracking a position (e.g. it was closed elsewhere)."""
        self.positions.pop(str(security_id), None)

    # -- per-price management ---------------------------------------------

    def on_price(
        self, security_id: str, price: float, *, is_last: bool = False
    ) -> list[ExitAction]:
        """Advance one managed position by a price tick.

        Returns the SELL actions (one per newly generated fill leg) the caller
        should place. Empty when nothing triggers.
        """
        security_id = str(security_id)
        tp = self.positions.get(security_id)
        if tp is None:
            return []

        tp.bar_idx += 1
        before = len(tp.pos["legs"])
        closed, _reason = step_long(
            tp.pos,
            high=price,
            low=price,
            close=price,
            bar_idx=tp.bar_idx,
            is_last=is_last,
            cfg=tp.cfg,
            end_reason="eod",
        )
        new_legs = tp.pos["legs"][before:]
        actions: list[ExitAction] = []
        for i, leg in enumerate(new_legs):
            # Only the final new leg can be the one that closed the position.
            is_closing = closed and i == len(new_legs) - 1
            actions.append(
                ExitAction(
                    security_id=security_id,
                    symbol=tp.symbol,
                    qty=int(leg["qty"]),
                    reason=str(leg["reason"]),
                    price=float(leg["price"]),
                    closed=is_closing,
                )
            )
        return actions

    def finalize(self, security_id: str) -> dict[str, float]:
        """Fold a closed position into blended P&L and stop tracking it."""
        security_id = str(security_id)
        tp = self.positions[security_id]
        pnl = finalize_long(tp.pos, self.costs, tp.product)
        self.drop(security_id)
        return pnl

    # -- introspection -----------------------------------------------------

    def snapshot(self) -> list[dict[str, Any]]:
        """Rows describing every tracked position, for display."""
        rows: list[dict[str, Any]] = []
        for tp in self.positions.values():
            pos = tp.pos
            rows.append(
                {
                    "symbol": tp.symbol,
                    "security_id": tp.security_id,
                    "product": tp.product,
                    "qty_open": pos["remaining_qty"],
                    "entry": round(pos["entry_fill"], 2),
                    "stop": round(pos["stop"], 2),
                    "target": round(pos["target"], 2),
                    "high_water": round(pos["high_water"], 2),
                    "partial_done": pos["partial_done"],
                }
            )
        return rows
