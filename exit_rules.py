"""Configurable exit-management rules shared by the trade simulations.

The swing and intraday backtests both enter long and then, bar by bar, look
for an exit. Historically that was a fixed ``stop -> target -> end-of-data``
ladder. This module layers optional, configurable exit rules on top of that so
users can compare how different exit styles change *net* expectancy:

* **Break-even after N R** — once price has run ``breakeven_r`` multiples of the
  initial risk in your favour, the stop is lifted to the entry price.
* **Trailing stop** — the stop trails ``trail_r`` R below the highest high seen
  since entry (a chandelier-style trail; because the initial risk is
  ``atr_multiplier x ATR`` this is ATR-proportional).
* **Partial profit-taking** — at ``partial_r`` R a fraction ``partial_pct`` of
  the position is sold and the remainder is left to run.
* **Time stop** — the position is closed after ``max_hold_bars`` bars.

All rules are opt-in; an :class:`ExitConfig` with everything at its default is
``inactive`` and reproduces the original fixed stop/target behaviour exactly.

The manager is *long-only* (matching the scanners) and deliberately conservative
within a bar: a stop touch is honoured before any profit target, and the
trailing/break-even stop is only tightened *after* the current bar's stop test
so a stop can never be raised into its own bar's high.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dhan_algo.backtest import TradingCosts


@dataclass
class ExitConfig:
    """User-tunable exit-management parameters (all optional)."""

    breakeven_r: float = 0.0          # move stop to entry after this many R
    trail_r: float = 0.0              # trail stop this many R below the high
    partial_r: float = 0.0            # take partial profit at this many R
    partial_pct: float = 0.0          # fraction (0-1] of qty to exit at partial_r
    max_hold_bars: int | None = None  # time stop, in bars held

    @property
    def active(self) -> bool:
        """True if any rule would alter the plain stop/target behaviour."""
        return bool(
            self.breakeven_r
            or self.trail_r
            or (self.partial_r and self.partial_pct)
            or self.max_hold_bars is not None
        )


def open_long(
    entry_fill: float,
    stop: float,
    target: float,
    entry_idx: int,
    entry_time: str,
    qty: int,
    entry_charge: float,
) -> dict[str, Any]:
    """Build the mutable position record the bar manager operates on."""
    return {
        "entry_fill": entry_fill,
        "stop": stop,
        "initial_stop": stop,
        "target": target,
        "initial_risk": entry_fill - stop,
        "qty": qty,
        "remaining_qty": qty,
        "high_water": entry_fill,
        "entry_idx": entry_idx,
        "entry_time": entry_time,
        "entry_charge": entry_charge,
        "legs": [],            # each: {"qty", "price", "reason"}
        "partial_done": False,
    }


def step_long(
    pos: dict[str, Any],
    high: float,
    low: float,
    close: float,
    bar_idx: int,
    *,
    is_last: bool,
    cfg: ExitConfig,
    end_reason: str,
) -> tuple[bool, str | None]:
    """Advance one bar for an open long position.

    Appends any fill legs to ``pos["legs"]`` and returns ``(closed, reason)``
    where *reason* is the exit reason of the final leg (``None`` while open).
    """
    # 1) Hard stop (uses the stop as it stood at the start of this bar).
    if low <= pos["stop"]:
        pos["legs"].append({"qty": pos["remaining_qty"], "price": pos["stop"], "reason": "stop"})
        pos["remaining_qty"] = 0
        return True, "stop"

    # 2) Partial profit-taking (leaves at least one share running).
    if not pos["partial_done"] and cfg.partial_r and cfg.partial_pct:
        level = pos["entry_fill"] + cfg.partial_r * pos["initial_risk"]
        if high >= level:
            pqty = int(pos["qty"] * cfg.partial_pct)
            pqty = max(0, min(pqty, pos["remaining_qty"] - 1))
            if pqty > 0:
                pos["legs"].append({"qty": pqty, "price": level, "reason": "partial"})
                pos["remaining_qty"] -= pqty
            pos["partial_done"] = True

    # 3) Profit target (closes whatever remains).
    if high >= pos["target"]:
        pos["legs"].append({"qty": pos["remaining_qty"], "price": pos["target"], "reason": "target"})
        pos["remaining_qty"] = 0
        return True, "target"

    # 4) Ratchet the stop up for *subsequent* bars (never into this bar's high).
    pos["high_water"] = max(pos["high_water"], high)
    new_stop = pos["stop"]
    if cfg.breakeven_r and pos["high_water"] >= pos["entry_fill"] + cfg.breakeven_r * pos["initial_risk"]:
        new_stop = max(new_stop, pos["entry_fill"])
    if cfg.trail_r:
        new_stop = max(new_stop, pos["high_water"] - cfg.trail_r * pos["initial_risk"])
    pos["stop"] = new_stop

    # 5) Time stop.
    if cfg.max_hold_bars is not None and (bar_idx - pos["entry_idx"]) >= cfg.max_hold_bars:
        pos["legs"].append({"qty": pos["remaining_qty"], "price": close, "reason": "time"})
        pos["remaining_qty"] = 0
        return True, "time"

    # 6) End of available data / session.
    if is_last:
        pos["legs"].append({"qty": pos["remaining_qty"], "price": close, "reason": end_reason})
        pos["remaining_qty"] = 0
        return True, end_reason

    return False, None


def finalize_long(
    pos: dict[str, Any],
    costs: TradingCosts | None,
    product: str,
) -> dict[str, float]:
    """Fold all fill legs into blended P&L, cost, exit price and R multiple.

    The entry (buy) charge is levied once on the full quantity; each sell leg is
    charged on its own quantity. Slippage is applied per leg via *costs*.
    """
    entry_fill = pos["entry_fill"]
    gross = 0.0
    sell_charge = 0.0
    filled_qty = 0
    weighted_exit = 0.0

    for leg in pos["legs"]:
        exit_fill = costs.slippage_price("SELL", leg["price"]) if costs else leg["price"]
        sell_charge += costs.total("SELL", exit_fill, leg["qty"], product) if costs else 0.0
        gross += (exit_fill - entry_fill) * leg["qty"]
        weighted_exit += exit_fill * leg["qty"]
        filled_qty += leg["qty"]

    cost = pos["entry_charge"] + sell_charge
    net = gross - cost
    avg_exit = weighted_exit / filled_qty if filled_qty else entry_fill
    initial_risk = pos["initial_risk"]
    r_mult = (gross / pos["qty"]) / initial_risk if initial_risk > 0 and pos["qty"] else 0.0

    return {
        "gross": gross,
        "cost": cost,
        "net": net,
        "avg_exit": avg_exit,
        "r_multiple": r_mult,
    }
