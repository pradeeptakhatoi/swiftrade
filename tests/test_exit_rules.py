"""Tests for the shared exit-management rules."""

from __future__ import annotations

from exit_rules import ExitConfig, finalize_long, open_long, step_long


def _pos(qty: int = 10, entry: float = 100.0, stop: float = 90.0, target: float = 120.0):
    """A fresh long: entry 100, risk 10 (1R=10), target at +2R."""
    return open_long(
        entry_fill=entry, stop=stop, target=target,
        entry_idx=0, entry_time="t0", qty=qty, entry_charge=0.0,
    )


def _run(pos, bars, cfg, end_reason="end"):
    """Feed (high, low, close) bars until the position closes; return reason."""
    n = len(bars)
    for i, (h, l, c) in enumerate(bars, start=1):
        closed, reason = step_long(
            pos, h, l, c, i, is_last=(i == n), cfg=cfg, end_reason=end_reason,
        )
        if closed:
            return reason
    return None


class TestActiveFlag:
    def test_default_is_inactive(self):
        assert ExitConfig().active is False

    def test_any_rule_activates(self):
        assert ExitConfig(breakeven_r=1).active
        assert ExitConfig(trail_r=2).active
        assert ExitConfig(partial_r=1, partial_pct=0.5).active
        assert ExitConfig(max_hold_bars=5).active

    def test_partial_needs_both_fields(self):
        assert ExitConfig(partial_r=1).active is False
        assert ExitConfig(partial_pct=0.5).active is False


class TestPlainStopTarget:
    def test_target_hit(self):
        pos = _pos()
        reason = _run(pos, [(121, 100, 120)], ExitConfig())
        assert reason == "target"
        pnl = finalize_long(pos, None, "CNC")
        assert pnl["gross"] == (120 - 100) * 10
        assert pnl["r_multiple"] == 2.0

    def test_stop_hit_takes_priority_in_same_bar(self):
        # A bar that touches both stop and target -> stop wins.
        pos = _pos()
        reason = _run(pos, [(121, 89, 100)], ExitConfig())
        assert reason == "stop"
        pnl = finalize_long(pos, None, "CNC")
        assert pnl["gross"] == (90 - 100) * 10

    def test_end_of_data(self):
        pos = _pos()
        reason = _run(pos, [(105, 99, 104)], ExitConfig(), end_reason="end")
        assert reason == "end"


class TestBreakeven:
    def test_stop_moves_to_entry_after_threshold(self):
        pos = _pos()
        cfg = ExitConfig(breakeven_r=1.0)
        # Bar 1 runs to +1R high (110) but closes below target -> stop lifts to 100.
        step_long(pos, 110, 100, 105, 1, is_last=False, cfg=cfg, end_reason="end")
        assert pos["stop"] == 100.0  # entry
        # Bar 2 dips to 100 -> stopped at break-even, no loss.
        closed, reason = step_long(pos, 106, 100, 101, 2, is_last=False, cfg=cfg, end_reason="end")
        assert closed and reason == "stop"
        pnl = finalize_long(pos, None, "CNC")
        assert pnl["gross"] == 0.0

    def test_not_triggered_below_threshold(self):
        pos = _pos()
        cfg = ExitConfig(breakeven_r=1.0)
        # High only reaches +0.5R (105) -> stop stays at original 90.
        step_long(pos, 105, 100, 103, 1, is_last=False, cfg=cfg, end_reason="end")
        assert pos["stop"] == 90.0


class TestTrailingStop:
    def test_stop_trails_below_high_water(self):
        pos = _pos()
        cfg = ExitConfig(trail_r=1.0)  # trail 1R (=10) below the high
        step_long(pos, 115, 100, 114, 1, is_last=False, cfg=cfg, end_reason="end")
        assert pos["stop"] == 105.0  # 115 - 10
        # Trail only ratchets up, never down.
        step_long(pos, 112, 106, 108, 2, is_last=False, cfg=cfg, end_reason="end")
        assert pos["stop"] == 105.0
        # New high 118 -> stop to 108.
        step_long(pos, 118, 108, 117, 3, is_last=False, cfg=cfg, end_reason="end")
        assert pos["stop"] == 108.0

    def test_trailing_locks_in_profit(self):
        pos = _pos(target=200.0)  # far target so the trail is what exits
        cfg = ExitConfig(trail_r=1.0)
        reason = _run(pos, [
            (130, 100, 129),   # high 130 -> stop 120
            (131, 119, 120),   # dips to 119 <= 120 -> stop
        ], cfg)
        assert reason == "stop"
        pnl = finalize_long(pos, None, "CNC")
        assert pnl["gross"] == (120 - 100) * 10  # locked +2R


class TestPartialProfit:
    def test_partial_then_runner_to_target(self):
        pos = _pos(qty=10)  # entry 100, 1R=10, target 120 (=+2R)
        cfg = ExitConfig(partial_r=1.0, partial_pct=0.5)
        # Bar reaches +1R (110) then, later bar, the target.
        step_long(pos, 111, 100, 110, 1, is_last=False, cfg=cfg, end_reason="end")
        assert pos["partial_done"] is True
        assert pos["remaining_qty"] == 5
        assert pos["legs"][0] == {"qty": 5, "price": 110.0, "reason": "partial"}
        closed, reason = step_long(pos, 121, 108, 120, 2, is_last=False, cfg=cfg, end_reason="end")
        assert closed and reason == "target"
        pnl = finalize_long(pos, None, "CNC")
        # 5 @ +10 (partial) + 5 @ +20 (target) = 50 + 100 = 150 gross.
        assert pnl["gross"] == 150.0
        # Blended R = (150/10)/10 = 1.5R.
        assert pnl["r_multiple"] == 1.5

    def test_partial_keeps_at_least_one_share(self):
        pos = _pos(qty=1)
        cfg = ExitConfig(partial_r=1.0, partial_pct=0.5)
        step_long(pos, 111, 100, 110, 1, is_last=False, cfg=cfg, end_reason="end")
        # Can't peel a share off a 1-lot; no partial leg, remainder intact.
        assert pos["remaining_qty"] == 1
        assert pos["partial_done"] is True
        assert pos["legs"] == []


class TestTimeStop:
    def test_time_exit_after_max_hold(self):
        pos = _pos(target=500.0)
        cfg = ExitConfig(max_hold_bars=3)
        reason = _run(pos, [
            (105, 99, 104),
            (106, 100, 105),
            (107, 101, 106),  # bar_idx 3 -> held 3 -> time exit
        ], cfg)
        assert reason == "time"


class TestFinalizeCosts:
    def test_costs_applied_per_leg(self):
        from dhan_algo.backtest import TradingCosts
        pos = _pos(qty=10)
        pos["entry_charge"] = 5.0
        _run(pos, [(121, 100, 120)], ExitConfig())
        pnl = finalize_long(pos, TradingCosts(), "CNC")
        assert pnl["cost"] > 5.0            # entry charge + sell charge
        assert pnl["net"] < pnl["gross"]
