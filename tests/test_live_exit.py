"""Tests for the live exit manager (pure, no broker)."""

from __future__ import annotations

import pytest

from dhan_algo.backtest import TradingCosts
from exit_rules import ExitConfig
from live_exit import ExitAction, LiveExitManager


def _mgr(**reg) -> tuple[LiveExitManager, str]:
    """A manager with one registered long (entry 100, stop 98, target 106)."""
    m = LiveExitManager()
    defaults = dict(
        entry=100.0, stop=98.0, target=106.0, qty=10,
        cfg=ExitConfig(), product="INTRA",
    )
    defaults.update(reg)
    m.register("SID1", "SYM", **defaults)
    return m, "SID1"


class TestRegister:
    def test_registers_position(self):
        m, sid = _mgr()
        assert sid in m.positions
        tp = m.positions[sid]
        assert tp.pos["qty"] == 10
        assert tp.pos["initial_risk"] == pytest.approx(2.0)

    def test_rejects_stop_at_or_above_entry(self):
        m = LiveExitManager()
        with pytest.raises(ValueError):
            m.register("S", "SYM", entry=100.0, stop=100.0, target=110.0, qty=1)
        with pytest.raises(ValueError):
            m.register("S", "SYM", entry=100.0, stop=101.0, target=110.0, qty=1)

    def test_rejects_non_positive_qty(self):
        m = LiveExitManager()
        with pytest.raises(ValueError):
            m.register("S", "SYM", entry=100.0, stop=98.0, target=110.0, qty=0)


class TestStopAndTarget:
    def test_no_action_between_stop_and_target(self):
        m, sid = _mgr()
        assert m.on_price(sid, 101.0) == []
        assert m.on_price(sid, 103.5) == []
        assert sid in m.positions  # still open

    def test_hard_stop_closes(self):
        m, sid = _mgr()
        actions = m.on_price(sid, 97.5)  # <= stop 98
        assert len(actions) == 1
        a = actions[0]
        assert isinstance(a, ExitAction)
        assert a.side == "SELL" and a.reason == "stop" and a.qty == 10
        assert a.closed is True
        pnl = m.finalize(sid)
        assert pnl["gross"] == pytest.approx((98.0 - 100.0) * 10)
        assert sid not in m.positions

    def test_target_closes_with_profit(self):
        m, sid = _mgr()
        actions = m.on_price(sid, 106.5)  # >= target 106
        assert len(actions) == 1
        assert actions[0].reason == "target" and actions[0].closed
        pnl = m.finalize(sid)
        assert pnl["gross"] == pytest.approx((106.0 - 100.0) * 10)
        assert pnl["r_multiple"] == pytest.approx(3.0)


class TestTrailing:
    def test_trailing_stop_locks_profit(self):
        # trail 1R (=2) below the high; far target so the trail is what exits.
        m, sid = _mgr(target=200.0, cfg=ExitConfig(trail_r=1.0))
        assert m.on_price(sid, 110.0) == []      # high 110 -> stop lifts to 108
        actions = m.on_price(sid, 107.9)         # dips below 108 -> stop
        assert len(actions) == 1
        assert actions[0].reason == "stop" and actions[0].closed
        pnl = m.finalize(sid)
        assert pnl["gross"] == pytest.approx((108.0 - 100.0) * 10)


class TestBreakeven:
    def test_stop_moves_to_entry(self):
        m, sid = _mgr(cfg=ExitConfig(breakeven_r=1.0))
        assert m.on_price(sid, 102.0) == []   # +1R high -> stop lifts to entry 100
        assert m.positions[sid].pos["stop"] == pytest.approx(100.0)
        actions = m.on_price(sid, 100.0)      # touches break-even
        assert actions and actions[0].reason == "stop"
        pnl = m.finalize(sid)
        assert pnl["gross"] == pytest.approx(0.0)


class TestPartial:
    def test_partial_then_target(self):
        m, sid = _mgr(cfg=ExitConfig(partial_r=1.0, partial_pct=0.5))
        # +1R = 102 -> sell half (5), position stays open.
        part = m.on_price(sid, 102.0)
        assert len(part) == 1
        assert part[0].reason == "partial" and part[0].qty == 5
        assert part[0].closed is False
        assert m.positions[sid].pos["remaining_qty"] == 5
        # target -> close the runner.
        rest = m.on_price(sid, 106.0)
        assert len(rest) == 1
        assert rest[0].reason == "target" and rest[0].qty == 5 and rest[0].closed
        pnl = m.finalize(sid)
        # 5 @ +2 (partial) + 5 @ +6 (target) = 10 + 30 = 40 gross.
        assert pnl["gross"] == pytest.approx(40.0)


class TestTimeStop:
    def test_time_exit_after_max_hold(self):
        # Wide levels; each on_price is one "bar", exit after 3.
        m, sid = _mgr(stop=1.0, target=1000.0, cfg=ExitConfig(max_hold_bars=3))
        assert m.on_price(sid, 100.5) == []
        assert m.on_price(sid, 100.6) == []
        actions = m.on_price(sid, 100.7)  # bar_idx 3 -> time exit
        assert actions and actions[0].reason == "time" and actions[0].closed


class TestManagement:
    def test_drop_stops_tracking(self):
        m, sid = _mgr()
        m.drop(sid)
        assert sid not in m.positions
        assert m.on_price(sid, 97.0) == []  # unknown id -> nothing

    def test_snapshot_reports_rows(self):
        m, sid = _mgr()
        rows = m.snapshot()
        assert len(rows) == 1
        assert rows[0]["symbol"] == "SYM"
        assert rows[0]["qty_open"] == 10

    def test_costs_reduce_net(self):
        m = LiveExitManager(costs=TradingCosts())
        m.register("S", "SYM", entry=100.0, stop=98.0, target=106.0, qty=10)
        m.on_price("S", 106.5)
        pnl = m.finalize("S")
        assert pnl["cost"] > 0.0
        assert pnl["net"] < pnl["gross"]
