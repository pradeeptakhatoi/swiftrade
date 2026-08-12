"""dhan_algo — algorithmic trading toolkit for DhanHQ."""

__version__ = "0.1.0"

from dhan_algo.client import get_client, ok, show_funds
from dhan_algo.market_data import ltp, WebSocketTicker
from dhan_algo.security_master import resolve_security_id
from dhan_algo.orders import place, show_positions, show_orders
from dhan_algo.risk import check_order, kill_switch
from dhan_algo.journal import record as journal_record
from dhan_algo.strategy import (
    MultiStrategy,
    Order,
    PollingTicker,
    SmaDemoMulti,
    Strategy,
    Ticker,
    run_multi_strategy_loop,
    run_ws_strategy_loop,
)
from dhan_algo.backtest import (
    BacktestResult,
    ReplayTicker,
    SimulatedFill,
    TradingCosts,
    load_csv,
    fetch_historical,
    run_backtest,
)

__all__ = [
    "get_client",
    "ok",
    "show_funds",
    "ltp",
    "WebSocketTicker",
    "resolve_security_id",
    "place",
    "show_positions",
    "show_orders",
    "check_order",
    "kill_switch",
    "journal_record",
    "MultiStrategy",
    "Order",
    "PollingTicker",
    "SmaDemoMulti",
    "Strategy",
    "Ticker",
    "run_multi_strategy_loop",
    "run_ws_strategy_loop",
    "BacktestResult",
    "ReplayTicker",
    "SimulatedFill",
    "TradingCosts",
    "load_csv",
    "fetch_historical",
    "run_backtest",
]
