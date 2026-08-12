"""Thin CLI entrypoint for dhan-algo."""

from __future__ import annotations

import argparse
import logging
import sys

from dhan_algo.config import Settings, get_settings

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dhan-algo",
        description="DhanHQ algorithmic trading CLI",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Enable live trading (overrides DHAN_LIVE env var)",
    )

    sub = parser.add_subparsers(dest="command")

    sub.add_parser("funds", help="Show account funds")

    ltp_p = sub.add_parser("ltp", help="Print LTP for a symbol")
    ltp_p.add_argument("symbol", help="Trading symbol (e.g. RELIANCE)")

    order_p = sub.add_parser("order", help="Place an order")
    order_p.add_argument("symbol", help="Trading symbol")
    order_p.add_argument("--side", required=True, choices=["BUY", "SELL"])
    order_p.add_argument("--qty", required=True, type=int)
    order_p.add_argument("--type", default="MARKET", choices=["MARKET", "LIMIT"])
    order_p.add_argument("--product", default="INTRA", choices=["INTRA", "CNC"])
    order_p.add_argument("--price", type=float, default=0.0)

    sub.add_parser("positions", help="Show current positions")
    sub.add_parser("orders", help="Show current orders")

    strat_p = sub.add_parser("strategy", help="Run the demo SMA strategy loop")
    strat_p.add_argument("symbols", nargs="+", help="Trading symbol(s)")
    strat_p.add_argument("--interval", type=int, default=None)
    strat_p.add_argument("--ws", action="store_true", help="Use WebSocket feed")

    bt_p = sub.add_parser("backtest", help="Backtest a strategy on historical data")
    bt_p.add_argument("symbols", nargs="+", help="Trading symbol(s)")
    bt_p.add_argument("--from", dest="from_date", default=None, help="Start date (YYYY-MM-DD)")
    bt_p.add_argument("--to", dest="to_date", default=None, help="End date (YYYY-MM-DD)")
    bt_p.add_argument("--interval", default="day", choices=["day", "minute"])
    bt_p.add_argument("--csv", dest="csv_path", default=None, help="CSV file path instead of API")
    bt_p.add_argument("--gross", action="store_true", help="Frictionless run (no costs/slippage)")
    bt_p.add_argument("--slippage-bps", type=float, default=5.0, help="Adverse slippage in basis points")
    bt_p.add_argument("--product", default="INTRA", choices=["INTRA", "CNC"], help="Product for cost model")

    sub.add_parser("kill-switch", help="Activate the Dhan kill switch")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        sys.exit(1)

    settings = get_settings()

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    if args.live:
        settings.dhan_live = True

    from dhan_algo.client import get_client, show_funds
    from dhan_algo.market_data import ltp
    from dhan_algo.orders import place, show_orders, show_positions
    from dhan_algo.risk import kill_switch
    from dhan_algo.security_master import resolve_security_id

    client = get_client(settings)
    mode = "LIVE -- orders will be SENT" if settings.dhan_live else "DRY_RUN (safe)"
    logger.info("Mode: %s", mode)

    if args.command == "funds":
        show_funds(client)

    elif args.command == "ltp":
        sid = resolve_security_id(client, args.symbol)
        if sid:
            price = ltp(client, sid)
            logger.info("%s (security_id=%s) LTP = %s", args.symbol, sid, price)

    elif args.command == "order":
        sid = resolve_security_id(client, args.symbol)
        if sid:
            place(
                client,
                sid,
                side=args.side,
                qty=args.qty,
                order_type=args.type,
                product=args.product,
                price=args.price,
                settings=settings,
            )

    elif args.command == "positions":
        show_positions(client)

    elif args.command == "orders":
        show_orders(client)

    elif args.command == "strategy":
        from dhan_algo.strategy import (
            SmaDemo,
            SmaDemoMulti,
            run_multi_strategy_loop,
            run_strategy_loop,
            run_ws_strategy_loop,
        )

        sids = []
        for sym in args.symbols:
            sid = resolve_security_id(client, sym)
            if not sid:
                sys.exit(f"Could not resolve symbol: {sym}")
            sids.append(sid)

        if args.interval is not None:
            settings.strategy_interval = args.interval

        if args.ws:
            strat = SmaDemoMulti() if len(sids) > 1 else SmaDemoMulti()
            run_ws_strategy_loop(strat, client, sids, settings=settings)
        elif len(sids) == 1:
            run_strategy_loop(SmaDemo(), client, sids[0], settings=settings)
        else:
            run_multi_strategy_loop(SmaDemoMulti(), client, sids, settings=settings)

    elif args.command == "backtest":
        from dhan_algo.backtest import (
            TradingCosts,
            fetch_historical,
            load_csv,
            run_backtest,
        )
        from dhan_algo.strategy import SmaDemo, SmaDemoMulti

        if args.csv_path:
            bars = load_csv(args.csv_path)
        else:
            if not args.from_date or not args.to_date:
                sys.exit("--from and --to are required when not using --csv")
            bars = {}
            for sym in args.symbols:
                sid = resolve_security_id(client, sym)
                if not sid:
                    sys.exit(f"Could not resolve symbol: {sym}")
                bars[sid] = fetch_historical(
                    client, sid,
                    from_date=args.from_date,
                    to_date=args.to_date,
                    interval=args.interval,
                )

        if len(bars) == 1:
            strategy: SmaDemo | SmaDemoMulti = SmaDemo()
        else:
            strategy = SmaDemoMulti()

        costs = None if args.gross else TradingCosts(slippage_pct=args.slippage_bps / 10000.0)
        result = run_backtest(
            strategy, bars, settings=settings,
            costs=costs, cost_product=args.product,
        )
        print(result.summary())

    elif args.command == "kill-switch":
        kill_switch(client)


if __name__ == "__main__":
    main()
