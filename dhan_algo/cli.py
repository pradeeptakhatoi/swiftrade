"""Thin CLI entrypoint for dhan-algo."""

from __future__ import annotations

import argparse
import sys

from dhan_algo.config import Settings, get_settings


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
    strat_p.add_argument("symbol", help="Trading symbol")
    strat_p.add_argument("--interval", type=int, default=None)

    sub.add_parser("kill-switch", help="Activate the Dhan kill switch")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Apply --live flag before loading settings
    settings = get_settings()
    if args.live:
        settings.dhan_live = True

    from dhan_algo.client import get_client, show_funds
    from dhan_algo.market_data import ltp
    from dhan_algo.orders import place, show_orders, show_positions
    from dhan_algo.risk import kill_switch
    from dhan_algo.security_master import resolve_security_id

    client = get_client(settings)
    mode = "LIVE -- orders will be SENT" if settings.dhan_live else "DRY_RUN (safe)"
    print(f"Mode: {mode}\n")

    if args.command == "funds":
        show_funds(client)

    elif args.command == "ltp":
        sid = resolve_security_id(client, args.symbol)
        if sid:
            price = ltp(client, sid)
            print(f"{args.symbol} (security_id={sid}) LTP = {price}")

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
        from dhan_algo.strategy import SmaDemo, run_strategy_loop

        sid = resolve_security_id(client, args.symbol)
        if not sid:
            sys.exit(f"Could not resolve symbol: {args.symbol}")
        if args.interval is not None:
            settings.strategy_interval = args.interval
        run_strategy_loop(SmaDemo(), client, sid, settings=settings)

    elif args.command == "kill-switch":
        kill_switch(client)


if __name__ == "__main__":
    main()
