"""SwiftTrade — Streamlit web UI for dhan_algo."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import asdict
from datetime import date, timedelta

import pandas as pd
import streamlit as st

from dhan_algo.config import Settings
from dhan_algo.client import get_client, ok
from dhan_algo.market_data import ltp
from dhan_algo.orders import calculate_position_size, place, place_bracket, place_with_sl_target
from dhan_algo.risk import kill_switch
from dhan_algo.security_master import resolve_security_id
from dhan_algo.strategy import Order, PollingTicker, SmaDemoMulti
from dhan_algo.backtest import fetch_historical, load_csv, run_backtest
import yfinance as yf

from intraday_scorer import score_single_intraday, score_universe_intraday
from swing_scorer import score_single, score_universe

# ---------------------------------------------------------------------------
# Stock universes — NIFTY 50 & NIFTY 100 constituents
# ---------------------------------------------------------------------------

NIFTY_50: list[str] = [
    "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK", "BAJAJ-AUTO",
    "BAJFINANCE", "BAJAJFINSV", "BHARTIARTL", "BPCL", "BRITANNIA",
    "CIPLA", "COALINDIA", "DIVISLAB", "DRREDDY", "EICHERMOT",
    "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE", "HEROMOTOCO",
    "HINDALCO", "HINDUNILVR", "ICICIBANK", "INDUSINDBK", "INFY",
    "ITC", "JSWSTEEL", "KOTAKBANK", "LT", "LTIM",
    "M&M", "MARUTI", "NESTLEIND", "NTPC", "ONGC",
    "POWERGRID", "RELIANCE", "SBILIFE", "SBIN", "SHRIRAMFIN",
    "SUNPHARMA", "TATAMOTORS", "TATASTEEL", "TATACONSUM", "TCS",
    "TECHM", "TITAN", "TRENT", "ULTRACEMCO", "WIPRO",
]

NIFTY_NEXT_50: list[str] = [
    "ABB", "ADANIENT", "AMBUJACEM", "ATGL", "AUBANK",
    "BANKBARODA", "BEL", "BHEL", "BOSCHLTD", "CANBK",
    "COLPAL", "DLF", "GAIL", "GODREJCP", "HAL",
    "HAVELLS", "ICICIPRULI", "ICICIGI", "IDFCFIRSTB", "IGL",
    "IOC", "IRCTC", "JINDALSTEL", "JIOFIN", "LUPIN",
    "MARICO", "MAXHEALTH", "NAUKRI", "NHPC", "OFSS",
    "PAGEIND", "PEL", "PERSISTENT", "PETRONET", "PFC",
    "PIDILITIND", "PNB", "POLYCAB", "RECLTD", "SAIL",
    "SBICARD", "SIEMENS", "TATAPOWER", "TORNTPHARM", "TVSMOTOR",
    "UNITDSPR", "VEDL", "ZOMATO", "MCDOWELL-N", "LTF",
]

NIFTY_100: list[str] = NIFTY_50 + NIFTY_NEXT_50

# ---------------------------------------------------------------------------
# Logging capture — collect log records for in-app display
# ---------------------------------------------------------------------------

_log_records: list[str] = []
_log_lock = threading.Lock()


class _StreamlitLogHandler(logging.Handler):
    """Appends formatted records to a module-level list."""

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        with _log_lock:
            _log_records.append(msg)
            # Keep last 500 lines
            if len(_log_records) > 500:
                del _log_records[:100]


def _install_log_handler() -> None:
    root = logging.getLogger()
    if not any(isinstance(h, _StreamlitLogHandler) for h in root.handlers):
        handler = _StreamlitLogHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
        root.addHandler(handler)
        root.setLevel(logging.DEBUG)


_install_log_handler()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_settings() -> Settings:
    """Build Settings from session-state credentials (entered at login).

    The dry-run / live-trading mode defaults to the ``DHAN_LIVE`` env var but
    can be overridden at runtime via the sidebar toggle (session state key
    ``live_trading``).
    """
    creds = st.session_state.get("credentials", {})
    settings = Settings(
        dhan_client_id=creds.get("client_id", ""),
        dhan_access_token=creds.get("access_token", ""),
        dhan_pin=creds.get("pin", ""),
        dhan_totp_secret=creds.get("totp_secret", ""),
    )
    override = st.session_state.get("live_trading")
    if override is not None:
        settings.dhan_live = override
    return settings


def _get_client():
    """Return cached dhanhq client, built from session credentials."""
    if "client" not in st.session_state:
        settings = _get_settings()
        st.session_state.client = get_client(settings)
    return st.session_state.client


def _is_logged_in() -> bool:
    return bool(st.session_state.get("credentials"))


def _logout() -> None:
    for key in ("credentials", "client"):
        st.session_state.pop(key, None)


def page_login() -> None:
    """Login page — user enters their Dhan credentials."""
    st.title("SwiftTrade")
    st.subheader("Login with your Dhan credentials")
    st.caption("Your credentials are stored only in your browser session and are never saved to disk.")

    auth_method = st.radio(
        "Authentication method",
        ["Access Token", "PIN + TOTP Secret"],
        help="Use Access Token (24h token from web.dhan.co) or automatic TOTP login.",
    )

    with st.form("login_form"):
        client_id = st.text_input("Dhan Client ID", placeholder="e.g. 1000000001")

        if auth_method == "Access Token":
            access_token = st.text_input("Access Token", type="password", placeholder="Paste your 24h token")
            pin = ""
            totp_secret = ""
        else:
            access_token = ""
            pin = st.text_input("PIN", type="password", placeholder="6-digit login PIN")
            totp_secret = st.text_input("TOTP Secret", type="password", placeholder="Base32 secret from web.dhan.co")

        submitted = st.form_submit_button("Login", type="primary")

    if submitted:
        if not client_id.strip():
            st.error("Client ID is required.")
            return

        if auth_method == "Access Token" and not access_token.strip():
            st.error("Access Token is required.")
            return

        if auth_method != "Access Token":
            if not pin.strip() or not totp_secret.strip():
                st.error("Both PIN and TOTP Secret are required.")
                return

        st.session_state.credentials = {
            "client_id": client_id.strip(),
            "access_token": access_token.strip(),
            "pin": pin.strip(),
            "totp_secret": totp_secret.strip(),
        }
        # Clear any stale client so it gets rebuilt with new credentials
        st.session_state.pop("client", None)

        # Verify credentials by trying to create a client
        try:
            _get_client()
            st.success("Logged in successfully!")
            time.sleep(0.5)
            st.rerun()
        except SystemExit as e:
            st.error(f"Login failed: {e}")
            _logout()
        except Exception as e:
            st.error(f"Login failed: {e}")
            _logout()


def _show_logs() -> None:
    """Render captured log lines in an expander."""
    with st.expander("Application Logs", expanded=False):
        with _log_lock:
            lines = list(_log_records[-100:])
        if lines:
            st.code("\n".join(lines), language="text")
        else:
            st.info("No log entries yet.")


# ---------------------------------------------------------------------------
# Strategy runner (background thread)
# ---------------------------------------------------------------------------


def _strategy_worker(
    client,
    security_ids: list[str],
    strategy: SmaDemoMulti,
    interval: int,
    segment: str | None,
) -> None:
    """Background thread that polls prices and runs strategy ticks."""
    ticker = PollingTicker(client, security_ids, segment)
    strategy.on_start(ticker, security_ids)
    logger = logging.getLogger("strategy_worker")
    logger.info("Strategy started for %s (interval=%ds)", security_ids, interval)

    while st.session_state.get("strategy_running", False):
        try:
            ticker.refresh_all()
            for sid in security_ids:
                price = ticker.get_ltp(sid)
                if price is not None:
                    st.session_state.setdefault("strategy_log", []).append(
                        f"[{time.strftime('%H:%M:%S')}] {sid} LTP={price:.2f}"
                    )
                orders = strategy.on_tick(ticker, sid, segment)
                for order in orders:
                    st.session_state.setdefault("strategy_log", []).append(
                        f"[{time.strftime('%H:%M:%S')}] SIGNAL: {order.side} {order.qty} of {order.security_id or sid}"
                    )
                    place(
                        client,
                        order.security_id or sid,
                        side=order.side,
                        qty=order.qty,
                        order_type=order.order_type,
                        product=order.product,
                        price=order.price,
                        segment=segment,
                    )
        except Exception:
            logger.exception("Error in strategy tick")
        time.sleep(interval)

    strategy.on_stop()
    logger.info("Strategy stopped.")


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def _fund_metric_map(data: dict) -> list[tuple[str, str]]:
    """Pick common fund fields for a compact metric row, falling back to raw keys."""
    preferred = [
        ("availabelBalance", "Available"),
        ("availableBalance", "Available"),
        ("sodLimit", "SOD limit"),
        ("collateralAmount", "Collateral"),
        ("utilizedAmount", "Utilized"),
        ("withdrawableBalance", "Withdrawable"),
    ]
    picked: list[tuple[str, str]] = []
    for key, label in preferred:
        if key in data and data[key] is not None:
            try:
                picked.append((label, f"{float(data[key]):,.0f}"))
            except (TypeError, ValueError):
                picked.append((label, str(data[key])))
    if not picked:
        for k, v in list(data.items())[:4]:
            picked.append((str(k), str(v)))
    return picked[:4]


def page_dashboard() -> None:
    settings = _get_settings()

    col1, col2 = st.columns([3, 2])
    with col1.container(border=True):
        st.caption("Account funds")
        try:
            client = _get_client()
            resp = client.get_fund_limits()
            if ok(resp):
                data = resp.get("data", {})
                if isinstance(data, dict) and data:
                    metrics = _fund_metric_map(data)
                    mcols = st.columns(len(metrics))
                    for mc, (label, value) in zip(mcols, metrics):
                        mc.metric(label, value)
                    with st.expander("Raw funds response"):
                        st.json(data)
                else:
                    st.write(data)
            else:
                st.error(f"Could not fetch funds: {resp}")
        except SystemExit as e:
            st.warning(f"Client not configured: {e}")
        except Exception as e:
            st.error(f"Error fetching funds: {e}")

    with col2.container(border=True):
        st.caption("Risk configuration")
        r1, r2 = st.columns(2)
        r1.metric("Max qty", settings.max_qty)
        r2.metric("Max order value", f"{settings.max_order_value:,.0f}")
        r3, r4 = st.columns(2)
        r3.metric("Max daily loss", f"{settings.max_daily_loss:,.0f}")
        r4.metric("Strategy interval", f"{settings.strategy_interval}s")


def page_market_data() -> None:
    mcol1, mcol2 = st.columns([3, 1], vertical_alignment="bottom")
    symbol = mcol1.text_input("Symbol", value="RELIANCE", help="e.g. RELIANCE, TCS, INFY")
    auto_refresh = mcol2.checkbox("Auto-refresh (5s)")
    data_source = st.session_state.get("data_source", "Yahoo Finance")

    if symbol:
        if data_source == "Yahoo Finance":
            try:
                ticker_yf = yf.Ticker(f"{symbol.upper()}.NS")
                price = ticker_yf.fast_info.last_price
                if price and price > 0:
                    st.caption("Source: Yahoo Finance (15-min delayed)")
                    st.metric(f"{symbol.upper()} LTP", f"{price:,.2f} INR")
                else:
                    st.warning("Could not fetch price from Yahoo Finance. Try Dhan API or check symbol.")
            except Exception as e:
                st.error(f"Yahoo Finance error: {e}")
        else:
            try:
                client = _get_client()
                sid = resolve_security_id(client, symbol)
                if sid is None:
                    st.error(f"Could not resolve symbol: {symbol}")
                    return
                st.caption(f"Source: Dhan API — Security ID: {sid}")
                price = ltp(client, sid)
                if price is not None:
                    st.metric(f"{symbol.upper()} LTP", f"{price:,.2f} INR")
                else:
                    st.warning("Could not fetch LTP.")
            except SystemExit as e:
                st.warning(f"Client not configured: {e}")
            except Exception as e:
                st.error(f"Error: {e}")

    if auto_refresh:
        time.sleep(5)
        st.rerun()


def page_place_order() -> None:
    settings = _get_settings()

    with st.form("order_form", border=True):
        c1, c2, c3 = st.columns(3)
        symbol = c1.text_input("Symbol", value="RELIANCE")
        side = c2.selectbox("Side", ["BUY", "SELL"])
        qty = c3.number_input("Qty", min_value=1, max_value=settings.max_qty, value=1)
        c4, c5, c6 = st.columns(3)
        order_type = c4.selectbox("Type", ["MARKET", "LIMIT"])
        product = c5.selectbox("Product", ["INTRA", "CNC"])
        price = c6.number_input("Price (LIMIT)", min_value=0.0, value=0.0, step=0.05)
        submitted = st.form_submit_button("Place order", type="primary", width="stretch")

    if submitted:
        try:
            client = _get_client()
            sid = resolve_security_id(client, symbol)
            if sid is None:
                st.error(f"Could not resolve symbol: {symbol}")
                return
            result = place(
                client,
                sid,
                side=side,
                qty=qty,
                order_type=order_type,
                product=product,
                price=price if order_type == "LIMIT" else 0.0,
            )
            if result is None:
                st.error("Order blocked by risk guards. Check logs for details.")
            elif result.get("status") == "dry_run":
                st.info(f"[DRY RUN] {result.get('plan', '')}")
            elif ok(result):
                st.success(f"Order placed: {result}")
            else:
                st.error(f"Order failed: {result}")
        except SystemExit as e:
            st.warning(f"Client not configured: {e}")
        except Exception as e:
            st.error(f"Error: {e}")


def page_positions_orders() -> None:
    try:
        client = _get_client()
    except SystemExit as e:
        st.warning(f"Client not configured: {e}")
        return
    except Exception as e:
        st.error(f"Error: {e}")
        return

    tab_pos, tab_ord = st.tabs(["Positions", "Orders"])

    with tab_pos:
        try:
            resp = client.get_positions()
            if ok(resp):
                data = resp.get("data", [])
                if data:
                    st.dataframe(pd.DataFrame(data), width="stretch", hide_index=True)
                else:
                    st.info("No open positions.")
            else:
                st.error(f"Could not fetch positions: {resp}")
        except Exception as e:
            st.error(f"Error fetching positions: {e}")

    with tab_ord:
        try:
            resp = client.get_order_list()
            if ok(resp):
                data = resp.get("data", [])
                if data:
                    st.dataframe(pd.DataFrame(data), width="stretch", hide_index=True)
                else:
                    st.info("No orders today.")
            else:
                st.error(f"Could not fetch orders: {resp}")
        except Exception as e:
            st.error(f"Error fetching orders: {e}")


def page_strategy() -> None:
    st.caption("SMA crossover demo — not investment advice. Uses Dhan API for real-time polling.")
    if st.session_state.get("data_source", "Yahoo Finance") == "Yahoo Finance":
        st.warning("Switch data source to Dhan API in the sidebar for live polling.", icon=":material/warning:")

    settings = _get_settings()

    running = st.session_state.get("strategy_running", False)

    symbols_input = st.text_input("Symbols (comma-separated)", value="RELIANCE, INFY")
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1], vertical_alignment="bottom")
    with col1:
        short_period = st.number_input("Short SMA", min_value=2, max_value=50, value=5)
    with col2:
        long_period = st.number_input("Long SMA", min_value=5, max_value=200, value=20)
    with col3:
        interval = st.number_input("Interval (s)", min_value=5, max_value=600, value=settings.strategy_interval)
    with col4:
        start_clicked = st.button(
            "Start", disabled=running, type="primary", width="stretch",
            icon=":material/play_arrow:",
        )
    with col5:
        stop_clicked = st.button(
            "Stop", disabled=not running, width="stretch",
            icon=":material/stop:",
        )

    if start_clicked:
        try:
            client = _get_client()
        except (SystemExit, Exception) as e:
            st.error(f"Client error: {e}")
            return

        symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
        security_ids = []
        for sym in symbols:
            sid = resolve_security_id(client, sym)
            if sid is None:
                st.error(f"Could not resolve: {sym}")
                return
            security_ids.append(sid)

        strategy = SmaDemoMulti(short_period=short_period, long_period=long_period, qty=1)
        st.session_state.strategy_running = True
        st.session_state.strategy_log = []
        thread = threading.Thread(
            target=_strategy_worker,
            args=(client, security_ids, strategy, interval, None),
            daemon=True,
        )
        thread.start()
        st.session_state.strategy_thread = thread
        st.rerun()

    if stop_clicked:
        st.session_state.strategy_running = False
        st.rerun()

    # Status + live log
    if running:
        st.success("Strategy running", icon=":material/sensors:")
    else:
        st.info("Strategy stopped", icon=":material/pause:")

    st.caption("Signal log")
    log_lines = st.session_state.get("strategy_log", [])
    if log_lines:
        st.code("\n".join(log_lines[-50:]), language="text")
    else:
        st.info("No signals yet.")

    if running:
        time.sleep(2)
        st.rerun()


def _yf_to_bars(df: pd.DataFrame, symbol: str) -> list[dict]:
    """Convert a yfinance OHLCV DataFrame to the bar dict format used by run_backtest."""
    bars = []
    for _, row in df.iterrows():
        ts = str(row.get("Datetime", row.get("Date", "")))
        bars.append({
            "timestamp": ts,
            "security_id": symbol,
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
            "volume": int(row.get("Volume", 0)),
        })
    return bars


def page_backtest() -> None:
    st.caption("Backtest — SMA crossover replay over historical data")

    data_source = st.session_state.get("data_source", "Yahoo Finance")
    tab_fetch, tab_csv = st.tabs([f"Fetch from {data_source}", "Upload CSV"])

    with tab_fetch:
        symbols_input = st.text_input("Symbols (comma-separated)", value="RELIANCE", key="bt_symbols")
        col1, col2, col3 = st.columns(3)
        with col1:
            from_date = st.date_input("From", value=date.today() - timedelta(days=90), key="bt_from")
        with col2:
            to_date = st.date_input("To", value=date.today(), key="bt_to")
        with col3:
            interval = st.selectbox("Interval", ["day", "minute"], key="bt_interval")

        if st.button(f"Run Backtest ({data_source})", key="bt_api_run"):
            symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
            all_bars: dict[str, list] = {}

            if data_source == "Yahoo Finance":
                tickers_yf = [f"{sym}.NS" for sym in symbols]
                yf_interval = "1d" if interval == "day" else "1m"
                days_back = max((date.today() - from_date).days, 1)
                period = f"{days_back}d" if days_back <= 730 else "2y"
                with st.spinner("Fetching historical data from Yahoo Finance..."):
                    try:
                        df_all = yf.download(
                            tickers_yf, period=period, interval=yf_interval,
                            auto_adjust=True, group_by="ticker", threads=True,
                        )
                    except Exception as e:
                        st.error(f"Yahoo Finance error: {e}")
                        return

                if df_all is None or df_all.empty:
                    st.error("No data fetched from Yahoo Finance.")
                    return

                multi = isinstance(df_all.columns, pd.MultiIndex)
                for sym in symbols:
                    ticker_yf = f"{sym}.NS"
                    try:
                        df_bars = df_all[ticker_yf] if multi else df_all
                        df_bars = df_bars.dropna(how="all").reset_index()
                        for col in ("Open", "High", "Low", "Close", "Volume"):
                            if col not in df_bars.columns:
                                lc = col.lower()
                                if lc in df_bars.columns:
                                    df_bars.rename(columns={lc: col}, inplace=True)
                        if len(df_bars) >= 2:
                            all_bars[sym] = _yf_to_bars(df_bars, sym)
                        else:
                            st.warning(f"Not enough data for {sym}")
                    except (KeyError, Exception) as e:
                        st.warning(f"Skipped {sym}: {e}")
            else:
                try:
                    client = _get_client()
                except (SystemExit, Exception) as e:
                    st.error(f"Client error: {e}")
                    return

                with st.spinner("Fetching historical data from Dhan API..."):
                    for sym in symbols:
                        sid = resolve_security_id(client, sym)
                        if sid is None:
                            st.error(f"Could not resolve: {sym}")
                            return
                        bars = fetch_historical(
                            client,
                            sid,
                            from_date=str(from_date),
                            to_date=str(to_date),
                            interval=interval,
                        )
                        if bars:
                            all_bars[sid] = bars
                        else:
                            st.warning(f"No data for {sym} ({sid})")

            if all_bars:
                _run_and_display_backtest(all_bars)

    with tab_csv:
        uploaded = st.file_uploader(
            "Upload CSV (columns: timestamp, security_id, open, high, low, close, volume)",
            type=["csv"],
        )
        if uploaded is not None and st.button("Run Backtest (CSV)", key="bt_csv_run"):
            import tempfile, os
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="wb") as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = tmp.name
            try:
                all_bars = load_csv(tmp_path)
                if all_bars:
                    _run_and_display_backtest(all_bars)
                else:
                    st.warning("CSV produced no bars.")
            finally:
                os.unlink(tmp_path)


def _run_and_display_backtest(all_bars: dict[str, list]) -> None:
    strategy = SmaDemoMulti(short_period=5, long_period=20, qty=1)
    with st.spinner("Running backtest..."):
        result = run_backtest(strategy, all_bars)

    with st.container(border=True):
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Trades", result.total_trades)
        c2.metric("Buys", result.buy_count)
        c3.metric("Sells", result.sell_count)
        c4.metric("Realized", f"{result.pnl:,.0f}")
        c5.metric("Unrealized", f"{result.unrealized_pnl:,.0f}")
        c6.metric("Net P&L", f"{result.pnl + result.unrealized_pnl:,.0f}")

    if result.positions:
        with st.expander("Open positions"):
            st.json(result.positions)

    if result.fills:
        st.caption("Fills")
        fills_data = [asdict(f) for f in result.fills]
        st.dataframe(pd.DataFrame(fills_data), width="stretch", hide_index=True)


def page_journal() -> None:
    settings = _get_settings()

    import os
    if not os.path.exists(settings.journal_path):
        st.info(f"No journal file found at `{settings.journal_path}`.")
        return

    try:
        df = pd.read_csv(settings.journal_path)
    except Exception as e:
        st.error(f"Error reading journal: {e}")
        return

    if df.empty:
        st.info("Journal is empty.")
        return

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        if "status" in df.columns:
            statuses = ["All"] + sorted(df["status"].dropna().unique().tolist())
            selected_status = st.selectbox("Status", statuses)
        else:
            selected_status = "All"
    with col2:
        if "side" in df.columns:
            sides = ["All"] + sorted(df["side"].dropna().unique().tolist())
            selected_side = st.selectbox("Side", sides)
        else:
            selected_side = "All"

    filtered = df.copy()
    if selected_status != "All":
        filtered = filtered[filtered["status"] == selected_status]
    if selected_side != "All":
        filtered = filtered[filtered["side"] == selected_side]

    st.dataframe(filtered, width="stretch", hide_index=True)
    st.caption(f"{len(filtered)} of {len(df)} entries shown")


def page_kill_switch() -> None:
    st.warning(
        "Activating the kill switch will **disable all trading** on your Dhan account "
        "for the rest of the trading day. This action cannot be undone.",
        icon=":material/dangerous:",
    )

    if "kill_confirm" not in st.session_state:
        st.session_state.kill_confirm = False

    if not st.session_state.kill_confirm:
        if st.button("Activate Kill Switch", type="primary"):
            st.session_state.kill_confirm = True
            st.rerun()
    else:
        st.error("Are you sure? This will disable ALL trading for the rest of the day.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Yes, activate kill switch", type="primary"):
                try:
                    client = _get_client()
                    resp = kill_switch(client)
                    if ok(resp):
                        st.success("Kill switch activated.")
                    else:
                        st.error(f"Kill switch response: {resp}")
                except SystemExit as e:
                    st.warning(f"Client not configured: {e}")
                except Exception as e:
                    st.error(f"Error: {e}")
                finally:
                    st.session_state.kill_confirm = False
        with col2:
            if st.button("Cancel"):
                st.session_state.kill_confirm = False
                st.rerun()


def sidebar_intraday() -> None:
    """Intraday scanner controls (rendered in the sidebar)."""
    universe = st.selectbox(
        "Universe",
        ["NIFTY 50", "NIFTY 100", "Custom"],
        key="intra_universe",
        help="Pre-built universe or enter your own symbols",
    )
    if universe == "Custom":
        st.text_input(
            "Symbols (comma-separated)",
            value="RELIANCE, TCS, INFY, HDFCBANK, SBIN",
            key="intra_symbols",
        )
    else:
        symbol_list = NIFTY_50 if universe == "NIFTY 50" else NIFTY_100
        st.session_state["intra_symbols"] = ", ".join(symbol_list)
        st.caption(f"{len(symbol_list)} stocks")

    st.segmented_control(
        "Candle interval (min)", [1, 5, 15], default=15, key="intra_interval"
    )

    with st.expander("Scoring weights"):
        st.slider("VWAP", 0.0, 5.0, 1.0, 0.5, key="w_vwap")
        st.slider("SuperTrend", 0.0, 5.0, 1.0, 0.5, key="w_st")
        st.slider("Momentum", 0.0, 5.0, 1.5, 0.5, key="w_mom")
        st.slider("Volume", 0.0, 5.0, 0.5, 0.5, key="w_vol")
        st.slider("ORB", 0.0, 5.0, 1.0, 0.5, key="w_orb")

    st.session_state["intra_do_scan"] = st.button(
        "Scan now", type="primary", key="intra_scan", width="stretch",
        icon=":material/radar:",
    )


def page_intraday() -> None:
    st.caption("Intraday scanner — VWAP, SuperTrend, momentum, volume & ORB")

    settings = _get_settings()

    symbols_input = st.session_state.get("intra_symbols", "RELIANCE, TCS, INFY, HDFCBANK, SBIN")
    interval_minutes = st.session_state.get("intra_interval", 15) or 15
    weights = {
        "vwap": st.session_state.get("w_vwap", 1.0),
        "supertrend": st.session_state.get("w_st", 1.0),
        "momentum": st.session_state.get("w_mom", 1.5),
        "volume": st.session_state.get("w_vol", 0.5),
        "orb": st.session_state.get("w_orb", 1.0),
    }
    params = {"interval_minutes": interval_minutes, "atr_multiplier": 1.0}

    if st.session_state.get("intra_do_scan"):
        symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
        if not symbols:
            st.warning("Enter at least one symbol.")
            return

        # Fetch intraday data via yfinance (batch download)
        data: dict[str, pd.DataFrame] = {}
        tickers_yf = [f"{sym}.NS" for sym in symbols]
        yf_interval = f"{interval_minutes}m"
        with st.spinner(f"Fetching intraday data for {len(symbols)} symbols..."):
            try:
                df_all = yf.download(
                    tickers_yf, period="5d", interval=yf_interval,
                    auto_adjust=True, group_by="ticker", threads=True,
                )
            except Exception as e:
                st.error(f"Error fetching data: {e}")
                return

        if df_all is None or df_all.empty:
            st.error("No data fetched for any symbol.")
            return

        multi = isinstance(df_all.columns, pd.MultiIndex)
        for sym in symbols:
            ticker_yf = f"{sym}.NS"
            try:
                df_bars = df_all[ticker_yf] if multi else df_all
                df_bars = df_bars.dropna(how="all").reset_index()
                for col in ("Open", "High", "Low", "Close", "Volume"):
                    if col not in df_bars.columns:
                        lc = col.lower()
                        if lc in df_bars.columns:
                            df_bars.rename(columns={lc: col}, inplace=True)
                if len(df_bars) >= 5:
                    data[sym] = df_bars
            except (KeyError, Exception):
                pass

        if not data:
            st.error("No data fetched for any symbol.")
            return

        # Score
        with st.spinner("Scoring..."):
            results = score_universe_intraday(data, weights, params)

        if not results:
            st.info("No symbols met the minimum bar requirement for scoring.")
            return

        st.session_state["intra_results"] = results

    # --- Display results ---
    results = st.session_state.get("intra_results")
    if not results:
        st.info("Click **Scan Now** to fetch data and score symbols.")
        return

    # Summary table
    display_cols = [
        "ticker", "name", "price", "change_pct", "score",
        "vwap_score", "supertrend_score", "momentum_score",
        "volume_score", "orb_score",
    ]
    df_display = pd.DataFrame(results)[display_cols]
    df_display.columns = [
        "Symbol", "Name", "Price", "Change %", "Score",
        "VWAP", "SuperTrend", "Momentum", "Volume", "ORB",
    ]

    # Color-code the score column
    def _highlight_score(val):
        if val >= 75:
            return "background-color: #1b5e20; color: white"
        elif val >= 50:
            return "background-color: #33691e; color: white"
        elif val >= 25:
            return "background-color: #e65100; color: white"
        else:
            return "background-color: #b71c1c; color: white"

    styled = df_display.style.map(_highlight_score, subset=["Score"])
    st.dataframe(styled, width="stretch", hide_index=True)

    # Symbol selector from scored results
    ticker_options = [r["ticker"] for r in results]
    selected_ticker = st.selectbox(
        "Symbol to trade",
        ticker_options,
        key="intra_trade_ticker",
    )

    # Find the result row for the selected ticker
    row = next((r for r in results if r["ticker"] == selected_ticker), None)
    if row is None:
        return

    entry = row.get("entry", 0)
    sl = row.get("stop_loss", 0)
    t1 = row.get("target1", 0)
    t2 = row.get("target2", 0)
    st_dir = row.get("st_direction", 0)
    above_orb = row.get("above_orb", 0)

    # Compact trade setup display
    st.markdown(
        f"""
<table style="width:100%; font-size:0.82rem; line-height:1.8; border-collapse:collapse;">
<tr style="border-bottom:1px solid #444;">
  <td><b>Score:</b> {row['score']:.1f}</td>
  <td><b>Price:</b> {row['price']:,.2f}</td>
  <td><b>RSI(7):</b> {row.get('rsi7', 0):.1f}</td>
  <td><b>Vol Ratio:</b> {row.get('vol_ratio', 0):.2f}</td>
</tr>
<tr style="border-bottom:1px solid #444;">
  <td><b>VWAP:</b> {row.get('vwap', 0):,.2f} ({row.get('vwap_pct', 0):+.2f}%)</td>
  <td><b>SuperTrend:</b> {"BULLISH" if st_dir == 1 else "BEARISH"}</td>
  <td><b>ORB:</b> {"ABOVE" if above_orb == 1 else "INSIDE/BELOW"}</td>
  <td><b>Change:</b> {row.get('change_pct', 0):+.2f}%</td>
</tr>
<tr>
  <td><b>Entry:</b> {entry:,.2f}</td>
  <td><b>Stop Loss:</b> {sl:,.2f}</td>
  <td><b>Target 1:</b> {t1:,.2f} (RR {row.get('rr1', 0)})</td>
  <td><b>Target 2:</b> {t2:,.2f} (RR {row.get('rr2', 0)})</td>
</tr>
</table>
""",
        unsafe_allow_html=True,
    )

    # Position sizing info
    _entry_default = round(entry if entry > 0 else row["price"], 2)
    _sl_default = round(sl, 2) if sl > 0 else round(row["price"] * 0.99, 2)
    _risk_per_share = abs(_entry_default - _sl_default)

    if settings.risk_per_trade > 0 and _risk_per_share > 0.01:
        auto_qty = calculate_position_size(
            _entry_default, _sl_default,
            settings.risk_per_trade, settings.max_qty, settings.max_order_value,
        )
        st.caption(
            f"Position sizing: risk {settings.risk_per_trade:.0f} INR / "
            f"risk per share {_risk_per_share:.2f} = **{auto_qty} shares** "
            f"(max qty: {settings.max_qty}, max value: {settings.max_order_value:,.0f})"
        )
    else:
        auto_qty = 1

    with st.form("intra_order_form", border=True):
        st.caption(f"Bracket order for {selected_ticker} (entry + SL + target)")
        fcol1, fcol2, fcol3, fcol4, fcol5 = st.columns(5)
        with fcol1:
            side = st.selectbox("Side", ["BUY", "SELL"], key="intra_side")
        with fcol2:
            qty = st.number_input(
                "Qty",
                min_value=1,
                max_value=settings.max_qty,
                value=max(auto_qty, 1),
                key="intra_qty",
            )
        with fcol3:
            entry_price = st.number_input(
                "Entry",
                min_value=0.01,
                value=_entry_default,
                step=0.05,
                key="intra_entry_price",
            )
        with fcol4:
            sl_price = st.number_input(
                "Stop loss",
                min_value=0.01,
                value=_sl_default,
                step=0.05,
                key="intra_sl_price",
            )
        with fcol5:
            target_price = st.number_input(
                "Target",
                min_value=0.01,
                value=round(t1, 2) if t1 > 0 else round(row["price"] * 1.01, 2),
                step=0.05,
                key="intra_target_price",
            )

        submitted = st.form_submit_button(
            "Place bracket order", type="primary", width="stretch",
        )

    if submitted:
        try:
            client = _get_client()
            sid = resolve_security_id(client, selected_ticker)
            if sid is None:
                st.error(f"Could not resolve symbol: {selected_ticker}")
                return

            result = place_bracket(
                client, sid,
                side=side, qty=qty,
                entry_price=entry_price,
                stop_loss_price=sl_price,
                target_price=target_price,
                settings=settings,
            )
            if result is None:
                st.error("Order blocked by risk guards.")
            elif result.get("status") == "dry_run":
                st.info(f"[DRY RUN] {result.get('plan', '')}")
            elif ok(result):
                st.success("Bracket order placed: Entry + SL + Target in one order")
                st.json(result)
            else:
                st.error(f"Bracket order failed: {result}")

        except SystemExit as e:
            st.warning(f"Client not configured: {e}")
        except Exception as e:
            st.error(f"Error placing order: {e}")


def sidebar_swing() -> None:
    """Swing scanner controls (rendered in the sidebar)."""
    universe = st.selectbox(
        "Universe",
        ["NIFTY 50", "NIFTY 100", "Custom"],
        key="swing_universe",
        help="Pre-built universe or enter your own symbols",
    )
    if universe == "Custom":
        st.text_input(
            "Symbols (comma-separated)",
            value="RELIANCE, TCS, INFY, HDFCBANK, SBIN",
            key="swing_symbols",
        )
    else:
        symbol_list = NIFTY_50 if universe == "NIFTY 50" else NIFTY_100
        st.session_state["swing_symbols"] = ", ".join(symbol_list)
        st.caption(f"{len(symbol_list)} stocks")

    with st.expander("Scoring weights"):
        st.slider("Trend", 0.0, 5.0, 1.0, 0.5, key="sw_trend")
        st.slider("Momentum", 0.0, 5.0, 1.0, 0.5, key="sw_mom")
        st.slider("Volume", 0.0, 5.0, 0.8, 0.5, key="sw_vol")
        st.slider("Breakout", 0.0, 5.0, 0.8, 0.5, key="sw_brk")
        st.slider("Volatility", 0.0, 5.0, 0.5, 0.5, key="sw_vlt")

    with st.expander("Advanced"):
        st.number_input(
            "Stop-loss ATR multiplier",
            min_value=0.5, max_value=3.0, value=1.5, step=0.1,
            key="sw_atr_mult",
        )
        st.number_input(
            "History (trading days)",
            min_value=250, max_value=500, value=365, step=10,
            key="sw_lookback",
            help="Number of calendar days of daily data to fetch",
        )

    st.session_state["swing_do_scan"] = st.button(
        "Scan now", type="primary", key="swing_scan", width="stretch",
        icon=":material/radar:",
    )


def page_swing() -> None:
    st.caption("Swing scanner — trend, momentum, volume, breakout & volatility")

    settings = _get_settings()

    symbols_input = st.session_state.get("swing_symbols", "RELIANCE, TCS, INFY, HDFCBANK, SBIN")
    weights = {
        "trend": st.session_state.get("sw_trend", 1.0),
        "momentum": st.session_state.get("sw_mom", 1.0),
        "volume": st.session_state.get("sw_vol", 0.8),
        "breakout": st.session_state.get("sw_brk", 0.8),
        "volatility": st.session_state.get("sw_vlt", 0.5),
    }
    params = {"atr_multiplier": st.session_state.get("sw_atr_mult", 1.5)}

    if st.session_state.get("swing_do_scan"):
        symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
        if not symbols:
            st.warning("Enter at least one symbol.")
            return

        # Fetch daily data via yfinance (batch download)
        data: dict[str, pd.DataFrame] = {}
        tickers_yf = [f"{sym}.NS" for sym in symbols]
        with st.spinner(f"Fetching daily data for {len(symbols)} symbols..."):
            try:
                df_all = yf.download(
                    tickers_yf, period="1y",
                    auto_adjust=True, group_by="ticker", threads=True,
                )
            except Exception as e:
                st.error(f"Error fetching data: {e}")
                return

        if df_all is None or df_all.empty:
            st.error("No data fetched for any symbol.")
            return

        multi = isinstance(df_all.columns, pd.MultiIndex)
        for sym in symbols:
            ticker_yf = f"{sym}.NS"
            try:
                df_bars = df_all[ticker_yf] if multi else df_all
                df_bars = df_bars.dropna(how="all").reset_index()
                for col in ("Open", "High", "Low", "Close", "Volume"):
                    if col not in df_bars.columns:
                        lc = col.lower()
                        if lc in df_bars.columns:
                            df_bars.rename(columns={lc: col}, inplace=True)
                if len(df_bars) >= 220:
                    data[sym] = df_bars
            except (KeyError, Exception):
                pass

        if not data:
            st.error("No data fetched for any symbol.")
            return

        # Score
        with st.spinner("Scoring..."):
            results = score_universe(data, weights, params)

        if not results:
            st.info("No symbols met the minimum bar requirement (220 daily bars needed).")
            return

        st.session_state["swing_results"] = results

    # --- Display results ---
    results = st.session_state.get("swing_results")
    if not results:
        st.info("Click **Scan Now** to fetch daily data and score symbols.")
        return

    # Summary table
    display_cols = [
        "ticker", "name", "price", "change_pct", "score",
        "trend_score", "momentum_score", "volume_score",
        "breakout_score", "volatility_score",
    ]
    df_display = pd.DataFrame(results)[display_cols]
    df_display.columns = [
        "Symbol", "Name", "Price", "Change %", "Score",
        "Trend", "Momentum", "Volume", "Breakout", "Volatility",
    ]

    def _highlight_score(val):
        if val >= 75:
            return "background-color: #1b5e20; color: white"
        elif val >= 50:
            return "background-color: #33691e; color: white"
        elif val >= 25:
            return "background-color: #e65100; color: white"
        else:
            return "background-color: #b71c1c; color: white"

    styled = df_display.style.map(_highlight_score, subset=["Score"])
    st.dataframe(styled, width="stretch", hide_index=True)

    ticker_options = [r["ticker"] for r in results]
    selected_ticker = st.selectbox(
        "Symbol to trade",
        ticker_options,
        key="swing_trade_ticker",
    )

    row = next((r for r in results if r["ticker"] == selected_ticker), None)
    if row is None:
        return

    # Compact trade setup: key stats + levels grouped in one bordered block.
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Score", f"{row['score']:.1f}")
        c2.metric("Price", f"{row['price']:,.2f}")
        c3.metric("RSI(14)", f"{row.get('rsi', 0):.1f}")
        c4.metric("Vol ratio", f"{row.get('vol_ratio', 0):.2f}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Entry", f"{row.get('entry', 0):,.2f}")
        c2.metric("Stop loss", f"{row.get('stop_loss', 0):,.2f}")
        c3.metric("Target 1", f"{row.get('target1', 0):,.2f}", f"RR {row.get('rr1', 0)}")
        c4.metric("Target 2", f"{row.get('target2', 0):,.2f}", f"RR {row.get('rr2', 0)}")

        st.caption(
            f"52wk high {row.get('pct_from_52wk', 0):.1f}%  ·  "
            f"MACD hist {row.get('macd_hist', 0):.4f}  ·  "
            f"%B {row.get('pct_b', 0):.2f}  ·  ATR {row.get('atr_pct', 0):.2f}%"
        )

    _sw_entry_default = round(row.get("entry", row["price"]), 2)
    _sw_sl_default = round(row.get("stop_loss", row["price"] * 0.97), 2)
    _sw_risk_per_share = abs(_sw_entry_default - _sw_sl_default)

    if settings.risk_per_trade > 0 and _sw_risk_per_share > 0.01:
        sw_auto_qty = calculate_position_size(
            _sw_entry_default, _sw_sl_default,
            settings.risk_per_trade, settings.max_qty, settings.max_order_value,
        )
        st.caption(
            f"Position sizing: risk {settings.risk_per_trade:.0f} INR / "
            f"risk per share {_sw_risk_per_share:.2f} = **{sw_auto_qty} shares** "
            f"(max qty: {settings.max_qty}, max value: {settings.max_order_value:,.0f})"
        )
    else:
        sw_auto_qty = 1

    with st.form("swing_order_form", border=True):
        st.caption(f"Place order for {selected_ticker} (entry + SL + target)")
        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        with r1c1:
            side = st.selectbox("Side", ["BUY", "SELL"], key="sw_side")
        with r1c2:
            order_type = st.selectbox("Type", ["LIMIT", "MARKET"], key="sw_otype")
        with r1c3:
            product = st.selectbox("Product", ["CNC", "INTRA"], key="sw_product",
                                   help="CNC for delivery (swing), INTRA for same-day")
        with r1c4:
            qty = st.number_input(
                "Qty", min_value=1, max_value=settings.max_qty,
                value=max(sw_auto_qty, 1), key="sw_qty",
            )

        r2c1, r2c2, r2c3, r2c4 = st.columns([1, 1, 1, 1])
        with r2c1:
            limit_price = st.number_input(
                "Entry", min_value=0.0, value=_sw_entry_default,
                step=0.05, key="sw_price",
            )
        with r2c2:
            sw_sl = st.number_input(
                "Stop loss", min_value=0.0, value=_sw_sl_default,
                step=0.05, key="sw_sl_price",
            )
        with r2c3:
            sw_target = st.number_input(
                "Target", min_value=0.0,
                value=round(row.get("target1", row["price"] * 1.05), 2),
                step=0.05, key="sw_target_price",
            )
        with r2c4:
            st.write("")
            submitted = st.form_submit_button(
                "Place order", type="primary", width="stretch",
            )

    if submitted:
        try:
            client = _get_client()
            sid = resolve_security_id(client, selected_ticker)
            if sid is None:
                st.error(f"Could not resolve symbol: {selected_ticker}")
                return

            if product == "INTRA":
                result = place_bracket(
                    client, sid,
                    side=side, qty=qty,
                    entry_price=limit_price if order_type == "LIMIT" else 0.0,
                    stop_loss_price=sw_sl,
                    target_price=sw_target,
                    settings=settings,
                )
                if result is None:
                    st.error("Order blocked by risk guards.")
                elif result.get("status") == "dry_run":
                    st.info(f"[DRY RUN] {result.get('plan', '')}")
                elif ok(result):
                    st.success("Bracket order placed: Entry + SL + Target in one order")
                    st.json(result)
                else:
                    st.error(f"Bracket order failed: {result}")
            else:
                result = place_with_sl_target(
                    client, sid,
                    side=side, qty=qty,
                    entry_price=limit_price if order_type == "LIMIT" else 0.0,
                    stop_loss_price=sw_sl,
                    target_price=sw_target,
                    order_type=order_type, product=product,
                    settings=settings,
                )
                for leg, resp in result.items():
                    if resp is None:
                        st.error(f"{leg.replace('_', ' ').title()} order blocked by risk guards.")
                    elif resp.get("status") == "dry_run":
                        st.info(f"[DRY RUN] {leg.replace('_', ' ').title()}: {resp.get('plan', '')}")
                    elif ok(resp):
                        st.success(f"{leg.replace('_', ' ').title()} order placed: {resp}")
                    else:
                        st.error(f"{leg.replace('_', ' ').title()} order failed: {resp}")
        except SystemExit as e:
            st.warning(f"Client not configured: {e}")
        except Exception as e:
            st.error(f"Error placing order: {e}")


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

PAGES = {
    "Dashboard": page_dashboard,
    "Swing Scanner": page_swing,
    "Intraday Scanner": page_intraday,
    "Market Data": page_market_data,
    "Place Order": page_place_order,
    "Positions & Orders": page_positions_orders,
    "Strategy": page_strategy,
    "Backtest": page_backtest,
    "Trade Journal": page_journal,
    "Kill Switch": page_kill_switch,
}

# Per-page controls rendered in the sidebar to keep the main area focused
# on results and actions.
SIDEBAR_CONTROLS = {
    "Swing Scanner": sidebar_swing,
    "Intraday Scanner": sidebar_intraday,
}

st.set_page_config(page_title="SwiftTrade", layout="wide", initial_sidebar_state="expanded")

if not _is_logged_in():
    page_login()
else:
    settings = _get_settings()
    creds = st.session_state.get("credentials", {})

    # Sidebar: navigation + global controls only (kept dense).
    with st.sidebar:
        page = st.radio(
            "Navigation", list(PAGES.keys()), label_visibility="collapsed"
        )

        st.segmented_control(
            "Data source",
            ["Yahoo Finance", "Dhan API"],
            key="data_source",
            default="Yahoo Finance",
            help=(
                "Yahoo Finance: free, ~15-min delayed, no login needed.\n"
                "Dhan API: real-time, requires Dhan credentials."
            ),
        )

        # Trading mode toggle. Seed from the DHAN_LIVE env value on first
        # load, then let the user flip it; _get_settings() reads this key.
        st.session_state.setdefault("live_trading", settings.dhan_live)
        live = st.toggle(
            "Live trading",
            key="live_trading",
            help="Off = dry run (no real orders). On = send real orders to Dhan.",
        )
        with st.container(horizontal=True, vertical_alignment="center"):
            if live:
                st.badge("LIVE", color="red", icon=":material/bolt:")
            else:
                st.badge("DRY RUN", color="green", icon=":material/shield:")
            if st.session_state.get("strategy_running", False):
                st.badge("Strategy running", color="blue")
        if live:
            st.warning("Real orders will be sent to your account.", icon=":material/warning:")
        st.caption(
            f"Qty ≤ {settings.max_qty}  ·  Value ≤ {settings.max_order_value:,.0f}  ·  "
            f"Loss ≤ {settings.max_daily_loss:,.0f}"
        )

        # Contextual controls for the active page render here to keep the
        # main area focused on results and actions.
        sidebar_controls = SIDEBAR_CONTROLS.get(page)
        if sidebar_controls is not None:
            st.divider()
            sidebar_controls()

        with st.container(horizontal=True, vertical_alignment="center"):
            st.caption(f"Client {creds.get('client_id', '')}")
            if st.button("Logout", icon=":material/logout:", width="stretch"):
                _logout()
                st.rerun()

    # Render selected page, then a collapsed logs expander.
    PAGES[page]()
    _show_logs()
