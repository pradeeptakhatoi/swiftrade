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
from dhan_algo.orders import place
from dhan_algo.risk import kill_switch
from dhan_algo.security_master import resolve_security_id
from dhan_algo.strategy import Order, PollingTicker, SmaDemoMulti
from dhan_algo.backtest import fetch_historical, load_csv, run_backtest
import yfinance as yf

from intraday_scorer import score_single_intraday, score_universe_intraday
from swing_scorer import score_single, score_universe

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
    """Build Settings from session-state credentials (entered at login)."""
    creds = st.session_state.get("credentials", {})
    return Settings(
        dhan_client_id=creds.get("client_id", ""),
        dhan_access_token=creds.get("access_token", ""),
        dhan_pin=creds.get("pin", ""),
        dhan_totp_secret=creds.get("totp_secret", ""),
    )


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


def page_dashboard() -> None:
    st.header("Dashboard")
    settings = _get_settings()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Account Funds")
        try:
            client = _get_client()
            resp = client.get_fund_limits()
            if ok(resp):
                data = resp.get("data", {})
                if isinstance(data, dict):
                    st.json(data)
                else:
                    st.write(data)
            else:
                st.error(f"Could not fetch funds: {resp}")
        except SystemExit as e:
            st.warning(f"Client not configured: {e}")
        except Exception as e:
            st.error(f"Error fetching funds: {e}")

    with col2:
        st.subheader("Configuration")
        st.metric("Max Qty", settings.max_qty)
        st.metric("Max Order Value", f"{settings.max_order_value:,.0f} INR")
        st.metric("Max Daily Loss", f"{settings.max_daily_loss:,.0f} INR")
        st.metric("Strategy Interval", f"{settings.strategy_interval}s")


def page_market_data() -> None:
    st.header("Market Data")

    symbol = st.text_input("Symbol", value="RELIANCE", help="e.g. RELIANCE, TCS, INFY")
    auto_refresh = st.checkbox("Auto-refresh (every 5s)")

    if symbol:
        try:
            client = _get_client()
            sid = resolve_security_id(client, symbol)
            if sid is None:
                st.error(f"Could not resolve symbol: {symbol}")
                return
            st.caption(f"Security ID: {sid}")
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
    st.header("Place Order")
    settings = _get_settings()

    with st.form("order_form"):
        symbol = st.text_input("Symbol", value="RELIANCE")
        col1, col2 = st.columns(2)
        with col1:
            side = st.selectbox("Side", ["BUY", "SELL"])
            qty = st.number_input("Quantity", min_value=1, max_value=settings.max_qty, value=1)
        with col2:
            order_type = st.selectbox("Order Type", ["MARKET", "LIMIT"])
            product = st.selectbox("Product", ["INTRA", "CNC"])
        price = st.number_input("Price (for LIMIT orders)", min_value=0.0, value=0.0, step=0.05)
        submitted = st.form_submit_button("Place Order")

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
    st.header("Positions & Orders")

    try:
        client = _get_client()
    except SystemExit as e:
        st.warning(f"Client not configured: {e}")
        return
    except Exception as e:
        st.error(f"Error: {e}")
        return

    st.subheader("Positions")
    try:
        resp = client.get_positions()
        if ok(resp):
            data = resp.get("data", [])
            if data:
                st.dataframe(pd.DataFrame(data), use_container_width=True)
            else:
                st.info("No open positions.")
        else:
            st.error(f"Could not fetch positions: {resp}")
    except Exception as e:
        st.error(f"Error fetching positions: {e}")

    st.subheader("Orders")
    try:
        resp = client.get_order_list()
        if ok(resp):
            data = resp.get("data", [])
            if data:
                st.dataframe(pd.DataFrame(data), use_container_width=True)
            else:
                st.info("No orders today.")
        else:
            st.error(f"Could not fetch orders: {resp}")
    except Exception as e:
        st.error(f"Error fetching orders: {e}")


def page_strategy() -> None:
    st.header("Strategy Runner")
    st.caption("SMA Crossover Demo — not investment advice")

    settings = _get_settings()

    symbols_input = st.text_input("Symbols (comma-separated)", value="RELIANCE, INFY")
    col1, col2, col3 = st.columns(3)
    with col1:
        short_period = st.number_input("Short SMA", min_value=2, max_value=50, value=5)
    with col2:
        long_period = st.number_input("Long SMA", min_value=5, max_value=200, value=20)
    with col3:
        interval = st.number_input("Interval (s)", min_value=5, max_value=600, value=settings.strategy_interval)

    running = st.session_state.get("strategy_running", False)

    col_start, col_stop = st.columns(2)
    with col_start:
        if st.button("Start Strategy", disabled=running, type="primary"):
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

    with col_stop:
        if st.button("Stop Strategy", disabled=not running):
            st.session_state.strategy_running = False
            st.rerun()

    # Status
    if running:
        st.success("Strategy is running...")
    else:
        st.info("Strategy is stopped.")

    # Live log
    st.subheader("Signal Log")
    log_lines = st.session_state.get("strategy_log", [])
    if log_lines:
        st.code("\n".join(log_lines[-50:]), language="text")
    else:
        st.info("No signals yet.")

    if running:
        time.sleep(2)
        st.rerun()


def page_backtest() -> None:
    st.header("Backtest")

    tab_api, tab_csv = st.tabs(["Fetch from API", "Upload CSV"])

    with tab_api:
        symbols_input = st.text_input("Symbols (comma-separated)", value="RELIANCE", key="bt_symbols")
        col1, col2, col3 = st.columns(3)
        with col1:
            from_date = st.date_input("From", value=date.today() - timedelta(days=90), key="bt_from")
        with col2:
            to_date = st.date_input("To", value=date.today(), key="bt_to")
        with col3:
            interval = st.selectbox("Interval", ["day", "minute"], key="bt_interval")

        if st.button("Run Backtest (API)", key="bt_api_run"):
            symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
            try:
                client = _get_client()
            except (SystemExit, Exception) as e:
                st.error(f"Client error: {e}")
                return

            all_bars: dict[str, list] = {}
            with st.spinner("Fetching historical data..."):
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

    st.subheader("Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Trades", result.total_trades)
    col2.metric("Buys", result.buy_count)
    col3.metric("Sells", result.sell_count)
    col4.metric("Net P&L", f"{result.pnl + result.unrealized_pnl:,.2f}")

    col5, col6 = st.columns(2)
    col5.metric("Realized P&L", f"{result.pnl:,.2f}")
    col6.metric("Unrealized P&L", f"{result.unrealized_pnl:,.2f}")

    if result.positions:
        st.subheader("Open Positions")
        st.json(result.positions)

    if result.fills:
        st.subheader("Fills")
        fills_data = [asdict(f) for f in result.fills]
        st.dataframe(pd.DataFrame(fills_data), use_container_width=True)


def page_journal() -> None:
    st.header("Trade Journal")
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
            selected_status = st.selectbox("Filter by Status", statuses)
        else:
            selected_status = "All"
    with col2:
        if "side" in df.columns:
            sides = ["All"] + sorted(df["side"].dropna().unique().tolist())
            selected_side = st.selectbox("Filter by Side", sides)
        else:
            selected_side = "All"

    filtered = df.copy()
    if selected_status != "All":
        filtered = filtered[filtered["status"] == selected_status]
    if selected_side != "All":
        filtered = filtered[filtered["side"] == selected_side]

    st.dataframe(filtered, use_container_width=True)
    st.caption(f"{len(filtered)} of {len(df)} entries shown")


def page_kill_switch() -> None:
    st.header("Kill Switch")
    st.warning(
        "Activating the kill switch will **disable all trading** on your Dhan account "
        "for the rest of the trading day. This action cannot be undone."
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


def page_intraday() -> None:
    st.header("Intraday Scanner & Trade")
    st.caption("Score stocks using VWAP, SuperTrend, Momentum, Volume & ORB — place orders from results")

    settings = _get_settings()

    # --- Configuration sidebar-like controls ---
    col_sym, col_interval = st.columns([3, 1])
    with col_sym:
        symbols_input = st.text_input(
            "Symbols (comma-separated)",
            value="RELIANCE, TCS, INFY, HDFCBANK, SBIN",
            key="intra_symbols",
            help="Enter NSE symbols to scan",
        )
    with col_interval:
        interval_minutes = st.selectbox(
            "Candle interval",
            [1, 5, 15],
            index=2,
            key="intra_interval",
            help="Intraday candle size in minutes",
        )

    # Weight sliders
    with st.expander("Scoring Weights", expanded=False):
        wcol1, wcol2, wcol3, wcol4, wcol5 = st.columns(5)
        with wcol1:
            w_vwap = st.slider("VWAP", 0.0, 5.0, 1.0, 0.5, key="w_vwap")
        with wcol2:
            w_supertrend = st.slider("SuperTrend", 0.0, 5.0, 1.0, 0.5, key="w_st")
        with wcol3:
            w_momentum = st.slider("Momentum", 0.0, 5.0, 1.5, 0.5, key="w_mom")
        with wcol4:
            w_volume = st.slider("Volume", 0.0, 5.0, 0.5, 0.5, key="w_vol")
        with wcol5:
            w_orb = st.slider("ORB", 0.0, 5.0, 1.0, 0.5, key="w_orb")

    weights = {
        "vwap": w_vwap,
        "supertrend": w_supertrend,
        "momentum": w_momentum,
        "volume": w_volume,
        "orb": w_orb,
    }
    params = {"interval_minutes": interval_minutes, "atr_multiplier": 1.0}

    if st.button("Scan Now", type="primary", key="intra_scan"):
        symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
        if not symbols:
            st.warning("Enter at least one symbol.")
            return

        try:
            client = _get_client()
        except (SystemExit, Exception) as e:
            st.error(f"Client error: {e}")
            return

        # Resolve security IDs (for order placement later)
        sym_to_sid: dict[str, str] = {}
        for sym in symbols:
            sid = resolve_security_id(client, sym)
            if sid is not None:
                sym_to_sid[sym] = sid

        # Fetch intraday data via yfinance
        data: dict[str, pd.DataFrame] = {}
        progress = st.progress(0, text="Fetching intraday data...")
        total = len(symbols)
        yf_interval = f"{interval_minutes}m"
        for idx, sym in enumerate(symbols):
            try:
                ticker_yf = f"{sym}.NS"
                df_bars = yf.Ticker(ticker_yf).history(
                    period="5d", interval=yf_interval, auto_adjust=True,
                )
                if df_bars is not None and len(df_bars) >= 5:
                    df_bars = df_bars.reset_index()
                    # Ensure standard OHLCV column names
                    for col in ("Open", "High", "Low", "Close", "Volume"):
                        if col not in df_bars.columns:
                            lc = col.lower()
                            if lc in df_bars.columns:
                                df_bars.rename(columns={lc: col}, inplace=True)
                    data[sym] = df_bars
                else:
                    st.warning(f"No intraday data for {sym}")
            except Exception as e:
                st.warning(f"Error fetching {sym}: {e}")
            progress.progress((idx + 1) / total, text=f"Fetched {sym}")

        progress.empty()

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
        st.session_state["intra_sym_to_sid"] = sym_to_sid

    # --- Display results ---
    results = st.session_state.get("intra_results")
    if not results:
        st.info("Click **Scan Now** to fetch data and score symbols.")
        return

    sym_to_sid = st.session_state.get("intra_sym_to_sid", {})

    # Summary table
    st.subheader("Scored Results")
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
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # --- Detailed view + order placement ---
    st.divider()
    st.subheader("Trade Setup & Order Placement")

    # Symbol selector from scored results
    ticker_options = [r["ticker"] for r in results]
    selected_ticker = st.selectbox(
        "Select symbol to trade",
        ticker_options,
        key="intra_trade_ticker",
    )

    # Find the result row for the selected ticker
    row = next((r for r in results if r["ticker"] == selected_ticker), None)
    if row is None:
        return

    # Trade setup display
    col_setup1, col_setup2, col_setup3, col_setup4 = st.columns(4)
    col_setup1.metric("Score", f"{row['score']:.1f}")
    col_setup2.metric("Price", f"{row['price']:,.2f}")
    col_setup3.metric("RSI(7)", f"{row.get('rsi7', 0):.1f}")
    col_setup4.metric("Vol Ratio", f"{row.get('vol_ratio', 0):.2f}")

    col_t1, col_t2, col_t3, col_t4 = st.columns(4)
    col_t1.metric("Entry", f"{row.get('entry', 0):,.2f}")
    col_t2.metric("Stop Loss", f"{row.get('stop_loss', 0):,.2f}")
    col_t3.metric("Target 1", f"{row.get('target1', 0):,.2f} (RR {row.get('rr1', 0)})")
    col_t4.metric("Target 2", f"{row.get('target2', 0):,.2f} (RR {row.get('rr2', 0)})")

    # Indicator summary
    col_ind1, col_ind2, col_ind3 = st.columns(3)
    with col_ind1:
        vwap_val = row.get("vwap", 0)
        vwap_pct = row.get("vwap_pct", 0)
        st.metric("VWAP", f"{vwap_val:,.2f}", delta=f"{vwap_pct:+.2f}%")
    with col_ind2:
        st_dir = row.get("st_direction", 0)
        st.metric("SuperTrend", "BULLISH" if st_dir == 1 else "BEARISH")
    with col_ind3:
        above = row.get("above_orb", 0)
        st.metric("ORB Status", "ABOVE" if above == 1 else "INSIDE/BELOW")

    # Order form
    st.markdown("---")
    with st.form("intra_order_form"):
        st.markdown(f"**Place order for {selected_ticker}**")
        ocol1, ocol2 = st.columns(2)
        with ocol1:
            side = st.selectbox("Side", ["BUY", "SELL"], key="intra_side")
            qty = st.number_input(
                "Quantity",
                min_value=1,
                max_value=settings.max_qty,
                value=1,
                key="intra_qty",
            )
        with ocol2:
            order_type = st.selectbox("Order Type", ["MARKET", "LIMIT"], key="intra_otype")
            limit_price = st.number_input(
                "Limit Price",
                min_value=0.0,
                value=round(row.get("entry", row["price"]), 2),
                step=0.05,
                key="intra_price",
                help="Used only for LIMIT orders",
            )

        submitted = st.form_submit_button("Place Order", type="primary")

    if submitted:
        try:
            client = _get_client()
            sid = sym_to_sid.get(selected_ticker)
            if sid is None:
                sid = resolve_security_id(client, selected_ticker)
            if sid is None:
                st.error(f"Could not resolve symbol: {selected_ticker}")
                return

            result = place(
                client,
                sid,
                side=side,
                qty=qty,
                order_type=order_type,
                product="INTRA",
                price=limit_price if order_type == "LIMIT" else 0.0,
            )
            if result is None:
                st.error("Order blocked by risk guards. Check logs for details.")
            elif result.get("status") == "dry_run":
                st.info(f"[DRY RUN] {result.get('plan', '')}")
            elif ok(result):
                st.success(f"Order placed successfully: {result}")
            else:
                st.error(f"Order failed: {result}")
        except SystemExit as e:
            st.warning(f"Client not configured: {e}")
        except Exception as e:
            st.error(f"Error placing order: {e}")


def page_swing() -> None:
    st.header("Swing Scanner & Trade")
    st.caption("Score stocks using Trend, Momentum, Volume, Breakout & Volatility — place orders from results")

    settings = _get_settings()

    symbols_input = st.text_input(
        "Symbols (comma-separated)",
        value="RELIANCE, TCS, INFY, HDFCBANK, SBIN",
        key="swing_symbols",
    )

    # Weight sliders
    with st.expander("Scoring Weights", expanded=False):
        wcol1, wcol2, wcol3, wcol4, wcol5 = st.columns(5)
        with wcol1:
            w_trend = st.slider("Trend", 0.0, 5.0, 1.0, 0.5, key="sw_trend")
        with wcol2:
            w_momentum = st.slider("Momentum", 0.0, 5.0, 1.0, 0.5, key="sw_mom")
        with wcol3:
            w_volume = st.slider("Volume", 0.0, 5.0, 0.8, 0.5, key="sw_vol")
        with wcol4:
            w_breakout = st.slider("Breakout", 0.0, 5.0, 0.8, 0.5, key="sw_brk")
        with wcol5:
            w_volatility = st.slider("Volatility", 0.0, 5.0, 0.5, 0.5, key="sw_vlt")

    with st.expander("Advanced Settings", expanded=False):
        acol1, acol2 = st.columns(2)
        with acol1:
            atr_mult = st.number_input(
                "Stop-loss ATR multiplier",
                min_value=0.5, max_value=3.0, value=1.5, step=0.1,
                key="sw_atr_mult",
            )
        with acol2:
            lookback_days = st.number_input(
                "History (trading days)",
                min_value=250, max_value=500, value=365, step=10,
                key="sw_lookback",
                help="Number of calendar days of daily data to fetch",
            )

    weights = {
        "trend": w_trend,
        "momentum": w_momentum,
        "volume": w_volume,
        "breakout": w_breakout,
        "volatility": w_volatility,
    }
    params = {"atr_multiplier": atr_mult}

    if st.button("Scan Now", type="primary", key="swing_scan"):
        symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
        if not symbols:
            st.warning("Enter at least one symbol.")
            return

        try:
            client = _get_client()
        except (SystemExit, Exception) as e:
            st.error(f"Client error: {e}")
            return

        # Resolve security IDs (for order placement later)
        sym_to_sid: dict[str, str] = {}
        for sym in symbols:
            sid = resolve_security_id(client, sym)
            if sid is not None:
                sym_to_sid[sym] = sid

        # Fetch daily data via yfinance
        data: dict[str, pd.DataFrame] = {}
        progress = st.progress(0, text="Fetching daily data...")
        total = len(symbols)
        for idx, sym in enumerate(symbols):
            try:
                ticker_yf = f"{sym}.NS"
                df_bars = yf.Ticker(ticker_yf).history(period="1y", auto_adjust=True)
                if df_bars is not None and len(df_bars) >= 220:
                    df_bars = df_bars.reset_index()
                    for col in ("Open", "High", "Low", "Close", "Volume"):
                        if col not in df_bars.columns:
                            lc = col.lower()
                            if lc in df_bars.columns:
                                df_bars.rename(columns={lc: col}, inplace=True)
                    data[sym] = df_bars
                else:
                    bars_n = len(df_bars) if df_bars is not None else 0
                    st.warning(f"{sym}: only {bars_n} daily bars (need 220+). Try a longer history.")
            except Exception as e:
                st.warning(f"Error fetching {sym}: {e}")
            progress.progress((idx + 1) / total, text=f"Fetched {sym}")

        progress.empty()

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
        st.session_state["swing_sym_to_sid"] = sym_to_sid

    # --- Display results ---
    results = st.session_state.get("swing_results")
    if not results:
        st.info("Click **Scan Now** to fetch daily data and score symbols.")
        return

    sym_to_sid = st.session_state.get("swing_sym_to_sid", {})

    # Summary table
    st.subheader("Scored Results")
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
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # --- Detailed view + order placement ---
    st.divider()
    st.subheader("Trade Setup & Order Placement")

    ticker_options = [r["ticker"] for r in results]
    selected_ticker = st.selectbox(
        "Select symbol to trade",
        ticker_options,
        key="swing_trade_ticker",
    )

    row = next((r for r in results if r["ticker"] == selected_ticker), None)
    if row is None:
        return

    # Trade setup display
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    col_s1.metric("Score", f"{row['score']:.1f}")
    col_s2.metric("Price", f"{row['price']:,.2f}")
    col_s3.metric("RSI(14)", f"{row.get('rsi', 0):.1f}")
    col_s4.metric("Vol Ratio", f"{row.get('vol_ratio', 0):.2f}")

    col_t1, col_t2, col_t3, col_t4 = st.columns(4)
    col_t1.metric("Entry", f"{row.get('entry', 0):,.2f}")
    col_t2.metric("Stop Loss", f"{row.get('stop_loss', 0):,.2f}")
    col_t3.metric("Target 1", f"{row.get('target1', 0):,.2f} (RR {row.get('rr1', 0)})")
    col_t4.metric("Target 2", f"{row.get('target2', 0):,.2f} (RR {row.get('rr2', 0)})")

    # Indicator summary
    col_i1, col_i2, col_i3, col_i4 = st.columns(4)
    with col_i1:
        st.metric("52wk High %", f"{row.get('pct_from_52wk', 0):.1f}%")
    with col_i2:
        st.metric("MACD Hist", f"{row.get('macd_hist', 0):.4f}")
    with col_i3:
        st.metric("Bollinger %B", f"{row.get('pct_b', 0):.2f}")
    with col_i4:
        st.metric("ATR %", f"{row.get('atr_pct', 0):.2f}%")

    # Order form
    st.markdown("---")
    with st.form("swing_order_form"):
        st.markdown(f"**Place order for {selected_ticker}**")
        ocol1, ocol2 = st.columns(2)
        with ocol1:
            side = st.selectbox("Side", ["BUY", "SELL"], key="sw_side")
            qty = st.number_input(
                "Quantity", min_value=1, max_value=settings.max_qty,
                value=1, key="sw_qty",
            )
        with ocol2:
            order_type = st.selectbox("Order Type", ["MARKET", "LIMIT"], key="sw_otype")
            product = st.selectbox("Product", ["CNC", "INTRA"], key="sw_product",
                                   help="CNC for delivery (swing), INTRA for same-day")
            limit_price = st.number_input(
                "Limit Price", min_value=0.0,
                value=round(row.get("entry", row["price"]), 2),
                step=0.05, key="sw_price",
            )

        submitted = st.form_submit_button("Place Order", type="primary")

    if submitted:
        try:
            client = _get_client()
            sid = sym_to_sid.get(selected_ticker)
            if sid is None:
                sid = resolve_security_id(client, selected_ticker)
            if sid is None:
                st.error(f"Could not resolve symbol: {selected_ticker}")
                return

            result = place(
                client, sid,
                side=side, qty=qty,
                order_type=order_type, product=product,
                price=limit_price if order_type == "LIMIT" else 0.0,
            )
            if result is None:
                st.error("Order blocked by risk guards. Check logs for details.")
            elif result.get("status") == "dry_run":
                st.info(f"[DRY RUN] {result.get('plan', '')}")
            elif ok(result):
                st.success(f"Order placed successfully: {result}")
            else:
                st.error(f"Order failed: {result}")
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

st.set_page_config(page_title="SwiftTrade", layout="wide")

if not _is_logged_in():
    page_login()
else:
    # Sidebar
    with st.sidebar:
        st.title("SwiftTrade")
        creds = st.session_state.get("credentials", {})
        st.caption(f"Client: {creds.get('client_id', '')}")
        if st.button("Logout"):
            _logout()
            st.rerun()

        st.divider()

        page = st.radio("Navigation", list(PAGES.keys()))

        st.divider()

        # Mode badge
        settings = _get_settings()
        if settings.dhan_live:
            st.error("LIVE TRADING")
        else:
            st.success("DRY RUN")

        st.caption(f"Max Qty: {settings.max_qty}")
        st.caption(f"Max Order Value: {settings.max_order_value:,.0f}")
        st.caption(f"Max Daily Loss: {settings.max_daily_loss:,.0f}")

        # Strategy status
        if st.session_state.get("strategy_running", False):
            st.info("Strategy: Running")

    # Render selected page
    PAGES[page]()

    # Logs at bottom of every page
    _show_logs()
