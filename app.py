"""SwiftTrade — Streamlit web UI for dhan_algo."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import asdict
from datetime import date, timedelta

import altair as alt
import pandas as pd
import streamlit as st

from dhan_algo.config import Settings
from dhan_algo.client import get_client, ok
from dhan_algo.market_data import ltp
from dhan_algo.orders import calculate_position_size, place, place_bracket, place_with_sl_target
from dhan_algo.risk import kill_switch
from dhan_algo.security_master import resolve_security_id
from dhan_algo.strategy import Order, PollingTicker, SmaDemoMulti
from dhan_algo.backtest import TradingCosts, fetch_historical, load_csv, run_backtest
import yfinance as yf

from exit_rules import ExitConfig
from intraday_scorer import score_single_intraday, score_universe_intraday
from intraday_backtest import simulate_intraday_universe
from swing_backtest import simulate_swing_universe
from swing_scorer import MIN_BARS, score_single, score_universe
from data.universe import ticker_display_name

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
# Symbol autosuggest — searchable NSE-equity symbol list from the security master
# ---------------------------------------------------------------------------

_SECURITY_CSV = "security_id_list.csv"


@st.cache_data(show_spinner=False)
def _symbol_master() -> tuple[list[str], dict[str, str]]:
    """Return (sorted NSE-equity symbols, {symbol: company name}).

    Reads the security master CSV, keeping only NSE cash-equity rows
    (series EQ/BE). Falls back to the NIFTY 100 list if the CSV is missing
    or unreadable.
    """
    try:
        df = pd.read_csv(
            _SECURITY_CSV,
            usecols=[
                "SEM_EXM_EXCH_ID",
                "SEM_TRADING_SYMBOL",
                "SEM_SERIES",
                "SM_SYMBOL_NAME",
            ],
            dtype=str,
        )
        df = df[
            (df["SEM_EXM_EXCH_ID"] == "NSE")
            & (df["SEM_SERIES"].isin(["EQ", "BE"]))
        ]
        symbols = sorted(df["SEM_TRADING_SYMBOL"].dropna().unique().tolist())
        names = dict(
            zip(df["SEM_TRADING_SYMBOL"], df["SM_SYMBOL_NAME"].fillna(""))
        )
    except Exception:
        symbols = list(NIFTY_100)
        names = {s: ticker_display_name(s) for s in symbols}
    if not symbols:
        symbols = list(NIFTY_100)
        names = {s: ticker_display_name(s) for s in symbols}
    return symbols, names


def _symbol_label(symbol: str) -> str:
    """Format ``SYMBOL — Company Name`` for a dropdown entry."""
    _, names = _symbol_master()
    name = names.get(symbol) or ticker_display_name(symbol)
    if name and name.upper() != symbol.upper():
        return f"{symbol} — {name}"
    return symbol


def symbol_selectbox(
    label: str,
    key: str,
    *,
    default: str = "RELIANCE",
    help: str | None = None,
    container=st,
) -> str:
    """Searchable single-symbol picker that also accepts typed-in symbols."""
    symbols, _ = _symbol_master()
    options = list(symbols)
    if default and default not in options:
        options = [default, *options]
    index = options.index(default) if default in options else None
    choice = container.selectbox(
        label,
        options,
        index=index,
        key=key,
        help=help,
        format_func=_symbol_label,
        accept_new_options=True,
        placeholder="Type to search symbol or company…",
    )
    return (choice or "").strip().upper()


def symbol_multiselect(
    label: str,
    key: str,
    *,
    default: list[str] | None = None,
    help: str | None = None,
    container=st,
) -> list[str]:
    """Searchable multi-symbol picker that also accepts typed-in symbols."""
    symbols, _ = _symbol_master()
    default = default or []
    options = list(symbols)
    for sym in default:
        if sym not in options:
            options.insert(0, sym)
    chosen = container.multiselect(
        label,
        options,
        default=default,
        key=key,
        help=help,
        format_func=_symbol_label,
        accept_new_options=True,
        placeholder="Type to search symbols…",
    )
    return [s.strip().upper() for s in chosen if s and s.strip()]


# ---------------------------------------------------------------------------
# P&L estimator — gross profit, itemised charges and net profit
# ---------------------------------------------------------------------------


def render_trade_pnl(
    entry: float,
    exit_price: float,
    qty: int,
    *,
    stop: float | None = None,
    product: str = "INTRA",
    container=st,
) -> None:
    """Show gross/net P&L at target and, when *stop* is given, at stop-loss.

    Assumes a long round-trip: BUY at *entry*, SELL at the exit. Charges are
    computed for both legs with the realistic NSE cost model.
    """
    if entry <= 0 or exit_price <= 0 or qty <= 0:
        return

    costs = TradingCosts()
    buy = costs.breakdown("BUY", entry, qty, product)

    def _scenario(sell_price: float) -> tuple[float, float, float, dict[str, float]]:
        sell = costs.breakdown("SELL", sell_price, qty, product)
        gross = (sell_price - entry) * qty
        total_charges = buy["total"] + sell["total"]
        return gross, total_charges, gross - total_charges, sell

    gross, total_charges, net, sell = _scenario(exit_price)
    charges = {
        "Brokerage": buy["brokerage"] + sell["brokerage"],
        "STT": buy["stt"] + sell["stt"],
        "Exchange txn": buy["exchange"] + sell["exchange"],
        "SEBI fee": buy["sebi"] + sell["sebi"],
        "Stamp duty": buy["stamp_duty"] + sell["stamp_duty"],
        "GST (18%)": buy["gst"] + sell["gst"],
    }

    container.markdown(f"**If target {exit_price:,.2f} is hit — profit**")
    m1, m2, m3 = container.columns(3)
    m1.metric("Gross P&L", f"{gross:,.2f}")
    m2.metric("Total charges", f"-{total_charges:,.2f}")
    m3.metric("Net P&L", f"{net:,.2f}", delta=f"{net - gross:,.2f}")

    if stop is not None and stop > 0:
        gross_s, charges_s, net_s, _ = _scenario(stop)
        container.markdown(f"**If stop-loss {stop:,.2f} is hit — loss**")
        s1, s2, s3 = container.columns(3)
        s1.metric("Gross P&L", f"{gross_s:,.2f}")
        s2.metric("Total charges", f"-{charges_s:,.2f}")
        s3.metric("Net P&L", f"{net_s:,.2f}", delta=f"{net_s - gross_s:,.2f}")
        if net_s < 0:
            container.caption(
                f"Net reward-to-risk ≈ {abs(net / net_s):.2f} : 1"
            )

    rows = "".join(
        f"<tr><td>{name}</td>"
        f"<td style='text-align:right'>{val:,.2f}</td></tr>"
        for name, val in charges.items()
    )
    container.markdown(
        f"""
<table style="width:100%; font-size:0.8rem; line-height:1.6; border-collapse:collapse;">
<tr style="border-bottom:1px solid #444;">
  <td><b>Charge (round trip at target)</b></td>
  <td style="text-align:right"><b>INR</b></td>
</tr>
{rows}
<tr style="border-top:1px solid #444;">
  <td><b>Total charges</b></td>
  <td style="text-align:right"><b>{total_charges:,.2f}</b></td>
</tr>
</table>
""",
        unsafe_allow_html=True,
    )
    container.caption(
        f"Buy {qty} @ {entry:,.2f} ({product}). "
        f"Estimated per NSE charges; actuals may differ."
    )


# ---------------------------------------------------------------------------
# Price charts — candlesticks with entry / stop / target levels
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False, ttl=300)
def _fetch_ohlc(symbol: str, interval: str, period: str) -> pd.DataFrame:
    """Fetch OHLC candles for *symbol* from Yahoo Finance (cached 5 min)."""
    try:
        df = yf.download(
            f"{symbol.upper()}.NS", period=period, interval=interval,
            progress=False, auto_adjust=False,
        )
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    dt_col = "Datetime" if "Datetime" in df.columns else "Date"
    df = df.rename(columns={dt_col: "Date"})
    keep = [c for c in ("Date", "Open", "High", "Low", "Close") if c in df.columns]
    return df[keep].dropna()


def _candlestick_chart(
    df: pd.DataFrame,
    *,
    entry: float | None = None,
    stop: float | None = None,
    target: float | None = None,
    height: int = 320,
) -> alt.LayerChart:
    """Build a candlestick chart with optional horizontal level lines."""
    df = df.copy()
    df["up"] = df["Close"] >= df["Open"]
    color = alt.condition(
        "datum.up", alt.value("#26a69a"), alt.value("#ef5350")
    )
    base = alt.Chart(df).encode(x=alt.X("Date:T", title=None))
    wick = base.mark_rule().encode(
        y=alt.Y("Low:Q", title="Price", scale=alt.Scale(zero=False)),
        y2="High:Q",
        color=color,
    )
    body = base.mark_bar().encode(y="Open:Q", y2="Close:Q", color=color)
    layers: list[alt.Chart] = [wick, body]

    for value, label, col in (
        (entry, "Entry", "#2962ff"),
        (stop, "Stop", "#ef5350"),
        (target, "Target", "#26a69a"),
    ):
        if value and value > 0:
            level = pd.DataFrame({"y": [value], "label": [f"{label} {value:,.2f}"]})
            layers.append(
                alt.Chart(level).mark_rule(strokeDash=[6, 4], color=col).encode(y="y:Q")
            )
            layers.append(
                alt.Chart(level).mark_text(
                    align="left", baseline="bottom", dx=3, color=col
                ).encode(y="y:Q", text="label:N", x=alt.value(3))
            )

    return alt.layer(*layers).properties(height=height)


def render_price_chart(
    symbol: str,
    *,
    entry: float | None = None,
    stop: float | None = None,
    target: float | None = None,
    interval: str = "1d",
    period: str = "6mo",
    container=st,
) -> None:
    """Fetch candles for *symbol* and draw them with entry/stop/target lines."""
    df = _fetch_ohlc(symbol, interval, period)
    if df.empty:
        container.info(f"No chart data available for {symbol}.")
        return
    chart = _candlestick_chart(df, entry=entry, stop=stop, target=target)
    container.altair_chart(chart, use_container_width=True)


def render_backtest_chart(all_bars: dict[str, list], fills: list, container=st) -> None:
    """Plot a tested symbol's price line with BUY/SELL trade markers."""
    symbols = list(all_bars.keys())
    if not symbols:
        return
    if len(symbols) == 1:
        sym = symbols[0]
    else:
        sym = container.selectbox("Chart symbol", symbols, key="bt_chart_sym")
    bars = all_bars.get(sym, [])
    if not bars:
        return

    price_df = pd.DataFrame(bars)
    price_df["timestamp"] = pd.to_datetime(price_df["timestamp"], errors="coerce")
    price_df = price_df.dropna(subset=["timestamp"])
    line = alt.Chart(price_df).mark_line(color="#42a5f5").encode(
        x=alt.X("timestamp:T", title=None),
        y=alt.Y("close:Q", title="Price", scale=alt.Scale(zero=False)),
    )
    layers: list[alt.Chart] = [line]

    fill_df = pd.DataFrame([asdict(f) for f in fills]) if fills else pd.DataFrame()
    if not fill_df.empty:
        fill_df = fill_df[fill_df["security_id"] == sym].copy()
        fill_df["timestamp"] = pd.to_datetime(fill_df["timestamp"], errors="coerce")
        fill_df = fill_df.dropna(subset=["timestamp"])
        if not fill_df.empty:
            markers = alt.Chart(fill_df).mark_point(
                size=90, filled=True, opacity=0.9
            ).encode(
                x="timestamp:T",
                y="price:Q",
                color=alt.Color(
                    "side:N",
                    scale=alt.Scale(domain=["BUY", "SELL"], range=["#26a69a", "#ef5350"]),
                    legend=alt.Legend(title="Trade"),
                ),
                shape=alt.Shape(
                    "side:N",
                    scale=alt.Scale(domain=["BUY", "SELL"], range=["triangle-up", "triangle-down"]),
                    legend=None,
                ),
                tooltip=["timestamp:T", "side:N", "price:Q", "qty:Q"],
            )
            layers.append(markers)

    container.altair_chart(alt.layer(*layers).properties(height=320), use_container_width=True)


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


# Config fields the user may edit at runtime via the Settings page. These
# default from the .env / built-in Settings values and are then overridable
# per browser session without touching disk.
EDITABLE_CONFIG_FIELDS = (
    "max_qty",
    "max_order_value",
    "max_daily_loss",
    "max_open_positions",
    "max_consecutive_losses",
    "max_position_pct",
    "trading_capital",
    "risk_per_trade",
    "strategy_interval",
    "log_level",
    "journal_path",
)


def _config_store() -> dict:
    """Session-persistent config overrides, seeded from .env / defaults."""
    if "config" not in st.session_state:
        base = Settings()
        st.session_state["config"] = {
            "max_qty": int(base.max_qty),
            "max_order_value": float(base.max_order_value),
            "max_daily_loss": float(base.max_daily_loss),
            "max_open_positions": int(base.max_open_positions),
            "max_consecutive_losses": int(base.max_consecutive_losses),
            "max_position_pct": float(base.max_position_pct),
            "trading_capital": float(base.trading_capital),
            "risk_per_trade": float(base.risk_per_trade),
            "strategy_interval": int(base.strategy_interval),
            "log_level": base.log_level,
            "journal_path": base.journal_path,
        }
    return st.session_state["config"]


def _get_settings() -> Settings:
    """Build Settings from session-state credentials (entered at login).

    The dry-run / live-trading mode defaults to the ``DHAN_LIVE`` env var but
    can be overridden at runtime via the sidebar toggle (session state key
    ``live_trading``). Risk limits and runtime config are overridden from the
    session config store edited on the Settings page.
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
    cfg = st.session_state.get("config")
    if cfg:
        for field in EDITABLE_CONFIG_FIELDS:
            if field in cfg:
                setattr(settings, field, cfg[field])
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
        r5, r6 = st.columns(2)
        r5.metric(
            "Max open positions",
            settings.max_open_positions or "—",
        )
        r6.metric(
            "Max loss streak",
            settings.max_consecutive_losses or "—",
        )


def page_market_data() -> None:
    mcol1, mcol2 = st.columns([3, 1], vertical_alignment="bottom")
    symbol = symbol_selectbox(
        "Symbol", key="md_symbol",
        help="Search by symbol or company name", container=mcol1,
    )
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
        symbol = symbol_selectbox("Symbol", key="po_symbol", container=c1)
        side = c2.selectbox("Side", ["BUY", "SELL"])
        qty = c3.number_input("Qty", min_value=1, max_value=settings.max_qty, value=1)
        c4, c5, c6 = st.columns(3)
        order_type = c4.selectbox("Type", ["MARKET", "LIMIT"])
        product = c5.selectbox("Product", ["INTRA", "CNC"])
        price = c6.number_input("Price (LIMIT)", min_value=0.0, value=0.0, step=0.05)
        submitted = st.form_submit_button("Place order", type="primary", width="stretch")

    if submitted:
        if not symbol:
            st.warning("Select a symbol first.")
            return
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

    strat_symbols = symbol_multiselect(
        "Symbols", key="strat_symbols", default=["RELIANCE", "INFY"],
        help="Search and add symbols to run the SMA strategy on",
    )
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

        symbols = strat_symbols
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


def _exit_config_controls(prefix: str, *, show_time_stop: bool = True) -> ExitConfig:
    """Render exit-management widgets and return the resulting ExitConfig.

    Rules are opt-in: with everything at 0 the simulation uses the plain
    ATR stop/target ladder. *prefix* namespaces the widget keys so the same
    controls can appear in more than one tab.
    """
    with st.expander("Exit management (optional)", expanded=False):
        st.caption(
            "Layer break-even, trailing, partial-profit and time exits on top "
            "of the ATR stop/target. R = the trade's initial risk (entry − stop)."
        )
        e1, e2 = st.columns(2)
        with e1:
            breakeven_r = st.number_input(
                "Break-even after (R)", min_value=0.0, max_value=10.0, step=0.25,
                value=0.0, key=f"{prefix}_be_r",
                help="Move the stop to entry once price is this many R in profit. 0 = off.",
            )
        with e2:
            trail_r = st.number_input(
                "Trailing stop (R below high)", min_value=0.0, max_value=10.0, step=0.25,
                value=0.0, key=f"{prefix}_trail_r",
                help="Trail the stop this many R below the highest high since entry. 0 = off.",
            )
        e3, e4 = st.columns(2)
        with e3:
            partial_r = st.number_input(
                "Partial profit at (R)", min_value=0.0, max_value=10.0, step=0.25,
                value=0.0, key=f"{prefix}_partial_r",
                help="Sell part of the position at this many R. 0 = off.",
            )
        with e4:
            partial_pct = st.slider(
                "Partial size (%)", 0, 90, 0, 5, key=f"{prefix}_partial_pct",
                help="Fraction of the position sold at the partial level.",
            )
        max_hold = 0
        if show_time_stop:
            max_hold = st.number_input(
                "Time stop (bars, 0 = none)", min_value=0, max_value=500,
                value=0, step=1, key=f"{prefix}_max_hold",
                help="Force-close a trade after this many bars.",
            )

    return ExitConfig(
        breakeven_r=float(breakeven_r),
        trail_r=float(trail_r),
        partial_r=float(partial_r),
        partial_pct=float(partial_pct) / 100.0,
        max_hold_bars=int(max_hold) or None,
    )


def page_backtest() -> None:
    st.caption("Backtest — SMA replay & intraday signal trade simulation")

    with st.expander("Cost model (slippage + charges)", expanded=False):
        st.session_state.setdefault("bt_apply_costs", True)
        st.toggle(
            "Apply realistic costs",
            key="bt_apply_costs",
            help="Model adverse slippage and NSE brokerage/taxes/fees. Off = frictionless.",
        )
        cc1, cc2 = st.columns(2)
        with cc1:
            st.number_input(
                "Slippage (bps)", min_value=0.0, max_value=100.0,
                value=5.0, step=0.5, key="bt_slippage_bps",
                help="Adverse basis points added to buys / subtracted from sells.",
            )
        with cc2:
            st.selectbox(
                "Product", ["INTRA", "CNC"], key="bt_product",
                help="INTRA (intraday) or CNC (delivery) — affects STT / stamp duty.",
            )

    data_source = st.session_state.get("data_source", "Yahoo Finance")
    tab_fetch, tab_csv, tab_intra, tab_swing = st.tabs(
        [f"Fetch from {data_source}", "Upload CSV",
         "Intraday simulation", "Swing simulation"]
    )

    with tab_fetch:
        bt_symbols = symbol_multiselect(
            "Symbols", key="bt_symbols", default=["RELIANCE"],
            help="Search and add symbols to backtest",
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            from_date = st.date_input("From", value=date.today() - timedelta(days=90), key="bt_from")
        with col2:
            to_date = st.date_input("To", value=date.today(), key="bt_to")
        with col3:
            interval = st.selectbox("Interval", ["day", "minute"], key="bt_interval")

        if st.button(f"Run Backtest ({data_source})", key="bt_api_run"):
            symbols = bt_symbols
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

    with tab_intra:
        st.caption(
            "Simulate the intraday scanner's signals as real trades: enter long "
            "on a high score, exit on ATR stop/target, square off at end of day."
        )
        itb_symbols = symbol_multiselect(
            "Symbols", key="itb_symbols",
            default=["RELIANCE", "TCS", "INFY"],
            help="Search and add symbols to simulate",
        )
        ic1, ic2, ic3 = st.columns(3)
        with ic1:
            itf = st.selectbox("Interval", ["15m", "5m", "1m"], key="itb_interval")
        with ic2:
            ilookback = st.number_input(
                "Lookback (days)", min_value=1, max_value=60, value=5, step=1,
                key="itb_lookback",
                help="Yahoo allows ~7d of 1m and ~60d of 5m/15m data.",
            )
        with ic3:
            ithresh = st.slider(
                "Entry score threshold", 0, 100, 70, 5, key="itb_threshold",
                help="Enter only when the intraday composite score is at or above this.",
            )
        ic4, ic5 = st.columns(2)
        with ic4:
            iatr = st.number_input(
                "Stop-loss ATR multiplier", min_value=0.5, max_value=3.0,
                value=1.0, step=0.1, key="itb_atr_mult",
            )
        with ic5:
            ireentry = st.checkbox(
                "Allow re-entries same day", value=True, key="itb_reentry",
            )

        itb_exit_cfg = _exit_config_controls("itb")

        if st.button("Run intraday simulation", key="itb_run", type="primary"):
            symbols = itb_symbols
            if not symbols:
                st.warning("Enter at least one symbol.")
                return

            interval_minutes = {"15m": 15, "5m": 5, "1m": 1}[itf]
            period_days = min(int(ilookback), 7 if itf == "1m" else 60)
            tickers_yf = [f"{sym}.NS" for sym in symbols]

            with st.spinner(f"Fetching {itf} bars for {len(symbols)} symbols..."):
                try:
                    df_all = yf.download(
                        tickers_yf, period=f"{period_days}d", interval=itf,
                        auto_adjust=True, group_by="ticker", threads=True,
                    )
                except Exception as e:
                    st.error(f"Yahoo Finance error: {e}")
                    return

            if df_all is None or df_all.empty:
                st.error("No intraday data fetched.")
                return

            multi = isinstance(df_all.columns, pd.MultiIndex)
            data: dict[str, pd.DataFrame] = {}
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
                    if len(df_bars) >= 21:
                        data[sym] = df_bars
                    else:
                        st.warning(f"Not enough bars for {sym}")
                except (KeyError, Exception) as e:
                    st.warning(f"Skipped {sym}: {e}")

            if not data:
                st.error("No symbols had enough intraday data.")
                return

            apply_costs = st.session_state.get("bt_apply_costs", True)
            costs = (
                TradingCosts(slippage_pct=st.session_state.get("bt_slippage_bps", 5.0) / 10000.0)
                if apply_costs else None
            )
            params = {"interval_minutes": interval_minutes, "atr_multiplier": iatr}

            with st.spinner("Simulating trades..."):
                result = simulate_intraday_universe(
                    data, params=params, score_threshold=float(ithresh),
                    costs=costs, product="INTRA", allow_reentry=ireentry,
                    exit_config=itb_exit_cfg,
                )
            st.session_state["itb_result"] = result

        result = st.session_state.get("itb_result")
        if result is not None:
            _display_intraday_sim(result)

    with tab_swing:
        st.caption(
            "Simulate the swing scanner's signals as real trades on daily bars: "
            "enter long on a high score, exit on ATR stop/target, hold across "
            "days until an optional max-hold cap or the end of data."
        )
        stb_symbols = symbol_multiselect(
            "Symbols", key="stb_symbols",
            default=["RELIANCE", "TCS", "INFY"],
            help="Search and add symbols to simulate",
        )
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            syears = st.selectbox(
                "History", ["2y", "5y", "10y"], key="stb_years",
                help="Daily lookback window. More history = more trades.",
            )
        with sc2:
            sthresh = st.slider(
                "Entry score threshold", 0, 100, 60, 5, key="stb_threshold",
                help="Enter only when the swing composite score is at or above this.",
            )
        with sc3:
            smaxhold = st.number_input(
                "Max hold (bars, 0 = none)", min_value=0, max_value=250,
                value=0, step=5, key="stb_maxhold",
                help="Force-close a trade after this many daily bars. 0 disables.",
            )
        sc4, sc5 = st.columns(2)
        with sc4:
            satr = st.number_input(
                "Stop-loss ATR multiplier", min_value=0.5, max_value=3.0,
                value=1.5, step=0.1, key="stb_atr_mult",
            )
        with sc5:
            sreentry = st.checkbox(
                "Allow re-entries", value=True, key="stb_reentry",
                help="Take another trade after a prior one closes on the same symbol.",
            )

        stb_exit_cfg = _exit_config_controls("stb", show_time_stop=False)

        if st.button("Run swing simulation", key="stb_run", type="primary"):
            symbols = stb_symbols
            if not symbols:
                st.warning("Enter at least one symbol.")
                return

            tickers_yf = [f"{sym}.NS" for sym in symbols]
            with st.spinner(f"Fetching {syears} daily bars for {len(symbols)} symbols..."):
                try:
                    df_all = yf.download(
                        tickers_yf, period=syears, interval="1d",
                        auto_adjust=True, group_by="ticker", threads=True,
                    )
                except Exception as e:
                    st.error(f"Yahoo Finance error: {e}")
                    return

            if df_all is None or df_all.empty:
                st.error("No daily data fetched.")
                return

            multi = isinstance(df_all.columns, pd.MultiIndex)
            data: dict[str, pd.DataFrame] = {}
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
                    if len(df_bars) >= MIN_BARS + 1:
                        data[sym] = df_bars
                    else:
                        st.warning(
                            f"Not enough daily history for {sym} "
                            f"(need > {MIN_BARS} bars)."
                        )
                except (KeyError, Exception) as e:
                    st.warning(f"Skipped {sym}: {e}")

            if not data:
                st.error("No symbols had enough daily data.")
                return

            apply_costs = st.session_state.get("bt_apply_costs", True)
            costs = (
                TradingCosts(slippage_pct=st.session_state.get("bt_slippage_bps", 5.0) / 10000.0)
                if apply_costs else None
            )
            params = {"atr_multiplier": satr}
            max_hold = int(smaxhold) or None

            with st.spinner("Simulating swing trades..."):
                result = simulate_swing_universe(
                    data, params=params, score_threshold=float(sthresh),
                    costs=costs, product="CNC", max_hold_bars=max_hold,
                    allow_reentry=sreentry, exit_config=stb_exit_cfg,
                )
            st.session_state["stb_result"] = result

        result = st.session_state.get("stb_result")
        if result is not None:
            _display_intraday_sim(result)


def _display_intraday_sim(result) -> None:
    if result.total_trades == 0:
        st.info(
            "No trades were triggered. Lower the entry score threshold or widen "
            "the lookback / interval."
        )
        return

    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Trades", result.total_trades)
        c2.metric("Win rate", f"{result.win_rate:.0f}%")
        c3.metric("Avg R", f"{result.avg_r:.2f}")
        pf = result.profit_factor
        c4.metric("Profit factor", "inf" if pf == float("inf") else f"{pf:.2f}")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Gross P&L", f"{result.gross_pnl:,.0f}")
        c6.metric("Costs", f"{result.total_costs:,.0f}")
        c7.metric(
            "Net P&L", f"{result.net_pnl:,.0f}",
            delta=f"{result.expectancy:,.1f}/trade",
        )
        c8.metric("Max drawdown", f"{result.max_drawdown:,.0f}")

    curve = result.equity_curve
    if curve:
        st.caption("Equity curve (cumulative net P&L)")
        st.line_chart(pd.DataFrame({"Net P&L": curve}), height=200)

    st.caption("Trades")
    trades_df = pd.DataFrame([asdict(t) for t in result.trades])
    st.dataframe(trades_df, width="stretch", hide_index=True)


def _run_and_display_backtest(all_bars: dict[str, list]) -> None:
    strategy = SmaDemoMulti(short_period=5, long_period=20, qty=1)

    apply_costs = st.session_state.get("bt_apply_costs", True)
    if apply_costs:
        costs = TradingCosts(slippage_pct=st.session_state.get("bt_slippage_bps", 5.0) / 10000.0)
        product = st.session_state.get("bt_product", "INTRA")
    else:
        costs = None
        product = None

    with st.spinner("Running backtest..."):
        result = run_backtest(strategy, all_bars, costs=costs, cost_product=product)

    with st.container(border=True):
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Trades", result.total_trades)
        c2.metric("Buys", result.buy_count)
        c3.metric("Sells", result.sell_count)
        c4.metric("Realized", f"{result.pnl:,.0f}")
        c5.metric("Unrealized", f"{result.unrealized_pnl:,.0f}")
        gross = result.pnl + result.unrealized_pnl
        c6.metric(
            "Net P&L", f"{result.net_pnl:,.0f}",
            delta=f"-{result.total_costs:,.0f} cost" if result.total_costs else None,
            delta_color="inverse",
        )

    if apply_costs:
        st.caption(
            f"Gross P&L {gross:,.0f}  −  Costs {result.total_costs:,.2f}  "
            f"=  Net {result.net_pnl:,.0f}"
        )

    st.caption("Price & trades")
    render_backtest_chart(all_bars, result.fills)

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
        chosen = symbol_multiselect(
            "Symbols", key="intra_symbols_ms",
            default=["RELIANCE", "TCS", "INFY", "HDFCBANK", "SBIN"],
        )
        st.session_state["intra_symbols"] = ", ".join(chosen)
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

    # Intraday price chart with the trade levels overlaid.
    _iv = int(st.session_state.get("intra_interval", 15))
    _yf_iv = {1: "1m", 5: "5m", 15: "15m"}.get(_iv, "15m")
    _yf_period = "5d" if _iv == 1 else "1mo"
    with st.expander(f"Price chart — {selected_ticker} ({_yf_iv})", expanded=True):
        render_price_chart(
            selected_ticker, entry=entry, stop=sl, target=t1,
            interval=_yf_iv, period=_yf_period,
        )

    # Position sizing info
    _entry_default = round(entry if entry > 0 else row["price"], 2)
    _sl_default = round(sl, 2) if sl > 0 else round(row["price"] * 0.99, 2)
    _target_default = round(t1, 2) if t1 > 0 else round(row["price"] * 1.01, 2)
    _risk_per_share = abs(_entry_default - _sl_default)

    if settings.risk_per_trade > 0 and _risk_per_share > 0.01:
        auto_qty = calculate_position_size(
            _entry_default, _sl_default,
            settings.risk_per_trade, settings.max_qty, settings.max_order_value,
            capital=settings.trading_capital, max_position_pct=settings.max_position_pct,
        )
        st.caption(
            f"Position sizing: risk {settings.risk_per_trade:.0f} INR / "
            f"risk per share {_risk_per_share:.2f} = **{auto_qty} shares** "
            f"(max qty: {settings.max_qty}, max value: {settings.max_order_value:,.0f})"
        )
    else:
        auto_qty = 1

    # Refresh the order-form fields when the selected symbol changes. Keyed
    # widgets keep their value in session_state, so the value= defaults are
    # ignored on rerun — update session_state before the widgets are built.
    if st.session_state.get("intra_form_ticker") != selected_ticker:
        st.session_state["intra_form_ticker"] = selected_ticker
        st.session_state["intra_qty"] = max(auto_qty, 1)
        st.session_state["intra_entry_price"] = _entry_default
        st.session_state["intra_sl_price"] = _sl_default
        st.session_state["intra_target_price"] = _target_default

    with st.container(border=True):
        st.caption(f"Bracket order for {selected_ticker} (entry + SL + target)")
        fcol1, fcol2, fcol3, fcol4, fcol5 = st.columns(5)
        with fcol1:
            side = st.selectbox("Side", ["BUY", "SELL"], key="intra_side")
        with fcol2:
            qty = st.number_input(
                "Qty",
                min_value=1,
                max_value=settings.max_qty,
                key="intra_qty",
            )
        with fcol3:
            entry_price = st.number_input(
                "Entry",
                min_value=0.01,
                step=0.05,
                key="intra_entry_price",
            )
        with fcol4:
            sl_price = st.number_input(
                "Stop loss",
                min_value=0.01,
                step=0.05,
                key="intra_sl_price",
            )
        with fcol5:
            target_price = st.number_input(
                "Target",
                min_value=0.01,
                step=0.05,
                key="intra_target_price",
            )

        with st.expander("Estimated P&L (target vs stop-loss)", expanded=True):
            render_trade_pnl(
                entry_price, target_price, qty,
                stop=sl_price, product="INTRA",
            )

        submitted = st.button(
            "Place bracket order", type="primary", width="stretch",
            key="intra_place",
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
        chosen = symbol_multiselect(
            "Symbols", key="swing_symbols_ms",
            default=["RELIANCE", "TCS", "INFY", "HDFCBANK", "SBIN"],
        )
        st.session_state["swing_symbols"] = ", ".join(chosen)
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

    # Daily price chart with the trade levels overlaid.
    with st.expander(f"Price chart — {selected_ticker} (daily, 6mo)", expanded=True):
        render_price_chart(
            selected_ticker,
            entry=row.get("entry", 0),
            stop=row.get("stop_loss", 0),
            target=row.get("target1", 0),
            interval="1d", period="6mo",
        )

    _sw_entry_default = round(row.get("entry", row["price"]), 2)
    _sw_sl_default = round(row.get("stop_loss", row["price"] * 0.97), 2)
    _sw_target_default = round(row.get("target1", row["price"] * 1.05), 2)
    _sw_risk_per_share = abs(_sw_entry_default - _sw_sl_default)

    if settings.risk_per_trade > 0 and _sw_risk_per_share > 0.01:
        sw_auto_qty = calculate_position_size(
            _sw_entry_default, _sw_sl_default,
            settings.risk_per_trade, settings.max_qty, settings.max_order_value,
            capital=settings.trading_capital, max_position_pct=settings.max_position_pct,
        )
        st.caption(
            f"Position sizing: risk {settings.risk_per_trade:.0f} INR / "
            f"risk per share {_sw_risk_per_share:.2f} = **{sw_auto_qty} shares** "
            f"(max qty: {settings.max_qty}, max value: {settings.max_order_value:,.0f})"
        )
    else:
        sw_auto_qty = 1

    # Refresh the order-form fields when the selected symbol changes (keyed
    # widgets otherwise retain the previous symbol's values across reruns).
    if st.session_state.get("sw_form_ticker") != selected_ticker:
        st.session_state["sw_form_ticker"] = selected_ticker
        st.session_state["sw_qty"] = max(sw_auto_qty, 1)
        st.session_state["sw_price"] = _sw_entry_default
        st.session_state["sw_sl_price"] = _sw_sl_default
        st.session_state["sw_target_price"] = _sw_target_default

    with st.container(border=True):
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
                key="sw_qty",
            )

        r2c1, r2c2, r2c3 = st.columns(3)
        with r2c1:
            limit_price = st.number_input(
                "Entry", min_value=0.0,
                step=0.05, key="sw_price",
            )
        with r2c2:
            sw_sl = st.number_input(
                "Stop loss", min_value=0.0,
                step=0.05, key="sw_sl_price",
            )
        with r2c3:
            sw_target = st.number_input(
                "Target", min_value=0.0,
                step=0.05, key="sw_target_price",
            )

        with st.expander("Estimated P&L (target vs stop-loss)", expanded=True):
            render_trade_pnl(
                limit_price if limit_price > 0 else _sw_entry_default,
                sw_target, qty, stop=sw_sl, product=product,
            )

        submitted = st.button(
            "Place order", type="primary", width="stretch", key="sw_place",
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


def sidebar_tomorrow() -> None:
    """Tomorrow's Picks controls (rendered in the sidebar)."""
    universe = st.selectbox(
        "Universe",
        ["NIFTY 100", "NIFTY 50"],
        key="tm_universe",
        help="Universe to scan for next-day swing candidates",
    )
    symbol_list = NIFTY_50 if universe == "NIFTY 50" else NIFTY_100
    st.caption(f"{len(symbol_list)} stocks")

    st.slider("Show top", 5, 30, 10, 1, key="tm_top_n")
    st.slider(
        "Minimum score", 0, 100, 60, 5, key="tm_min_score",
        help="Only list candidates scoring at or above this swing score",
    )
    st.number_input(
        "Stop-loss ATR multiplier",
        min_value=0.5, max_value=3.0, value=1.5, step=0.1,
        key="tm_atr_mult",
    )

    st.session_state["tomorrow_do_scan"] = st.button(
        "Find picks", type="primary", key="tomorrow_scan", width="stretch",
        icon=":material/trending_up:",
    )


def page_tomorrow() -> None:
    st.caption("Tomorrow's Picks — top-ranked swing candidates for next-day trading")

    universe = st.session_state.get("tm_universe", "NIFTY 100")
    symbols = NIFTY_50 if universe == "NIFTY 50" else NIFTY_100
    top_n = int(st.session_state.get("tm_top_n", 10))
    min_score = float(st.session_state.get("tm_min_score", 60))
    weights = {
        "trend": 1.0, "momentum": 1.0, "volume": 0.8,
        "breakout": 0.8, "volatility": 0.5,
    }
    params = {"atr_multiplier": st.session_state.get("tm_atr_mult", 1.5)}

    if st.session_state.get("tomorrow_do_scan"):
        data: dict[str, pd.DataFrame] = {}
        tickers_yf = [f"{sym}.NS" for sym in symbols]
        with st.spinner(f"Fetching daily data for {len(symbols)} stocks..."):
            try:
                df_all = yf.download(
                    tickers_yf, period="1y",
                    auto_adjust=True, group_by="ticker", threads=True,
                )
            except Exception as e:
                st.error(f"Error fetching data: {e}")
                return

        if df_all is None or df_all.empty:
            st.error("No data fetched.")
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
            st.error("No data met the minimum bar requirement (220 daily bars).")
            return

        with st.spinner("Scoring universe..."):
            results = score_universe(data, weights, params)

        st.session_state["tomorrow_results"] = results
        st.session_state["tomorrow_scanned"] = len(data)

    results = st.session_state.get("tomorrow_results")
    if not results:
        st.info("Click **Find picks** to scan the universe and rank next-day candidates.")
        return

    picks = [r for r in results if r["score"] >= min_score][:top_n]
    scanned = st.session_state.get("tomorrow_scanned", len(results))

    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        c1.metric("Scanned", scanned)
        c2.metric(f"Picks (score ≥ {min_score:.0f})", len(picks))
        c3.metric("Top score", f"{picks[0]['score']:.1f}" if picks else "—")

    if not picks:
        st.warning(
            f"No candidates scored ≥ {min_score:.0f}. Lower the minimum score in the sidebar."
        )
        return

    # Ranked list with entry/stop/target levels for each pick.
    rows = []
    for i, r in enumerate(picks, start=1):
        rows.append({
            "Rank": i,
            "Symbol": r["ticker"],
            "Name": r["name"],
            "Price": r["price"],
            "Change %": r["change_pct"],
            "Score": r["score"],
            "Entry": r.get("entry", 0),
            "Stop": r.get("stop_loss", 0),
            "Target 1": r.get("target1", 0),
            "Target 2": r.get("target2", 0),
            "R:R": r.get("rr1", 0),
            "RSI": r.get("rsi", 0),
        })
    df_picks = pd.DataFrame(rows)

    def _highlight_score(val):
        if val >= 75:
            return "background-color: #1b5e20; color: white"
        elif val >= 50:
            return "background-color: #33691e; color: white"
        elif val >= 25:
            return "background-color: #e65100; color: white"
        else:
            return "background-color: #b71c1c; color: white"

    styled = df_picks.style.map(_highlight_score, subset=["Score"])
    st.dataframe(styled, width="stretch", hide_index=True)

    st.caption(
        "Long-biased swing setups (entry = last close, stop = ATR-based, "
        "targets = 2:1 / 3:1). Data may be delayed. Not financial advice — "
        "confirm on your own analysis before trading. Use the **Swing Scanner** "
        "page to size and place an order for any of these symbols."
    )

    # Price chart with trade levels for a chosen pick.
    pick_syms = [p["ticker"] for p in picks]
    chart_sym = st.selectbox("Chart symbol", pick_syms, key="tm_chart_sym")
    chart_row = next((p for p in picks if p["ticker"] == chart_sym), None)
    if chart_row is not None:
        with st.expander(f"Price chart — {chart_sym} (daily, 6mo)", expanded=True):
            render_price_chart(
                chart_sym,
                entry=chart_row.get("entry", 0),
                stop=chart_row.get("stop_loss", 0),
                target=chart_row.get("target1", 0),
                interval="1d", period="6mo",
            )


def page_settings() -> None:
    st.caption("Settings — risk limits & runtime configuration")

    cfg = _config_store()
    levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
    cur_level = str(cfg.get("log_level", "INFO")).upper()

    with st.form("settings_form", border=True):
        st.markdown("**Risk limits** — enforced before every order")
        c1, c2, c3 = st.columns(3)
        with c1:
            max_qty = st.number_input(
                "Max quantity / order", min_value=1, step=1,
                value=int(cfg["max_qty"]),
                help="Orders above this quantity are blocked.",
            )
        with c2:
            max_order_value = st.number_input(
                "Max order value (INR)", min_value=0.0, step=1000.0,
                value=float(cfg["max_order_value"]),
                help="Orders whose notional exceeds this are blocked.",
            )
        with c3:
            max_daily_loss = st.number_input(
                "Max daily loss (INR)", min_value=0.0, step=1000.0,
                value=float(cfg["max_daily_loss"]),
                help="Trading is halted once realised loss reaches this.",
            )

        st.markdown("**Trading halts** — 0 disables a guard")
        h1, h2 = st.columns(2)
        with h1:
            max_open_positions = st.number_input(
                "Max open positions", min_value=0, step=1,
                value=int(cfg.get("max_open_positions", 0)),
                help="New entries are blocked once this many positions are open.",
            )
        with h2:
            max_consecutive_losses = st.number_input(
                "Max consecutive losses", min_value=0, step=1,
                value=int(cfg.get("max_consecutive_losses", 0)),
                help="Entries are paused after this many losing trades in a row.",
            )

        st.markdown("**Position sizing & strategy**")
        c4, c5 = st.columns(2)
        with c4:
            risk_per_trade = st.number_input(
                "Risk per trade (INR)", min_value=0.0, step=100.0,
                value=float(cfg["risk_per_trade"]),
                help="Auto position-sizing budget. 0 = enter quantity manually.",
            )
        with c5:
            strategy_interval = st.number_input(
                "Strategy poll interval (s)", min_value=5, max_value=3600, step=5,
                value=int(cfg["strategy_interval"]),
            )

        c8, c9 = st.columns(2)
        with c8:
            trading_capital = st.number_input(
                "Trading capital (INR)", min_value=0.0, step=10000.0,
                value=float(cfg.get("trading_capital", 0.0)),
                help="Capital base for percentage-of-capital position sizing. 0 = off.",
            )
        with c9:
            max_position_pct = st.number_input(
                "Max position size (% of capital)", min_value=0.0, max_value=100.0,
                step=1.0, value=float(cfg.get("max_position_pct", 0.0)),
                help="Caps each position's notional at this % of capital. 0 = off.",
            )

        st.markdown("**Runtime**")
        c6, c7 = st.columns(2)
        with c6:
            log_level = st.selectbox(
                "Log level", levels,
                index=levels.index(cur_level) if cur_level in levels else 1,
            )
        with c7:
            journal_path = st.text_input(
                "Trade journal file", value=cfg["journal_path"],
                help="CSV path where order attempts are logged.",
            )

        col_save, col_reset = st.columns(2)
        save = col_save.form_submit_button(
            "Save settings", type="primary", width="stretch",
        )
        reset = col_reset.form_submit_button("Reset to defaults", width="stretch")

    if save:
        cfg.update({
            "max_qty": int(max_qty),
            "max_order_value": float(max_order_value),
            "max_daily_loss": float(max_daily_loss),
            "max_open_positions": int(max_open_positions),
            "max_consecutive_losses": int(max_consecutive_losses),
            "max_position_pct": float(max_position_pct),
            "trading_capital": float(trading_capital),
            "risk_per_trade": float(risk_per_trade),
            "strategy_interval": int(strategy_interval),
            "log_level": log_level,
            "journal_path": journal_path.strip() or "trades.csv",
        })
        logging.getLogger().setLevel(getattr(logging, log_level, logging.INFO))
        st.success("Settings saved for this session.")
        st.rerun()

    if reset:
        st.session_state.pop("config", None)
        logging.getLogger().setLevel(logging.DEBUG)
        st.success("Settings reset to .env / built-in defaults.")
        st.rerun()

    st.caption(
        "Changes apply immediately to this browser session. To persist them, "
        "set the matching variables in your `.env` file."
    )


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

PAGES = {
    "Dashboard": page_dashboard,
    "Tomorrow's Picks": page_tomorrow,
    "Swing Scanner": page_swing,
    "Intraday Scanner": page_intraday,
    "Market Data": page_market_data,
    "Place Order": page_place_order,
    "Positions & Orders": page_positions_orders,
    "Strategy": page_strategy,
    "Backtest": page_backtest,
    "Trade Journal": page_journal,
    "Settings": page_settings,
    "Kill Switch": page_kill_switch,
}

# Per-page controls rendered in the sidebar to keep the main area focused
# on results and actions.
SIDEBAR_CONTROLS = {
    "Tomorrow's Picks": sidebar_tomorrow,
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
