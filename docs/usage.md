# Usage Guide

## Prerequisites

- Python 3.12+
- A [DhanHQ](https://dhan.co) trading account with API access

## Installation

```bash
git clone <repo-url> && cd swiftrade
pip install -e ".[dev]"
```

Or with Poetry:

```bash
poetry install
```

## Configuration

Copy the example environment file and fill in your credentials:

```bash
cp .env.example .env
```

### Required

| Variable | Description |
|---|---|
| `DHAN_CLIENT_ID` | Your Dhan broker client ID |

### Authentication (choose one)

**Manual token** — paste a 24-hour token from [web.dhan.co](https://web.dhan.co):

| Variable | Description |
|---|---|
| `DHAN_ACCESS_TOKEN` | 24-hour API token |

**Automatic (TOTP)** — generates a fresh token on each run:

| Variable | Description |
|---|---|
| `DHAN_PIN` | 6-digit login PIN |
| `DHAN_TOTP_SECRET` | Base32 TOTP secret from web.dhan.co > Profile > DhanHQ Trading APIs > Optional Settings > Enable TOTP |

### Optional

| Variable | Default | Description |
|---|---|---|
| `DHAN_LIVE` | `0` | Set to `1` to send real orders |
| `MAX_QTY` | `50` | Max shares per order |
| `MAX_ORDER_VALUE` | `50000` | Max notional value per order (INR) |
| `MAX_DAILY_LOSS` | `10000` | Daily realized loss cap (INR) |
| `STRATEGY_INTERVAL` | `60` | Strategy polling interval (seconds) |
| `LOG_LEVEL` | `INFO` | DEBUG, INFO, WARNING, ERROR, CRITICAL |
| `JOURNAL_PATH` | `trades.csv` | Path for the trade journal CSV |
| `FEED_MODE` | `poll` | `poll` or `ws` (WebSocket) |
| `DHAN_PROXY` | *(empty)* | HTTPS proxy for order calls only, e.g. `http://STATIC_IP:8080` (see [Static IP for orders](#static-ip-for-orders)) |

## Starting the app

### Web dashboard

```bash
streamlit run app.py
```

Opens at http://localhost:8501. The dashboard provides access to all features through a sidebar menu.

### CLI

```bash
dhan-algo --help
```

## Web dashboard pages

### Dashboard

Account funds overview, current configuration, and risk limit display.

### Swing Scanner

Scores NSE stocks for swing trading setups using daily timeframe data.

- Select a universe: NIFTY 50, NIFTY 100, or enter custom symbols
- Adjust indicator weights with sliders (trend, momentum, volume, breakout, volatility)
- Results are color-coded by score: green (strong), yellow, orange, red (weak)
- Expand any result to see the detailed trade setup with entry, stop-loss (1.5x ATR), and targets (2:1 and 3:1 risk-reward)
- Place orders directly from the scan results

### Intraday Scanner

Scores stocks for intraday trades using minute-level data.

- Choose candle interval: 1, 5, or 15 minutes
- Indicators scored: VWAP, SuperTrend, momentum (RSI-7, MACD), volume breakout, and Opening Range Breakout (ORB)
- Trade setup uses tighter parameters than swing: 1.0x ATR stop-loss, 1.5:1 and 2.5:1 targets
- Entry, stop-loss, and target fields auto-fill from the analysis
- Supports placing entry + stop-loss + target orders together

### Strategy Runner

Run the built-in SMA crossover demo strategy on one or more symbols. Start/stop controls and a live signal log are provided. This is a demo strategy — see the README for how to write a custom strategy.

### Market Data

Look up the last traded price for any NSE symbol. Supports Yahoo Finance (free, ~15-min delayed) or Dhan API (real-time). Optional auto-refresh every 5 seconds.

### Place Order

Manual order entry form with:

- Symbol lookup
- BUY / SELL side
- MARKET / LIMIT order types
- INTRA (intraday) / CNC (delivery) products
- Quantity and price inputs
- Risk guard feedback before submission

### Positions & Orders

View current open positions and order history.

### Backtest

Test strategies against historical data:

- Fetch data via Dhan API or Yahoo Finance
- Upload your own CSV
- Run the SMA crossover strategy and view results (P&L, trade count, fills)

### Trade Journal

Browse all past trade attempts with filters:

- Status: placed, blocked, dry_run, failed
- Side: BUY, SELL

### Kill Switch

Emergency halt for all trading activity. Requires two-step confirmation.

## CLI commands

```
dhan-algo funds                              # show account funds
dhan-algo ltp RELIANCE                       # print last traded price
dhan-algo order RELIANCE --side BUY --qty 1  # place order (dry-run by default)
dhan-algo positions                          # show open positions
dhan-algo orders                             # show order history
dhan-algo strategy RELIANCE --interval 30    # run SMA demo strategy loop
dhan-algo backtest RELIANCE --from 2024-01-01 --to 2024-12-31
dhan-algo kill-switch                        # emergency halt
```

Pass `--live` to any command to send real orders instead of dry-run.

## Dry-run vs live

Dry-run is **on by default**. Orders are validated, risk-checked, and logged — but never sent to the broker.

To go live, set **either**:

- `DHAN_LIVE=1` in your `.env`, **or**
- pass `--live` on the CLI

## Static IP for orders

Dhan requires a **whitelisted static IP** for order placement — you'll get
error **DH-905** if orders come from an unlisted IP. This applies only to
order placement; data APIs (scanners, quotes, historical) work from any IP,
so most users hit this only when going live.

Home and office connections usually have a *dynamic* public IP that changes.
`DHAN_PROXY` lets you route **only the order calls** through a fixed-IP HTTPS
proxy, so orders exit from the whitelisted IP while data traffic stays direct
(fast, and unaffected if the proxy is down).

### Setup

1. Stand up an HTTPS proxy on a host that has a static public IP:
   - A small cloud VPS (AWS EC2 + Elastic IP, Oracle Cloud Always-Free,
     DigitalOcean, Hetzner) running `tinyproxy` or `squid`, **or**
   - a static-IP VPN/proxy service.
2. Whitelist that static IP in the Dhan portal:
   **web.dhan.co > Profile > DhanHQ Trading APIs**.
3. Point the app at the proxy in `.env`:

   ```bash
   DHAN_PROXY=http://STATIC_IP:8080
   DHAN_LIVE=1
   ```

Leave `DHAN_PROXY` empty (the default) and orders go out directly, unchanged.

Only `place_order` and `place_super_order` are routed through the proxy. The
proxy is applied and restored on the shared client session under a lock, so
the Strategy Runner and Exit Manager worker threads never clobber each other
mid-order.

> **Tip:** The most robust option is to run the whole app on the static-IP
> VPS. You then don't need a proxy at all — just whitelist the VPS IP — and
> the Exit Manager can run around the clock.

## Running tests

```bash
pytest
```

All tests use mocks — no live API calls are made.
