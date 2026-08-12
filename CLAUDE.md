# SwiftTrade

Algorithmic trading toolkit for the Indian stock market (NSE) using DhanHQ as the brokerage API.

## Tech Stack

- Python 3.12+
- Streamlit (web dashboard)
- DhanHQ API (broker integration)
- Yahoo Finance (historical/delayed market data)
- Poetry (dependency management)
- pytest (testing, all tests use mocks)

## How to Run

- **Web dashboard**: `streamlit run app.py` (opens at http://localhost:8501)
- **CLI**: `dhan-algo --help`
- **Tests**: `pytest`

## Project Structure

- `app.py` — Main Streamlit web UI (all dashboard pages)
- `dhan_algo/` — Core trading engine package
  - `auth.py` — TOTP/token authentication
  - `cli.py` — Command-line interface
  - `client.py` — DhanHQ client wrapper
  - `config.py` — Settings/config management (pydantic-settings, reads .env)
  - `journal.py` — Trade journal CSV recorder
  - `market_data.py` — Market data & WebSocket ticker
  - `orders.py` — Order placement with risk guards
  - `risk.py` — Risk management checks
  - `security_master.py` — Security ID resolver
  - `strategy.py` — Strategy framework & SMA demo
  - `backtest.py` — Backtesting engine
- `indicators/` — Technical indicators library (EMA, RSI, MACD, ATR, Bollinger Bands, VWAP, SuperTrend, ORB, breakout, volume)
- `intraday_scorer.py` — Intraday trading signal scoring
- `swing_scorer.py` — Swing trading signal scoring
- `data/universe.py` — Ticker display name mapping
- `docs/usage.md` — Full usage guide
- `tests/` — Unit tests

## Key Features

- **Swing Scanner**: Scores stocks on daily timeframe (trend, momentum, volume, breakout, volatility). ATR-based trade setups with 2:1 and 3:1 R:R targets.
- **Intraday Scanner**: Scores stocks on 1/5/15-min candles (VWAP, SuperTrend, momentum, volume, ORB). Tighter ATR-based setups.
- **Strategy Runner**: SMA crossover demo strategy with start/stop controls.
- **Order Management**: BUY/SELL with MARKET/LIMIT/SL types, INTRA/CNC products.
- **Risk Management**: MAX_QTY, MAX_ORDER_VALUE, MAX_DAILY_LOSS guards checked before every order.
- **Backtesting**: Historical data replay with SMA strategy, supports Dhan API, Yahoo Finance, or CSV upload.
- **Trade Journal**: CSV log of all order attempts (placed/blocked/dry_run/failed).
- **Kill Switch**: Emergency trading halt with 2-step confirmation.

## Configuration

Config via `.env` file (see `.env.example`). Key variables:
- `DHAN_CLIENT_ID` (required) — Dhan broker ID
- `DHAN_ACCESS_TOKEN` or `DHAN_PIN` + `DHAN_TOTP_SECRET` — authentication
- `DHAN_LIVE` (default `0`) — set to `1` for real orders, otherwise dry-run
- `MAX_QTY`, `MAX_ORDER_VALUE`, `MAX_DAILY_LOSS` — risk limits
- `FEED_MODE` — `poll` or `ws` (WebSocket)

## Important Notes

- DRY_RUN is on by default — no real orders are sent unless DHAN_LIVE=1 or --live flag is passed.
- 24-hour token expiry — tokens from Dhan expire daily. Use TOTP flow for auto-renewal.
- Static IP requirement — order placement requires a whitelisted IP (error DH-905 if not).
- Custom strategies: subclass `dhan_algo.strategy.Strategy` and implement `evaluate()`.
