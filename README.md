# SwiftTrade

Algorithmic trading toolkit for the Indian stock market (NSE) using DhanHQ. Features a Streamlit web dashboard with swing and intraday scanners, a CLI for automation, backtesting, and built-in risk management. DRY_RUN by default — no orders are sent unless you explicitly opt in.

## Quick start

```bash
git clone <repo-url> && cd swiftrade
pip install -e ".[dev]"
cp .env.example .env
# Edit .env with your DhanHQ credentials
```

### Web dashboard (recommended)

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

### CLI

```bash
dhan-algo --help
```

See [docs/usage.md](docs/usage.md) for detailed usage instructions.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `DHAN_CLIENT_ID` | *(required)* | Your Dhan client ID |
| `DHAN_ACCESS_TOKEN` | `""` | 24-hour API token (or use TOTP flow) |
| `DHAN_PIN` | `""` | 6-digit login PIN for TOTP auth |
| `DHAN_TOTP_SECRET` | `""` | Base32 TOTP secret from Dhan |
| `DHAN_LIVE` | `0` | Set to `1` to send real orders |
| `MAX_QTY` | `50` | Max shares per order |
| `MAX_ORDER_VALUE` | `50000` | Max notional value per order (INR) |
| `MAX_DAILY_LOSS` | `10000` | Daily realized loss cap (INR) |
| `STRATEGY_INTERVAL` | `60` | Strategy polling interval (seconds) |

## Dry-run vs live

DRY_RUN is **on by default**. Orders are validated, risk-checked, and printed — but never sent.

To go live, set **either**:
- `DHAN_LIVE=1` in your `.env`, **or**
- pass `--live` on the command line

## Token lifecycle

**Manual**: paste a 24-hour token from [web.dhan.co](https://web.dhan.co) into `DHAN_ACCESS_TOKEN`. Refresh daily.

**Automatic (TOTP)**: set `DHAN_PIN` (your 6-digit login PIN) and `DHAN_TOTP_SECRET` (the base32 secret from web.dhan.co > Profile > DhanHQ Trading APIs > Optional Settings > Enable TOTP). The tool generates a fresh token on each run.

## DhanHQ gotchas

- **24-hour token expiry** — tokens expire daily. Use the TOTP flow or re-paste.
- **Static IP requirement** — order placement requires a whitelisted static IP. Error `DH-905` means your IP isn't whitelisted. Market data calls are unaffected.

## CLI usage

```
dhan-algo funds                              # show account funds
dhan-algo ltp RELIANCE                       # print LTP
dhan-algo order RELIANCE --side BUY --qty 1  # place order (dry-run)
dhan-algo positions                          # show positions
dhan-algo orders                             # show orders
dhan-algo strategy RELIANCE --interval 30    # run SMA demo loop
dhan-algo kill-switch                        # emergency halt
```

## Running tests

```bash
pytest
```

All tests use mocks — no live API calls.

## Writing a custom strategy

Subclass `dhan_algo.strategy.Strategy` and implement `evaluate()`:

```python
from dhan_algo.strategy import Strategy, Order

class MyStrategy(Strategy):
    def evaluate(self, client, security_id, segment) -> Order | None:
        # Your logic here
        return Order(side="BUY", qty=1)
```

Then run it with `run_strategy_loop(MyStrategy(), client, security_id)`.
