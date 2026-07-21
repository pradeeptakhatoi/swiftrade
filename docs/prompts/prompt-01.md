This repo has a single working file, dhan_trader.py — a tested DhanHQ (Indian
broker) starter that connects, fetches funds/LTP, resolves security IDs, and
places orders with a DRY_RUN default and risk guards (MAX_QTY, MAX_ORDER_VALUE).
Restructure it into a small, production-sane Python package WITHOUT changing its
safe defaults. Requirements:

Structure
- Convert to a package: dhan_algo/ (client, market_data, orders, security_master,
  risk, config) plus a thin cli.py entrypoint. Keep it lean — this is a
  script-first tool, not a framework. No web server yet.
- pyproject.toml (or requirements.txt) pinning dhanhq; target Python 3.11+.

Config & secrets
- Pydantic-settings (or a small env loader) reading DHAN_CLIENT_ID,
  DHAN_ACCESS_TOKEN, DHAN_LIVE, and the risk limits. Add .env.example.
- Confirm .gitignore already excludes .env — never read or print token values.

Token lifecycle
- Add an auth module that regenerates the 24-hour access token from a stored
  API key + secret using Dhan's TOTP-based flow, so I don't paste a token daily.
  Stub the TOTP secret via env; document where to get the API key on web.dhan.co.

Safety (keep and harden)
- Preserve DRY_RUN-by-default and the risk guards. Add a daily-loss cap and a
  single kill_switch() entrypoint. Make live mode require an explicit flag.

Strategy hook
- Add a minimal, pluggable strategy loop: poll LTP on an interval, evaluate a
  simple signal (e.g. SMA cross or % move), and route any order through the
  EXISTING dry-run + guard path. One example strategy, clearly marked as a demo
  not a recommendation.

Docs & tests
- README: setup, env vars, dry-run vs live, and the DhanHQ gotchas — 24h token
  expiry and the static-IP requirement for orders (error DH-905 from a
  non-whitelisted IP; market data is unaffected).
- pytest tests for the risk guards and security-id resolver, mocking the SDK so
  no live calls run in CI.

Do not add the FastAPI backend or iOS app yet — leave room for them but keep
this PR focused on a clean, testable CLI trading tool. Show me the proposed file
layout before writing code.