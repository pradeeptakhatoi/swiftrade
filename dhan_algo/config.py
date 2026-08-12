"""Centralised settings loaded from environment / .env file."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    dhan_client_id: str = ""
    dhan_access_token: str = ""
    dhan_pin: str = ""
    dhan_totp_secret: str = ""
    dhan_live: bool = False
    max_qty: int = 50
    max_order_value: float = 50_000
    max_daily_loss: float = 10_000
    strategy_interval: int = 60
    log_level: str = "INFO"
    risk_per_trade: float = 0  # INR risk per trade; 0 = manual qty
    journal_path: str = "trades.csv"
    feed_mode: str = "poll"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
