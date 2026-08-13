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
    max_open_positions: int = 0  # cap on concurrent open positions; 0 = unlimited
    max_consecutive_losses: int = 0  # halt after N losing trades in a row; 0 = off
    max_position_pct: float = 0  # max % of capital per position; 0 = off
    trading_capital: float = 0  # capital base for % sizing; 0 = off
    strategy_interval: int = 60
    log_level: str = "INFO"
    risk_per_trade: float = 0  # INR risk per trade; 0 = manual qty
    journal_path: str = "trades.csv"
    feed_mode: str = "poll"
    default_data_source: str = "Dhan API"  # "Dhan API" or "Yahoo Finance"
    security_master_path: str = "security_id_list.csv"  # offline scrip-master fallback
    dhan_proxy: str = ""  # HTTPS proxy for order calls (e.g. static-IP egress)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
