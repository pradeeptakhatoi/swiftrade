"""Pluggable strategy hook with a polling loop, SMA demo, and ORB breakout."""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import pandas as pd
from dhanhq import dhanhq

from dhan_algo.config import Settings, get_settings
from dhan_algo.market_data import ltp
from dhan_algo.orders import calculate_position_size, place, place_bracket, place_with_sl_target

logger = logging.getLogger(__name__)


@dataclass
class Order:
    """A lightweight order intent returned by a strategy."""

    side: str  # "BUY" or "SELL"
    qty: int
    order_type: str = "MARKET"
    product: str = "INTRA"
    price: float = 0.0
    security_id: str = ""
    stop_loss: float = 0.0
    target: float = 0.0
    trailing_jump: float = 0.0


# ---------------------------------------------------------------------------
# Ticker protocol – abstracts price sources
# ---------------------------------------------------------------------------


@runtime_checkable
class Ticker(Protocol):
    """Uniform interface for price feeds (polling, WebSocket, replay)."""

    def get_ltp(self, security_id: str, segment: str | None = None) -> float | None:
        ...

    @property
    def watchlist(self) -> dict[str, float]:
        ...


class PollingTicker:
    """HTTP-based ticker that polls ``ltp()`` for each symbol."""

    def __init__(self, client: dhanhq, security_ids: list[str], segment: str | None = None):
        self._client = client
        self._security_ids = list(security_ids)
        self._segment = segment
        self._prices: dict[str, float] = {}

    def get_ltp(self, security_id: str, segment: str | None = None) -> float | None:
        return self._prices.get(security_id)

    @property
    def watchlist(self) -> dict[str, float]:
        return dict(self._prices)

    def refresh_all(self) -> None:
        """Poll LTP for every tracked security and update the cache."""
        for sid in self._security_ids:
            price = ltp(self._client, sid, self._segment)
            if price is not None:
                self._prices[sid] = price


class Strategy(ABC):
    """Protocol for pluggable strategies."""

    @abstractmethod
    def evaluate(
        self, client: dhanhq, security_id: str, segment: str | None
    ) -> Order | None:
        """Return an Order to place, or None to do nothing this tick."""


def _dispatch_order(
    client: dhanhq,
    order: Order,
    default_security_id: str,
    segment: str | None,
    settings: Settings,
) -> dict | None:
    """Route an Order through the appropriate placement function."""
    sid = order.security_id or default_security_id
    if order.stop_loss > 0 and order.target > 0:
        if order.product.upper() in ("INTRA", "INTRADAY"):
            return place_bracket(
                client, sid,
                side=order.side, qty=order.qty,
                entry_price=order.price,
                stop_loss_price=order.stop_loss,
                target_price=order.target,
                trailing_jump=order.trailing_jump,
                segment=segment, settings=settings,
            )
        return place_with_sl_target(
            client, sid,
            side=order.side, qty=order.qty,
            entry_price=order.price,
            stop_loss_price=order.stop_loss,
            target_price=order.target,
            order_type=order.order_type, product=order.product,
            segment=segment, settings=settings,
        )
    return place(
        client, sid,
        side=order.side, qty=order.qty,
        order_type=order.order_type, product=order.product,
        price=order.price, segment=segment, settings=settings,
    )


def run_strategy_loop(
    strategy: Strategy,
    client: dhanhq,
    security_id: str,
    settings: Settings | None = None,
    segment: str | None = None,
) -> None:
    """Poll LTP at the configured interval, route orders through place()."""
    settings = settings or get_settings()
    logger.info(
        "Starting strategy loop for security %s (interval=%ds). Ctrl+C to stop.",
        security_id,
        settings.strategy_interval,
    )
    try:
        while True:
            order = strategy.evaluate(client, security_id, segment)
            if order is not None:
                _dispatch_order(client, order, security_id, segment, settings)
            time.sleep(settings.strategy_interval)
    except KeyboardInterrupt:
        logger.info("Strategy loop stopped.")


class SmaDemo(Strategy):
    """DEMO ONLY -- not investment advice.

    Maintains a rolling window of LTP values, computes short/long SMA,
    generates a BUY signal on golden cross and SELL on death cross.
    """

    def __init__(self, short_period: int = 5, long_period: int = 20, qty: int = 1):
        self.short_period = short_period
        self.long_period = long_period
        self.qty = qty
        self._prices: deque[float] = deque(maxlen=long_period)
        self._prev_short: float | None = None
        self._prev_long: float | None = None

    def evaluate(
        self, client: dhanhq, security_id: str, segment: str | None
    ) -> Order | None:
        price = ltp(client, security_id, segment)
        if price is None:
            return None

        self._prices.append(price)
        if len(self._prices) < self.long_period:
            logger.debug(
                "[SMA] collecting prices (%d/%d)...",
                len(self._prices),
                self.long_period,
            )
            return None

        prices = list(self._prices)
        short_sma = sum(prices[-self.short_period :]) / self.short_period
        long_sma = sum(prices) / self.long_period
        logger.debug("[SMA] short=%.2f  long=%.2f  ltp=%.2f", short_sma, long_sma, price)

        signal: Order | None = None
        if (
            self._prev_short is not None
            and self._prev_long is not None
        ):
            if self._prev_short <= self._prev_long and short_sma > long_sma:
                logger.info("[SMA] golden cross -> BUY")
                signal = Order(side="BUY", qty=self.qty)
            elif self._prev_short >= self._prev_long and short_sma < long_sma:
                logger.info("[SMA] death cross -> SELL")
                signal = Order(side="SELL", qty=self.qty)

        self._prev_short = short_sma
        self._prev_long = long_sma
        return signal


# ---------------------------------------------------------------------------
# Multi-symbol strategy framework
# ---------------------------------------------------------------------------


class MultiStrategy(ABC):
    """Base class for strategies that handle multiple symbols via ticks."""

    @abstractmethod
    def on_tick(self, ticker: Ticker, security_id: str, segment: str | None) -> list[Order]:
        """Called for each price update. Return zero or more orders."""

    def on_start(self, ticker: Ticker, security_ids: list[str]) -> None:  # noqa: B027
        """Optional hook called once before the loop begins."""

    def on_stop(self) -> None:  # noqa: B027
        """Optional hook called when the loop ends."""


class _StrategyAdapter(MultiStrategy):
    """Wraps a single-symbol ``Strategy`` as a ``MultiStrategy``."""

    def __init__(self, strategy: Strategy, client: dhanhq):
        self._strategy = strategy
        self._client = client

    def on_tick(self, ticker: Ticker, security_id: str, segment: str | None) -> list[Order]:
        order = self._strategy.evaluate(self._client, security_id, segment)
        if order is not None:
            order.security_id = security_id
            return [order]
        return []


class SmaDemoMulti(MultiStrategy):
    """Runs independent ``SmaDemo`` instances per symbol."""

    def __init__(self, short_period: int = 5, long_period: int = 20, qty: int = 1):
        self.short_period = short_period
        self.long_period = long_period
        self.qty = qty
        self._delegates: dict[str, SmaDemo] = {}

    def _get_delegate(self, security_id: str) -> SmaDemo:
        if security_id not in self._delegates:
            self._delegates[security_id] = SmaDemo(
                short_period=self.short_period,
                long_period=self.long_period,
                qty=self.qty,
            )
        return self._delegates[security_id]

    def on_tick(self, ticker: Ticker, security_id: str, segment: str | None) -> list[Order]:
        price = ticker.get_ltp(security_id, segment)
        if price is None:
            return []
        delegate = self._get_delegate(security_id)
        # Feed the price directly into the delegate's internal deque
        delegate._prices.append(price)
        if len(delegate._prices) < delegate.long_period:
            logger.debug(
                "[SmaDemoMulti] %s collecting (%d/%d)",
                security_id, len(delegate._prices), delegate.long_period,
            )
            return []

        prices = list(delegate._prices)
        short_sma = sum(prices[-delegate.short_period:]) / delegate.short_period
        long_sma = sum(prices) / delegate.long_period

        signal: list[Order] = []
        if delegate._prev_short is not None and delegate._prev_long is not None:
            if delegate._prev_short <= delegate._prev_long and short_sma > long_sma:
                logger.info("[SmaDemoMulti] %s golden cross -> BUY", security_id)
                signal = [Order(side="BUY", qty=self.qty, security_id=security_id)]
            elif delegate._prev_short >= delegate._prev_long and short_sma < long_sma:
                logger.info("[SmaDemoMulti] %s death cross -> SELL", security_id)
                signal = [Order(side="SELL", qty=self.qty, security_id=security_id)]

        delegate._prev_short = short_sma
        delegate._prev_long = long_sma
        return signal


# ---------------------------------------------------------------------------
# ORB Breakout strategy
# ---------------------------------------------------------------------------


class OrbBreakoutStrategy(MultiStrategy):
    """Intraday ORB breakout with SuperTrend + volume confirmation.

    On start, fetches today's intraday candles via yfinance and computes the
    opening range, ATR, SuperTrend direction, and volume ratio.  On each tick,
    checks whether the current price has broken above/below the ORB levels
    with confirmation filters.  Generates bracket orders (entry + SL + target)
    with ATR-based risk/reward.

    Only one trade per direction per day to avoid whipsaws.
    """

    def __init__(
        self,
        qty: int = 1,
        interval_minutes: int = 15,
        atr_multiplier: float = 1.0,
        min_volume_ratio: float = 1.0,
        risk_per_trade: float = 0,
        symbol_map: dict[str, str] | None = None,
        settings: Settings | None = None,
    ):
        self.qty = qty
        self.interval_minutes = interval_minutes
        self.atr_multiplier = atr_multiplier
        self.min_volume_ratio = min_volume_ratio
        self.risk_per_trade = risk_per_trade
        self._symbol_map = symbol_map or {}  # security_id -> ticker symbol
        self._settings = settings
        self._state: dict[str, dict] = {}

    def on_start(self, ticker: Ticker, security_ids: list[str]) -> None:
        """Fetch intraday data and compute ORB + indicators for each symbol."""
        from indicators import compute_intraday

        for sid in security_ids:
            sym = self._symbol_map.get(sid)
            if not sym:
                logger.warning("[ORB] No symbol mapping for %s, skipping", sid)
                continue
            try:
                import yfinance as yf

                df = yf.download(
                    f"{sym}.NS", period="5d",
                    interval=f"{self.interval_minutes}m",
                    auto_adjust=True, progress=False,
                )
                if df is None or df.empty:
                    logger.warning("[ORB] No data for %s", sym)
                    continue

                df = df.reset_index()
                # Normalise column names
                for col in ("Open", "High", "Low", "Close", "Volume"):
                    if col not in df.columns:
                        lc = col.lower()
                        if lc in df.columns:
                            df.rename(columns={lc: col}, inplace=True)

                if len(df) < 10:
                    logger.warning("[ORB] Too few bars for %s (%d)", sym, len(df))
                    continue

                df = compute_intraday(df, self.interval_minutes)
                last = df.iloc[-1]

                self._state[sid] = {
                    "symbol": sym,
                    "orb_high": float(df["orb_high"].iloc[-1]),
                    "orb_low": float(df["orb_low"].iloc[-1]),
                    "atr": float(df["atr7"].dropna().iloc[-1]),
                    "st_direction": int(last.get("st_direction", 0)),
                    "vol_ratio": float(last.get("vol_ratio", 0)),
                    "long_traded": False,
                    "short_traded": False,
                }
                st = self._state[sid]
                logger.info(
                    "[ORB] %s ready: ORB high=%.2f low=%.2f ATR=%.2f ST=%s vol=%.2f",
                    sym, st["orb_high"], st["orb_low"], st["atr"],
                    "BULL" if st["st_direction"] == 1 else "BEAR",
                    st["vol_ratio"],
                )
            except Exception:
                logger.exception("[ORB] Failed to initialise %s", sym)

    def on_tick(self, ticker: Ticker, security_id: str, segment: str | None) -> list[Order]:
        state = self._state.get(security_id)
        if state is None:
            return []

        price = ticker.get_ltp(security_id, segment)
        if price is None:
            return []

        orb_high = state["orb_high"]
        orb_low = state["orb_low"]
        atr = state["atr"]
        st_dir = state["st_direction"]
        vol_ratio = state["vol_ratio"]

        # ---- LONG breakout ----
        if (
            not state["long_traded"]
            and price > orb_high
            and st_dir == 1
            and vol_ratio >= self.min_volume_ratio
        ):
            risk = self.atr_multiplier * atr
            sl = price - risk
            target = price + 2 * risk  # 2:1 R:R

            qty = self.qty
            if self.risk_per_trade > 0 and risk > 0.01:
                s = self._settings or get_settings()
                qty = calculate_position_size(
                    price, sl, self.risk_per_trade, s.max_qty, s.max_order_value,
                    capital=s.trading_capital, max_position_pct=s.max_position_pct,
                )
                qty = max(qty, 1)

            state["long_traded"] = True
            logger.info(
                "[ORB] %s LONG breakout: entry=%.2f SL=%.2f target=%.2f qty=%d",
                state["symbol"], price, sl, target, qty,
            )
            return [Order(
                side="BUY", qty=qty, order_type="LIMIT", product="INTRA",
                price=price, security_id=security_id,
                stop_loss=sl, target=target,
            )]

        # ---- SHORT breakout ----
        if (
            not state["short_traded"]
            and price < orb_low
            and st_dir == -1
            and vol_ratio >= self.min_volume_ratio
        ):
            risk = self.atr_multiplier * atr
            sl = price + risk
            target = price - 2 * risk  # 2:1 R:R

            qty = self.qty
            if self.risk_per_trade > 0 and risk > 0.01:
                s = self._settings or get_settings()
                qty = calculate_position_size(
                    price, sl, self.risk_per_trade, s.max_qty, s.max_order_value,
                    capital=s.trading_capital, max_position_pct=s.max_position_pct,
                )
                qty = max(qty, 1)

            state["short_traded"] = True
            logger.info(
                "[ORB] %s SHORT breakout: entry=%.2f SL=%.2f target=%.2f qty=%d",
                state["symbol"], price, sl, target, qty,
            )
            return [Order(
                side="SELL", qty=qty, order_type="LIMIT", product="INTRA",
                price=price, security_id=security_id,
                stop_loss=sl, target=target,
            )]

        return []


def run_multi_strategy_loop(
    strategy: MultiStrategy,
    client: dhanhq,
    security_ids: list[str],
    settings: Settings | None = None,
    segment: str | None = None,
) -> None:
    """Poll LTP for multiple symbols, dispatch ticks to a ``MultiStrategy``."""
    settings = settings or get_settings()
    ticker = PollingTicker(client, security_ids, segment)
    strategy.on_start(ticker, security_ids)
    logger.info(
        "Starting multi-strategy loop for %s (interval=%ds). Ctrl+C to stop.",
        security_ids, settings.strategy_interval,
    )
    try:
        while True:
            ticker.refresh_all()
            for sid in security_ids:
                orders = strategy.on_tick(ticker, sid, segment)
                for order in orders:
                    _dispatch_order(client, order, sid, segment, settings)
            time.sleep(settings.strategy_interval)
    except KeyboardInterrupt:
        strategy.on_stop()
        logger.info("Multi-strategy loop stopped.")


def run_ws_strategy_loop(
    strategy: MultiStrategy,
    client: dhanhq,
    security_ids: list[str],
    settings: Settings | None = None,
    segment: str | None = None,
) -> None:
    """Drive a ``MultiStrategy`` from a WebSocket feed."""
    settings = settings or get_settings()

    from dhan_algo.market_data import WebSocketTicker

    def _on_tick(security_id: str, seg: str | None) -> None:
        orders = strategy.on_tick(ws_ticker, security_id, seg)
        for order in orders:
            _dispatch_order(client, order, security_id, seg, settings)

    ws_ticker = WebSocketTicker(client, security_ids, segment=segment, on_tick_callback=_on_tick)
    strategy.on_start(ws_ticker, security_ids)
    ws_ticker.start()

    logger.info("WebSocket strategy loop running for %s. Ctrl+C to stop.", security_ids)
    stop_event = threading.Event()
    try:
        stop_event.wait()
    except KeyboardInterrupt:
        ws_ticker.stop()
        strategy.on_stop()
        logger.info("WebSocket strategy loop stopped.")
