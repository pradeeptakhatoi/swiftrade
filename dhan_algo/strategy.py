"""Pluggable strategy hook with a polling loop and SMA demo."""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from dhanhq import dhanhq

from dhan_algo.config import Settings, get_settings
from dhan_algo.market_data import ltp
from dhan_algo.orders import place

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
                place(
                    client,
                    security_id,
                    side=order.side,
                    qty=order.qty,
                    order_type=order.order_type,
                    product=order.product,
                    price=order.price,
                    segment=segment,
                    settings=settings,
                )
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
                    place(
                        client, order.security_id or sid,
                        side=order.side, qty=order.qty,
                        order_type=order.order_type, product=order.product,
                        price=order.price, segment=segment, settings=settings,
                    )
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
            place(
                client, order.security_id or security_id,
                side=order.side, qty=order.qty,
                order_type=order.order_type, product=order.product,
                price=order.price, segment=seg, settings=settings,
            )

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
