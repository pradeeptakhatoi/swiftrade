"""Pluggable strategy hook with a polling loop and SMA demo."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass

from dhanhq import dhanhq

from dhan_algo.config import Settings, get_settings
from dhan_algo.market_data import ltp
from dhan_algo.orders import place


@dataclass
class Order:
    """A lightweight order intent returned by a strategy."""

    side: str  # "BUY" or "SELL"
    qty: int
    order_type: str = "MARKET"
    product: str = "INTRA"
    price: float = 0.0


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
    print(
        f"Starting strategy loop for security {security_id} "
        f"(interval={settings.strategy_interval}s). Ctrl+C to stop."
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
        print("\nStrategy loop stopped.")


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
            print(
                f"[SMA] collecting prices ({len(self._prices)}/{self.long_period})..."
            )
            return None

        prices = list(self._prices)
        short_sma = sum(prices[-self.short_period :]) / self.short_period
        long_sma = sum(prices) / self.long_period
        print(f"[SMA] short={short_sma:.2f}  long={long_sma:.2f}  ltp={price:.2f}")

        signal: Order | None = None
        if (
            self._prev_short is not None
            and self._prev_long is not None
        ):
            if self._prev_short <= self._prev_long and short_sma > long_sma:
                print("[SMA] golden cross -> BUY")
                signal = Order(side="BUY", qty=self.qty)
            elif self._prev_short >= self._prev_long and short_sma < long_sma:
                print("[SMA] death cross -> SELL")
                signal = Order(side="SELL", qty=self.qty)

        self._prev_short = short_sma
        self._prev_long = long_sma
        return signal
