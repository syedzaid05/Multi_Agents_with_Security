"""A tiny async token-bucket rate limiter.

One limiter is created per provider so a burst against Groq does not starve
Gemini. Buckets refill continuously at ``rate_per_minute / 60`` tokens/second,
capped at the burst size (the per-minute allowance).
"""

from __future__ import annotations

import asyncio
import time

__all__ = ["RateLimiter"]


class RateLimiter:
    """Async, in-process token bucket."""

    def __init__(
        self,
        rate_per_minute: int,
        *,
        burst: int | None = None,
        clock=time.monotonic,
    ) -> None:
        self.rate_per_minute = max(1, int(rate_per_minute))
        self.capacity = float(burst if burst is not None else self.rate_per_minute)
        self._refill_per_second = self.rate_per_minute / 60.0
        self._tokens = self.capacity
        self._updated = clock()
        self._clock = clock
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = self._clock()
        elapsed = max(0.0, now - self._updated)
        self._updated = now
        self._tokens = min(self.capacity, self._tokens + elapsed * self._refill_per_second)

    async def acquire(self, tokens: float = 1.0) -> float:
        """Block until ``tokens`` are available; return the seconds waited."""
        waited = 0.0
        async with self._lock:
            while True:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return waited
                deficit = tokens - self._tokens
                delay = deficit / self._refill_per_second
                waited += delay
                await asyncio.sleep(delay)

    @property
    def available_tokens(self) -> float:
        self._refill()
        return self._tokens
