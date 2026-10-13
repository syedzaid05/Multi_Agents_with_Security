from __future__ import annotations

import pytest

from multi_agent_research.llm.ratelimit import RateLimiter


async def test_consumes_tokens_without_waiting():
    limiter = RateLimiter(60, clock=lambda: 0.0)  # capacity 60
    waited = await limiter.acquire(50)
    assert waited == 0.0
    assert limiter.available_tokens == pytest.approx(10.0, abs=0.01)


async def test_refills_over_time():
    now = {"t": 0.0}
    limiter = RateLimiter(60, clock=lambda: now["t"])  # 1 token / second
    await limiter.acquire(60)
    assert limiter.available_tokens == pytest.approx(0.0, abs=0.01)
    now["t"] = 10.0
    assert limiter.available_tokens == pytest.approx(10.0, abs=0.01)


async def test_bucket_is_capped_at_capacity():
    now = {"t": 0.0}
    limiter = RateLimiter(30, clock=lambda: now["t"])
    now["t"] = 10_000.0
    assert limiter.available_tokens == pytest.approx(30.0, abs=0.01)
