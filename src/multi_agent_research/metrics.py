"""Lightweight in-process metrics + latency tracking.

Not a Prometheus replacement - just enough structure to show per-stage latency
and counters in logs and in the Streamlit UI.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

__all__ = ["Metrics", "Span"]


@dataclass(slots=True)
class Span:
    """A timed unit of work (an agent run, an LLM call, a tool fetch...)."""

    name: str
    duration_ms: float
    ok: bool = True
    meta: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {"name": self.name, "duration_ms": round(self.duration_ms, 2), **self.meta}


@dataclass
class Metrics:
    """Collects spans, counters and observations for one pipeline run."""

    spans: list[Span] = field(default_factory=list)
    counters: dict[str, int] = field(default_factory=dict)
    observations: dict[str, list[float]] = field(default_factory=dict)
    _clock = staticmethod(time.perf_counter)

    # -- recording ----------------------------------------------------------
    @contextmanager
    def span(self, name: str, **meta: object) -> Iterator[dict[str, object]]:
        """Time a block; records a :class:`Span` on exit.

        The yielded dict can be mutated inside the block to attach extra
        metadata or to mark the span as failed (``ok=False``).
        """
        bucket: dict[str, object] = dict(meta)
        started = self._clock()
        error: BaseException | None = None
        try:
            yield bucket
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            error = exc
            bucket.setdefault("ok", False)
            raise
        finally:
            duration_ms = (self._clock() - started) * 1000.0
            self.spans.append(
                Span(
                    name=name,
                    duration_ms=duration_ms,
                    ok=bool(bucket.pop("ok", True)),
                    meta=bucket,
                )
            )
            if error is None:
                self.observe(f"stage_ms.{name}", duration_ms)
            else:
                self.increment(f"stage_error.{name}")

    def increment(self, name: str, amount: int = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) + amount

    def observe(self, name: str, value: float) -> None:
        self.observations.setdefault(name, []).append(value)

    # -- reporting ----------------------------------------------------------
    @property
    def total_ms(self) -> float:
        return sum(span.duration_ms for span in self.spans)

    def stage_timings(self) -> dict[str, float]:
        """Sum of durations per span name (milliseconds)."""
        totals: dict[str, float] = {}
        for span in self.spans:
            totals[span.name] = totals.get(span.name, 0.0) + span.duration_ms
        return {k: round(v, 2) for k, v in totals.items()}

    def summary(self) -> dict[str, object]:
        return {
            "total_ms": round(self.total_ms, 2),
            "spans": len(self.spans),
            "counters": dict(self.counters),
            "stage_timings_ms": self.stage_timings(),
        }
