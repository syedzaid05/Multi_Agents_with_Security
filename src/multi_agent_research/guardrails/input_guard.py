"""Input guardrails.

Validates and sanitises a user query before any agent sees it:

* rejects empty / whitespace-only input,
* enforces a maximum length (guards against context-stuffing and cost blowups),
* strips control characters,
* blocks high-severity prompt-injection attempts and flags medium ones.
"""

from __future__ import annotations

import re

from .injection import detect_injection
from .models import GuardrailResult, Severity, Violation

__all__ = ["InputGuard"]

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class InputGuard:
    """Stateless-ish input validator (configuration is immutable per instance)."""

    def __init__(
        self,
        *,
        max_chars: int = 4000,
        min_chars: int = 2,
        block_on_high: bool = True,
    ) -> None:
        self.max_chars = max_chars
        self.min_chars = min_chars
        self.block_on_high = block_on_high

    def check(self, text: str, *, location: str = "input") -> GuardrailResult:
        violations: list[Violation] = []
        raw = text or ""

        cleaned = _CONTROL_CHARS.sub(" ", raw).strip()
        if cleaned != raw.strip():
            violations.append(
                Violation(
                    "control_chars_stripped",
                    "Control characters were removed from the input.",
                    Severity.LOW,
                    location,
                )
            )

        if len(cleaned) < self.min_chars:
            return GuardrailResult(
                allowed=False,
                text=cleaned,
                violations=violations
                + [
                    Violation(
                        "empty_input",
                        "The query is empty or too short to research.",
                        Severity.HIGH,
                        location,
                    )
                ],
                details={"reason": "empty_input"},
            )

        truncated = False
        if len(cleaned) > self.max_chars:
            cleaned = cleaned[: self.max_chars]
            truncated = True
            violations.append(
                Violation(
                    "input_too_long",
                    f"Input exceeded {self.max_chars} characters and was truncated.",
                    Severity.MEDIUM,
                    location,
                )
            )

        injection = detect_injection(cleaned, location=location)
        violations.extend(injection)

        has_high = any(v.severity is Severity.HIGH for v in violations)
        allowed = not (self.block_on_high and has_high)

        return GuardrailResult(
            allowed=allowed,
            text=cleaned,
            violations=violations,
            details={
                "truncated": truncated,
                "injection_detected": bool(injection),
                "blocked_reason": "prompt_injection" if has_high and not allowed else None,
            },
        )
