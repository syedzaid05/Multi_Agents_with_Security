"""Output guardrails.

Checks a generated answer before it is returned to the user:

* blocks empty output,
* truncates over-long output,
* blocks obvious system-prompt leakage (heuristic + known secret snippets),
* warns when an answer cites no sources (citation coverage is a requirement of
  this system, but a warning keeps the pipeline usable in demo mode).
"""

from __future__ import annotations

import re

from .models import GuardrailResult, Severity, Violation

__all__ = ["OutputGuard"]

_LEAK_PATTERNS = [
    re.compile(r"you are (a|an) (helpful|expert|ai|assistant)", re.IGNORECASE),
    re.compile(r"\bsystem prompt\b", re.IGNORECASE),
    re.compile(r"<<<\s*untrusted_source_start\s*>>>", re.IGNORECASE),
]
_CITATION_MARKER = re.compile(r"\[\d+\]")


class OutputGuard:
    def __init__(
        self,
        *,
        max_chars: int = 12000,
        require_citations: bool = True,
        secrets: list[str] | None = None,
    ) -> None:
        self.max_chars = max_chars
        self.require_citations = require_citations
        self.secrets = [s for s in (secrets or []) if s]

    def check(self, text: str, *, location: str = "output") -> GuardrailResult:
        violations: list[Violation] = []
        cleaned = (text or "").strip()

        if not cleaned:
            return GuardrailResult(
                allowed=False,
                text="",
                violations=[
                    Violation(
                        "empty_output",
                        "The model returned an empty answer.",
                        Severity.HIGH,
                        location,
                    )
                ],
                details={"reason": "empty_output"},
            )

        truncated = False
        if len(cleaned) > self.max_chars:
            cleaned = cleaned[: self.max_chars].rstrip() + "\n\n[output truncated]"
            truncated = True
            violations.append(
                Violation(
                    "output_too_long",
                    f"Output exceeded {self.max_chars} characters and was truncated.",
                    Severity.LOW,
                    location,
                )
            )

        leaked = [p.pattern for p in _LEAK_PATTERNS if p.search(cleaned)]
        leaked_secret = [s for s in self.secrets if s in cleaned]
        if leaked or leaked_secret:
            violations.append(
                Violation(
                    "prompt_leakage",
                    "Output appears to leak system prompt content or secrets.",
                    Severity.HIGH,
                    location,
                )
            )

        has_citations = bool(_CITATION_MARKER.search(cleaned))
        if self.require_citations and not has_citations:
            violations.append(
                Violation(
                    "missing_citations",
                    "Answer contains no [n] source citations.",
                    Severity.LOW,
                    location,
                )
            )

        blocked = any(v.severity is Severity.HIGH for v in violations)
        return GuardrailResult(
            allowed=not blocked,
            text=cleaned,
            violations=violations,
            details={
                "truncated": truncated,
                "has_citations": has_citations,
                "blocked_reason": "prompt_leakage" if blocked else None,
            },
        )
