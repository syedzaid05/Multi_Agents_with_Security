"""Prompt-injection detection and untrusted-content isolation.

Two defences are provided:

1. :func:`detect_injection` - heuristic regex scan for common jailbreak /
   instruction-override phrasings, returning graded :class:`Violation`s.
2. :func:`wrap_untrusted` - fences external text (Wikipedia extracts, web
   snippets) behind nonce-guarded delimiters after stripping any forged copies
   of those delimiters, so scraped content cannot "break out" and issue
   instructions.

This is defence-in-depth, not a silver bullet; the system prompt also instructs
models to treat fenced content as data only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .models import Severity, Violation

__all__ = [
    "INJECTION_PATTERNS",
    "detect_injection",
    "sanitize_untrusted",
    "wrap_untrusted",
]

UNTRUSTED_START = "<<<UNTRUSTED_SOURCE_START>>>"
UNTRUSTED_END = "<<<UNTRUSTED_SOURCE_END>>>"


@dataclass(frozen=True)
class _Pattern:
    code: str
    regex: re.Pattern[str]
    severity: Severity
    message: str


def _compile(patterns: list[tuple[str, str, Severity, str]]) -> list[_Pattern]:
    return [
        _Pattern(code, re.compile(rx, re.IGNORECASE | re.DOTALL), sev, msg)
        for code, rx, sev, msg in patterns
    ]


INJECTION_PATTERNS: list[_Pattern] = _compile(
    [
        (
            "instruction_override",
            r"\b(ignore|disregard|forget|override|bypass)\b[^.\n]{0,40}?"
            r"\b(previous|prior|above|earlier|all|any)\b[^.\n]{0,20}?\b"
            r"(instruction|instructions|prompt|prompts|rules|direction|directions)\b",
            Severity.HIGH,
            "Attempt to override prior instructions.",
        ),
        (
            "system_prompt_exfiltration",
            r"\b(reveal|show|print|repeat|expose|leak|display)\b[^.\n]{0,40}?"
            r"\b(system|developer|hidden|initial|original)\b[^.\n]{0,20}?"
            r"\b(prompt|message|instruction|instructions|rules)\b",
            Severity.HIGH,
            "Attempt to exfiltrate the system prompt.",
        ),
        (
            "role_hijack",
            r"\b(you are now|from now on,? you|act as|pretend to be|roleplay as|"
            r"new persona|your new (role|identity))\b",
            Severity.MEDIUM,
            "Attempt to reassign the model's role/persona.",
        ),
        (
            "jailbreak",
            r"\b(jailbreak|do anything now|dan mode|developer mode|"
            r"unrestricted mode|without any restrictions|no restrictions)\b",
            Severity.HIGH,
            "Known jailbreak phrasing.",
        ),
        (
            "delimiter_escape",
            r"(<<<\s*untrusted_source_(start|end)\s*>>>|```\s*(system|end))",
            Severity.HIGH,
            "Attempt to forge untrusted-content delimiters.",
        ),
        (
            "tool_abuse",
            r"\b(execute|run)\b[^.\n]{0,30}\b(shell|command|cmd|powershell|bash|"
            r"os\.system|subprocess)\b",
            Severity.HIGH,
            "Attempt to trigger command execution.",
        ),
    ]
)


def detect_injection(text: str, *, location: str = "input") -> list[Violation]:
    """Return violations for any injection patterns found in ``text``."""
    if not text:
        return []
    violations: list[Violation] = []
    for pattern in INJECTION_PATTERNS:
        match = pattern.regex.search(text)
        if match:
            violations.append(
                Violation(
                    code=pattern.code,
                    message=pattern.message,
                    severity=pattern.severity,
                    location=location,
                )
            )
    return violations


def sanitize_untrusted(text: str) -> str:
    """Strip forged delimiter tokens so external text cannot escape its fence."""
    cleaned = text.replace(UNTRUSTED_START, "").replace(UNTRUSTED_END, "")
    # Also neutralise case/whitespace variants that could slip past an exact match.
    cleaned = re.sub(
        r"<<<\s*untrusted_source_(start|end)\s*>>>",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned.strip()


def wrap_untrusted(text: str, source: str) -> str:
    """Fence external text as inert data for inclusion in a prompt."""
    safe = sanitize_untrusted(text)
    return (
        f"{UNTRUSTED_START} source={source!r}\n"
        f"{safe}\n"
        f"{UNTRUSTED_END}"
    )
