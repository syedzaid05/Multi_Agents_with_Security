"""Shared guardrail result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

__all__ = ["GuardrailResult", "Severity", "Violation"]


class Severity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    @property
    def weight(self) -> float:
        return {"low": 0.1, "medium": 0.35, "high": 0.7}[self.value]


@dataclass(slots=True)
class Violation:
    code: str
    message: str
    severity: Severity = Severity.LOW
    location: str = "input"

    def as_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity.value,
            "location": self.location,
        }


@dataclass
class GuardrailResult:
    """Outcome of a guardrail check.

    ``text`` is the (possibly sanitised/truncated) text that downstream stages
    should use. When ``allowed`` is ``False`` the caller must not proceed.
    """

    allowed: bool
    text: str
    violations: list[Violation] = field(default_factory=list)
    details: dict[str, object] = field(default_factory=dict)

    @property
    def risk_score(self) -> float:
        score = sum(v.severity.weight for v in self.violations)
        return round(min(1.0, score), 3)

    @property
    def highest_severity(self) -> Severity | None:
        if not self.violations:
            return None
        order = {Severity.LOW: 0, Severity.MEDIUM: 1, Severity.HIGH: 2}
        return max(self.violations, key=lambda v: order[v.severity]).severity

    @property
    def blocked(self) -> bool:
        return not self.allowed

    def as_dict(self) -> dict[str, object]:
        return {
            "allowed": self.allowed,
            "risk_score": self.risk_score,
            "violations": [v.as_dict() for v in self.violations],
            "details": self.details,
        }
