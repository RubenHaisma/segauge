from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Estimate:
    """A point estimate with a confidence interval."""

    value: float
    ci_low: float
    ci_high: float

    def __str__(self) -> str:
        return f"{self.value:.4g} [{self.ci_low:.4g}, {self.ci_high:.4g}]"
