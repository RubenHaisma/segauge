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


@dataclass(frozen=True)
class RankStat:
    """One model's ranking behaviour under case-resampling. Rank 1 is best.

    ``p_best`` is the fraction of bootstrap resamples in which this model ranked
    first; a high mean Dice with a low ``p_best`` means the lead is not robust to
    which cases happened to be in the test set.
    """

    name: str
    score: float
    p_best: float
    mean_rank: float
    rank_ci_low: float
    rank_ci_high: float

    def __str__(self) -> str:
        return (
            f"{self.name}: {self.score:.4g} "
            f"(P(rank 1)={self.p_best:.0%}, "
            f"rank {self.mean_rank:.1f} [{self.rank_ci_low:.0f}, "
            f"{self.rank_ci_high:.0f}])"
        )


@dataclass(frozen=True)
class RankingResult:
    """Ranking-stability of several models over aligned per-case scores."""

    higher_is_better: bool
    n_cases: int
    n_resamples: int
    stats: list[RankStat]
    metric: str = ""

    def __str__(self) -> str:
        head = f"ranking by {self.metric or 'metric'} (n={self.n_cases})"
        return "\n".join([head, *(f"  {s}" for s in self.stats)])


@dataclass(frozen=True)
class PairedComparison:
    """Bootstrap of the paired per-case difference between two models.

    ``distinguishable`` is False when the confidence interval of the mean
    per-case difference straddles zero, i.e. the two models are not
    statistically separable on this test set. This is the check that catches
    the common case of a reported "winner" that sits inside the runner-up's CI.
    """

    a: str
    b: str
    delta: float
    ci_low: float
    ci_high: float
    favored: str | None
    distinguishable: bool

    def __str__(self) -> str:
        if not self.distinguishable:
            return f"{self.a} vs {self.b}: indistinguishable (Δ={self.delta:.4g})"
        return (
            f"{self.a} vs {self.b}: {self.favored} better "
            f"(Δ={self.delta:.4g} [{self.ci_low:.4g}, {self.ci_high:.4g}])"
        )
