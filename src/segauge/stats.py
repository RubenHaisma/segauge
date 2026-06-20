"""Bootstrap confidence intervals.

A Dice of 0.85 on 12 cases is not the same claim as 0.85 on 1200. segauge
attaches a confidence interval to every aggregate metric so the reader can
tell the difference. CIs are deterministic given a seed, because a trust tool
must be reproducible.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
import numpy.typing as npt
from scipy.stats import rankdata

from segauge.types import Estimate, PairedComparison, RankingResult, RankStat


def bootstrap_ci(
    values: npt.ArrayLike,
    statistic: Callable[[npt.NDArray[np.float64]], float] = np.mean,
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 0,
    drop_nonfinite: bool = True,
) -> Estimate:
    """Percentile bootstrap CI for an aggregate statistic over per-case values.

    Args:
        values: per-case metric values (e.g. one Dice per patient).
        statistic: aggregate to bootstrap; defaults to the mean.
        confidence: two-sided confidence level (0.95 -> 2.5%/97.5%).
        n_resamples: number of bootstrap resamples.
        seed: RNG seed; identical inputs always give identical CIs.
        drop_nonfinite: drop NaN/inf before resampling (e.g. inf distances
            from empty-mask cases). The point estimate is over the kept values.

    Returns:
        An :class:`Estimate` of (value, ci_low, ci_high). With fewer than two
        usable values the interval collapses to the point estimate.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")

    arr = np.asarray(values, dtype=np.float64).ravel()
    if drop_nonfinite:
        arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        nan = float("nan")
        return Estimate(value=nan, ci_low=nan, ci_high=nan)

    point = float(statistic(arr))
    if arr.size == 1:
        return Estimate(value=point, ci_low=point, ci_high=point)

    rng = np.random.default_rng(seed)
    n = arr.size
    boot = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        boot[i] = statistic(arr[rng.integers(0, n, n)])

    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(boot, [alpha, 1.0 - alpha])
    return Estimate(value=point, ci_low=float(lo), ci_high=float(hi))


def ranking_stability(
    scores_by_model: Mapping[str, npt.ArrayLike],
    *,
    higher_is_better: bool = True,
    n_resamples: int = 2000,
    seed: int = 0,
    metric: str = "",
) -> RankingResult:
    """How stable is the ranking of several models to which cases were tested?

    Maier-Hein et al. (2018) showed that removing a single test case changed the
    rank of most teams in a majority of segmentation challenges. This resamples
    the *shared* set of cases (a paired bootstrap: the same resampled case
    indices are applied to every model, because they were all scored on the same
    cases) and records each model's rank distribution.

    Args:
        scores_by_model: ``{model_name: per_case_values}``. Every model must
            have the same number of cases, aligned in the same order.
        higher_is_better: True for Dice/NSD, False for HD95/ASSD.
        n_resamples: number of bootstrap resamples.
        seed: RNG seed; identical inputs always give identical rankings.
        metric: optional label carried into the result for display.

    Returns:
        A :class:`RankingResult` whose ``stats`` are sorted best-first by point
        score. Cases where *any* model is non-finite are dropped so that all
        models are ranked on a common, comparable set of cases.
    """
    names = list(scores_by_model)
    if not names:
        raise ValueError("need at least one model")

    rows = [np.asarray(scores_by_model[n], dtype=np.float64).ravel() for n in names]
    lengths = {r.size for r in rows}
    if len(lengths) != 1:
        raise ValueError(
            f"all models must have the same number of cases; got {lengths}"
        )

    mat = np.vstack(rows)  # (n_models, n_cases)
    finite_cols = np.all(np.isfinite(mat), axis=0)
    mat = mat[:, finite_cols]
    n_models, n_cases = mat.shape

    point = mat.mean(axis=1) if n_cases else np.full(n_models, np.nan)
    order_key = (lambda v: -v) if higher_is_better else (lambda v: v)

    if n_cases == 0:
        stats = [
            RankStat(names[j], float("nan"), float("nan"), float("nan"),
                     float("nan"), float("nan"))
            for j in range(n_models)
        ]
        return RankingResult(higher_is_better, 0, n_resamples, stats, metric)

    rng = np.random.default_rng(seed)
    ranks_accum = np.empty((n_resamples, n_models), dtype=np.float64)
    best_credit = np.zeros(n_models, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n_cases, n_cases)
        means = mat[:, idx].mean(axis=1)
        ranks = rankdata(order_key(means), method="average")  # 1 = best
        ranks_accum[i] = ranks
        best = ranks == ranks.min()
        best_credit += best / best.sum()  # split credit across ties

    p_best = best_credit / n_resamples
    mean_rank = ranks_accum.mean(axis=0)
    lo = np.quantile(ranks_accum, 0.025, axis=0)
    hi = np.quantile(ranks_accum, 0.975, axis=0)

    order = np.argsort(order_key(point))
    stats = [
        RankStat(
            name=names[j],
            score=float(point[j]),
            p_best=float(p_best[j]),
            mean_rank=float(mean_rank[j]),
            rank_ci_low=float(lo[j]),
            rank_ci_high=float(hi[j]),
        )
        for j in order
    ]
    return RankingResult(higher_is_better, n_cases, n_resamples, stats, metric)


def paired_significance(
    a_scores: npt.ArrayLike,
    b_scores: npt.ArrayLike,
    *,
    a_name: str = "A",
    b_name: str = "B",
    higher_is_better: bool = True,
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 0,
) -> PairedComparison:
    """Are two models statistically separable on this test set?

    Bootstraps the mean of the per-case difference ``a - b``. If the confidence
    interval straddles zero the two models are not distinguishable on these
    cases. This is the check behind the finding that the reported "winner" is
    often inside the runner-up's interval.

    The two score arrays must be aligned per case (same cases, same order).
    """
    a = np.asarray(a_scores, dtype=np.float64).ravel()
    b = np.asarray(b_scores, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError("paired comparison needs aligned per-case scores")

    mask = np.isfinite(a) & np.isfinite(b)
    diffs = a[mask] - b[mask]
    est = bootstrap_ci(
        diffs, confidence=confidence, n_resamples=n_resamples, seed=seed,
        drop_nonfinite=False,
    )
    distinguishable = not (est.ci_low <= 0.0 <= est.ci_high)
    favored: str | None = None
    if distinguishable:
        a_is_better = (est.value > 0) == higher_is_better
        favored = a_name if a_is_better else b_name

    return PairedComparison(
        a=a_name,
        b=b_name,
        delta=float(est.value),
        ci_low=float(est.ci_low),
        ci_high=float(est.ci_high),
        favored=favored,
        distinguishable=distinguishable,
    )
