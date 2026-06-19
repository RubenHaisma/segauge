"""Bootstrap confidence intervals.

A Dice of 0.85 on 12 cases is not the same claim as 0.85 on 1200. segauge
attaches a confidence interval to every aggregate metric so the reader can
tell the difference. CIs are deterministic given a seed, because a trust tool
must be reproducible.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from segauge.types import Estimate


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
