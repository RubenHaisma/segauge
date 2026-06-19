from __future__ import annotations

import numpy as np
import pytest

from segauge.stats import bootstrap_ci


def test_constant_collapses_to_point():
    est = bootstrap_ci([0.8] * 20)
    assert est.value == pytest.approx(0.8)
    assert est.ci_low == pytest.approx(0.8)
    assert est.ci_high == pytest.approx(0.8)


def test_interval_brackets_point():
    rng = np.random.default_rng(1)
    values = rng.normal(0.85, 0.05, size=200)
    est = bootstrap_ci(values)
    assert est.ci_low <= est.value <= est.ci_high
    assert est.value == pytest.approx(float(np.mean(values)))


def test_deterministic_given_seed():
    values = [0.1, 0.5, 0.9, 0.3, 0.7]
    a = bootstrap_ci(values, seed=42)
    b = bootstrap_ci(values, seed=42)
    assert (a.value, a.ci_low, a.ci_high) == (b.value, b.ci_low, b.ci_high)


def test_single_value_collapses():
    est = bootstrap_ci([0.42])
    assert est.value == pytest.approx(0.42)
    assert est.ci_low == est.ci_high == pytest.approx(0.42)


def test_empty_is_nan():
    est = bootstrap_ci([])
    assert np.isnan(est.value)
    assert np.isnan(est.ci_low)
    assert np.isnan(est.ci_high)


def test_drops_nonfinite():
    est = bootstrap_ci([1.0, 2.0, 3.0, float("inf"), float("nan")])
    assert est.value == pytest.approx(2.0)


def test_invalid_confidence_raises():
    with pytest.raises(ValueError, match="confidence"):
        bootstrap_ci([1.0, 2.0], confidence=1.5)
