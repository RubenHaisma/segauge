from __future__ import annotations

import numpy as np
import pytest

from segauge.core import Case, evaluate


def _cube(shape=(24, 24, 24)):
    m = np.zeros(shape, dtype=bool)
    m[4:14, 4:14, 4:14] = True
    return m


def test_evaluate_identical_is_perfect():
    m = _cube()
    res = evaluate([Case("a", m, m, spacing=1.0)], n_resamples=200)
    summary = res.summary()
    assert summary["dice"].value == pytest.approx(1.0)
    assert summary["iou"].value == pytest.approx(1.0)
    assert summary["assd"].value == pytest.approx(0.0, abs=1e-6)
    assert summary["det_f1"].value == pytest.approx(1.0)


def test_no_detection_drops_detection_metrics():
    m = _cube()
    res = evaluate([Case("a", m, m, spacing=1.0)], detection=False, n_resamples=50)
    assert "det_f1" not in res.summary()


def test_subgroups():
    m = _cube()
    cases = [
        Case("a", m, m, spacing=1.0, metadata={"site": "X"}),
        Case("b", m, m, spacing=1.0, metadata={"site": "Y"}),
    ]
    res = evaluate(cases, n_resamples=200)
    sub = res.by_subgroup("site")
    assert set(sub) == {"X", "Y"}
    assert sub["X"]["dice"].value == pytest.approx(1.0)
    with pytest.raises(KeyError, match="unknown metadata"):
        res.by_subgroup("nope")


def test_to_dict_shape():
    m = _cube()
    res = evaluate(
        [Case("a", m, m, spacing=1.0, metadata={"site": "X"})], n_resamples=100
    )
    d = res.to_dict()
    assert d["n_cases"] == 1
    assert "dice" in d["summary"]
    assert "site" in d["subgroups"]
    assert len(d["per_case"]) == 1


def test_ground_truth_spacing_is_authoritative():
    # imperfect prediction so distance is non-zero and spacing matters
    gt = np.zeros((40, 40, 40), dtype=bool)
    pred = np.zeros((40, 40, 40), dtype=bool)
    gt[10:30, 10:30, 10:30] = True
    pred[12:30, 10:30, 10:30] = True  # one face shifted by 2 voxels
    res1 = evaluate([Case("a", pred, gt, spacing=1.0)], n_resamples=50)
    res2 = evaluate([Case("a", pred, gt, spacing=2.0)], n_resamples=50)
    assert res2.summary()["assd"].value > res1.summary()["assd"].value
