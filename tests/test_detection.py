from __future__ import annotations

import numpy as np
import pytest

from segauge.metrics.detection import detection_scores


def _two_lesions(shape=(40, 40)):
    m = np.zeros(shape, dtype=bool)
    m[5:10, 5:10] = True
    m[25:32, 25:32] = True
    return m


def test_perfect_detection():
    gt = _two_lesions()
    pred = _two_lesions()
    s = detection_scores(pred, gt)
    assert (s.tp, s.fp, s.fn) == (2, 0, 0)
    assert s.f1 == pytest.approx(1.0)


def test_missed_lesion_is_false_negative():
    gt = _two_lesions()
    pred = np.zeros((40, 40), dtype=bool)
    pred[5:10, 5:10] = True  # only the first lesion
    s = detection_scores(pred, gt)
    assert (s.tp, s.fp, s.fn) == (1, 0, 1)
    assert s.recall == pytest.approx(0.5)
    assert s.precision == pytest.approx(1.0)


def test_spurious_lesion_is_false_positive():
    gt = np.zeros((40, 40), dtype=bool)
    gt[5:10, 5:10] = True
    pred = _two_lesions()  # extra lesion the gt does not have
    s = detection_scores(pred, gt)
    assert (s.tp, s.fp, s.fn) == (1, 1, 0)
    assert s.precision == pytest.approx(0.5)
    assert s.recall == pytest.approx(1.0)


def test_below_threshold_overlap_does_not_match():
    gt = np.zeros((40, 40), dtype=bool)
    pred = np.zeros((40, 40), dtype=bool)
    gt[5:15, 5:15] = True  # 100 voxels
    pred[14:24, 14:24] = True  # touches gt at a single voxel -> tiny IoU
    s = detection_scores(pred, gt, iou_threshold=0.1)
    assert (s.tp, s.fp, s.fn) == (0, 1, 1)


def test_empty_pred_recall_zero():
    gt = _two_lesions()
    pred = np.zeros((40, 40), dtype=bool)
    s = detection_scores(pred, gt)
    assert (s.tp, s.fp, s.fn) == (0, 0, 2)
    assert s.recall == pytest.approx(0.0)


def test_both_empty_is_perfect():
    empty = np.zeros((40, 40), dtype=bool)
    s = detection_scores(empty, empty)
    assert (s.tp, s.fp, s.fn) == (0, 0, 0)
    assert s.f1 == pytest.approx(1.0)
