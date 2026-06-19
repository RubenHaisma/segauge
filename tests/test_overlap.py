from __future__ import annotations

import numpy as np
import pytest

from segauge.metrics.overlap import confusion, dice, iou


def test_dice_identical_is_one():
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:8, 2:8] = True
    assert dice(mask, mask) == pytest.approx(1.0)
    assert iou(mask, mask) == pytest.approx(1.0)


def test_dice_disjoint_is_zero():
    a = np.zeros((10, 10), dtype=bool)
    b = np.zeros((10, 10), dtype=bool)
    a[0:3, 0:3] = True
    b[7:10, 7:10] = True
    assert dice(a, b) == pytest.approx(0.0)
    assert iou(a, b) == pytest.approx(0.0)


def test_dice_half_overlap_known_value():
    # gt = 100 voxels, pred = same 100 shifted to share 50 -> Dice = 2*50/(100+100)
    gt = np.zeros((10, 20), dtype=bool)
    pred = np.zeros((10, 20), dtype=bool)
    gt[:, 0:10] = True
    pred[:, 5:15] = True
    # intersection cols 5..9 -> 10*5 = 50; each has 100
    assert dice(pred, gt) == pytest.approx(2 * 50 / (100 + 100))
    assert iou(pred, gt) == pytest.approx(50 / 150)


def test_both_empty_is_perfect():
    empty = np.zeros((8, 8), dtype=bool)
    assert dice(empty, empty) == pytest.approx(1.0)
    assert iou(empty, empty) == pytest.approx(1.0)


def test_one_empty_is_zero():
    empty = np.zeros((8, 8), dtype=bool)
    full = np.ones((8, 8), dtype=bool)
    assert dice(full, empty) == pytest.approx(0.0)
    assert iou(full, empty) == pytest.approx(0.0)


def test_confusion_counts():
    gt = np.array([[1, 1], [0, 0]], dtype=bool)
    pred = np.array([[1, 0], [1, 0]], dtype=bool)
    tp, fp, fn, tn = confusion(pred, gt)
    assert (tp, fp, fn, tn) == (1, 1, 1, 1)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="same shape"):
        dice(np.zeros((4, 4), dtype=bool), np.zeros((4, 5), dtype=bool))
