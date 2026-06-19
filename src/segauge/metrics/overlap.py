"""Overlap metrics: exact, integer-counted, no discretization games."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def _validate_pair(
    pred: npt.ArrayLike, gt: npt.ArrayLike
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    pred_arr = np.asarray(pred).astype(bool)
    gt_arr = np.asarray(gt).astype(bool)
    if pred_arr.shape != gt_arr.shape:
        raise ValueError(
            f"pred and gt must have the same shape, got "
            f"{pred_arr.shape} and {gt_arr.shape}"
        )
    return pred_arr, gt_arr


def confusion(pred: npt.ArrayLike, gt: npt.ArrayLike) -> tuple[int, int, int, int]:
    """Return (tp, fp, fn, tn) voxel counts."""
    pred_arr, gt_arr = _validate_pair(pred, gt)
    tp = int(np.count_nonzero(pred_arr & gt_arr))
    fp = int(np.count_nonzero(pred_arr & ~gt_arr))
    fn = int(np.count_nonzero(~pred_arr & gt_arr))
    tn = int(np.count_nonzero(~pred_arr & ~gt_arr))
    return tp, fp, fn, tn


def dice(pred: npt.ArrayLike, gt: npt.ArrayLike) -> float:
    """Dice similarity coefficient (== F1 on voxels).

    Two empty masks agree perfectly and return 1.0 (Metrics Reloaded
    convention), since "correctly predicting nothing where there is nothing"
    is a perfect score, not an undefined one.
    """
    tp, fp, fn, _ = confusion(pred, gt)
    denom = 2 * tp + fp + fn
    if denom == 0:
        return 1.0
    return 2 * tp / denom


def iou(pred: npt.ArrayLike, gt: npt.ArrayLike) -> float:
    """Intersection over union (Jaccard index)."""
    tp, fp, fn, _ = confusion(pred, gt)
    denom = tp + fp + fn
    if denom == 0:
        return 1.0
    return tp / denom
