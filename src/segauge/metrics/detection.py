"""Per-lesion detection metrics.

Voxel overlap answers "how much did it overlap?". Clinicians ask "did it find
the tumor?". This module labels connected components and matches them one-to-one
by IoU, so you get per-lesion precision / recall / F1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class DetectionScores:
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return 1.0 if denom == 0 else self.tp / denom

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return 1.0 if denom == 0 else self.tp / denom

    @property
    def f1(self) -> float:
        denom = 2 * self.tp + self.fp + self.fn
        return 1.0 if denom == 0 else 2 * self.tp / denom


def _component_ious(
    pred: npt.NDArray[np.bool_],
    gt: npt.NDArray[np.bool_],
    connectivity: int | None,
) -> tuple[npt.NDArray[np.float64], int, int]:
    """IoU matrix between predicted and ground-truth connected components."""
    from scipy import ndimage

    structure = None
    if connectivity is not None:
        structure = ndimage.generate_binary_structure(pred.ndim, connectivity)

    pred_lbl, n_pred = ndimage.label(pred, structure=structure)
    gt_lbl, n_gt = ndimage.label(gt, structure=structure)

    if n_pred == 0 or n_gt == 0:
        return np.zeros((n_pred, n_gt), dtype=np.float64), n_pred, n_gt

    # intersection counts via a single scatter-add over the label pair grid
    inter = np.zeros((n_pred + 1, n_gt + 1), dtype=np.int64)
    np.add.at(inter, (pred_lbl.ravel(), gt_lbl.ravel()), 1)
    inter = inter[1:, 1:]  # drop background row/col

    pred_sizes = np.bincount(pred_lbl.ravel(), minlength=n_pred + 1)[1:]
    gt_sizes = np.bincount(gt_lbl.ravel(), minlength=n_gt + 1)[1:]

    union = pred_sizes[:, None] + gt_sizes[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
    return iou.astype(np.float64), n_pred, n_gt


def detection_scores(
    pred: npt.ArrayLike,
    gt: npt.ArrayLike,
    iou_threshold: float = 0.1,
    connectivity: int | None = None,
) -> DetectionScores:
    """Greedy one-to-one lesion matching by IoU.

    A predicted component and a ground-truth component may match if their IoU
    is at least ``iou_threshold``. Matches are assigned greedily, highest IoU
    first, one prediction to one ground-truth lesion. Matched ground-truth
    lesions are true positives; unmatched predictions are false positives;
    unmatched ground-truth lesions are false negatives.
    """
    pred_arr = np.asarray(pred).astype(bool)
    gt_arr = np.asarray(gt).astype(bool)
    if pred_arr.shape != gt_arr.shape:
        raise ValueError(
            f"pred and gt must have the same shape, got "
            f"{pred_arr.shape} and {gt_arr.shape}"
        )

    iou, n_pred, n_gt = _component_ious(pred_arr, gt_arr, connectivity)

    matched_pred: set[int] = set()
    matched_gt: set[int] = set()
    if n_pred and n_gt:
        # candidate pairs above threshold, sorted by IoU descending
        pi, gi = np.where(iou >= iou_threshold)
        order = np.argsort(iou[pi, gi])[::-1]
        for idx in order:
            p, g = int(pi[idx]), int(gi[idx])
            if p in matched_pred or g in matched_gt:
                continue
            matched_pred.add(p)
            matched_gt.add(g)

    tp = len(matched_gt)
    fp = n_pred - len(matched_pred)
    fn = n_gt - len(matched_gt)
    return DetectionScores(tp=tp, fp=fp, fn=fn)
