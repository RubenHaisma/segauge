"""Compute metrics for a single pair of masks (Dice, HD95, ASSD, NSD, detection).

python examples/single_pair.py
"""

from __future__ import annotations

import numpy as np

import segauge as sg


def main() -> None:
    gt = np.zeros((40, 40, 40), dtype=bool)
    gt[10:30, 10:30, 10:30] = True
    pred = np.zeros((40, 40, 40), dtype=bool)
    pred[12:30, 10:30, 10:30] = True

    spacing = (1.0, 1.0, 3.0)  # anisotropic, thick-slice

    print(f"Dice: {sg.dice(pred, gt):.4f}")
    print(f"IoU:  {sg.iou(pred, gt):.4f}")

    surface = sg.surface_metrics(pred, gt, spacing=spacing, nsd_tolerance=1.0)
    for k, v in surface.items():
        print(f"{k:5s} {v:.4f}")

    det = sg.detection_scores(pred, gt)
    print(f"detection F1: {det.f1:.4f}  (tp={det.tp}, fp={det.fp}, fn={det.fn})")


if __name__ == "__main__":
    main()
