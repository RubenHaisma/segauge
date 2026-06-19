"""Evaluate NIfTI segmentations and write a report.

python examples/evaluate_nifti.py
"""

from __future__ import annotations

import numpy as np

import segauge as sg


def main() -> None:
    # In practice, pass file paths: sg.Case("p1", pred="pred.nii.gz", gt="gt.nii.gz").
    # Here we build two synthetic 3D cases inline so the example runs anywhere.
    gt = np.zeros((40, 40, 40), dtype=bool)
    gt[10:30, 10:30, 10:30] = True
    pred_good = gt.copy()
    pred_rough = np.zeros((40, 40, 40), dtype=bool)
    pred_rough[12:30, 10:30, 10:30] = True  # one face off by 2 voxels

    sp = (1.0, 1.0, 1.0)
    cases = [
        sg.Case("good", pred_good, gt, spacing=sp, metadata={"site": "A"}),
        sg.Case("rough", pred_rough, gt, spacing=sp, metadata={"site": "B"}),
    ]
    result = sg.evaluate(cases, nsd_tolerance=1.0)

    for name, est in result.summary().items():
        print(f"{name:14s} {est}")

    print("\nby site:")
    for site, metrics in result.by_subgroup("site").items():
        dice = metrics["dice"].value
        hd95 = metrics["hd95"].value
        print(f"  {site}: dice {dice:.3f}, hd95 {hd95:.3f}")


if __name__ == "__main__":
    main()
