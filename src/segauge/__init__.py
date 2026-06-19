"""segauge — the honest gauge for medical image segmentation.

Geometrically-correct, DICOM-native evaluation metrics with confidence
intervals, per-lesion detection, and subgroup slicing.

    import segauge as sg

    result = sg.evaluate([
        sg.Case("patient_001", pred="pred.nii.gz", gt="gt.nii.gz",
                metadata={"site": "A"}),
    ])
    print(result.summary())
    result.to_html("report.html")
"""

from __future__ import annotations

from segauge.core import Case, EvaluationResult, evaluate
from segauge.io import load_mask, load_rtstruct
from segauge.metrics.detection import detection_scores
from segauge.metrics.distance import (
    assd,
    compute_surface_distances,
    hausdorff,
    hausdorff95,
    masd,
    nsd,
    surface_metrics,
)
from segauge.metrics.overlap import confusion, dice, iou
from segauge.stats import bootstrap_ci
from segauge.types import Estimate

__version__ = "0.1.0.dev0"

__all__ = [
    "Case",
    "Estimate",
    "EvaluationResult",
    "__version__",
    "assd",
    "bootstrap_ci",
    "compute_surface_distances",
    "confusion",
    "detection_scores",
    "dice",
    "evaluate",
    "hausdorff",
    "hausdorff95",
    "iou",
    "load_mask",
    "load_rtstruct",
    "masd",
    "nsd",
    "surface_metrics",
]
