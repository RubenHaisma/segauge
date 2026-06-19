"""Orchestration: evaluate a dataset of (pred, gt) cases into a report.

This is the layer a user actually calls. It runs every metric per case,
aggregates with bootstrap confidence intervals, and can slice every metric by
subgroup so you can see *where* a model fails, not just an average that hides
it.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from segauge.io import MaskSource, Spacing, load_mask
from segauge.metrics.detection import detection_scores
from segauge.metrics.distance import surface_metrics
from segauge.metrics.overlap import dice, iou
from segauge.stats import bootstrap_ci
from segauge.types import Estimate

OVERLAP_METRICS = ("dice", "iou")
DISTANCE_METRICS = ("hd", "hd95", "assd", "masd", "nsd")
DETECTION_METRICS = ("det_f1", "det_precision", "det_recall")
ALL_METRICS = OVERLAP_METRICS + DISTANCE_METRICS + DETECTION_METRICS

# "up" = higher is better, "down" = lower is better. Used for report rendering.
METRIC_DIRECTION = {
    "dice": "up",
    "iou": "up",
    "nsd": "up",
    "det_f1": "up",
    "det_precision": "up",
    "det_recall": "up",
    "hd": "down",
    "hd95": "down",
    "assd": "down",
    "masd": "down",
}

META_PREFIX = "meta."


@dataclass
class Case:
    """One evaluation case: a prediction, a ground truth, and its metadata."""

    case_id: str
    pred: MaskSource
    gt: MaskSource
    spacing: float | Spacing | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Per-case metric rows plus the config needed to aggregate them."""

    rows: list[dict[str, object]]
    metric_names: list[str]
    metadata_keys: list[str]
    confidence: float = 0.95
    n_resamples: int = 2000
    seed: int = 0

    def _values(self, metric: str, rows: Sequence[dict[str, object]]) -> list[float]:
        return [float(r[metric]) for r in rows if metric in r]

    def _ci(self, values: Sequence[float]) -> Estimate:
        return bootstrap_ci(
            values,
            confidence=self.confidence,
            n_resamples=self.n_resamples,
            seed=self.seed,
        )

    def summary(self) -> dict[str, Estimate]:
        """Aggregate estimate (with CI) for each metric over all cases."""
        return {m: self._ci(self._values(m, self.rows)) for m in self.metric_names}

    def by_subgroup(self, key: str) -> dict[object, dict[str, Estimate]]:
        """Per-metric estimates broken down by a metadata field."""
        col = f"{META_PREFIX}{key}"
        if key not in self.metadata_keys:
            raise KeyError(f"unknown metadata field {key!r}; have {self.metadata_keys}")
        groups: dict[object, list[dict[str, object]]] = defaultdict(list)
        for row in self.rows:
            groups[row.get(col)].append(row)
        return {
            value: {m: self._ci(self._values(m, rs)) for m in self.metric_names}
            for value, rs in groups.items()
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "n_cases": len(self.rows),
            "config": {
                "confidence": self.confidence,
                "n_resamples": self.n_resamples,
                "seed": self.seed,
            },
            "summary": {
                m: {"value": e.value, "ci_low": e.ci_low, "ci_high": e.ci_high}
                for m, e in self.summary().items()
            },
            "subgroups": {
                key: {
                    str(value): {
                        m: {"value": e.value, "ci_low": e.ci_low, "ci_high": e.ci_high}
                        for m, e in metrics.items()
                    }
                    for value, metrics in self.by_subgroup(key).items()
                }
                for key in self.metadata_keys
            },
            "per_case": self.rows,
        }

    def to_html(self, path: str, title: str = "segauge evaluation report") -> None:
        from segauge.report import render_html

        html = render_html(self, title=title)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(html)


def evaluate(
    cases: Iterable[Case],
    *,
    nsd_tolerance: float = 1.0,
    detection: bool = True,
    detection_iou: float = 0.1,
    label: int | None = None,
    segment_number: int = 1,
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 0,
) -> EvaluationResult:
    """Run all metrics over every case and return an aggregatable result.

    Ground-truth spacing is authoritative: if a case does not set ``spacing``
    explicitly, the spacing read from the ground-truth file is used for the
    distance metrics.
    """
    rows: list[dict[str, object]] = []
    metadata_keys: list[str] = []

    metric_names = list(OVERLAP_METRICS + DISTANCE_METRICS)
    if detection:
        metric_names += list(DETECTION_METRICS)

    for case in cases:
        gt_mask, gt_spacing = load_mask(
            case.gt, label=label, spacing=case.spacing, segment_number=segment_number
        )
        pred_mask, _ = load_mask(
            case.pred, label=label, spacing=case.spacing, segment_number=segment_number
        )
        spacing = case.spacing if case.spacing is not None else gt_spacing

        row: dict[str, object] = {"case_id": case.case_id}
        row["dice"] = dice(pred_mask, gt_mask)
        row["iou"] = iou(pred_mask, gt_mask)
        row.update(
            surface_metrics(
                pred_mask, gt_mask, spacing=spacing, nsd_tolerance=nsd_tolerance
            )
        )
        if detection:
            scores = detection_scores(pred_mask, gt_mask, iou_threshold=detection_iou)
            row["det_f1"] = scores.f1
            row["det_precision"] = scores.precision
            row["det_recall"] = scores.recall

        for meta_key, meta_value in case.metadata.items():
            row[f"{META_PREFIX}{meta_key}"] = meta_value
            if meta_key not in metadata_keys:
                metadata_keys.append(meta_key)

        rows.append(row)

    return EvaluationResult(
        rows=rows,
        metric_names=metric_names,
        metadata_keys=metadata_keys,
        confidence=confidence,
        n_resamples=n_resamples,
        seed=seed,
    )
