# segauge: medical image segmentation metrics in Python

**segauge** is a Python library for evaluating medical image segmentation. It computes Dice, IoU, Hausdorff distance (HD), **HD95**, average symmetric surface distance (**ASSD**), **Normalized Surface Dice (NSD)**, and **per-lesion detection F1**, and puts a **bootstrap confidence interval** on every number. It reads **NIfTI, DICOM-SEG, RTSTRUCT, and NumPy** directly, so you can evaluate the output of nnU-Net, MONAI, or TotalSegmentator without a lossy conversion.

```bash
pip install segauge
```

```python
import segauge as sg

result = sg.evaluate([
    sg.Case("patient_001", pred="pred.nii.gz", gt="gt.nii.gz", metadata={"scanner": "siemens"}),
])
print(result.summary())            # every metric with a 95% confidence interval
print(result.by_subgroup("scanner"))   # see where a model quietly fails
result.to_html("report.html")      # one self-contained report
```

## What you can do with it

- [Compute HD95 in Python](compute-hd95-in-python.md) — 95th-percentile Hausdorff distance, on a surface mesh at true voxel spacing.
- [Normalized Surface Dice (NSD)](normalized-surface-dice-nsd.md) — surface agreement within a clinical tolerance.
- [Evaluate a DICOM-SEG](evaluate-dicom-seg.md) — read a DICOM segmentation object directly.
- [Dice with confidence intervals](confidence-intervals.md) — because 0.85 on 12 cases ≠ 0.85 on 1200.
- [Per-lesion detection F1](per-lesion-detection-f1.md) — "did it find the tumor?", not just voxel overlap.
- [How segauge compares to MONAI and Metrics Reloaded](vs-monai-metrics-reloaded.md).

## Why segauge

It bundles four things no existing library provides together: confidence intervals on every metric, per-lesion detection, subgroup/fairness slicing, and native DICOM-SEG/RTSTRUCT input. Distance metrics are computed on a surface mesh at true voxel spacing rather than on the voxel grid, following the [MeshMetrics](https://arxiv.org/abs/2509.05670) method; in benchmarking this reduces HD95 error under anisotropic, thick-slice spacing. See the [comparison](vs-monai-metrics-reloaded.md).

segauge is an evaluation tool for developers and researchers. It is not a medical device.
