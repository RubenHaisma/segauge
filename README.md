# segauge

**DICOM-native evaluation for medical image segmentation: surface-mesh distance metrics, per-lesion detection, fairness slices, and a confidence interval on every number.**

[![PyPI](https://img.shields.io/pypi/v/segauge.svg)](https://pypi.org/project/segauge/)
[![Python](https://img.shields.io/pypi/pyversions/segauge.svg)](https://pypi.org/project/segauge/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

segauge computes Dice, IoU, Hausdorff distance (HD), **HD95**, average symmetric surface distance (**ASSD**), **Normalized Surface Dice (NSD)**, and **per-lesion detection F1** for 3D medical image segmentation, and puts a **bootstrap confidence interval** on every number. It reads **NIfTI, DICOM-SEG, RTSTRUCT, and NumPy** directly, so you can evaluate the output of nnU-Net, MONAI, or TotalSegmentator without a lossy conversion.

It exists because the metrics you report should be the metrics that are true, and today they often aren't.

```bash
pip install segauge
```

```python
import segauge as sg

result = sg.evaluate([
    sg.Case("patient_001", pred="pred.nii.gz", gt="gt.nii.gz", metadata={"scanner": "siemens"}),
    sg.Case("patient_002", pred="pred2.dcm",   gt="gt2.dcm",   metadata={"scanner": "ge"}),
])

print(result.summary())          # every metric with a 95% CI
print(result.by_subgroup("scanner"))   # where does the model quietly fail?
result.to_html("report.html")    # one self-contained report
```

Or from the command line:

```bash
segauge eval --pred preds/ --gt labels/ --metadata cases.csv --report report.html
```

## Why not just use MONAI or Metrics Reloaded?

segauge bundles four things no incumbent provides together: a **confidence interval** on every metric, **per-lesion detection**, **subgroup/fairness slicing**, and **native DICOM-SEG/RTSTRUCT** input.

It also computes distance metrics on a surface mesh rather than the voxel grid: it extracts the object surface with marching cubes **at true voxel spacing** and integrates over the surface, following the [MeshMetrics](https://arxiv.org/abs/2509.05670) method (Podobnik & Vrtovec, 2025). In our own benchmark (`benchmarks/mesh_vs_grid.py`) this measurably reduces **HD95** error under anisotropic, thick-slice spacing, where grid rasterization hurts most; for mean distance (ASSD) on smooth shapes the two methods are comparable. A formal validation suite against the MeshMetrics reference is in progress; we report what the benchmark shows, not more.

| | segauge | MONAI | Metrics Reloaded | seg-metrics | DeepMind surface-distance |
|---|:---:|:---:|:---:|:---:|:---:|
| Surface-mesh distances (vs voxel grid) | ✅ | ❌ | ❌ | ❌ | ❌ |
| Confidence intervals | ✅ | ❌ | ❌ | ❌ | ❌ |
| Per-lesion detection F1 | ✅ | partial | ✅ | ❌ | ❌ |
| Subgroup / fairness slicing | ✅ | ❌ | ❌ | ❌ | ❌ |
| DICOM-SEG / RTSTRUCT native | ✅ | partial | ❌ | ❌ | ❌ |
| `pip install` | ✅ | ✅ | ❌ | ✅ | ✅ |

## Metrics

- **Overlap:** Dice (DSC), IoU (Jaccard) — exact, integer-counted.
- **Surface distance:** Hausdorff (HD), HD95, ASSD, MASD, Normalized Surface Dice (NSD) — mesh-based, spacing-aware, area-weighted.
- **Per-lesion detection:** precision, recall, F1 via connected-component matching, so you can answer "did it find the tumor?" not just "how much voxel overlap?"
- Every aggregate carries a deterministic **bootstrap confidence interval**.

## Inputs

NIfTI (`.nii`, `.nii.gz`), DICOM-SEG, RTSTRUCT, NumPy arrays, and `.npy`. Voxel spacing is read from the file header and used for the distance metrics, so a segmentation from a clinical pipeline is evaluated as-is.

## FAQ

**How do I compute HD95 correctly in Python?** `segauge.surface_metrics(pred, gt, spacing)` returns HD, HD95, ASSD, MASD, and NSD computed on the surface mesh, not the voxel grid.

**Can it evaluate DICOM-SEG / RTSTRUCT directly?** Yes. `pip install segauge[dicom]` and pass a `.dcm` SEG, or use `segauge.load_rtstruct(series_dir, rtstruct, roi_name)`.

**Does it work with nnU-Net / TotalSegmentator / MONAI outputs?** Yes. Point it at the NIfTI (or DICOM) they produce.

**Why a confidence interval?** A Dice of 0.85 on 12 cases is not the same claim as 0.85 on 1200. segauge makes the difference visible.

**2D images?** Overlap and detection metrics work in any dimension; surface-distance metrics are 3D in v0.1 (2D is planned for v0.2).

## The segauge benchmark

[**segauge-benchmark**](https://github.com/RubenHaisma/segauge-benchmark) is an independent, reproducible leaderboard for medical image segmentation models, built on segauge. It runs real models (TotalSegmentator, MONAI, MOOSE, CT-FM) on public data and reports every score with a confidence interval, a **ranking-stability** test, and a pairwise significance check, so you can tell a real lead from sampling noise. segauge 0.2.0 adds the `ranking_stability` and `paired_significance` functions that power it.

## Status

Released (`0.2.0`), built in the open. segauge is an evaluation tool for developers and researchers. It is **not a medical device** and produces no diagnosis.

## References

- Maier-Hein, Reinke et al. *Metrics Reloaded.* Nature Methods (2024). https://www.nature.com/articles/s41592-023-02151-z
- Podobnik & Vrtovec. *MeshMetrics.* arXiv:2509.05670 (2025). https://arxiv.org/abs/2509.05670

## License

Apache-2.0.
