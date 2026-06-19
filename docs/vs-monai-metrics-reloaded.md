# segauge vs MONAI, Metrics Reloaded, seg-metrics, and surface-distance

If you are looking for a Python library to evaluate medical image segmentation, here is an honest comparison of segauge with the common alternatives.

| | segauge | MONAI | Metrics Reloaded | seg-metrics | DeepMind surface-distance |
|---|:---:|:---:|:---:|:---:|:---:|
| Dice / IoU | ✅ | ✅ | ✅ | ✅ | ✅ |
| HD95 / ASSD / NSD | ✅ | ✅ | ✅ | ✅ | ✅ |
| Surface-mesh distances (vs voxel grid) | ✅ | ❌ | ❌ | ❌ | ❌ |
| Confidence intervals | ✅ | ❌ | ❌ | ❌ | ❌ |
| Per-lesion detection F1 | ✅ | partial | ✅ | ❌ | ❌ |
| Subgroup / fairness slicing | ✅ | ❌ | ❌ | ❌ | ❌ |
| DICOM-SEG / RTSTRUCT input | ✅ | partial | ❌ | ❌ | ❌ |
| One-command CLI + HTML report | ✅ | ❌ | ❌ | ❌ | ❌ |
| `pip install` | ✅ | ✅ | ❌ (git only) | ✅ | ✅ |

## What segauge adds

The metric *menu* is similar everywhere; the standard for *which* metrics to report is set by [Metrics Reloaded](https://www.nature.com/articles/s41592-023-02151-z). What segauge bundles that the others do not, together, is the reporting layer:

- **Confidence intervals** on every metric, by default.
- **Per-lesion detection F1**, so you can report lesion-level sensitivity.
- **Subgroup / fairness slicing**, to find where a model fails (scanner, site, demographic).
- **Native DICOM-SEG / RTSTRUCT input**, to evaluate what a clinical pipeline produced.
- A single `segauge eval` command and a self-contained HTML report.

## On distance metrics

segauge computes distance metrics on a surface mesh at true voxel spacing, following the [MeshMetrics](https://arxiv.org/abs/2509.05670) method, rather than on the voxel grid. In our own benchmark (`benchmarks/mesh_vs_grid.py`) this measurably reduces **HD95** error under anisotropic, thick-slice spacing; for mean distance (ASSD) on smooth shapes the methods are comparable. We report what the benchmark shows, not more: distance metrics are implementation-sensitive on curved surfaces, and segauge is transparent about it.

## When to use something else

If you only need Dice in a training loop, MONAI's metrics are fine and already in your stack. If you are choosing which metrics are appropriate for your task, read Metrics Reloaded first. segauge is for the step after: producing honest, reportable, DICOM-aware evaluation results.
