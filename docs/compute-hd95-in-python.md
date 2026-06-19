# How to compute HD95 (95th-percentile Hausdorff distance) in Python

**HD95** is the 95th percentile of the symmetric surface distances between a predicted and a ground-truth segmentation. It is preferred over the maximum Hausdorff distance (HD) because it is far less sensitive to a single outlier voxel, which is why Metrics Reloaded recommends it for most segmentation tasks.

With segauge:

```python
import segauge as sg

metrics = sg.surface_metrics(pred_mask, gt_mask, spacing=(1.0, 1.0, 3.0))
print(metrics["hd95"])   # 95th-percentile Hausdorff distance, in mm
print(metrics["hd"])     # maximum Hausdorff distance
print(metrics["assd"])   # average symmetric surface distance
```

`pred_mask` and `gt_mask` are boolean (or 0/1) 3D NumPy arrays. `spacing` is the physical voxel size in the same axis order as the arrays; it matters, because HD95 is a physical distance.

## Why segauge's HD95 is different

Most libraries compute Hausdorff distance on the **voxel grid**: they take the surface voxels and measure distances between voxel centres. That bakes in discretization error, which is largest for thick-slice (anisotropic) data.

segauge extracts the object surface with marching cubes at the true voxel spacing and measures distances over that surface, following the [MeshMetrics](https://arxiv.org/abs/2509.05670) method. In benchmarking on data with a known answer, this reduces HD95 error under anisotropic spacing.

## From files (NIfTI / DICOM-SEG)

```python
import segauge as sg

result = sg.evaluate([sg.Case("p1", pred="pred.nii.gz", gt="gt.nii.gz")])
print(result.summary()["hd95"])   # HD95 with a bootstrap confidence interval
```

## Confidence intervals

When you evaluate more than one case, every metric (including HD95) comes back with a [bootstrap confidence interval](confidence-intervals.md), so you can report HD95 honestly.

See also: [Normalized Surface Dice](normalized-surface-dice-nsd.md), [the API reference](api.md).
