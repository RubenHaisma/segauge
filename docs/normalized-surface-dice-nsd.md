# Normalized Surface Dice (NSD) in Python

**Normalized Surface Dice (NSD)**, sometimes called surface Dice, measures what fraction of the two surfaces lie within a given **tolerance** of each other. Unlike volumetric Dice, it answers a boundary question: "is the contour correct to within the clinically acceptable margin?" NSD = 1.0 means every part of the surface agrees to within the tolerance.

```python
import segauge as sg

metrics = sg.surface_metrics(pred_mask, gt_mask, spacing=(1.0, 1.0, 3.0), nsd_tolerance=1.0)
print(metrics["nsd"])   # fraction of surface within 1.0 mm
```

The `nsd_tolerance` is in physical units (the same units as `spacing`). Pick it from the clinically acceptable boundary error for your task (often 1–2 mm).

## Surface-based, spacing-aware

segauge computes NSD on a marching-cubes surface at true voxel spacing and weights by surface area, rather than counting boundary voxels on the grid. That makes the tolerance a real physical distance, which matters for anisotropic, thick-slice data.

## With confidence intervals, over a dataset

```python
import segauge as sg

result = sg.evaluate(
    [sg.Case("p1", pred="pred.nii.gz", gt="gt.nii.gz")],
    nsd_tolerance=1.0,
)
print(result.summary()["nsd"])   # NSD with a 95% confidence interval
```

See also: [HD95](compute-hd95-in-python.md), [the API reference](api.md).
