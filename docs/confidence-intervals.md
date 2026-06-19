# Segmentation Dice (and HD95) with confidence intervals

A mean Dice of 0.85 over 12 cases is a very different claim from 0.85 over 1200 cases, but papers usually report just the number. segauge attaches a **bootstrap confidence interval** to every aggregate metric so the reader can tell the difference, and so you can report results honestly.

```python
import segauge as sg

result = sg.evaluate([
    sg.Case("p1", pred="p1.nii.gz", gt="g1.nii.gz"),
    sg.Case("p2", pred="p2.nii.gz", gt="g2.nii.gz"),
    # ...
])

for name, est in result.summary().items():
    print(f"{name}: {est.value:.3f} [{est.ci_low:.3f}, {est.ci_high:.3f}]")
# dice: 0.910 [0.882, 0.937]
# hd95: 3.20 [2.10, 4.80]
# ...
```

Every metric, Dice and IoU and HD95 and ASSD and NSD and detection F1, comes with a 95% interval. The interval is **deterministic** given a seed, because a tool you trust to report results must be reproducible.

## Confidence interval for your own values

```python
import segauge as sg

per_case_dice = [0.88, 0.91, 0.79, 0.93, 0.85]
est = sg.bootstrap_ci(per_case_dice)
print(est)   # 0.872 [0.812, 0.918]
```

## Configure it

```python
result = sg.evaluate(cases, confidence=0.95, n_resamples=2000, seed=0)
```

See also: [Per-lesion detection F1](per-lesion-detection-f1.md), [the API reference](api.md).
