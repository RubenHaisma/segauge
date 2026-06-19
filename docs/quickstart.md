# Quickstart

## Install

```bash
pip install segauge
# DICOM-SEG / RTSTRUCT support:
pip install "segauge[dicom]"
```

## Evaluate a dataset (Python)

```python
import segauge as sg

cases = [
    sg.Case("patient_001", pred="preds/001.nii.gz", gt="labels/001.nii.gz",
            metadata={"scanner": "siemens", "site": "A"}),
    sg.Case("patient_002", pred="preds/002.nii.gz", gt="labels/002.nii.gz",
            metadata={"scanner": "ge", "site": "B"}),
]

result = sg.evaluate(cases, nsd_tolerance=1.0)

print(result.summary())              # {'dice': 0.91 [0.88, 0.94], 'hd95': ...}
print(result.by_subgroup("scanner")) # per-scanner breakdown
result.to_html("report.html")        # self-contained report
result.to_dict()                     # machine-readable results
```

## Evaluate from the command line

```bash
segauge eval --pred preds/ --gt labels/ \
    --metadata cases.csv \
    --report report.html --json results.json
```

`--pred` and `--gt` can be single files or directories (matched by filename). `cases.csv` needs a `case_id` column plus any columns you want to slice by.

## Single pair of masks

```python
import numpy as np
import segauge as sg

dice = sg.dice(pred_mask, gt_mask)
metrics = sg.surface_metrics(pred_mask, gt_mask, spacing=(1.0, 1.0, 3.0))
# -> {'hd': ..., 'hd95': ..., 'assd': ..., 'masd': ..., 'nsd': ...}
```

## Supported inputs

NIfTI (`.nii`, `.nii.gz`), DICOM-SEG, RTSTRUCT, NumPy arrays, and `.npy`. Voxel spacing is read from the file header and used for the distance metrics.
