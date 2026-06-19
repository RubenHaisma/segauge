# Evaluate a DICOM-SEG segmentation in Python

A **DICOM-SEG** (DICOM Segmentation object) is what clinical and research pipelines actually produce. segauge reads it directly, so you can evaluate it as-is without converting to NIfTI first (which can lose or alter the geometry).

```bash
pip install "segauge[dicom]"
```

```python
import segauge as sg

# read a DICOM-SEG to a boolean mask plus its physical voxel spacing
mask, spacing = sg.load_mask("segmentation.dcm")
print(mask.shape, spacing)

# pick a specific segment (DICOM-SEG can hold several)
mask, spacing = sg.load_mask("segmentation.dcm", segment_number=2)
```

## Evaluate prediction vs ground truth, both DICOM-SEG

```python
import segauge as sg

result = sg.evaluate([
    sg.Case("patient_001", pred="pred_seg.dcm", gt="gt_seg.dcm"),
])
print(result.summary())        # Dice, HD95, ASSD, NSD, detection F1, all with CIs
result.to_html("report.html")
```

The voxel spacing is read from the DICOM-SEG itself, so distance metrics (HD95, ASSD, NSD) are computed in real physical units.

## RTSTRUCT

For RTSTRUCT (contours, common in radiotherapy), point segauge at the structure set and its referenced image series:

```python
mask, spacing = sg.load_rtstruct("ct_series/", "structures.dcm", roi_name="GTV")
```

Slice spacing is computed from the actual slice positions (`ImagePositionPatient`), not the `SpacingBetweenSlices` tag, which is often stale or missing.

See also: [Quickstart](quickstart.md), [the API reference](api.md).
