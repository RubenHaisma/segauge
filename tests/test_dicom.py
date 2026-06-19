"""Real-world DICOM-SEG loading.

Validates the loader against an actual DICOM-SEG file (see tests/data/README),
not synthetic data: the headline "DICOM-native" claim has to hold on a file a
clinical tool would produce. Skipped if highdicom is not installed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("highdicom")

from segauge.io import load_mask  # noqa: E402

FIXTURE = Path(__file__).parent / "data" / "seg_image_ct_binary.dcm"


def test_load_real_dicom_seg():
    mask, spacing = load_mask(FIXTURE)
    assert mask.dtype == bool
    assert mask.shape == (3, 16, 16)
    assert int(mask.sum()) == 638
    # anisotropic real-world spacing, read from the file header
    assert spacing[0] == pytest.approx(1.25, abs=1e-2)
    assert spacing[1] == pytest.approx(0.488, abs=1e-2)
    assert spacing[2] == pytest.approx(0.488, abs=1e-2)


def test_loader_matches_highdicom_get_volume():
    """The loader must not mangle what highdicom decodes."""
    import highdicom as hd

    mask, spacing = load_mask(FIXTURE)
    seg = hd.seg.segread(str(FIXTURE))
    vol = seg.get_volume(
        segment_numbers=[seg.segment_numbers[0]], combine_segments=True
    )
    assert mask.shape == vol.array.shape
    assert np.array_equal(mask, np.asarray(vol.array) != 0)
    assert tuple(round(s, 5) for s in spacing) == tuple(
        round(float(s), 5) for s in vol.spacing
    )


def _make_ct_series(series_dir, n=4, slice_spacing=3.0):
    """Write a small synthetic CT series with known geometry."""
    import copy

    from pydicom import dcmread
    from pydicom.data import get_testdata_file
    from pydicom.uid import generate_uid

    template = dcmread(get_testdata_file("CT_small.dcm"))
    rows, cols = int(template.Rows), int(template.Columns)
    study, series, frame = generate_uid(), generate_uid(), generate_uid()
    for k in range(n):
        ds = copy.deepcopy(template)
        ds.SOPInstanceUID = generate_uid()
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        ds.StudyInstanceUID = study
        ds.SeriesInstanceUID = series
        ds.FrameOfReferenceUID = frame
        ds.InstanceNumber = k + 1
        ds.PixelSpacing = [1.0, 1.0]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.ImagePositionPatient = [0.0, 0.0, float(k * slice_spacing)]
        ds.SliceThickness = slice_spacing
        ds.PixelData = np.zeros((rows, cols), np.int16).tobytes()
        ds.save_as(series_dir / f"ct_{k}.dcm")
    return rows, cols, n


def test_rtstruct_round_trip(tmp_path):
    pytest.importorskip("rt_utils")
    from rt_utils import RTStructBuilder

    from segauge.io import load_rtstruct

    series_dir = tmp_path / "series"
    series_dir.mkdir()
    # CT_small carries a stale SpacingBetweenSlices; the true slice gap is 3.0
    rows, cols, n = _make_ct_series(series_dir, n=4, slice_spacing=3.0)

    mask = np.zeros((rows, cols, n), dtype=bool)
    mask[40:90, 40:90, 1:3] = True
    rt = RTStructBuilder.create_new(dicom_series_path=str(series_dir))
    rt.add_roi(mask=mask, name="lesion")
    rt.save(str(tmp_path / "rt.dcm"))

    loaded, spacing = load_rtstruct(str(series_dir), str(tmp_path / "rt.dcm"), "lesion")
    assert loaded.shape == (rows, cols, n)
    dice = 2 * (loaded & mask).sum() / (loaded.sum() + mask.sum())
    assert dice > 0.95  # contouring is lossy; axis-aligned cubes round-trip well
    # slice spacing must come from the true geometry (3.0), not the stale tag
    assert spacing == pytest.approx((1.0, 1.0, 3.0))
