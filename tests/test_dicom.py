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
