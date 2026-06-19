from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest

from segauge.io import load_mask


def test_load_array_defaults_to_unit_spacing():
    arr = np.zeros((5, 5), dtype=int)
    arr[1:3, 1:3] = 1
    mask, spacing = load_mask(arr)
    assert mask.dtype == bool
    assert mask.sum() == 4
    assert spacing == (1.0, 1.0)


def test_load_npy_with_label(tmp_path):
    arr = np.zeros((4, 4), dtype=int)
    arr[0, 0] = 2
    arr[1, 1] = 3
    path = tmp_path / "x.npy"
    np.save(path, arr)

    assert load_mask(path)[0].sum() == 2  # any non-zero
    assert load_mask(path, label=2)[0].sum() == 1
    assert load_mask(path, label=9)[0].sum() == 0


def test_load_nifti_reads_spacing_from_header(tmp_path):
    data = np.zeros((6, 6, 6), dtype=np.uint8)
    data[2:4, 2:4, 2:4] = 1
    img = nib.Nifti1Image(data, affine=np.diag([1.5, 2.0, 3.0, 1.0]))
    path = tmp_path / "m.nii.gz"
    nib.save(img, str(path))

    mask, spacing = load_mask(path)
    assert mask.sum() == 8
    assert spacing == pytest.approx((1.5, 2.0, 3.0))


def test_spacing_override_and_validation():
    arr = np.ones((3, 3), dtype=int)
    assert load_mask(arr, spacing=2.0)[1] == (2.0, 2.0)
    assert load_mask(arr, spacing=(1.0, 4.0))[1] == (1.0, 4.0)
    with pytest.raises(ValueError, match="entries"):
        load_mask(arr, spacing=(1.0, 2.0, 3.0))
