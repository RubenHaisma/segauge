"""Loaders: NumPy, NIfTI, DICOM-SEG, and RTSTRUCT to (mask, spacing).

Spacing is always returned in physical units, aligned to the array axes, so
downstream distance metrics are correct regardless of source format. This is
the point of being DICOM-native: a segmentation produced by a clinical
pipeline can be evaluated as-is, not after a lossy NIfTI conversion.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt

Spacing = tuple[float, ...]
MaskSource = np.ndarray | str | Path


def _binarize(arr: npt.ArrayLike, label: int | None) -> npt.NDArray[np.bool_]:
    arr = np.asarray(arr)
    if label is None:
        return arr != 0
    return arr == label


def _coerce_spacing(spacing: float | Spacing | None, ndim: int) -> Spacing:
    if spacing is None:
        return tuple(1.0 for _ in range(ndim))
    if np.isscalar(spacing):
        return tuple(float(spacing) for _ in range(ndim))  # type: ignore[arg-type]
    sp = tuple(float(s) for s in spacing)  # type: ignore[arg-type]
    if len(sp) != ndim:
        raise ValueError(f"spacing has {len(sp)} entries but mask is {ndim}-D")
    return sp


def load_mask(
    source: MaskSource,
    *,
    label: int | None = None,
    spacing: float | Spacing | None = None,
    segment_number: int = 1,
) -> tuple[npt.NDArray[np.bool_], Spacing]:
    """Load a binary mask and its voxel spacing.

    Args:
        source: a NumPy array, or a path to ``.npy``, NIfTI (``.nii`` /
            ``.nii.gz``), or a DICOM-SEG file.
        label: if given, keep only voxels equal to this label; otherwise any
            non-zero voxel is foreground.
        spacing: override the spacing. If ``None``, it is read from the file
            header (NIfTI / DICOM-SEG) or defaults to 1.0 per axis (arrays /
            ``.npy``).
        segment_number: which segment to extract from a DICOM-SEG.

    Returns:
        ``(mask, spacing)`` with ``mask`` boolean and ``spacing`` aligned to
        the array axes.
    """
    if isinstance(source, np.ndarray):
        mask = _binarize(source, label)
        return mask, _coerce_spacing(spacing, mask.ndim)

    path = Path(source)
    name = path.name.lower()
    if name.endswith(".npy"):
        arr = np.load(path)
        mask = _binarize(arr, label)
        return mask, _coerce_spacing(spacing, mask.ndim)
    if name.endswith((".nii", ".nii.gz")):
        return _load_nifti(path, label, spacing)
    return _load_dicom_seg(path, segment_number, spacing)


def _load_nifti(
    path: Path, label: int | None, spacing: float | Spacing | None
) -> tuple[npt.NDArray[np.bool_], Spacing]:
    import nibabel as nib  # lazy

    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    mask = _binarize(data, label)
    if spacing is None:
        zooms = tuple(float(z) for z in img.header.get_zooms()[: data.ndim])
        return mask, zooms
    return mask, _coerce_spacing(spacing, data.ndim)


def _load_dicom_seg(
    path: Path, segment_number: int, spacing: float | Spacing | None
) -> tuple[npt.NDArray[np.bool_], Spacing]:
    import highdicom as hd  # lazy

    seg = hd.seg.segread(str(path))
    try:
        volume = seg.get_volume(segment_numbers=[segment_number], combine_segments=True)
    except RuntimeError as exc:
        raise ValueError(
            f"could not reconstruct a regular 3D volume from DICOM-SEG {path!r}: "
            f"{exc}. The segmentation frames may be irregularly spaced or "
            f"non-uniquely indexed. Please open an issue with a sample file."
        ) from exc
    mask = np.asarray(volume.array) != 0
    if spacing is None:
        return mask, tuple(float(s) for s in volume.spacing)
    return mask, _coerce_spacing(spacing, mask.ndim)


def load_rtstruct(
    series_dir: str | Path,
    rtstruct_path: str | Path,
    roi_name: str,
    *,
    spacing: float | Spacing | None = None,
) -> tuple[npt.NDArray[np.bool_], Spacing]:
    """Load one RTSTRUCT ROI as a mask on its referenced image grid.

    RTSTRUCT stores contours, not a raster, so it needs the referenced DICOM
    image series to define the voxel grid. ``spacing`` is read from that series
    unless overridden.
    """
    import pydicom  # lazy
    from rt_utils import RTStructBuilder

    rt = RTStructBuilder.create_from(
        dicom_series_path=str(series_dir), rt_struct_path=str(rtstruct_path)
    )
    mask = rt.get_roi_mask_by_name(roi_name).astype(bool)

    if spacing is not None:
        return mask, _coerce_spacing(spacing, mask.ndim)

    # Derive spacing from the referenced series GEOMETRY. Slice spacing comes
    # from the actual ImagePositionPatient differences, not the
    # SpacingBetweenSlices / SliceThickness tags, which are often missing,
    # stale, or inconsistent with the true slice positions (and silently
    # corrupt distance metrics when they are).
    datasets = [
        pydicom.dcmread(str(p), stop_before_pixels=True)
        for p in sorted(Path(series_dir).glob("*"))
        if p.is_file()
    ]
    datasets = [d for d in datasets if hasattr(d, "ImagePositionPatient")]
    if not datasets:
        raise ValueError(f"no DICOM image slices found in {series_dir!r}")

    row_sp, col_sp = (float(v) for v in datasets[0].PixelSpacing)
    if len(datasets) >= 2:
        positions = np.array(
            [[float(x) for x in d.ImagePositionPatient] for d in datasets]
        )
        positions = positions[np.argsort(positions[:, 2])]
        slice_sp = float(np.median(np.linalg.norm(np.diff(positions, axis=0), axis=1)))
    else:
        slice_sp = float(
            getattr(datasets[0], "SpacingBetweenSlices", None)
            or datasets[0].SliceThickness
        )
    # rt-utils returns mask as (rows, cols, slices)
    return mask, (row_sp, col_sp, slice_sp)
