"""Geometrically-correct, spacing-aware surface-distance metrics.

The whole reason segauge exists. Every widely-used library computes
distance-based metrics on the voxel grid, which bakes in discretization
error. segauge instead extracts the object surface with marching cubes at
true voxel spacing and integrates distances *over the surface*, area-weighted,
which converges to the continuous-domain value. Validated against the
MeshMetrics reference implementation (dev-only oracle).

References:
    Podobnik & Vrtovec, "MeshMetrics", arXiv:2509.05670 (2025)
    Maier-Hein et al., "Metrics Reloaded", Nature Methods (2024)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

Spacing = tuple[float, ...]

_NOT_3D = (
    "surface-distance metrics require 3D masks (medical volumes); got {ndim}D. "
    "2D distance support is planned for v0.2. Overlap (dice/iou) and detection "
    "metrics work in any dimension."
)


@dataclass(frozen=True)
class SurfaceDistances:
    """Area-weighted directed surface distances, both directions.

    ``d_pg`` are distances from surface elements of ``pred`` to the surface of
    ``gt``; ``w_pg`` are the corresponding surface-element areas (the weights
    that make the integral continuous rather than a vertex count). ``d_gp`` /
    ``w_gp`` are the reverse direction.
    """

    d_pg: npt.NDArray[np.float64]
    w_pg: npt.NDArray[np.float64]
    d_gp: npt.NDArray[np.float64]
    w_gp: npt.NDArray[np.float64]

    @property
    def both(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Concatenated symmetric (distance, weight) arrays."""
        return (
            np.concatenate([self.d_pg, self.d_gp]),
            np.concatenate([self.w_pg, self.w_gp]),
        )


def _normalize_spacing(spacing: float | Spacing, ndim: int) -> Spacing:
    if np.isscalar(spacing):
        return tuple(float(spacing) for _ in range(ndim))  # type: ignore[arg-type]
    spacing = tuple(float(s) for s in spacing)  # type: ignore[arg-type]
    if len(spacing) != ndim:
        raise ValueError(f"spacing has {len(spacing)} entries but mask is {ndim}-D")
    return spacing


def _extract_mesh(mask: npt.NDArray[np.bool_], spacing: Spacing):
    """Marching-cubes surface of a binary mask, in physical coordinates.

    The mask is zero-padded by one voxel so objects touching the array border
    still produce a closed surface.
    """
    import trimesh
    from skimage.measure import marching_cubes  # lazy: heavy import

    padded = np.pad(mask.astype(np.float32), 1, mode="constant", constant_values=0.0)
    verts, faces, _normals, _values = marching_cubes(
        np.ascontiguousarray(padded), level=0.5, spacing=spacing
    )
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def _surface_samples(mesh) -> npt.NDArray[np.float64]:
    """Dense point sampling of a surface: vertices plus face centroids.

    After the mesh is subdivided to ~one voxel, this is a dense, uniform point
    cloud covering the surface, suitable as a KD-tree target for nearest-point
    distance queries.
    """
    return np.vstack(
        [
            np.asarray(mesh.vertices, dtype=np.float64),
            np.asarray(mesh.triangles_center, dtype=np.float64),
        ]
    )


def _directed(
    mesh_from, tree_to
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Area-weighted distances from one surface to another.

    Distances are measured from the face centroids of ``mesh_from`` (the
    area-weighted quadrature points) to the nearest sampled point on the other
    surface, via a KD-tree (``tree_to``). Because both meshes are pre-subdivided
    to ~one voxel, the sampling is dense and uniform, so this matches the true
    point-to-surface distance to sub-voxel accuracy, which is the most precision
    that voxel-resolution input supports, while scaling to clinical-size volumes
    (exact point-to-triangle search does not).
    """
    centroids = np.asarray(mesh_from.triangles_center, dtype=np.float64)
    areas = np.asarray(mesh_from.area_faces, dtype=np.float64)
    dist, _ = tree_to.query(centroids, workers=-1)
    return np.asarray(dist, dtype=np.float64), areas


def compute_surface_distances(
    pred: npt.ArrayLike, gt: npt.ArrayLike, spacing: float | Spacing = 1.0
) -> SurfaceDistances:
    """Compute area-weighted symmetric surface distances between two masks.

    Raises ``ValueError`` if either mask is empty (no surface to measure);
    callers that want the graceful-degradation convention should use
    :func:`surface_metrics`.
    """
    pred_arr = np.asarray(pred).astype(bool)
    gt_arr = np.asarray(gt).astype(bool)
    if pred_arr.shape != gt_arr.shape:
        raise ValueError(
            f"pred and gt must have the same shape, got "
            f"{pred_arr.shape} and {gt_arr.shape}"
        )
    if pred_arr.ndim != 3:
        raise ValueError(_NOT_3D.format(ndim=pred_arr.ndim))
    if not pred_arr.any() or not gt_arr.any():
        raise ValueError(
            "cannot compute surface distances when a mask is empty; "
            "use surface_metrics() for the graceful-degradation convention"
        )
    from scipy.spatial import cKDTree

    sp = _normalize_spacing(spacing, pred_arr.ndim)
    # Subdivide to ~one voxel so the surface sampling is dense and uniform
    # everywhere (including flat faces, which marching cubes leaves as large
    # triangles), then measure nearest-point distances with a KD-tree.
    max_edge = min(sp)
    mesh_p = _extract_mesh(pred_arr, sp).subdivide_to_size(max_edge)
    mesh_g = _extract_mesh(gt_arr, sp).subdivide_to_size(max_edge)
    tree_p = cKDTree(_surface_samples(mesh_p))
    tree_g = cKDTree(_surface_samples(mesh_g))
    d_pg, w_pg = _directed(mesh_p, tree_g)
    d_gp, w_gp = _directed(mesh_g, tree_p)
    return SurfaceDistances(d_pg=d_pg, w_pg=w_pg, d_gp=d_gp, w_gp=w_gp)


def _weighted_quantile(
    values: npt.NDArray[np.float64], weights: npt.NDArray[np.float64], q: float
) -> float:
    """Weighted quantile via the interpolated empirical CDF."""
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cum = np.cumsum(weights) - 0.5 * weights
    total = weights.sum()
    if total == 0:
        return float("nan")
    cum /= total
    return float(np.interp(q, cum, values))


def hausdorff(sd: SurfaceDistances) -> float:
    """Maximum (100th-percentile) symmetric surface distance."""
    return float(max(sd.d_pg.max(), sd.d_gp.max()))


def hausdorff95(sd: SurfaceDistances) -> float:
    """95th-percentile symmetric surface distance (HD95), area-weighted."""
    values, weights = sd.both
    return _weighted_quantile(values, weights, 0.95)


def assd(sd: SurfaceDistances) -> float:
    """Average symmetric surface distance, area-weighted over both surfaces."""
    values, weights = sd.both
    return float(np.average(values, weights=weights))


def masd(sd: SurfaceDistances) -> float:
    """Mean average surface distance: mean of the two directed means.

    Distinct from ASSD when the two surfaces have very different areas
    (Metrics Reloaded keeps them separate, so we do too).
    """
    return float(
        0.5
        * (np.average(sd.d_pg, weights=sd.w_pg) + np.average(sd.d_gp, weights=sd.w_gp))
    )


def nsd(sd: SurfaceDistances, tolerance: float) -> float:
    """Normalized Surface Dice at ``tolerance`` (physical units), area-weighted.

    Fraction of the combined surface area that lies within ``tolerance`` of the
    other surface. NSD == 1.0 means every surface element agrees to within the
    clinically acceptable tolerance.
    """
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    values, weights = sd.both
    total = weights.sum()
    if total == 0:
        return float("nan")
    return float(weights[values <= tolerance].sum() / total)


def surface_metrics(
    pred: npt.ArrayLike,
    gt: npt.ArrayLike,
    spacing: float | Spacing = 1.0,
    nsd_tolerance: float = 1.0,
) -> dict[str, float]:
    """All surface metrics at once, with graceful degradation on empty masks.

    Conventions (Metrics Reloaded):
        * both masks empty  -> perfect agreement: distances 0, NSD 1.0
        * exactly one empty -> worst case: distances +inf, NSD 0.0
    """
    pred_arr = np.asarray(pred).astype(bool)
    gt_arr = np.asarray(gt).astype(bool)
    if pred_arr.ndim != 3:
        raise ValueError(_NOT_3D.format(ndim=pred_arr.ndim))
    pred_empty = not pred_arr.any()
    gt_empty = not gt_arr.any()

    if pred_empty and gt_empty:
        return {"hd": 0.0, "hd95": 0.0, "assd": 0.0, "masd": 0.0, "nsd": 1.0}
    if pred_empty or gt_empty:
        inf = float("inf")
        return {"hd": inf, "hd95": inf, "assd": inf, "masd": inf, "nsd": 0.0}

    sd = compute_surface_distances(pred_arr, gt_arr, spacing)
    return {
        "hd": hausdorff(sd),
        "hd95": hausdorff95(sd),
        "assd": assd(sd),
        "masd": masd(sd),
        "nsd": nsd(sd, nsd_tolerance),
    }
