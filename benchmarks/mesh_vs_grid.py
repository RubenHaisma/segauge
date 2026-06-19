"""Mesh-correct vs grid-based surface distance: the thesis, with numbers.

Grid-based tools (MONAI, seg-metrics, pymia, MedPy, DeepMind surface-distance)
measure distances between voxel surfaces. segauge measures them on a marching-
cubes surface mesh at true spacing. On geometry with a *known* answer, the gap
shows up, and it widens with anisotropic spacing, exactly where it matters
clinically (thick-slice CT/MR).

Run:  uv run python benchmarks/mesh_vs_grid.py
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from segauge.metrics.distance import surface_metrics


def sphere(shape, center, radius, spacing):
    grids = np.ogrid[tuple(slice(0, s) for s in shape)]
    dist2 = sum(
        ((g - c) * sp) ** 2 for g, c, sp in zip(grids, center, spacing, strict=True)
    )
    return dist2 <= radius**2


def _surface(mask):
    return mask & ~ndimage.binary_erosion(mask)


def grid_metrics(pred, gt, spacing, tol):
    """Classic grid surface-distance, the way incumbent tools compute it."""
    sa, sb = _surface(pred), _surface(gt)
    dt_to_b = ndimage.distance_transform_edt(~sb, sampling=spacing)
    dt_to_a = ndimage.distance_transform_edt(~sa, sampling=spacing)
    d_ab, d_ba = dt_to_b[sa], dt_to_a[sb]
    allv = np.concatenate([d_ab, d_ba])
    nsd = ((d_ab <= tol).sum() + (d_ba <= tol).sum()) / (len(d_ab) + len(d_ba))
    return {
        "hd95": float(np.percentile(allv, 95)),
        "assd": float(allv.mean()),
        "nsd": float(nsd),
    }


def run(label, shape, spacing, r_inner, r_outer):
    center = tuple(s / 2 for s in shape)
    inner = sphere(shape, center, r_inner, spacing)
    outer = sphere(shape, center, r_outer, spacing)
    truth = r_outer - r_inner  # true symmetric surface distance, in physical units
    mesh = surface_metrics(inner, outer, spacing=spacing, nsd_tolerance=1.0)
    grid = grid_metrics(inner, outer, spacing, tol=1.0)
    print(f"\n{label}  (spacing {spacing}, true ASSD = {truth:.1f})")
    print(f"  {'metric':6} {'mesh':>10} {'grid':>10} {'truth':>10}")
    for m, t in [("assd", truth), ("hd95", truth)]:
        print(f"  {m:6} {mesh[m]:>10.3f} {grid[m]:>10.3f} {t:>10.3f}")
    err_mesh = abs(mesh["assd"] - truth)
    err_grid = abs(grid["assd"] - truth)
    print(f"  ASSD abs error:  mesh {err_mesh:.3f}  vs  grid {err_grid:.3f}")


if __name__ == "__main__":
    run("isotropic 1mm", (90, 90, 90), (1.0, 1.0, 1.0), 30, 33)
    run("anisotropic 1x1x3mm", (90, 90, 34), (1.0, 1.0, 3.0), 30, 33)
