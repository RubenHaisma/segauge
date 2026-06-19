"""Oracle tests against the MeshMetrics reference implementation.

These run only when MeshMetrics (a dev-only, git-installed package) is present;
they are skipped otherwise, so the normal suite and CI stay self-contained.

What they encode is deliberately honest:
  * on polyhedral (flat-faced) geometry, segauge and MeshMetrics agree closely;
  * on CURVED surfaces, mesh distance is implementation-sensitive: segauge
    (skimage marching cubes) and MeshMetrics (VTK) differ by ~12-17%, and on a
    sphere with a known analytic answer segauge is at least as close to the
    truth as the reference. There is no single "correct" number for discretized
    curved surfaces, and the test documents that rather than pretending.
"""

from __future__ import annotations

import numpy as np
import pytest

MeshMetrics = pytest.importorskip("MeshMetrics")

from segauge.metrics.distance import surface_metrics  # noqa: E402


def _reference(ref, pred, spacing, tau=1.0):
    dm = MeshMetrics.DistanceMetrics(verbose=False)
    dm.set_input(ref=ref.astype(bool), pred=pred.astype(bool), spacing=spacing)
    return {
        "hd": dm.hd(100.0),
        "hd95": dm.hd(95.0),
        "assd": dm.assd(),
        "masd": dm.masd(),
        "nsd": dm.nsd(tau),
    }


def _sphere(shape, center, radius):
    grids = np.ogrid[tuple(slice(0, s) for s in shape)]
    dist2 = sum((g - c) ** 2 for g, c in zip(grids, center, strict=True))
    return dist2 <= radius**2


def test_matches_reference_on_polyhedral_geometry():
    ref = np.zeros((40, 40, 40), dtype=bool)
    pred = np.zeros((40, 40, 40), dtype=bool)
    ref[10:30, 10:30, 10:30] = True
    pred[12:30, 10:30, 10:30] = True  # one flat face shifted by 2

    sg = surface_metrics(pred, ref, spacing=(1.0, 1.0, 1.0), nsd_tolerance=1.0)
    ref_m = _reference(ref, pred, (1.0, 1.0, 1.0))

    assert sg["hd"] == pytest.approx(ref_m["hd"], abs=1e-6)
    assert sg["hd95"] == pytest.approx(ref_m["hd95"], abs=1e-6)
    assert sg["assd"] == pytest.approx(ref_m["assd"], rel=0.05)
    assert sg["nsd"] == pytest.approx(ref_m["nsd"], abs=0.02)


def test_curved_surface_divergence_is_bounded_and_documented():
    # Concentric spheres: the true symmetric surface distance is exactly 3.0.
    shape, center = (60, 60, 60), (30, 30, 30)
    ref = _sphere(shape, center, 18)
    pred = _sphere(shape, center, 15)

    sg = surface_metrics(pred, ref, spacing=(1.0, 1.0, 1.0))["assd"]
    ref_assd = _reference(ref, pred, (1.0, 1.0, 1.0))["assd"]
    truth = 3.0

    # the two mesh implementations differ on curved surfaces, but bounded
    assert abs(sg - ref_assd) / ref_assd < 0.25
    # and segauge is at least as close to the analytic truth as the reference
    assert abs(sg - truth) <= abs(ref_assd - truth) + 1e-9
