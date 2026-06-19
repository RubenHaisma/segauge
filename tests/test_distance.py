from __future__ import annotations

import numpy as np
import pytest

from segauge.metrics.distance import (
    compute_surface_distances,
    surface_metrics,
)


def test_identical_sphere_zero_distance(make_sphere):
    sph = make_sphere((50, 50, 50), (25, 25, 25), 15)
    m = surface_metrics(sph, sph, spacing=1.0, nsd_tolerance=1.0)
    assert m["assd"] == pytest.approx(0.0, abs=1e-6)
    assert m["hd"] == pytest.approx(0.0, abs=1e-6)
    assert m["nsd"] == pytest.approx(1.0, abs=1e-6)


def test_concentric_spheres_recover_radial_offset(make_sphere):
    # surfaces differ by a uniform radial 3 voxels -> ASSD ~ 3
    inner = make_sphere((60, 60, 60), (30, 30, 30), 15)
    outer = make_sphere((60, 60, 60), (30, 30, 30), 18)
    m = surface_metrics(inner, outer, spacing=1.0)
    assert m["assd"] == pytest.approx(3.0, abs=0.6)
    assert m["hd"] == pytest.approx(3.0, abs=1.0)
    assert m["masd"] == pytest.approx(3.0, abs=0.6)


def test_spacing_scales_distance(make_sphere):
    # same geometry, isotropic spacing 2.0 -> distances double
    inner = make_sphere((60, 60, 60), (30, 30, 30), 15)
    outer = make_sphere((60, 60, 60), (30, 30, 30), 18)
    m = surface_metrics(inner, outer, spacing=2.0)
    assert m["assd"] == pytest.approx(6.0, abs=1.2)


def test_nsd_tolerance_behaviour(make_sphere):
    inner = make_sphere((60, 60, 60), (30, 30, 30), 15)
    outer = make_sphere((60, 60, 60), (30, 30, 30), 18)
    # tolerance below the 3-voxel gap -> little surface agrees
    nsd_tight = surface_metrics(inner, outer, nsd_tolerance=1.0)["nsd"]
    # tolerance above the gap -> all surface agrees
    nsd_loose = surface_metrics(inner, outer, nsd_tolerance=5.0)["nsd"]
    assert nsd_tight < 0.5
    assert nsd_loose == pytest.approx(1.0, abs=1e-6)


def test_both_empty_is_perfect():
    empty = np.zeros((20, 20, 20), dtype=bool)
    m = surface_metrics(empty, empty)
    assert m == {"hd": 0.0, "hd95": 0.0, "assd": 0.0, "masd": 0.0, "nsd": 1.0}


def test_one_empty_is_worst(make_sphere):
    empty = np.zeros((40, 40, 40), dtype=bool)
    sph = make_sphere((40, 40, 40), (20, 20, 20), 10)
    m = surface_metrics(sph, empty)
    assert m["assd"] == float("inf")
    assert m["hd95"] == float("inf")
    assert m["nsd"] == 0.0


def test_compute_surface_distances_empty_raises():
    empty = np.zeros((20, 20, 20), dtype=bool)
    full = np.ones((20, 20, 20), dtype=bool)
    with pytest.raises(ValueError, match="empty"):
        compute_surface_distances(full, empty)


def test_2d_mask_raises_clear_error():
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:10, 5:10] = True
    with pytest.raises(ValueError, match="require 3D masks"):
        surface_metrics(mask, mask)


def test_negative_tolerance_raises(make_sphere):
    sph = make_sphere((40, 40, 40), (20, 20, 20), 10)
    from segauge.metrics.distance import compute_surface_distances, nsd

    sd = compute_surface_distances(sph, sph)
    with pytest.raises(ValueError, match="non-negative"):
        nsd(sd, -1.0)
