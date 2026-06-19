from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pytest

Shape = Sequence[int]


def _sphere(shape: Shape, center: Sequence[float], radius: float) -> np.ndarray:
    grids = np.ogrid[tuple(slice(0, s) for s in shape)]
    dist2 = sum((g - c) ** 2 for g, c in zip(grids, center, strict=False))
    return dist2 <= radius**2


def _cube(shape: Shape, lo: Sequence[int], hi: Sequence[int]) -> np.ndarray:
    mask = np.zeros(tuple(shape), dtype=bool)
    mask[tuple(slice(a, b) for a, b in zip(lo, hi, strict=False))] = True
    return mask


@pytest.fixture
def make_sphere() -> Callable[..., np.ndarray]:
    return _sphere


@pytest.fixture
def make_cube() -> Callable[..., np.ndarray]:
    return _cube
