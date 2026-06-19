from __future__ import annotations

import numpy as np

from segauge.core import Case, evaluate
from segauge.report import render_html


def test_render_html_contains_sections():
    m = np.zeros((16, 16, 16), dtype=bool)
    m[2:10, 2:10, 2:10] = True
    res = evaluate(
        [Case("a", m, m, spacing=1.0, metadata={"site": "X"})], n_resamples=50
    )
    html = render_html(res)
    assert "<html" in html.lower()
    assert "Summary" in html
    assert "dice" in html
    assert "By site" in html
    assert "Per case" in html
