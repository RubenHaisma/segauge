from __future__ import annotations

import json

import numpy as np
import pytest

from segauge.cli import main


def _cube():
    a = np.zeros((20, 20, 20), dtype=np.uint8)
    a[3:13, 3:13, 3:13] = 1
    return a


def test_cli_eval_directories(tmp_path):
    pred_dir = tmp_path / "pred"
    gt_dir = tmp_path / "gt"
    pred_dir.mkdir()
    gt_dir.mkdir()
    a = _cube()
    for cid in ("case1", "case2"):
        np.save(pred_dir / f"{cid}.npy", a)
        np.save(gt_dir / f"{cid}.npy", a)

    out = tmp_path / "r.json"
    rc = main(
        [
            "eval",
            "--pred",
            str(pred_dir),
            "--gt",
            str(gt_dir),
            "--json",
            str(out),
            "--resamples",
            "100",
        ]
    )
    assert rc == 0
    data = json.loads(out.read_text())
    assert data["n_cases"] == 2
    assert data["summary"]["dice"]["value"] == 1.0


def test_cli_metadata_and_report(tmp_path):
    pred_dir = tmp_path / "pred"
    gt_dir = tmp_path / "gt"
    pred_dir.mkdir()
    gt_dir.mkdir()
    a = _cube()
    for cid in ("p1", "p2"):
        np.save(pred_dir / f"{cid}.npy", a)
        np.save(gt_dir / f"{cid}.npy", a)
    meta = tmp_path / "m.csv"
    meta.write_text("case_id,site\np1,A\np2,B\n")
    report = tmp_path / "r.html"

    rc = main(
        [
            "eval",
            "--pred",
            str(pred_dir),
            "--gt",
            str(gt_dir),
            "--metadata",
            str(meta),
            "--report",
            str(report),
            "--resamples",
            "50",
        ]
    )
    assert rc == 0
    assert report.exists()
    assert "By site" in report.read_text()


def test_cli_no_matching_cases_errors(tmp_path):
    pred_dir = tmp_path / "pred"
    gt_dir = tmp_path / "gt"
    pred_dir.mkdir()
    gt_dir.mkdir()
    np.save(pred_dir / "x.npy", _cube())
    np.save(gt_dir / "y.npy", _cube())
    with pytest.raises(SystemExit):
        main(["eval", "--pred", str(pred_dir), "--gt", str(gt_dir)])
