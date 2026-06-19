"""Command-line interface: ``segauge eval``.

Point it at a prediction and a ground truth (single files or matching
directories), get honest numbers with confidence intervals, and optionally a
report you can hand to a reviewer.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from segauge.core import Case, evaluate

_MASK_SUFFIXES = (".nii.gz", ".nii", ".npy", ".dcm")


def _case_id(path: Path) -> str:
    name = path.name
    for suffix in _MASK_SUFFIXES:
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _collect(path: Path) -> dict[str, Path]:
    """Map case-id -> file for a directory, or a single {id: file}."""
    if path.is_dir():
        out: dict[str, Path] = {}
        for child in sorted(path.iterdir()):
            if child.is_file() and child.name.lower().endswith(_MASK_SUFFIXES):
                out[_case_id(child)] = child
        return out
    return {_case_id(path): path}


def _load_metadata(path: Path | None) -> dict[str, dict[str, object]]:
    if path is None:
        return {}
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or "case_id" not in reader.fieldnames:
            raise SystemExit("metadata CSV must have a 'case_id' column")
        return {
            row["case_id"]: {k: v for k, v in row.items() if k != "case_id"}
            for row in reader
        }


def _build_cases(
    pred: Path, gt: Path, metadata: dict[str, dict[str, object]]
) -> list[Case]:
    preds = _collect(pred)
    gts = _collect(gt)
    shared = sorted(set(preds) & set(gts))
    if not shared:
        raise SystemExit(
            f"no matching case ids between --pred ({len(preds)}) and "
            f"--gt ({len(gts)}); files are matched by name without extension"
        )
    missing = set(preds) ^ set(gts)
    if missing:
        print(
            f"warning: {len(missing)} unmatched case(s) ignored: "
            f"{', '.join(sorted(missing)[:5])}"
            f"{'...' if len(missing) > 5 else ''}",
            file=sys.stderr,
        )
    return [
        Case(case_id=cid, pred=preds[cid], gt=gts[cid], metadata=metadata.get(cid, {}))
        for cid in shared
    ]


def _print_summary(result) -> None:
    summary = result.summary()
    width = max(len(m) for m in summary) if summary else 6
    print(f"\n{len(result.rows)} case(s), {round(result.confidence * 100)}% CI\n")
    print(f"{'metric'.ljust(width)}   value   CI")
    for name, est in summary.items():
        val = "n/a" if est.value != est.value else f"{est.value:.4g}"  # NaN check
        lo = "n/a" if est.ci_low != est.ci_low else f"{est.ci_low:.4g}"
        hi = "n/a" if est.ci_high != est.ci_high else f"{est.ci_high:.4g}"
        print(f"{name.ljust(width)}   {val:>6}   [{lo}, {hi}]")
    print()


def _cmd_eval(args: argparse.Namespace) -> int:
    metadata = _load_metadata(Path(args.metadata) if args.metadata else None)
    cases = _build_cases(Path(args.pred), Path(args.gt), metadata)
    result = evaluate(
        cases,
        nsd_tolerance=args.nsd_tolerance,
        detection=not args.no_detection,
        detection_iou=args.detection_iou,
        label=args.label,
        segment_number=args.segment_number,
        confidence=args.confidence,
        n_resamples=args.resamples,
        seed=args.seed,
    )
    _print_summary(result)
    if args.report:
        result.to_html(args.report)
        print(f"report written to {args.report}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(result.to_dict(), fh, indent=2, default=str)
        print(f"json written to {args.json}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="segauge",
        description="The honest gauge for medical image segmentation.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ev = sub.add_parser("eval", help="evaluate predictions against ground truth")
    ev.add_argument("--pred", required=True, help="prediction file or directory")
    ev.add_argument("--gt", required=True, help="ground-truth file or directory")
    ev.add_argument("--report", help="write a self-contained HTML report here")
    ev.add_argument("--json", help="write machine-readable results here")
    ev.add_argument("--metadata", help="CSV with a 'case_id' column for subgroups")
    ev.add_argument(
        "--nsd-tolerance",
        type=float,
        default=1.0,
        help="NSD tolerance in physical units (default 1.0)",
    )
    ev.add_argument(
        "--detection-iou",
        type=float,
        default=0.1,
        help="IoU threshold for per-lesion matching (default 0.1)",
    )
    ev.add_argument(
        "--no-detection", action="store_true", help="skip per-lesion detection metrics"
    )
    ev.add_argument(
        "--label",
        type=int,
        default=None,
        help="keep only this label value as foreground",
    )
    ev.add_argument(
        "--segment-number",
        type=int,
        default=1,
        help="DICOM-SEG segment number to extract (default 1)",
    )
    ev.add_argument("--confidence", type=float, default=0.95)
    ev.add_argument("--resamples", type=int, default=2000)
    ev.add_argument("--seed", type=int, default=0)
    ev.set_defaults(func=_cmd_eval)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
