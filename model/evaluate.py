"""
evaluate.py – Compute validity rate and per-condition breakdown.

Usage:
    python evaluate.py --predictions path/to/output.txt \
                       [--ground_truth path/to/data.txt]

Validity metric: % of predicted dates that satisfy ALL four conditions.
"""

from __future__ import annotations
import argparse
import os
import sys
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.tokenizer import validate_date, parse_line


def resolve_project_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    cwd_path = os.path.abspath(path)
    if os.path.exists(cwd_path):
        return cwd_path
    return os.path.join(PROJECT_ROOT, path)


def evaluate(predictions_path: str, ground_truth_path: str | None = None):
    predictions_path = resolve_project_path(predictions_path)
    if not os.path.isfile(predictions_path):
        raise FileNotFoundError(
            f"Missing predictions file: {predictions_path}\n"
            f"Create it first with:\n"
            f"  python model/predict.py -i data/example_input.txt -o predictions.txt --model lstm"
        )
    with open(predictions_path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    n_total = len(lines)
    n_valid = 0
    per_day    = defaultdict(lambda: [0, 0])
    per_month  = defaultdict(lambda: [0, 0])
    per_decade = defaultdict(lambda: [0, 0])
    per_leap   = defaultdict(lambda: [0, 0])
    failures   = []

    for line in lines:
        rec = parse_line(line)
        if rec is None or rec["date"] is None:
            failures.append(line)
            continue
        ok = validate_date(rec["date"], rec["day"], rec["month"],
                           rec["leap"], rec["decade"])
        per_day   [rec["day"]]   [0 if ok else 1] += 1
        per_month [rec["month"]] [0 if ok else 1] += 1
        per_decade[rec["decade"]][0 if ok else 1] += 1
        per_leap  [rec["leap"]]  [0 if ok else 1] += 1
        if ok:
            n_valid += 1
        else:
            failures.append(line)

    validity = n_valid / n_total if n_total else 0.0
    print(f"\n{'='*55}")
    print(f"  Total predictions : {n_total}")
    print(f"  Valid             : {n_valid}  ({validity*100:.1f}%)")
    print(f"{'='*55}")

    def report_breakdown(d, label):
        print(f"\n  {label} breakdown:")
        for key in sorted(d):
            ok, fail = d[key]
            tot = ok + fail
            pct = ok / tot * 100 if tot else 0
            print(f"    {key:>8} : {ok:5}/{tot:5}  ({pct:5.1f}%)")

    report_breakdown(per_day,    "Day")
    report_breakdown(per_month,  "Month")
    report_breakdown(per_leap,   "Leap")
    report_breakdown(per_decade, "Decade")

    if failures:
        print(f"\n  First 5 failures:")
        for fl in failures[:5]:
            print(f"    {fl}")

    return validity


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions",    required=True)
    parser.add_argument("--ground_truth",   default=None)
    args = parser.parse_args()
    evaluate(args.predictions, args.ground_truth)
