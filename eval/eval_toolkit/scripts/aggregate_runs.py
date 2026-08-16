"""Aggregate repeated E1 runs into median + range, and expose per-item stability.

E1 is not deterministic. The faithfulness judge returns slightly different
scores for identical input (observed: 0.7 / 0.0 / 0.3 for the same item across
three runs), and because the gate compares that score against a threshold, a
small drift flips whether the gate passes. That decides whether a retry fires,
which decides which attempt is reported, which decides which nodes are cited -
so one judge wobble can change the final label.

Reporting a single run therefore overstates precision. This prints what a
benchmark with a stochastic component should report: the median, the observed
range, and how many items actually held their label across every run.

    python scripts/aggregate_runs.py ground_truth/results_a.json ground_truth/results_b.json ...
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
from collections import Counter, defaultdict
from typing import Any, Dict, List

CATEGORIES = [
    "a_answerable_good",
    "b_out_of_scope",
    "c_low_trust",
    "d_temporal_invalid",
]


def _fmt(values: List[float]) -> str:
    """median [min-max], or a bare value when every run agreed."""
    lo, hi = min(values), max(values)
    med = statistics.median(values)
    if lo == hi:
        return f"{med:g}"
    return f"{med:g}  [{lo:g}-{hi:g}]"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("results", nargs="+", help="Two or more results JSON files.")
    ap.add_argument("--out", default="", help="Optional path to write the summary JSON.")
    args = ap.parse_args()

    if len(args.results) < 2:
        raise SystemExit("Need at least two runs to report a range.")

    runs = []
    for path in args.results:
        p = pathlib.Path(path)
        if not p.is_absolute():
            p = pathlib.Path(__file__).resolve().parents[1] / p
        runs.append((p.name, json.loads(p.read_text())))

    n = len(runs)
    print(f"\nAggregating {n} runs of E1\n" + "=" * 66)
    for name, _ in runs:
        print(f"  {name}")

    acc = [r["summary"]["overall_accuracy"] for _, r in runs]
    fp = [r["summary"]["false_pass"]["count"] for _, r in runs]
    fb = [r["summary"]["false_block"]["count"] for _, r in runs]

    print("\nHeadline metrics (median [range])")
    print("-" * 66)
    print(f"  overall accuracy   {_fmt(acc)}")
    print(f"  false-pass         {_fmt(fp)}")
    print(f"  false-block        {_fmt(fb)}")

    print("\nPer category (correct out of total)")
    print("-" * 66)
    for cat in CATEGORIES:
        vals = [r["summary"]["per_category"][cat]["correct"] for _, r in runs]
        total = runs[0][1]["summary"]["per_category"][cat]["total"]
        print(f"  {cat:<22} {_fmt(vals)} / {total}")

    # Per-item stability: the number that says how much of the score is solid.
    labels: Dict[str, List[str]] = defaultdict(list)
    for _, r in runs:
        for item in r["per_item"]:
            labels[item["id"]].append(item["predicted"])

    stable = [i for i, v in labels.items() if len(set(v)) == 1]
    unstable = sorted(i for i, v in labels.items() if len(set(v)) > 1)

    print(f"\nPer-item stability: {len(stable)}/{len(labels)} identical across all {n} runs")
    print("-" * 66)
    if unstable:
        for i in unstable:
            counts = Counter(labels[i])
            spread = ", ".join(f"{k} x{v}" for k, v in counts.most_common())
            print(f"  {i:<6} {spread}")
    else:
        print("  every item held its label")

    summary: Dict[str, Any] = {
        "runs": [name for name, _ in runs],
        "n_runs": n,
        "overall_accuracy": {
            "median": statistics.median(acc), "min": min(acc), "max": max(acc)
        },
        "false_pass": {"median": statistics.median(fp), "min": min(fp), "max": max(fp)},
        "false_block": {"median": statistics.median(fb), "min": min(fb), "max": max(fb)},
        "per_category": {
            cat: {
                "median": statistics.median(
                    [r["summary"]["per_category"][cat]["correct"] for _, r in runs]
                ),
                "min": min(r["summary"]["per_category"][cat]["correct"] for _, r in runs),
                "max": max(r["summary"]["per_category"][cat]["correct"] for _, r in runs),
                "total": runs[0][1]["summary"]["per_category"][cat]["total"],
            }
            for cat in CATEGORIES
        },
        "items_stable": len(stable),
        "items_total": len(labels),
        "unstable_items": {i: labels[i] for i in unstable},
    }

    if args.out:
        out = pathlib.Path(args.out)
        if not out.is_absolute():
            out = pathlib.Path(__file__).resolve().parents[1] / out
        out.write_text(json.dumps(summary, indent=2))
        print(f"\n-> {out}")


if __name__ == "__main__":
    main()
