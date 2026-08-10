"""E3 — gate ablation, WITHOUT re-running the LLM.

Each benchmark item already records the three gate signals (trust_score,
temporal_status, faithfulness) for its final answer. The gate is
    passed = (trust >= min_trust) AND (temporal is VALID) AND (faith >= min_faith)
so we can re-derive the pass/fail decision with any single gate DISABLED,
holding retrieval + answer + scores fixed. This isolates each gate's
contribution (a clean controlled ablation) and costs zero inference - the fan
stays quiet.

Reports the false-pass rate under: all gates on (baseline), trust off, temporal
off, faithfulness off. A gate that matters should make false-pass RISE when
removed.

    python scripts/ablation.py --results ground_truth/results_chunk_vector_parsefix.json
"""
from __future__ import annotations

import argparse
import json
import os

FAILING_TEMPORAL = {"OUTDATED", "SUPERSEDED", "CONFLICTED"}


def gate_pass(item, min_trust, min_faith, use_trust, use_temporal, use_faith) -> bool:
    ok = True
    if use_trust:
        ok = ok and item["trust_score"] >= min_trust
    if use_temporal:
        ok = ok and item["temporal_status"] not in FAILING_TEMPORAL
    if use_faith:
        ok = ok and item["faithfulness"] >= min_faith
    return ok


def evaluate(items, min_trust, min_faith, **flags):
    """Return (false_pass_ids, false_block_ids) under the given gate config."""
    fp, fb = [], []
    for it in items:
        passed = gate_pass(it, min_trust, min_faith, **flags)
        exp = it["expected"]
        # A non-answerable item that now PASSES is a false-pass; an answerable
        # item that now fails is a false-block.
        if exp != "PASS" and passed:
            fp.append(it["id"])
        if exp == "PASS" and not passed:
            fb.append(it["id"])
    return fp, fb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="ground_truth/results_chunk_vector_parsefix.json")
    args = ap.parse_args()

    items = json.load(open(args.results))["per_item"]
    min_trust = float(os.environ.get("KG_MIN_TRUST_SCORE", "0.4"))
    min_faith = float(os.environ.get("KG_MIN_FAITHFULNESS", "0.7"))
    n_nonpass = sum(1 for it in items if it["expected"] != "PASS")
    n_pass = sum(1 for it in items if it["expected"] == "PASS")

    configs = [
        ("all gates ON (baseline)", dict(use_trust=True, use_temporal=True, use_faith=True)),
        ("trust gate OFF", dict(use_trust=False, use_temporal=True, use_faith=True)),
        ("temporal gate OFF", dict(use_trust=True, use_temporal=False, use_faith=True)),
        ("faithfulness gate OFF", dict(use_trust=True, use_temporal=True, use_faith=False)),
        ("ALL gates OFF (no verification)", dict(use_trust=False, use_temporal=False, use_faith=False)),
    ]
    print(f"ablation on {len(items)} items | min_trust={min_trust} min_faith={min_faith}\n")
    print(f"  {'config':<34} {'false-pass':>12} {'false-block':>12}")
    rows = []
    for name, flags in configs:
        fp, fb = evaluate(items, min_trust, min_faith, **flags)
        fpr = round(len(fp) / max(n_nonpass, 1), 3)
        fbr = round(len(fb) / max(n_pass, 1), 3)
        rows.append({"config": name, "false_pass": len(fp), "false_pass_rate": fpr,
                     "false_pass_ids": fp, "false_block": len(fb), "false_block_rate": fbr})
        print(f"  {name:<34} {len(fp):>3} ({fpr:>5}) {len(fb):>6} ({fbr:>5})")

    base_fp = rows[0]["false_pass"]
    print("\n  contribution of each gate (extra bad answers it catches vs baseline):")
    for r in rows[1:4]:
        gate = r["config"].replace(" OFF", "")
        print(f"    {gate:<26} catches {r['false_pass'] - base_fp} additional false-pass"
              f"  (ids newly leaked: {[i for i in r['false_pass_ids'] if i not in rows[0]['false_pass_ids']]})")

    with open("ground_truth/ablation.json", "w") as f:
        json.dump({"min_trust": min_trust, "min_faith": min_faith, "configs": rows}, f, indent=2)
    print("\n  -> ground_truth/ablation.json")


if __name__ == "__main__":
    main()
