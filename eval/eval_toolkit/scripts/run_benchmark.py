"""E1 benchmark runner — score kg-agent's gate decisions against ground truth.

Reads ground_truth.jsonl, runs each question through kg-agent's real
verification pipeline (AgenticVerifier.verify), maps the output to a predicted
gate outcome, and scores it against the expected outcome. Reports per-category
accuracy plus the two headline rates:
  * false-pass  — a non-answerable/flawed question that PASSED (dangerous)
  * false-block — an answerable-good question that was needlessly blocked

Run against YOUR LOCAL snapshot copy, with the REAL answer-gen model (not mock,
or category-b/NO_INFO is meaningless). Example:

    export NEO4J_URI=bolt://localhost:7687
    export KG_LLM_PROVIDER=ollama KG_LLM_MODEL=hermes3:3b OLLAMA_URL=http://localhost:11434
    export KG_JUDGE_PROVIDER=ollama KG_JUDGE_MODEL=hermes3:3b KG_JUDGE_OLLAMA_URL=http://localhost:11434
    export KG_ENTITY_LABEL=Topic KG_CHUNK_LABEL=Description \
           KG_CHUNK_TO_ENTITY_PATTERN='(c:{chunk_label})<-[:HAS_DESCRIPTION]-(:Type)<-[:HAS_TYPE]-(e:{entity_label})'
    python scripts/run_benchmark.py --gt ground_truth/ground_truth.jsonl --out ground_truth/results.json
    python scripts/run_benchmark.py ... --limit-per-cat 2      # quick pilot
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from collections import defaultdict

# Make kg_agent importable (repo root is three levels up from this file).
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from kg_agent.agentic_verifier import AgenticVerifier  # noqa: E402
from kg_agent.config import get_config  # noqa: E402
from kg_agent.neo4j_client import Neo4jClient  # noqa: E402

FAILING_TEMPORAL = {"OUTDATED", "SUPERSEDED", "CONFLICTED"}
NO_INFO_MARKERS = [
    "no supporting context", "does not contain", "not contain information",
    "no information", "did not produce", "could not find", "cannot find",
    "not found", "no relevant information", "does not mention", "not mentioned",
    "no data", "insufficient", "does not give information",
    "does not provide", "do not have enough", "not have enough data",
    "does not specify", "no details about", "not available in the",
]


def predict_outcome(result) -> str:
    """Map a VerifiedAnswer to one of the four expected_gate_outcome labels."""
    answer = (result.answer or "").lower()
    is_no_info = any(m in answer for m in NO_INFO_MARKERS)

    if result.passed:
        return "PASS"
    # Not passed -> figure out WHY, in priority order.
    if result.temporal_validity_status in FAILING_TEMPORAL:
        return "TEMPORAL_FLAGGED"
    if is_no_info:
        return "NO_INFO"
    if result.disclaimer:
        return "RELEASE_WITH_DISCLAIMER"
    return "BLOCKED_OTHER"  # blocked but not clearly one of the above


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="ground_truth/ground_truth.jsonl")
    ap.add_argument("--out", default="ground_truth/results.json")
    ap.add_argument("--limit-per-cat", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    items = [json.loads(l) for l in open(args.gt) if l.strip()]
    if args.limit_per_cat:
        by_cat = defaultdict(list)
        for it in items:
            by_cat[it["category"]].append(it)
        items = [x for cat in by_cat.values() for x in cat[: args.limit_per_cat]]

    cfg = get_config()
    print(f"NEO4J_URI={cfg.neo4j.uri} | answer-gen={cfg.llm.provider}:{cfg.llm.model} "
          f"| judge={cfg.judge.provider or 'main'}:{cfg.judge.model or cfg.llm.model}")
    print(f"scoring {len(items)} questions...\n")

    per_item = []
    with Neo4jClient.from_config(cfg) as client:
        verifier = AgenticVerifier(client, cfg)
        for i, it in enumerate(items, 1):
            t0 = time.time()
            r = verifier.verify(it["question"])
            predicted = predict_outcome(r)
            expected = it["expected_gate_outcome"]
            ok = predicted == expected
            per_item.append({
                "id": it["id"], "category": it["category"],
                "question": it["question"],
                "expected": expected, "predicted": predicted, "correct": ok,
                "passed": r.passed, "trust_score": r.trust_score,
                "temporal_status": r.temporal_validity_status,
                "faithfulness": r.faithfulness,
                "answer_snippet": (r.answer or "")[:140],
                "seconds": round(time.time() - t0, 1),
            })
            mark = "OK " if ok else "XX "
            print(f"  {mark}[{it['id']}] {it['category']:<18} exp={expected:<24} "
                  f"got={predicted:<24} ({per_item[-1]['seconds']}s)")

    # ---- aggregate ----
    cats = defaultdict(lambda: {"total": 0, "correct": 0})
    false_pass = []   # expected != PASS but predicted PASS
    false_block = []  # expected PASS but predicted != PASS
    for r in per_item:
        c = cats[r["category"]]
        c["total"] += 1
        c["correct"] += int(r["correct"])
        if r["expected"] != "PASS" and r["predicted"] == "PASS":
            false_pass.append(r["id"])
        if r["expected"] == "PASS" and r["predicted"] != "PASS":
            false_block.append(r["id"])

    n_nonpass = sum(1 for r in per_item if r["expected"] != "PASS")
    n_pass = sum(1 for r in per_item if r["expected"] == "PASS")
    summary = {
        "n_questions": len(per_item),
        "overall_accuracy": round(sum(r["correct"] for r in per_item) / max(len(per_item), 1), 3),
        "per_category": {
            cat: {"correct": v["correct"], "total": v["total"],
                  "accuracy": round(v["correct"] / v["total"], 3)}
            for cat, v in sorted(cats.items())
        },
        "false_pass": {"count": len(false_pass), "ids": false_pass,
                       "rate_over_nonpass": round(len(false_pass) / max(n_nonpass, 1), 3)},
        "false_block": {"count": len(false_block), "ids": false_block,
                        "rate_over_answerable": round(len(false_block) / max(n_pass, 1), 3)},
    }

    report = {"summary": summary, "per_item": per_item}
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nfull report -> {args.out}")


if __name__ == "__main__":
    main()
