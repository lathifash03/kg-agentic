"""Run the temporal/trust items, which each need their own injected graph state.

Nothing in this corpus is naturally OUTDATED, SUPERSEDED or CONFLICTED - the
oldest node is days old, there are no supersession edges, and Topic.name is
unique. Those branches can only be exercised by injecting state, so each item
in ``level3_temporal_trust.jsonl`` carries its own ``injection.cypher``.

Per item: snapshot -> inject -> verify -> restore. The snapshot is an exact
property map of every Topic, so restoring cannot drift values the way an
ad-hoc cleanup does (an earlier manual cleanup reset created_at to now(),
silently changing node ages).

REFUSES any non-localhost target: injection is a destructive write.

    python eval/paper_suite/scripts/run_temporal_suite.py \
        --out results/level3_temporal_trust.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from urllib.parse import urlparse

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from kg_agent.agentic_verifier import AgenticVerifier  # noqa: E402
from kg_agent.config import get_config  # noqa: E402
from kg_agent.neo4j_client import Neo4jClient  # noqa: E402
from run_suite import (  # noqa: E402
    check_evidence, check_expectations, cross_paper_entities, load_papers, score_retrieval,
)

SUITE_DIR = pathlib.Path(__file__).resolve().parents[1]


def snapshot(client) -> dict:
    """Exact property map of every Topic, plus the set of existing edge ids."""
    nodes = {
        r["eid"]: r["props"]
        for r in client.run_read(
            "MATCH (e:Topic) RETURN elementId(e) AS eid, properties(e) AS props"
        )
    }
    rels = {
        r["rid"]
        for r in client.run_read("MATCH (:Topic)-[x]->(:Topic) RETURN elementId(x) AS rid")
    }
    return {"nodes": nodes, "rels": rels}


def restore(client, snap: dict) -> dict:
    """Undo an injection: drop new nodes/edges, reset changed properties."""
    now = snapshot(client)
    new_nodes = set(now["nodes"]) - set(snap["nodes"])
    new_rels = now["rels"] - snap["rels"]
    if new_nodes:
        client.run_write(
            "MATCH (e:Topic) WHERE elementId(e) IN $ids DETACH DELETE e",
            ids=list(new_nodes),
        )
    if new_rels:
        client.run_write(
            "MATCH (:Topic)-[x]->(:Topic) WHERE elementId(x) IN $ids DELETE x",
            ids=list(new_rels),
        )
    changed = [
        {"eid": eid, "props": props}
        for eid, props in snap["nodes"].items()
        if eid in now["nodes"] and now["nodes"][eid] != props
    ]
    if changed:
        # `SET e = row.props` replaces the whole property map, so keys added by
        # the injection are removed rather than left behind.
        client.run_write(
            "UNWIND $rows AS row MATCH (e:Topic) WHERE elementId(e) = row.eid SET e = row.props",
            rows=changed,
        )
    return {"deleted_nodes": len(new_nodes), "deleted_rels": len(new_rels),
            "reset_nodes": len(changed)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="ground_truth/level3_temporal_trust.jsonl")
    ap.add_argument("--out", default="results/level3_temporal_trust.json")
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    cfg = get_config()
    gt_path = SUITE_DIR / args.gt
    items = [json.loads(l) for l in gt_path.read_text().splitlines() if l.strip()]
    if args.only:
        items = [i for i in items if i["id"] == args.only]

    # The guard is about WRITES, not about the endpoint. Items that read the
    # graph's own contradiction/supersession layer inject nothing and are safe
    # anywhere; only injected items need a local sandbox. Checking the selected
    # items (not the whole file) means `--only T03` can run against a shared KG
    # while `--only T01` still refuses.
    host = urlparse(cfg.neo4j.uri).hostname or ""
    writes = [i["id"] for i in items if i.get("requires_injection", True)]
    if writes and host not in ("localhost", "127.0.0.1", "::1"):
        sys.exit(
            f"Refusing to inject into non-local target {cfg.neo4j.uri!r}. "
            f"Items needing injection: {', '.join(writes)}"
        )
    papers = load_papers()

    print(f"NEO4J_URI={cfg.neo4j.uri}  (local, injection allowed)")
    print(f"running {len(items)} injected items...\n")

    per_item = []
    with Neo4jClient.from_config(cfg) as client:
        verifier = AgenticVerifier(client, cfg)
        for it in items:
            # Items that read the graph's own :Contradiction / :SUPERSEDES layer
            # need no injection - and must not be given one, or the test would
            # be measuring the fixture instead of the real pipeline output.
            inject = it.get("injection") if it.get("requires_injection", True) else None
            snap = snapshot(client)
            t0 = time.time()
            try:
                if inject:
                    client.run_write(inject["cypher"])
                r = verifier.verify(it["question"])
                rec = {
                    "id": it["id"], "question": it["question"],
                    "expected_papers": it["expected_papers"],
                    "expected_behavior": it["expected_behavior"],
                    **score_retrieval(it, r, papers),
                    **check_evidence(it, r.answer),
                    **check_expectations(it, r),
                    "cross_paper_entities": cross_paper_entities(r),
                    "answer": r.answer, "trust_score": r.trust_score,
                    "temporal_status": r.temporal_validity_status,
                    "faithfulness": r.faithfulness, "passed": r.passed,
                    "retries": r.retries, "strategy": r.strategy,
                    "n_sources": len(r.sources_used or []),
                    "seconds": round(time.time() - t0, 1),
                }
            finally:
                rec_restore = restore(client, snap) if inject else {"skipped": "no injection"}
            rec["restore"] = rec_restore
            per_item.append(rec)
            checks = [v for k, v in rec.items() if k.endswith("_ok") and v is not None]
            print(f"  {'OK ' if all(checks) else 'XX '}[{it['id']}] "
                  f"status={rec['temporal_status']:<11}(want {it.get('expected_temporal_status')}) "
                  f"trust={rec['trust_score']:<7} gate={rec['passed']} "
                  f"retries={rec['retries']} restore={rec_restore} ({rec['seconds']}s)")

    n = max(len(per_item), 1)
    summary = {
        "n_questions": len(per_item),
        "temporal_status_mismatches": [
            {"id": r["id"], "want": r["expected_temporal_status"], "got": r["temporal_status"]}
            for r in per_item if r["temporal_status_ok"] is False
        ],
        "gate_mismatches": [
            {"id": r["id"], "want": r["expected_gate_passed"], "got": r["passed"]}
            for r in per_item if r["gate_ok"] is False
        ],
        "trust_mismatches": [
            {"id": r["id"], "got": r["trust_score"]} for r in per_item if r["trust_ok"] is False
        ],
        "mean_trust": round(sum(r["trust_score"] for r in per_item) / n, 3),
        "gate_passed_rate": round(sum(r["passed"] for r in per_item) / n, 3),
    }
    out_path = SUITE_DIR / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "per_item": per_item}, indent=2))
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nfull report -> {out_path}")


if __name__ == "__main__":
    main()
