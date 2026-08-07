"""Step 3 helper — Propose candidate questions for categories (a) and (b).

(a) answerable-good : sampled from REAL graph nodes with the richest text,
    so you curate questions instead of inventing them.
(b) out-of-scope    : candidate terms are VERIFIED absent from the graph
    (keyword check on name + description) before being proposed.

Output: prints candidates + writes ground_truth/candidates.jsonl
You then EDIT/CURATE them into ground_truth/ground_truth.jsonl.

Usage:
    python scripts/propose_questions.py --n-a 15 --n-b 15
"""
from __future__ import annotations

import argparse
import json
import os
import sys

try:
    from neo4j import GraphDatabase
except ImportError:
    sys.exit("pip install neo4j")

ENTITY_LABEL = os.environ.get("KG_ENTITY_LABEL", "Topic")
NAME_PROP = os.environ.get("KG_ENTITY_NAME_PROP", "name")
CHUNK_LABEL = os.environ.get("KG_CHUNK_LABEL", "Description")
TEXT_PROP = os.environ.get("KG_CHUNK_TEXT_PROP", "text")

# Terms that plausibly belong to a lab/technical domain but should NOT be
# in the graph. Each is verified absent before being proposed. EDIT freely.
OUT_OF_SCOPE_TERMS = [
    "quantum annealing budget", "wet-lab safety audit", "CRISPR protocol",
    "wind tunnel calibration", "satellite downlink schedule",
    "clinical trial enrollment", "blockchain consensus policy",
    "underwater drone maintenance", "particle accelerator shift",
    "greenhouse irrigation controller", "MRI coil replacement",
    "volcanic sensor network", "cheese fermentation log",
    "opera rehearsal booking", "submarine cable repair",
    "beekeeping rota", "asphalt mix ratio", "glacier core storage",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-a", type=int, default=15)
    ap.add_argument("--n-b", type=int, default=15)
    ap.add_argument("--out", default="ground_truth/candidates.jsonl")
    args = ap.parse_args()

    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    auth = (os.environ.get("NEO4J_USERNAME", "neo4j"),
            os.environ.get("NEO4J_PASSWORD", ""))
    db = os.environ.get("NEO4J_DATABASE", "neo4j")
    driver = GraphDatabase.driver(uri, auth=auth)

    candidates = []
    with driver.session(database=db) as s:
        # ---- (a) answerable-good candidates, one per DISTINCT Topic -----
        # ADAPTED to Nabhyla's schema: text lives on :Description nodes two hops
        # from the Topic (Topic-[:HAS_TYPE]->Type-[:HAS_DESCRIPTION]->Description).
        #
        # NOTE ON THE STRUCTURE (see cardinality diagnostic): the Topic<->
        # Description mapping is heavily many-to-many because Type is a COARSE
        # shared classification (one Description reaches ~27 Topics of the same
        # Type). There is no clean 1:1 pairing to dedup on. So we emit one row
        # per DISTINCT Topic that has any linked Description, ranked by the
        # richest Description available to it, and leave the human to pick
        # natural, standalone topics (and drop fragment-y names) during curation.
        rows = s.run(
            f"""
            MATCH (n:{ENTITY_LABEL})-[:HAS_TYPE]->(:Type)-[:HAS_DESCRIPTION]->(d:{CHUNK_LABEL})
            WHERE n.injected IS NULL AND d.{TEXT_PROP} IS NOT NULL
            WITH n, max(size(d.{TEXT_PROP})) AS best_len,
                 head(collect(d.{TEXT_PROP})) AS sample
            RETURN elementId(n) AS id, n.{NAME_PROP} AS name,
                   sample AS desc, best_len AS len
            ORDER BY best_len DESC, name LIMIT $k
            """, k=args.n_a)
        for r in rows:
            candidates.append({
                "category": "a_answerable_good",
                "question": f"What is {r['name']}?",   # EDIT into natural phrasing
                "expected_gate_outcome": "PASS",
                "evidence_node_ids": [r["id"]],
                "notes": f"desc_len={r['len']}; snippet={r['desc'][:120]!r}",
            })

        # ---- (b) verified-absent terms ---------------------------------
        kept = 0
        for term in OUT_OF_SCOPE_TERMS:
            if kept >= args.n_b:
                break
            hit = s.run(
                f"""
                MATCH (n:{ENTITY_LABEL})
                WHERE toLower(n.{NAME_PROP}) CONTAINS toLower($t)
                   OR toLower(coalesce(n.description,'')) CONTAINS toLower($t)
                RETURN count(n) AS c
                """, t=term).single()["c"]
            if hit == 0:
                candidates.append({
                    "category": "b_out_of_scope",
                    "question": f"What does the lab documentation say about {term}?",
                    "expected_gate_outcome": "NO_INFO",
                    "evidence_node_ids": [],
                    "notes": f"verified absent: 0 name/description matches for {term!r}",
                })
                kept += 1
            else:
                print(f"skip (found in graph!): {term}")
    driver.close()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for c in candidates:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"\nwrote {len(candidates)} candidates -> {args.out}")
    print("NEXT: curate them (rephrase questions naturally, drop weak ones), "
          "then merge into ground_truth/ground_truth.jsonl")


if __name__ == "__main__":
    main()
