"""Step 1 — Freeze a KG snapshot and record its fingerprint.

Every ground-truth label refers to ONE frozen snapshot. This script:
  1. Records a fingerprint (counts + label/rel breakdown) as JSON
  2. Prints the dump command to run for a full backup

Usage:
    python scripts/freeze_snapshot.py                # fingerprint only
    python scripts/freeze_snapshot.py --out snap.json

Reads the same env vars as kg_agent (NEO4J_URI, NEO4J_USERNAME,
NEO4J_PASSWORD, NEO4J_DATABASE, KG_ENTITY_LABEL).
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

try:
    from neo4j import GraphDatabase
except ImportError:
    sys.exit("pip install neo4j")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ground_truth/snapshot_fingerprint.json")
    args = ap.parse_args()

    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USERNAME", "neo4j")
    pwd = os.environ.get("NEO4J_PASSWORD", "")
    db = os.environ.get("NEO4J_DATABASE", "neo4j")

    driver = GraphDatabase.driver(uri, auth=(user, pwd))
    with driver.session(database=db) as s:
        node_count = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        rel_count = s.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
        labels = {
            r["label"]: r["c"]
            for r in s.run(
                "MATCH (n) UNWIND labels(n) AS label "
                "RETURN label, count(*) AS c ORDER BY c DESC"
            )
        }
        rels = {
            r["t"]: r["c"]
            for r in s.run(
                "MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS c ORDER BY c DESC"
            )
        }
        # metadata coverage: how many nodes already carry the contract fields
        meta = s.run(
            "MATCH (n) RETURN "
            "count(n.created_at) AS created_at, "
            "count(n.source_type) AS source_type, "
            "count(n.confidence_score) AS confidence_score"
        ).single()

    fingerprint = {
        "frozen_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "neo4j_uri": uri,
        "database": db,
        "node_count": node_count,
        "relationship_count": rel_count,
        "labels": labels,
        "relationship_types": rels,
        "metadata_coverage": dict(meta),
        "note": "All ground-truth labels in this directory refer to THIS snapshot.",
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(fingerprint, f, indent=2)

    print(json.dumps(fingerprint, indent=2))
    print("\n--- full dump (run this for a restorable backup) ---")
    print("Docker:  docker exec <neo4j-container> neo4j-admin database dump "
          f"{db} --to-path=/backups")
    print("Local :  neo4j-admin database dump", db, "--to-path=./backups")
    driver.close()


if __name__ == "__main__":
    main()
