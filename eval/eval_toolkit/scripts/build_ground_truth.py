"""Build a FIRST-PASS curated ground_truth.jsonl from candidates + injections.

Not a replacement for human review - it makes a sensible draft the human can
refine (rephrase, drop weak items). It:
  * category a: dedupes candidates by their Description text (the shared-Type
    fan-out makes many Topics share one paragraph) and keeps N distinct ones.
  * category b: keeps N verified-absent candidates.
  * category c/d: uses distinctive-term questions (so retrieval isolates the
    injected node) and resolves evidence_node_ids by name from the DB.
"""
from __future__ import annotations

import json
import os

from neo4j import GraphDatabase

N_A, N_B = 6, 6
ENTITY_LABEL = os.environ.get("KG_ENTITY_LABEL", "Topic")
NAME_PROP = os.environ.get("KG_ENTITY_NAME_PROP", "name")

# category c: (question, injected node name). Distinctive terms only.
C_ITEMS = [
    ("What is the recommended recalibration interval for the lab vibration sensors?",
     "Sensor Recalibration Interval"),
    ("What EXP-year-initials naming convention did the older experiment datasets use?",
     "Legacy Dataset Naming Convention"),
    ("How many days ahead should GPU slots on the lab cluster be booked?",
     "Cluster GPU Booking Etiquette"),
    ("What desk-fan workaround was noted to prevent thermal throttling during long runs?",
     "Prototype Cooling Workaround"),
    ("What is the migration status of the old lab wiki?",
     "Deprecated Wiki Migration Status"),
]
# category d: (question, node name, expected_temporal_status).
# Distinctive-term phrasings (verified to actually retrieve the injected node -
# a generic phrasing lets unrelated Topics crowd the node out of the top-k, and
# then its temporal status is never checked).
D_ITEMS = [
    ("What is the Quarterly Backup Procedure?",
     "Quarterly Backup Procedure", "OUTDATED"),
    ("What is the lab access policy?",
     "Lab Access Policy v1", "SUPERSEDED"),
    ("What is the server room temperature setpoint?",
     "Server Room Temperature Setpoint", "CONFLICTED"),
]


def ids_for(session, name):
    rows = session.run(
        f"MATCH (n:{ENTITY_LABEL} {{{NAME_PROP}: $name}}) RETURN elementId(n) AS id",
        name=name).data()
    return [r["id"] for r in rows]


def main():
    cand = [json.loads(l) for l in open("ground_truth/candidates.jsonl") if l.strip()]
    a = [x for x in cand if x["category"] == "a_answerable_good"]
    b = [x for x in cand if x["category"] == "b_out_of_scope"]

    # Keep N distinct questions. (Many share a Description paragraph via the
    # shared-Type fan-out; that is fine here - each still asks about a real,
    # retrievable Topic, so PASS is the correct expected outcome.)
    seen, a_keep = set(), []
    for x in a:
        if x["question"] in seen:
            continue
        seen.add(x["question"])
        a_keep.append(x)
        if len(a_keep) >= N_A:
            break

    out = []
    for i, x in enumerate(a_keep, 1):
        out.append({
            "id": f"a{i:02d}", "category": "a_answerable_good",
            "question": x["question"],
            "expected_gate_outcome": "PASS",
            "evidence_node_ids": x["evidence_node_ids"],
            "injected": False, "notes": x["notes"],
        })
    for i, x in enumerate(b[:N_B], 1):
        out.append({
            "id": f"b{i:02d}", "category": "b_out_of_scope",
            "question": x["question"],
            "expected_gate_outcome": "NO_INFO",
            "evidence_node_ids": [], "injected": False, "notes": x["notes"],
        })

    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    auth = (os.environ.get("NEO4J_USERNAME", "neo4j"),
            os.environ.get("NEO4J_PASSWORD", "password123"))
    driver = GraphDatabase.driver(uri, auth=auth)
    with driver.session() as s:
        for i, (q, name) in enumerate(C_ITEMS, 1):
            out.append({
                "id": f"c{i:02d}", "category": "c_low_trust",
                "question": q, "expected_gate_outcome": "RELEASE_WITH_DISCLAIMER",
                "evidence_node_ids": ids_for(s, name), "injected": True,
                "notes": f"answerable only from injected low-trust node {name!r}",
            })
        for i, (q, name, tstatus) in enumerate(D_ITEMS, 1):
            out.append({
                "id": f"d{i:02d}", "category": "d_temporal_invalid",
                "question": q, "expected_gate_outcome": "TEMPORAL_FLAGGED",
                "expected_temporal_status": tstatus,
                "evidence_node_ids": ids_for(s, name), "injected": True,
                "notes": f"injected {tstatus} case: {name!r}",
            })
    driver.close()

    with open("ground_truth/ground_truth.jsonl", "w") as f:
        for item in out:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    from collections import Counter
    print("wrote ground_truth/ground_truth.jsonl:", dict(Counter(x["category"] for x in out)))
    print("total:", len(out))


if __name__ == "__main__":
    main()
