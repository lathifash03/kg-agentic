"""Step 2d — Inject TEMPORAL-INVALID cases for ground-truth category (d).

ADAPTED to Nabhyla's schema AND to kg-agent's actual detection contract.

Schema: each fact is a full 2-hop subgraph
    (Topic)-[:HAS_TYPE]->(Type)-[:HAS_DESCRIPTION]->(Description {text})
so the retriever (which reads Description.text) can actually surface it.

Contract fixes vs the generic template (verified against kg_agent):
  1. created_at is a real Neo4j datetime (a Python datetime object passed as a
     parameter), NOT an ISO string. temporal_validity ignores string
     timestamps, so a string would never be flagged OUTDATED.
  2. SUPERSEDED uses  (old)-[:SUPERSEDED_BY]->(new)  — kg-agent matches an OLD
     node pointing to a NEWER one via SUPERSEDED_BY (config default). The
     generic template's (new)-[:SUPERSEDES]->(old) is the reverse name AND
     direction.
  3. CONFLICTED creates two nodes with the SAME name (different descriptions),
     distinguished only by a private conflict_variant key so MERGE keeps them
     separate. kg-agent flags a conflict when >1 entity shares a name.

Three controlled failure modes: OUTDATED, SUPERSEDED, CONFLICTED. Questions
answerable only from these nodes must have their temporal status detected
(expected_gate_outcome = TEMPORAL_FLAGGED).

All nodes carry  injected: true  +  injection_batch  for cleanup:
    MATCH (n {injected: true}) DETACH DELETE n

Usage:
    python scripts/inject_temporal.py --dry-run
    python scripts/inject_temporal.py
"""
from __future__ import annotations

import argparse
import datetime
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
# Match kg-agent's config default so the SUPERSEDED edge is actually detected.
SUPERSEDES_REL = os.environ.get("KG_SUPERSEDES_REL", "SUPERSEDED_BY")
OUTDATED_DAYS = int(os.environ.get("KG_OUTDATED_THRESHOLD_DAYS", "30"))
BATCH = f"temporal_{datetime.date.today().isoformat()}"


def now_utc() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


# Attach a dedicated Type + Description (with the retrievable text) to a Topic
# already bound as `t` in the caller's query. Kept as a fragment so each case
# builds the exact 2-hop shape without repeating the boilerplate.
def subgraph(topic_alias: str, type_name: str, desc_name: str) -> str:
    return f"""
    MERGE (ty_{topic_alias}:Type {{{NAME_PROP}: '{type_name}'}})
      SET ty_{topic_alias}.injected = true, ty_{topic_alias}.injection_batch = $batch
    MERGE (d_{topic_alias}:`{CHUNK_LABEL}` {{{NAME_PROP}: '{desc_name}'}})
      SET d_{topic_alias}.`{TEXT_PROP}` = {topic_alias}.description,
          d_{topic_alias}.injected = true, d_{topic_alias}.injection_batch = $batch,
          d_{topic_alias}.created_by = 'eval_injection'
    MERGE ({topic_alias})-[:HAS_TYPE]->(ty_{topic_alias})
    MERGE (ty_{topic_alias})-[:HAS_DESCRIPTION]->(d_{topic_alias})
    """


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    old = now_utc() - datetime.timedelta(days=OUTDATED_DAYS * 3)  # safely stale
    fresh = now_utc() - datetime.timedelta(days=1)

    plan = [
        ("OUTDATED node", f"'Quarterly Backup Procedure' created_at={old.date()} (2-hop)"),
        ("SUPERSEDED pair",
         f"'Lab Access Policy v1' ({old.date()}) -[:{SUPERSEDES_REL}]-> "
         f"'Lab Access Policy v2' ({fresh.date()}) (2-hop each)"),
        ("CONFLICTED pair",
         "two 'Server Room Temperature Setpoint' nodes (19C vs 24C), same name (2-hop each)"),
    ]
    if args.dry_run:
        for kind, desc in plan:
            print(f"WOULD CREATE {kind}: {desc}")
        return

    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    auth = (os.environ.get("NEO4J_USERNAME", "neo4j"),
            os.environ.get("NEO4J_PASSWORD", ""))
    db = os.environ.get("NEO4J_DATABASE", "neo4j")

    driver = GraphDatabase.driver(uri, auth=auth)
    with driver.session(database=db) as s:
        # 1) OUTDATED -----------------------------------------------------
        s.run(
            f"""
            MERGE (t:{ENTITY_LABEL} {{{NAME_PROP}: 'Quarterly Backup Procedure'}})
            SET t.description = 'The quarterly data backup runs on the first '
                              + 'Friday of the quarter and archives to the NAS '
                              + 'pool. (Synthetic: intentionally stale.)',
                t.source_type = 'meeting', t.confidence_score = 0.8,
                t.created_at = $old, t.created_by = 'eval_injection',
                t.injected = true, t.injection_batch = $batch
            {subgraph('t', 'Injected Classification: Backup', 'Injected Description: Quarterly Backup')}
            """, old=old, batch=BATCH)
        print("created OUTDATED (2-hop): Quarterly Backup Procedure")

        # 2) SUPERSEDED ---------------------------------------------------
        s.run(
            f"""
            MERGE (v1:{ENTITY_LABEL} {{{NAME_PROP}: 'Lab Access Policy v1'}})
            SET v1.description = 'Lab access requires a physical key signed out '
                               + 'from the department office. (Synthetic old.)',
                v1.source_type = 'paper', v1.confidence_score = 0.9,
                v1.created_at = $old, v1.created_by = 'eval_injection',
                v1.injected = true, v1.injection_batch = $batch
            MERGE (v2:{ENTITY_LABEL} {{{NAME_PROP}: 'Lab Access Policy v2'}})
            SET v2.description = 'Lab access now uses the campus smart-card '
                               + 'system; physical keys are retired. (Synthetic new.)',
                v2.source_type = 'paper', v2.confidence_score = 0.9,
                v2.created_at = $fresh, v2.created_by = 'eval_injection',
                v2.injected = true, v2.injection_batch = $batch
            {subgraph('v1', 'Injected Classification: Access v1', 'Injected Description: Lab Access v1')}
            {subgraph('v2', 'Injected Classification: Access v2', 'Injected Description: Lab Access v2')}
            MERGE (v1)-[r:{SUPERSEDES_REL}]->(v2)
            SET r.injected = true
            """, old=old, fresh=fresh, batch=BATCH)
        print(f"created SUPERSEDED (2-hop): v1 -[:{SUPERSEDES_REL}]-> v2")

        # 3) CONFLICTED (same name, different description) ----------------
        s.run(
            f"""
            MERGE (a:{ENTITY_LABEL} {{{NAME_PROP}: 'Server Room Temperature Setpoint', conflict_variant: 'A'}})
            SET a.description = 'The server room temperature setpoint is 19 '
                              + 'degrees C per facilities guidance. (Synthetic A.)',
                a.source_type = 'meeting', a.confidence_score = 0.8,
                a.created_at = $fresh, a.created_by = 'eval_injection',
                a.injected = true, a.injection_batch = $batch
            MERGE (b:{ENTITY_LABEL} {{{NAME_PROP}: 'Server Room Temperature Setpoint', conflict_variant: 'B'}})
            SET b.description = 'The server room temperature setpoint is 24 '
                              + 'degrees C to save energy. (Synthetic B.)',
                b.source_type = 'meeting', b.confidence_score = 0.8,
                b.created_at = $fresh, b.created_by = 'eval_injection',
                b.injected = true, b.injection_batch = $batch
            {subgraph('a', 'Injected Classification: Setpoint A', 'Injected Description: Setpoint 19C')}
            {subgraph('b', 'Injected Classification: Setpoint B', 'Injected Description: Setpoint 24C')}
            """, fresh=fresh, batch=BATCH)
        print("created CONFLICTED (2-hop): two same-name 'Server Room Temperature Setpoint'")

    driver.close()
    print(f"\nBatch tag: {BATCH}")
    print("Remove all: MATCH (n {injected: true}) DETACH DELETE n")


if __name__ == "__main__":
    main()
