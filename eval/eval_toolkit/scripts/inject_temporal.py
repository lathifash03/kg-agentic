"""Step 2d — Inject TEMPORAL-INVALID cases for ground-truth category (d).

CHUNK-PROFILE variant + kg-agent's real detection contract.

Each fact is  (:Chunk {text, embedding}) -[:MENTIONS]-> (:Topic)  so vector
retrieval (over the Chunk layer) can surface it with precise attribution.
Temporal signals live on the Topic / its edges:
  1. OUTDATED   - Topic.created_at pushed back > KG_OUTDATED_THRESHOLD_DAYS
                  (a real Neo4j datetime, not a string - strings are ignored).
  2. SUPERSEDED - (old)-[:SUPERSEDED_BY]->(new)  (kg-agent matches old->new via
                  the configured supersedes rel; default SUPERSEDED_BY).
  3. CONFLICTED - two Topics with the SAME name, different descriptions.

    python scripts/inject_temporal.py --dry-run
    python scripts/inject_temporal.py
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import urllib.request

try:
    from neo4j import GraphDatabase
except ImportError:
    sys.exit("pip install neo4j")

ENTITY_LABEL = os.environ.get("KG_ENTITY_LABEL", "Topic")
NAME_PROP = os.environ.get("KG_ENTITY_NAME_PROP", "name")
SUPERSEDES_REL = os.environ.get("KG_SUPERSEDES_REL", "SUPERSEDED_BY")
OUTDATED_DAYS = int(os.environ.get("KG_OUTDATED_THRESHOLD_DAYS", "30"))
EMBED_URL = os.environ.get("KG_EMBED_URL") or os.environ.get("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.environ.get("KG_EMBED_MODEL", "mxbai-embed-large")
BATCH = f"temporal_{datetime.date.today().isoformat()}"


def embed(text: str):
    payload = json.dumps({"model": EMBED_MODEL, "prompt": text}).encode()
    req = urllib.request.Request(f"{EMBED_URL}/api/embeddings", data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode()).get("embedding")


# Create a Topic (with given props) + a Chunk that MENTIONS it, embedding the
# chunk text. `merge_props` adds keys to the Topic's MERGE pattern so that two
# same-NAME topics (the CONFLICTED pair) stay distinct instead of collapsing.
def make_topic_chunk(session, name, text, source_type, conf, created_at, chunk_id,
                     merge_props=None):
    vec = embed(text)
    if not vec:
        sys.exit(f"embedding failed for {name!r}")
    merge_extra = ""
    params = {}
    if merge_props:
        for i, (k, v) in enumerate(merge_props.items()):
            merge_extra += f", `{k}`: $mk{i}"
            params[f"mk{i}"] = v
    session.run(
        f"""
        MERGE (t:{ENTITY_LABEL} {{{NAME_PROP}: $name{merge_extra}}})
        SET t.description = $text, t.source_type = $src, t.confidence_score = $conf,
            t.created_at = $created_at, t.created_by = 'eval_injection',
            t.injected = true, t.injection_batch = $batch
        MERGE (c:Chunk {{chunk_id: $chunk_id}})
        SET c.text = $text, c.embedding = $vec, c.created_by = 'eval_injection',
            c.injected = true, c.injection_batch = $batch
        MERGE (c)-[:MENTIONS]->(t)
        """,
        name=name, text=text, src=source_type, conf=conf, created_at=created_at,
        vec=vec, chunk_id=chunk_id, batch=BATCH, **params)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    now = datetime.datetime.now(datetime.timezone.utc)
    old = now - datetime.timedelta(days=OUTDATED_DAYS * 3)
    fresh = now - datetime.timedelta(days=1)

    if args.dry_run:
        print(f"WOULD CREATE OUTDATED   : 'Quarterly Backup Procedure' created_at={old.date()} (Chunk->MENTIONS)")
        print(f"WOULD CREATE SUPERSEDED : 'Lab Access Policy v1'({old.date()}) -[:{SUPERSEDES_REL}]-> v2({fresh.date()})")
        print(f"WOULD CREATE CONFLICTED : two 'Server Room Temperature Setpoint' (19C vs 24C), same name")
        print(f"\nembedding via {EMBED_MODEL} @ {EMBED_URL}")
        return

    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    auth = (os.environ.get("NEO4J_USERNAME", "neo4j"), os.environ.get("NEO4J_PASSWORD", ""))
    db = os.environ.get("NEO4J_DATABASE", "neo4j")

    driver = GraphDatabase.driver(uri, auth=auth)
    with driver.session(database=db) as s:
        # 1) OUTDATED
        make_topic_chunk(s, "Quarterly Backup Procedure",
                         "The quarterly data backup runs on the first Friday of the quarter "
                         "and archives to the NAS pool. (Synthetic: intentionally stale.)",
                         "meeting", 0.8, old, "inject_temporal_backup")
        print("created OUTDATED (Chunk->MENTIONS): Quarterly Backup Procedure")

        # 2) SUPERSEDED  (v1 old -> v2 new)
        make_topic_chunk(s, "Lab Access Policy v1",
                         "Lab access requires a physical key signed out from the department "
                         "office. (Synthetic old policy.)", "paper", 0.9, old,
                         "inject_temporal_access_v1")
        make_topic_chunk(s, "Lab Access Policy v2",
                         "Lab access now uses the campus smart-card system; physical keys are "
                         "retired. (Synthetic new policy.)", "paper", 0.9, fresh,
                         "inject_temporal_access_v2")
        s.run(f"""
            MATCH (v1:{ENTITY_LABEL} {{{NAME_PROP}: 'Lab Access Policy v1'}})
            MATCH (v2:{ENTITY_LABEL} {{{NAME_PROP}: 'Lab Access Policy v2'}})
            MERGE (v1)-[r:{SUPERSEDES_REL}]->(v2) SET r.injected = true
        """)
        print(f"created SUPERSEDED (Chunk->MENTIONS): v1 -[:{SUPERSEDES_REL}]-> v2")

        # 3) CONFLICTED (same name, different description/variant)
        make_topic_chunk(s, "Server Room Temperature Setpoint",
                         "The server room temperature setpoint is 19 degrees C per facilities "
                         "guidance. (Synthetic conflict A.)", "meeting", 0.8, fresh,
                         "inject_temporal_setpoint_a", merge_props={"conflict_variant": "A"})
        make_topic_chunk(s, "Server Room Temperature Setpoint",
                         "The server room temperature setpoint is 24 degrees C to save energy. "
                         "(Synthetic conflict B.)", "meeting", 0.8, fresh,
                         "inject_temporal_setpoint_b", merge_props={"conflict_variant": "B"})
        print("created CONFLICTED (Chunk->MENTIONS): two same-name 'Server Room Temperature Setpoint'")

    driver.close()
    print(f"\nBatch tag: {BATCH}\nRemove all: MATCH (n {{injected: true}}) DETACH DELETE n")


if __name__ == "__main__":
    main()
