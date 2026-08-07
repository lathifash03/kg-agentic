"""Step 2c — Inject LOW-TRUST cases for ground-truth category (c).

CHUNK-PROFILE variant: retrieval runs over the Layer-2 Chunk substrate
((:Chunk {text, embedding}) -[:MENTIONS]-> (:Topic)), which the ontology
designates as the search layer and which gives PRECISE entity attribution
(a chunk mentions the specific topics it is about, unlike the coarse
Topic->Type->Description join). So each synthetic fact is created as:

    (:Chunk {text, embedding}) -[:MENTIONS]-> (:Topic {low provenance})

The Chunk is embedded with the SAME model the real chunks use (mxbai-embed-large)
so vector search can find it. Low provenance (source_type=discussion/
auto_extracted, low confidence) lives on the Topic, where trust scoring reads it.
Each injected Chunk mentions exactly ONE Topic, so retrieving it surfaces only
the low-trust node - the point of the fix.

Env: KG_EMBED_URL (or OLLAMA_URL) + KG_EMBED_MODEL for embedding.

    python scripts/inject_low_trust.py --dry-run
    python scripts/inject_low_trust.py
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
EMBED_URL = os.environ.get("KG_EMBED_URL") or os.environ.get("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.environ.get("KG_EMBED_MODEL", "mxbai-embed-large")
BATCH = f"lowtrust_{datetime.date.today().isoformat()}"

LOW_TRUST_FACTS = [
    {"name": "Sensor Recalibration Interval",
     "text": "During an informal discussion it was suggested that the lab's vibration "
             "sensors should be recalibrated every 45 days, though nobody has confirmed "
             "this against the vendor manual.",
     "source_type": "discussion", "confidence_score": 0.35},
    {"name": "Legacy Dataset Naming Convention",
     "text": "An auto-extracted note claims older experiment datasets were named with the "
             "pattern EXP-<year>-<initials>, but the extraction confidence was low and no "
             "source document is linked.",
     "source_type": "auto_extracted", "confidence_score": 0.30},
    {"name": "Cluster GPU Booking Etiquette",
     "text": "Discussion notes suggest GPU slots on the lab cluster should be booked at "
             "most 3 days ahead; this was mentioned once in a chat and never formalized.",
     "source_type": "discussion", "confidence_score": 0.40},
    {"name": "Prototype Cooling Workaround",
     "text": "A hallway conversation recorded that pointing a desk fan at the prototype "
             "rack keeps thermal throttling away during long runs. Unverified, anecdotal.",
     "source_type": "discussion", "confidence_score": 0.30},
    {"name": "Deprecated Wiki Migration Status",
     "text": "An automatically extracted fragment states that the old lab wiki was 'mostly "
             "migrated' to the new system in spring, without listing which pages remain.",
     "source_type": "auto_extracted", "confidence_score": 0.35},
]

INJECT_CYPHER = f"""
    MERGE (t:{ENTITY_LABEL} {{{NAME_PROP}: $name}})
    SET t.description = $text, t.source_type = $source_type,
        t.confidence_score = $conf, t.created_at = datetime(),
        t.created_by = 'eval_injection', t.injected = true, t.injection_batch = $batch
    MERGE (c:Chunk {{chunk_id: $chunk_id}})
    SET c.text = $text, c.embedding = $vec,
        c.created_by = 'eval_injection', c.injected = true, c.injection_batch = $batch
    MERGE (c)-[:MENTIONS]->(t)
"""


def embed(text: str):
    payload = json.dumps({"model": EMBED_MODEL, "prompt": text}).encode()
    req = urllib.request.Request(f"{EMBED_URL}/api/embeddings", data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode()).get("embedding")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        for f in LOW_TRUST_FACTS:
            print(f"WOULD CREATE (:Chunk{{text,embedding}})-[:MENTIONS]->"
                  f"(:{ENTITY_LABEL} {{{NAME_PROP}: {f['name']!r}, "
                  f"source_type: {f['source_type']!r}, conf: {f['confidence_score']}}})")
        print(f"\nembedding via {EMBED_MODEL} @ {EMBED_URL}")
        return

    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    auth = (os.environ.get("NEO4J_USERNAME", "neo4j"), os.environ.get("NEO4J_PASSWORD", ""))
    db = os.environ.get("NEO4J_DATABASE", "neo4j")

    driver = GraphDatabase.driver(uri, auth=auth)
    with driver.session(database=db) as s:
        for f in LOW_TRUST_FACTS:
            vec = embed(f["text"])
            if not vec:
                sys.exit(f"embedding failed for {f['name']!r} - check KG_EMBED_URL/{EMBED_MODEL}")
            s.run(INJECT_CYPHER, name=f["name"], text=f["text"], source_type=f["source_type"],
                  conf=f["confidence_score"], vec=vec,
                  chunk_id=f"inject_lowtrust_{f['name']}", batch=BATCH)
            print(f"created (Chunk->MENTIONS->Topic, dim={len(vec)}): {f['name']}")
    driver.close()
    print(f"\nBatch tag: {BATCH}\nRemove all: MATCH (n {{injected: true}}) DETACH DELETE n")


if __name__ == "__main__":
    main()
