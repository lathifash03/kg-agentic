"""Step 2c — Inject LOW-TRUST cases for ground-truth category (c).

ADAPTED to Nabhyla's schema: the retriever reads text from separate
``:Description`` nodes two hops from the entity
(``Topic-[:HAS_TYPE]->Type-[:HAS_DESCRIPTION]->Description {text}``), NOT from a
``description`` property on the entity. So each synthetic fact is created as
that full subgraph, or it would be invisible to retrieval. The low provenance
that makes it category-c (source_type=discussion/auto_extracted, low
confidence) lives on the Topic, where trust scoring reads it.

Every injected node gets  injected: true  +  injection_batch  so the whole
batch is auditable and removable:

    MATCH (n {injected: true}) DETACH DELETE n

Usage:
    python scripts/inject_low_trust.py --dry-run
    python scripts/inject_low_trust.py
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
# The text-node label + property the retriever reads (Nabhyla: Description.text).
CHUNK_LABEL = os.environ.get("KG_CHUNK_LABEL", "Description")
TEXT_PROP = os.environ.get("KG_CHUNK_TEXT_PROP", "text")
BATCH = f"lowtrust_{datetime.date.today().isoformat()}"

# --- EDIT ME: fictional-but-plausible facts. They must NOT overlap with real
# graph content (that would contaminate category a).
LOW_TRUST_FACTS = [
    {
        "name": "Sensor Recalibration Interval",
        "text": "During an informal discussion it was suggested that the lab's "
                "vibration sensors should be recalibrated every 45 days, "
                "though nobody has confirmed this against the vendor manual.",
        "source_type": "discussion",
        "confidence_score": 0.35,
    },
    {
        "name": "Legacy Dataset Naming Convention",
        "text": "An auto-extracted note claims older experiment datasets were "
                "named with the pattern EXP-<year>-<initials>, but the "
                "extraction confidence was low and no source document is linked.",
        "source_type": "auto_extracted",
        "confidence_score": 0.30,
    },
    {
        "name": "Cluster GPU Booking Etiquette",
        "text": "Discussion notes suggest GPU slots on the lab cluster should "
                "be booked at most 3 days ahead; this was mentioned once in "
                "a chat and never formalized.",
        "source_type": "discussion",
        "confidence_score": 0.40,
    },
    {
        "name": "Prototype Cooling Workaround",
        "text": "A hallway conversation recorded that pointing a desk fan at "
                "the prototype rack keeps thermal throttling away during long "
                "runs. Unverified, anecdotal.",
        "source_type": "discussion",
        "confidence_score": 0.30,
    },
    {
        "name": "Deprecated Wiki Migration Status",
        "text": "An automatically extracted fragment states that the old lab "
                "wiki was 'mostly migrated' to the new system in spring, "
                "without listing which pages remain.",
        "source_type": "auto_extracted",
        "confidence_score": 0.35,
    },
]

# One synthetic fact -> one Topic + its OWN Type + Description, matching the
# 2-hop shape. A dedicated Type per fact avoids the shared-Type fan-out that
# would otherwise link a Description back to unrelated real Topics.
INJECT_CYPHER = f"""
    MERGE (t:{ENTITY_LABEL} {{{NAME_PROP}: $name}})
    SET t.description = $text,
        t.source_type = $source_type,
        t.confidence_score = $conf,
        t.created_at = datetime(),
        t.created_by = 'eval_injection',
        t.injected = true,
        t.injection_batch = $batch
    MERGE (ty:Type {{{NAME_PROP}: $type_name}})
    SET ty.injected = true, ty.injection_batch = $batch
    MERGE (d:`{CHUNK_LABEL}` {{{NAME_PROP}: $desc_name}})
    SET d.`{TEXT_PROP}` = $text,
        d.created_by = 'eval_injection',
        d.injected = true, d.injection_batch = $batch
    MERGE (t)-[:HAS_TYPE]->(ty)
    MERGE (ty)-[:HAS_DESCRIPTION]->(d)
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        for f in LOW_TRUST_FACTS:
            print(f"WOULD CREATE (:{ENTITY_LABEL} {{{NAME_PROP}: {f['name']!r}, "
                  f"source_type: {f['source_type']!r}, "
                  f"confidence_score: {f['confidence_score']}}})"
                  f"  + :Type + :{CHUNK_LABEL}{{{TEXT_PROP}}} (2-hop)")
        return

    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    auth = (os.environ.get("NEO4J_USERNAME", "neo4j"),
            os.environ.get("NEO4J_PASSWORD", ""))
    db = os.environ.get("NEO4J_DATABASE", "neo4j")

    driver = GraphDatabase.driver(uri, auth=auth)
    with driver.session(database=db) as s:
        for f in LOW_TRUST_FACTS:
            s.run(
                INJECT_CYPHER,
                name=f["name"], text=f["text"],
                source_type=f["source_type"], conf=f["confidence_score"],
                type_name=f"Injected Classification: {f['name']}",
                desc_name=f"Injected Description: {f['name']}",
                batch=BATCH,
            )
            print("created (2-hop):", f["name"])
    driver.close()
    print(f"\nBatch tag: {BATCH}")
    print("Remove all: MATCH (n {injected: true}) DETACH DELETE n")


if __name__ == "__main__":
    main()
