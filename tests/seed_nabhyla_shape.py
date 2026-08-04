"""Disposable LOCAL seed data shaped like Nabhyla's real KG schema.

Nabhyla's graph (confirmed via a read-only recon export, see
docs/NABHYLA_CONNECT.md) has **no** ``:Chunk``/``:Document``/``MENTIONS`` at
all. Its shape is a concept-extraction graph from a single thesis::

    (:Topic)-[:HAS_TYPE]->(:Type)-[:HAS_DESCRIPTION]->(:Description {text})
    (:Topic)-[:HAS_SUBTOPIC]->(:Topic)
    (:Topic)-[:HAS_SOURCE]->(:Source)
    (:Agent)-[:WRITES_ABOUT]->(:Topic)
    (:Agent)-[:ROLE_IN_PAPER]->(:Role)

This script recreates a small slice of that exact shape on a LOCAL, disposable
Neo4j (e.g. the Docker workaround in README's Troubleshooting section) so the
new configurable chunk/entity-path retrieval (``KG_CHUNK_LABEL`` /
``KG_CHUNK_TEXT_PROP`` / ``KG_CHUNK_TO_ENTITY_PATTERN``) can be exercised end
to end against real Cypher, WITHOUT ever touching Nabhyla's shared instance.

SAFETY: refuses to run if ``NEO4J_URI`` looks like Nabhyla's known Tailscale
address - this script performs writes (``MERGE``) and must only ever run
against your own local/disposable database.

Run directly (after pointing .env at a local Neo4j with the Nabhyla-shaped
config from .env.example's "Profil skema Nabhyla" block)::

    python tests/seed_nabhyla_shape.py            # seed
    python tests/seed_nabhyla_shape.py --clear    # remove everything it created
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

sys.path.insert(0, ".")

from kg_agent.config import get_config  # noqa: E402
from kg_agent.neo4j_client import Neo4jClient  # noqa: E402

MARKER = "seed_nabhyla_shape"

# Known address of Nabhyla's shared, colleague-owned instance. This script
# writes data, so it must never be pointed at it.
_FORBIDDEN_URI_SUBSTRINGS = ("100.95.227.48",)

# (topic_name, type_name, description_text)
# Mirrors the real recon sample (Order Picking Process Optimization thesis):
# a Topic classified under a Type, with the substantive text living on the
# Description node two hops away - never directly on the Topic itself.
TOPICS = [
    (
        "Pnp",
        "Existing Research",
        "PNP (Pick-and-Pass) is an order picking method where totes move "
        "through a sequence of fixed zones, each staffed by a picker "
        "responsible only for items located in that zone.",
    ),
    (
        "Throughput",
        "Result",
        "Throughput is measured as the number of order lines completed per "
        "hour; the two-phase assignment method improved throughput by 18% "
        "over the FCFS baseline in the simulated scenario.",
    ),
    (
        "Pod Status Cycle",
        "Existing Research",
        "A pod cycles through Idle, Requested, InTransit and AtStation "
        "states as the RMFS controller assigns it to picking tasks.",
    ),
]


def seed(client: Neo4jClient, cfg) -> None:
    """Create the Topic/Type/Description slice plus one Source/Agent."""
    now = datetime.now(timezone.utc)

    client.run_write(
        """
        MERGE (s:Source {name: 'Thesis Demo Source', id: 'Thesis Demo Source'})
        SET s.created_by = $marker, s.created_at = $now
        """,
        marker=MARKER,
        now=now,
    )
    client.run_write(
        """
        MERGE (a:Agent {name: 'Demo Author', id: 'Demo Author'})
        SET a.created_by = $marker, a.created_at = $now
        """,
        marker=MARKER,
        now=now,
    )

    for topic_name, type_name, text in TOPICS:
        client.run_write(
            """
            MERGE (t:Topic {name: $topic_name, id: $topic_name})
            SET t.created_by = $marker, t.created_at = $now
            MERGE (ty:Type {name: $type_name, id: $type_name, domain: 'paper'})
            SET ty.created_by = $marker
            MERGE (t)-[:HAS_TYPE]->(ty)
            MERGE (d:Description {
                name: 'Description ' + $topic_name,
                id: 'Description ' + $topic_name,
                text: $text,
                typeName: $type_name,
                topicName: $topic_name
            })
            SET d.created_by = $marker, d.created_at = $now
            MERGE (ty)-[:HAS_DESCRIPTION]->(d)
            WITH t
            MATCH (s:Source {name: 'Thesis Demo Source'})
            MERGE (t)-[:HAS_SOURCE]->(s)
            WITH t
            MATCH (a:Agent {name: 'Demo Author'})
            MERGE (a)-[:WRITES_ABOUT]->(t)
            """,
            topic_name=topic_name,
            type_name=type_name,
            text=text,
            marker=MARKER,
            now=now,
        )
        print(f"  seeded Topic {topic_name!r} (type={type_name})")

    # One subtopic edge, mirroring the real HAS_SUBTOPIC hierarchy.
    client.run_write(
        """
        MATCH (a:Topic {name: 'Pnp'}), (b:Topic {name: 'Throughput'})
        MERGE (a)-[:HAS_SUBTOPIC]->(b)
        """
    )


def clear(client: Neo4jClient) -> None:
    """Delete only the nodes this script created."""
    result = client.run_write(
        "MATCH (n) WHERE n.created_by = $marker DETACH DELETE n",
        marker=MARKER,
    )
    print(f"  cleared seed nodes: {result}")


def main() -> None:
    """Seed or clear the local Nabhyla-shaped fixture."""
    parser = argparse.ArgumentParser(
        description="Seed a LOCAL, disposable Nabhyla-shaped KG fixture."
    )
    parser.add_argument("--clear", action="store_true", help="Remove seeded nodes.")
    args = parser.parse_args()

    cfg = get_config()
    if any(bad in cfg.neo4j.uri for bad in _FORBIDDEN_URI_SUBSTRINGS):
        raise SystemExit(
            f"Refusing to run: NEO4J_URI ({cfg.neo4j.uri!r}) looks like Nabhyla's "
            "shared instance. This script WRITES data and must only target your "
            "own local/disposable Neo4j. Point .env at a local instance first."
        )

    with Neo4jClient.from_config(cfg) as client:
        if not client.verify_connectivity():
            raise SystemExit("Could not connect to Neo4j - check your .env")
        if args.clear:
            clear(client)
        else:
            seed(client, cfg)
            print(
                "\nDone. Now query with the Nabhyla schema profile from "
                ".env.example, e.g.:\n"
                "  python -m kg_agent.cli --query \"What is PNP?\""
            )


if __name__ == "__main__":
    main()
