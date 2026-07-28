"""Disposable seed data for demoing the verified-answer path end to end.

This repo has no ingestion pipeline, so an empty Neo4j gives the verifier
nothing to retrieve. This script MERGEs a handful of entities and ``:Chunk``
nodes - the exact ``(:Chunk {text})-[:MENTIONS]->(entity)`` shape the retriever
expects - so at least one run exercises the real trust/temporal/faithfulness
gates against actual content.

It is a demo helper, deliberately **not** wired into the app. Run it directly::

    python tests/seed_demo_data.py            # seed
    python tests/seed_demo_data.py --clear    # remove everything it created

Everything it writes carries ``created_by='seed_demo_data'`` so ``--clear``
removes exactly its own nodes and nothing else in the graph.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, ".")

from kg_agent.config import get_config  # noqa: E402
from kg_agent.neo4j_client import Neo4jClient  # noqa: E402

MARKER = "seed_demo_data"

# (name, description, source_type, confidence, age_in_days)
# The ages are chosen to exercise the temporal gate: two fresh, one stale enough
# to trip KG_OUTDATED_THRESHOLD_DAYS (default 30).
ENTITIES = [
    (
        "Robotic Mobile Fulfillment System",
        "A Robotic Mobile Fulfillment System (RMFS) is a warehouse automation "
        "system where mobile robots carry movable shelf racks to stationary "
        "picking stations, so human pickers never walk to the inventory.",
        "paper",
        0.9,
        3,
    ),
    (
        "Picking Station",
        "A picking station is the fixed workstation in an RMFS where a human "
        "operator picks items from racks delivered by the robots.",
        "paper",
        0.85,
        5,
    ),
    (
        "Mobile Robot",
        "Mobile robots in an RMFS drive underneath a storage rack, lift it and "
        "transport it to a picking station.",
        "meeting",
        0.7,
        10,
    ),
    (
        "Legacy Conveyor Layout",
        "An older fixed-conveyor warehouse layout that the RMFS design replaced.",
        "auto_extracted",
        0.4,
        400,  # deliberately stale: should be flagged OUTDATED
    ),
]


def seed(client: Neo4jClient, cfg) -> None:
    """Create the demo entities and their supporting text chunks."""
    label = cfg.schema.entity_label
    name_prop = cfg.schema.entity_name_property
    now = datetime.now(timezone.utc)

    for name, description, source_type, confidence, age_days in ENTITIES:
        # Pass a tz-aware datetime OBJECT, not an ISO string: the Neo4j driver
        # maps it to a real temporal value. A plain string is stored as text,
        # which the temporal-validity and recency checks (both requiring an
        # actual datetime) silently ignore - the entity would look timeless.
        stamp = now - timedelta(days=age_days)
        client.run_write(
            f"""
            MERGE (e:`{label}` {{{name_prop}: $name}})
            SET e.description = $description,
                e.source_type = $source_type,
                e.confidence_score = $confidence,
                e.created_at = $stamp,
                e.updated_at = $stamp,
                e.created_by = $marker
            WITH e
            // The retriever looks for (:Chunk {{text}})-[:MENTIONS]->(entity);
            // matching that shape is what makes the entity retrievable.
            MERGE (c:Chunk {{text: $description}})
            SET c.created_at = $stamp,
                c.created_by = $marker
            MERGE (c)-[r:MENTIONS]->(e)
            SET r.created_at = $stamp
            """,
            name=name,
            description=description,
            source_type=source_type,
            confidence=confidence,
            stamp=stamp,
            marker=MARKER,
        )
        print(f"  seeded {name!r} (source_type={source_type}, age={age_days}d)")


def clear(client: Neo4jClient) -> None:
    """Delete only the nodes this script created."""
    result = client.run_write(
        """
        MATCH (n) WHERE n.created_by = $marker
        DETACH DELETE n
        """,
        marker=MARKER,
    )
    print(f"  cleared seed nodes: {result}")


def main() -> None:
    """Seed or clear the demo data."""
    parser = argparse.ArgumentParser(description="Seed disposable demo KG data.")
    parser.add_argument("--clear", action="store_true", help="Remove seeded nodes.")
    args = parser.parse_args()

    cfg = get_config()
    with Neo4jClient.from_config(cfg) as client:
        if not client.verify_connectivity():
            raise SystemExit("Could not connect to Neo4j - check your .env")
        if args.clear:
            clear(client)
        else:
            seed(client, cfg)
            print("\nDone. Now run the Phase 1 + Phase 3 setup so the gates see it:")
            print("  python -m kg_agent.cli --setup --query 'What is an RMFS?'")


if __name__ == "__main__":
    main()
