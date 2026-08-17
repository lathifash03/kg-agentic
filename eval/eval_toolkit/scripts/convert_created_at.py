"""Convert ISO-string timestamps into real Neo4j temporal values.

The chunker pipeline writes ``created_at`` as an ISO 8601 *string*. Neo4j
stores it as STRING, and every consumer silently degrades:

* ``to_datetime()`` returns ``None`` for a str, so ``age_days`` is ``None`` and
  the trust recency factor is pinned at 1.0;
* ``e.created_at < datetime()`` never matches, so temporal queries find nothing;
* ``freshness_reference()`` treats the node as timeless, reporting VALID.

Phase 1 does NOT fix this: ``apply_entity_temporal_metadata`` uses
``coalesce(e.created_at, datetime())``, and a present-but-wrongly-typed value
satisfies ``coalesce``, so the string survives the migration untouched.

Safety: refuses any non-localhost target unless ``--allow-remote`` is passed.
Writing to a shared KG needs a deliberate act, not a typo.

Usage::

    python eval/eval_toolkit/scripts/convert_created_at.py --target bolt://localhost:7687 --dry-run
    python eval/eval_toolkit/scripts/convert_created_at.py --target bolt://localhost:7687 --apply
"""

from __future__ import annotations

import argparse
from urllib.parse import urlparse

from neo4j import GraphDatabase

FIELDS = ("created_at", "updated_at", "validated_at", "trust_scored_at")


def survey(session):
    """Report, per label and field, how many values are STRING vs temporal."""
    rows = []
    for field in FIELDS:
        for r in session.run(
            f"MATCH (n) WHERE n.`{field}` IS NOT NULL "
            f"RETURN labels(n) AS labels, valueType(n.`{field}`) AS type, count(*) AS n "
            "ORDER BY n DESC"
        ):
            rows.append({"field": field, **r.data()})
    return rows


def convert(session, field: str, label: str, apply: bool) -> int:
    """Convert one label/field from ISO string to Neo4j datetime."""
    match = (
        f"MATCH (n:`{label}`) WHERE n.`{field}` IS NOT NULL "
        f"AND valueType(n.`{field}`) STARTS WITH 'STRING'"
    )
    if not apply:
        return session.run(f"{match} RETURN count(n) AS n").single()["n"]
    # datetime() parses ISO 8601; a value it cannot parse raises and aborts the
    # transaction rather than writing a half-converted graph.
    res = session.run(
        f"{match} SET n.`{field}` = datetime(n.`{field}`) RETURN count(n) AS n"
    )
    return res.single()["n"]


def main() -> None:
    p = argparse.ArgumentParser(description="Convert ISO-string timestamps to Neo4j temporals.")
    p.add_argument("--target", required=True)
    p.add_argument("--user", default="neo4j")
    p.add_argument("--password", default="password123")
    p.add_argument("--apply", action="store_true", help="Write. Without this, dry-run only.")
    p.add_argument("--allow-remote", action="store_true", help="Permit a non-localhost target.")
    args = p.parse_args()

    host = urlparse(args.target).hostname or ""
    if host not in ("localhost", "127.0.0.1", "::1") and not args.allow_remote:
        raise SystemExit(
            f"Refusing to touch non-local target {args.target!r} without --allow-remote. "
            "This is how a shared KG gets written to by accident."
        )

    driver = GraphDatabase.driver(args.target, auth=(args.user, args.password))
    try:
        with driver.session() as s:
            print(f"target: {args.target}   mode: {'APPLY' if args.apply else 'DRY-RUN'}\n")
            print("before:")
            for row in survey(s):
                print(f"  {row['field']:<16} {str(row['labels']):<14} {row['type']:<18} {row['n']}")

            # Only labels that actually carry a string value need touching.
            targets = {
                (row["field"], row["labels"][0])
                for row in survey(s)
                if str(row["type"]).startswith("STRING")
            }
            print("\nconverting:" if targets else "\nnothing to convert.")
            total = 0
            for field, label in sorted(targets):
                n = convert(s, field, label, args.apply)
                total += n
                verb = "converted" if args.apply else "would convert"
                print(f"  {verb} {n} :{label}.{field}")

            if args.apply:
                print("\nafter:")
                for row in survey(s):
                    print(f"  {row['field']:<16} {str(row['labels']):<14} {row['type']:<18} {row['n']}")
                left = [r for r in survey(s) if str(r["type"]).startswith("STRING")]
                if left:
                    raise SystemExit(f"FAILED: {len(left)} string timestamp group(s) remain: {left}")
                print("\nVerified: no STRING timestamps remain.")
            print(f"\ntotal: {total}")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
