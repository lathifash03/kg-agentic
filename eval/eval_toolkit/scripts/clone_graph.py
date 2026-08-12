"""Clone a Neo4j graph: read SOURCE read-only, write to TARGET.

Referenced by ``docs/NABHYLA_CONNECT.md``. The original lived in a session
scratchpad and was lost; this is the permanent version.

The source is only ever touched with ``execute_read`` -- this script cannot
write to the live Nabhyla graph even if misconfigured. The target is wiped and
rebuilt, so point it at a local sandbox, never at a shared KG.

Usage::

    python eval/eval_toolkit/scripts/clone_graph.py \
        --source bolt://100.110.179.78:7687 \
        --target bolt://localhost:7688

Node identity is carried across with a temporary ``_cloneId`` property (the
source elementId), which is indexed for the relationship pass and dropped at
the end.
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
import time
from typing import Any, Dict, List

from neo4j import GraphDatabase
from neo4j.time import Date, DateTime, Duration, Time

TEMP_KEY = "_cloneId"

# Neo4j temporal types are not JSON-serialisable and must survive a round trip
# intact: ``created_at`` drives the recency term of the trust score, so turning
# it into a bare string would silently change every score on restore.
_TEMPORAL = {"DateTime": DateTime, "Date": Date, "Time": Time, "Duration": Duration}


def encode(value: Any) -> Any:
    """Make a property JSON-safe, tagging Neo4j temporal types for decode()."""
    name = type(value).__name__
    if name in _TEMPORAL:
        return {"__neo4j_type__": name, "value": str(value)}
    if isinstance(value, list):
        return [encode(v) for v in value]
    return value


def _normalise_fractional(raw: str) -> str:
    """Trim sub-second digits to the 6 that :mod:`datetime` accepts.

    Neo4j stores nanosecond precision and renders 9 fractional digits, which
    ``datetime.fromisoformat`` rejects on Python < 3.11. Microseconds are far
    finer than anything the trust recency term needs.
    """
    return re.sub(r"(\.\d{6})\d+", r"\1", raw)


def decode(value: Any) -> Any:
    """Inverse of :func:`encode`."""
    if isinstance(value, dict) and "__neo4j_type__" in value:
        kind, raw = value["__neo4j_type__"], _normalise_fractional(value["value"])
        if kind == "DateTime":
            return DateTime.from_native(datetime.datetime.fromisoformat(raw))
        if kind == "Date":
            return Date.from_native(datetime.date.fromisoformat(raw))
        if kind == "Time":
            return Time.from_native(datetime.time.fromisoformat(raw))
        return Duration.from_iso_format(raw)
    if isinstance(value, list):
        return [decode(v) for v in value]
    return value


def _is_file(uri: str) -> bool:
    return uri.startswith("file://")


def _file_path(uri: str) -> str:
    return uri[len("file://") :]


def _retry(fn, *, attempts: int = 5, delay: float = 2.0, what: str = "operation"):
    """Run ``fn`` with retries -- the Nabhyla endpoint is intermittently down."""
    last: Exception | None = None
    for i in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last = exc
            print(f"  ! {what} failed (attempt {i}/{attempts}): {exc}", file=sys.stderr)
            if i < attempts:
                time.sleep(delay * i)
    raise RuntimeError(f"{what} failed after {attempts} attempts") from last


def read_nodes(driver) -> List[Dict[str, Any]]:
    """Read every node from the source (read-only transaction)."""

    def work(tx):
        rows = tx.run(
            "MATCH (n) RETURN elementId(n) AS eid, labels(n) AS labels, properties(n) AS props"
        )
        return [r.data() for r in rows]

    with driver.session() as s:
        return _retry(lambda: s.execute_read(work), what="read nodes")


def read_rels(driver) -> List[Dict[str, Any]]:
    """Read every relationship from the source (read-only transaction)."""

    def work(tx):
        rows = tx.run(
            "MATCH (a)-[r]->(b) "
            "RETURN elementId(a) AS start, elementId(b) AS end, type(r) AS type, "
            "properties(r) AS props"
        )
        return [r.data() for r in rows]

    with driver.session() as s:
        return _retry(lambda: s.execute_read(work), what="read relationships")


def wipe_target(driver) -> None:
    """Delete all nodes/relationships in the target."""
    with driver.session() as s:
        s.run("MATCH (n) CALL (n) { DETACH DELETE n } IN TRANSACTIONS OF 5000 ROWS")


def write_nodes(driver, nodes: List[Dict[str, Any]], batch: int = 500) -> int:
    """Recreate nodes in the target, batched by label set."""
    with driver.session() as s:
        s.run(f"CREATE INDEX clone_tmp IF NOT EXISTS FOR (n:`_Clone`) ON (n.`{TEMP_KEY}`)")
        by_labels: Dict[tuple, List[Dict[str, Any]]] = {}
        for n in nodes:
            by_labels.setdefault(tuple(sorted(n["labels"])), []).append(n)

        written = 0
        for labels, group in by_labels.items():
            label_str = "".join(f":`{l}`" for l in labels) + ":`_Clone`"
            cypher = (
                f"UNWIND $rows AS row CREATE (n{label_str}) "
                f"SET n = row.props, n.`{TEMP_KEY}` = row.eid"
            )
            for i in range(0, len(group), batch):
                chunk = group[i : i + batch]
                s.run(cypher, rows=chunk)
                written += len(chunk)
            print(f"  nodes {':'.join(labels)}: {len(group)}")
        return written


def write_rels(driver, rels: List[Dict[str, Any]], batch: int = 500) -> int:
    """Recreate relationships in the target, batched by type."""
    with driver.session() as s:
        by_type: Dict[str, List[Dict[str, Any]]] = {}
        for r in rels:
            by_type.setdefault(r["type"], []).append(r)

        written = 0
        for rtype, group in by_type.items():
            cypher = (
                "UNWIND $rows AS row "
                f"MATCH (a:`_Clone` {{`{TEMP_KEY}`: row.start}}) "
                f"MATCH (b:`_Clone` {{`{TEMP_KEY}`: row.end}}) "
                f"CREATE (a)-[r:`{rtype}`]->(b) SET r = row.props"
            )
            for i in range(0, len(group), batch):
                chunk = group[i : i + batch]
                s.run(cypher, rows=chunk)
                written += len(chunk)
            print(f"  rels  {rtype}: {len(group)}")
        return written


def cleanup(driver) -> None:
    """Drop the temporary clone marker label, property and index."""
    with driver.session() as s:
        s.run(
            f"MATCH (n:`_Clone`) CALL (n) {{ REMOVE n:`_Clone` REMOVE n.`{TEMP_KEY}` }} "
            "IN TRANSACTIONS OF 5000 ROWS"
        )
        s.run("DROP INDEX clone_tmp IF EXISTS")


def read_vector_indexes(source) -> List[Dict[str, Any]]:
    """Read vector-index definitions from the source."""
    with source.session() as s:
        return _retry(
            lambda: [
                r.data()
                for r in s.run(
                    "SHOW VECTOR INDEXES YIELD name, labelsOrTypes, properties, options "
                    "RETURN name, labelsOrTypes, properties, options"
                )
            ],
            what="read vector indexes",
        )


def create_vector_indexes(target, rows: List[Dict[str, Any]]) -> None:
    """Recreate vector indexes on the target (dimensions + similarity preserved)."""
    with target.session() as t:
        for idx in rows:
            cfg = (idx.get("options") or {}).get("indexConfig", {})
            dims = cfg.get("vector.dimensions")
            sim = cfg.get("vector.similarity_function", "cosine")
            label = idx["labelsOrTypes"][0]
            prop = idx["properties"][0]
            t.run(
                f"CREATE VECTOR INDEX `{idx['name']}` IF NOT EXISTS "
                f"FOR (n:`{label}`) ON (n.`{prop}`) "
                "OPTIONS {indexConfig: {`vector.dimensions`: $d, "
                "`vector.similarity_function`: $s}}",
                d=dims,
                s=str(sim).lower(),
            )
            print(f"  vector index '{idx['name']}' on :{label}({prop}) dims={dims} sim={sim}")


def main() -> None:
    p = argparse.ArgumentParser(description="Clone a Neo4j graph (read-only source).")
    p.add_argument("--source", required=True, help="Source bolt URI (read-only).")
    p.add_argument("--target", required=True, help="Target bolt URI (WIPED and rebuilt).")
    p.add_argument("--source-user", default="neo4j")
    p.add_argument("--source-password", default="password123")
    p.add_argument("--target-user", default="neo4j")
    p.add_argument("--target-password", default="password123")
    args = p.parse_args()

    if args.source == args.target:
        raise SystemExit("Refusing to run: source and target are the same URI.")

    src = tgt = None
    try:
        # ---- read side: a bolt graph, or a JSON dump written by an earlier run
        if _is_file(args.source):
            path = _file_path(args.source)
            print(f"Reading dump {path}...")
            with open(path) as fh:
                payload = json.load(fh)
            nodes = [
                {**n, "props": {k: decode(v) for k, v in n["props"].items()}}
                for n in payload["nodes"]
            ]
            rels = [
                {**r, "props": {k: decode(v) for k, v in r["props"].items()}}
                for r in payload["relationships"]
            ]
            vector_indexes = payload.get("vector_indexes", [])
        else:
            src = GraphDatabase.driver(
                args.source, auth=(args.source_user, args.source_password)
            )
            print(f"Reading source {args.source} (read-only)...")
            nodes = read_nodes(src)
            rels = read_rels(src)
            vector_indexes = read_vector_indexes(src)
        print(f"  read {len(nodes)} nodes, {len(rels)} relationships")

        # ---- write side: a bolt graph, or a JSON dump (backup)
        if _is_file(args.target):
            path = _file_path(args.target)
            payload = {
                "nodes": [
                    {**n, "props": {k: encode(v) for k, v in n["props"].items()}}
                    for n in nodes
                ],
                "relationships": [
                    {**r, "props": {k: encode(v) for k, v in r["props"].items()}}
                    for r in rels
                ],
                "vector_indexes": vector_indexes,
            }
            with open(path, "w") as fh:
                json.dump(payload, fh)
            print(f"\nDone. Wrote dump {path}: {len(nodes)} nodes, {len(rels)} rels")
            return

        tgt = GraphDatabase.driver(args.target, auth=(args.target_user, args.target_password))
        print(f"Wiping target {args.target}...")
        wipe_target(tgt)

        print("Writing nodes...")
        n_written = write_nodes(tgt, nodes)
        print("Writing relationships...")
        r_written = write_rels(tgt, rels)

        print("Creating vector indexes...")
        create_vector_indexes(tgt, vector_indexes)

        print("Cleaning up temporary clone markers...")
        cleanup(tgt)

        with tgt.session() as s:
            n = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            r = s.run("MATCH ()-[x]->() RETURN count(x) AS c").single()["c"]
        print(f"\nDone. target nodes={n} (wrote {n_written}), rels={r} (wrote {r_written})")
        if n != len(nodes) or r != len(rels):
            raise SystemExit(f"MISMATCH: source {len(nodes)}/{len(rels)} vs target {n}/{r}")
        print("Verified: clone matches source counts exactly.")
    finally:
        if src is not None:
            src.close()
        if tgt is not None:
            tgt.close()


if __name__ == "__main__":
    main()
