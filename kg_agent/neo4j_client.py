"""Neo4j connection, base read queries and the Phase 1 temporal-metadata migration.

This module is the single data-access layer for the agentic verification stack.
It wraps the official ``neo4j`` driver (a transitive dependency of
``langchain-neo4j`` already used by the pipeline) and exposes:

* connection management (context-manager friendly, with explicit ``close``);
* generic ``run_read`` / ``run_write`` helpers;
* **Phase 1** - idempotent migrations that stamp temporal metadata onto every
  entity node and semantic relationship, plus index creation;
* reusable entity getters consumed by Phases 2-4.

Run it directly to inspect the graph or apply Phase 1::

    python -m kg_agent.neo4j_client            # read-only status report
    python -m kg_agent.neo4j_client --migrate  # apply Phase 1 metadata + indexes
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from neo4j import Driver, GraphDatabase

from kg_agent.config import Config, get_config

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def escape_label(label: str) -> str:
    """Backtick-escape a Neo4j label so it is safe to interpolate into Cypher.

    Labels cannot be passed as query parameters, so they must be inlined. The
    value comes from trusted config, but we still escape embedded backticks to
    avoid breaking the query.
    """
    return "`" + label.replace("`", "``") + "`"


# Backward-compatible alias - kept in case anything outside this module still
# imports the original underscore-prefixed name.
_escape_label = escape_label


def to_datetime(value: Any) -> Optional[datetime]:
    """Convert a Neo4j temporal value into a timezone-aware Python ``datetime``.

    Parameters
    ----------
    value
        A ``neo4j.time.DateTime``, a Python ``datetime``, or ``None``.

    Returns
    -------
    Optional[datetime]
        A timezone-aware UTC-comparable datetime, or ``None`` if ``value`` is
        ``None``. Naive datetimes are assumed to be UTC.
    """
    if value is None:
        return None
    if hasattr(value, "to_native"):  # neo4j.time.DateTime
        value = value.to_native()
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    return None


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #
class Neo4jClient:
    """Thin, reusable wrapper around the Neo4j Python driver.

    The client owns a single :class:`neo4j.Driver` (a connection pool) and is
    safe to share across the temporal-validity, trust-scoring and verifier
    modules. Use it as a context manager to guarantee the driver is closed::

        with Neo4jClient.from_config(get_config()) as client:
            client.run_phase1_migration()
    """

    def __init__(self, driver: Driver, config: Config) -> None:
        self._driver = driver
        self.config = config
        self._label = _escape_label(config.schema.entity_label)
        self._name_prop = config.schema.entity_name_property
        self._id_prop = config.schema.entity_id_property

    # -- construction / lifecycle ----------------------------------------- #
    @classmethod
    def from_config(cls, config: Optional[Config] = None) -> "Neo4jClient":
        """Create a client from a :class:`Config` (defaults to ``get_config()``)."""
        config = config or get_config()
        # Temporal/trust properties are intentionally null until populated, so
        # "unknown property key" notifications are expected and noisy. Disable
        # that notification category where the driver supports it.
        kwargs: Dict[str, Any] = {"auth": (config.neo4j.username, config.neo4j.password)}
        try:
            driver = GraphDatabase.driver(
                config.neo4j.uri,
                notifications_disabled_categories=["UNRECOGNIZED", "DEPRECATION"],
                **kwargs,
            )
        except (TypeError, ValueError):  # older driver without the kwarg
            logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)
            driver = GraphDatabase.driver(config.neo4j.uri, **kwargs)
        return cls(driver, config)

    def close(self) -> None:
        """Close the underlying driver and release all pooled connections."""
        self._driver.close()

    def __enter__(self) -> "Neo4jClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def verify_connectivity(self) -> bool:
        """Return ``True`` if the database is reachable, ``False`` otherwise."""
        try:
            self._driver.verify_connectivity()
            return True
        except Exception as exc:  # pragma: no cover - network dependent
            logger.warning("Neo4j connectivity check failed: %s", exc)
            return False

    # -- generic query helpers -------------------------------------------- #
    def run_read(self, cypher: str, **params: Any) -> List[Dict[str, Any]]:
        """Execute a read query and return rows as a list of dictionaries.

        Parameters
        ----------
        cypher
            A Cypher read statement.
        **params
            Query parameters bound by name.

        Returns
        -------
        List[Dict[str, Any]]
            One dictionary per result record.
        """
        with self._driver.session(database=self.config.neo4j.database) as session:
            result = session.run(cypher, **params)
            return [record.data() for record in result]

    def run_write(self, cypher: str, **params: Any) -> Dict[str, Any]:
        """Execute a write query and return the result summary counters.

        Returns
        -------
        Dict[str, Any]
            A dictionary of mutation counters (e.g. ``properties_set``,
            ``indexes_added``) extracted from the query summary.
        """
        with self._driver.session(database=self.config.neo4j.database) as session:
            result = session.run(cypher, **params)
            records = [record.data() for record in result]
            summary = result.consume()
            counters = summary.counters
            return {
                "records": records,
                "properties_set": counters.properties_set,
                "labels_added": counters.labels_added,
                "relationships_created": counters.relationships_created,
                "nodes_created": counters.nodes_created,
                "indexes_added": counters.indexes_added,
            }

    # ------------------------------------------------------------------ #
    # Phase 1 - temporal metadata migration
    # ------------------------------------------------------------------ #
    def apply_entity_temporal_metadata(self) -> int:
        """Stamp temporal metadata onto every entity node (idempotent).

        Adds, *only where missing*: ``created_at``, ``updated_at``,
        ``source_type``, ``confidence_score``, ``is_outdated`` and
        ``created_by``. ``validated_at`` is intentionally left null until a
        node is actually validated (a null property is simply absent in Neo4j).

        Returns
        -------
        int
            The number of entity nodes matched by the migration.
        """
        defaults = self.config.temporal_defaults
        created_at_expr = "datetime()" if defaults.set_created_at_to_now else "null"
        cypher = f"""
            MATCH (e:{self._label})
            SET e.created_at       = coalesce(e.created_at, {created_at_expr}),
                e.updated_at       = coalesce(e.updated_at, {created_at_expr}),
                e.source_type      = coalesce(e.source_type, $source_type),
                e.confidence_score = coalesce(e.confidence_score, $confidence),
                e.is_outdated      = coalesce(e.is_outdated, false),
                e.created_by       = coalesce(e.created_by, $created_by)
            RETURN count(e) AS entities
        """
        result = self.run_write(
            cypher,
            source_type=defaults.default_source_type,
            confidence=defaults.default_confidence_score,
            created_by=defaults.default_created_by,
        )
        count = result["records"][0]["entities"] if result["records"] else 0
        logger.info("Phase 1: temporal metadata applied to %s entities", count)
        return count

    def apply_relationship_temporal_metadata(self) -> int:
        """Stamp temporal metadata onto semantic relationships (idempotent).

        Adds, *only where missing*: ``created_at``, ``valid_from``,
        ``confidence``. ``valid_until`` and ``superseded_by`` are left null
        until a relationship is actually closed/superseded. Structural plumbing
        relationships (``MENTIONS``/``NEXT``/``PART_OF`` by default) are skipped.

        Returns
        -------
        int
            The number of relationships matched by the migration.
        """
        defaults = self.config.temporal_defaults
        schema = self.config.schema
        created_at_expr = "datetime()" if defaults.set_created_at_to_now else "null"

        if schema.temporal_relationships_entity_only:
            match = f"MATCH (a:{self._label})-[r]->(b:{self._label})"
        else:
            match = "MATCH ()-[r]->()"

        cypher = f"""
            {match}
            WHERE NOT type(r) IN $structural
            SET r.created_at  = coalesce(r.created_at, {created_at_expr}),
                r.valid_from  = coalesce(r.valid_from, r.created_at, {created_at_expr}),
                r.confidence  = coalesce(r.confidence, $rel_conf)
            RETURN count(r) AS relationships
        """
        result = self.run_write(
            cypher,
            structural=schema.structural_relationship_types,
            rel_conf=defaults.default_relationship_confidence,
        )
        count = result["records"][0]["relationships"] if result["records"] else 0
        logger.info("Phase 1: temporal metadata applied to %s relationships", count)
        return count

    def create_temporal_indexes(self) -> List[str]:
        """Create indexes used for fast temporal/entity querying (idempotent).

        Creates range indexes on ``created_at``, the entity name property and
        the trust-score property. ``CREATE INDEX ... IF NOT EXISTS`` makes this
        safe to call repeatedly.

        Returns
        -------
        List[str]
            The names of the indexes that were ensured.
        """
        trust_prop = self.config.trust.trust_score_property
        statements = {
            "entity_created_at": f"CREATE INDEX entity_created_at IF NOT EXISTS "
            f"FOR (e:{self._label}) ON (e.created_at)",
            "entity_name": f"CREATE INDEX entity_name IF NOT EXISTS "
            f"FOR (e:{self._label}) ON (e.{self._name_prop})",
            "entity_trust_score": f"CREATE INDEX entity_trust_score IF NOT EXISTS "
            f"FOR (e:{self._label}) ON (e.{trust_prop})",
        }
        ensured: List[str] = []
        for name, stmt in statements.items():
            self.run_write(stmt)
            ensured.append(name)
        logger.info("Phase 1: ensured indexes %s", ensured)
        return ensured

    def run_phase1_migration(self) -> Dict[str, Any]:
        """Run the full Phase 1 migration: metadata on nodes + relationships + indexes.

        Returns
        -------
        Dict[str, Any]
            Summary with keys ``entities``, ``relationships`` and ``indexes``.
        """
        summary = {
            "entities": self.apply_entity_temporal_metadata(),
            "relationships": self.apply_relationship_temporal_metadata(),
            "indexes": self.create_temporal_indexes(),
        }
        logger.info("Phase 1 migration complete: %s", summary)
        return summary

    # ------------------------------------------------------------------ #
    # Base read queries (consumed by Phases 2-4)
    # ------------------------------------------------------------------ #
    def _entity_return_clause(self, var: str = "e") -> str:
        """Build the shared RETURN projection for an entity node bound to ``var``."""
        trust_prop = self.config.trust.trust_score_property
        return f"""
            elementId({var})                          AS element_id,
            {var}.{self._id_prop}                     AS id,
            {var}.{self._name_prop}                   AS name,
            [l IN labels({var}) WHERE l <> $entity_label] AS types,
            {var}.description                         AS description,
            {var}.created_at                          AS created_at,
            {var}.updated_at                          AS updated_at,
            {var}.validated_at                        AS validated_at,
            {var}.source_type                         AS source_type,
            {var}.confidence_score                    AS confidence_score,
            {var}.is_outdated                         AS is_outdated,
            {var}.created_by                          AS created_by,
            {var}.superseded_by                       AS superseded_by,
            {var}.pagerank                            AS pagerank,
            {var}.{trust_prop}                        AS trust_score
        """

    def _normalise_entity(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Convert raw Neo4j temporal values in an entity row to Python datetimes."""
        for key in ("created_at", "updated_at", "validated_at"):
            row[key] = to_datetime(row.get(key))
        return row

    def get_all_entities(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return every entity node as a normalised dictionary.

        Parameters
        ----------
        limit
            Optional cap on the number of entities returned.

        Returns
        -------
        List[Dict[str, Any]]
            Entity rows with Python-native datetimes (see ``to_datetime``).
        """
        cypher = f"""
            MATCH (e:{self._label})
            RETURN {self._entity_return_clause('e')}
            {"LIMIT $limit" if limit is not None else ""}
        """
        params: Dict[str, Any] = {"entity_label": self.config.schema.entity_label}
        if limit is not None:
            params["limit"] = limit
        return [self._normalise_entity(r) for r in self.run_read(cypher, **params)]

    def get_entities_by_names(self, names: List[str]) -> List[Dict[str, Any]]:
        """Return entity nodes whose name matches any value in ``names``.

        Matching is case-insensitive on the configured name property. This is
        the primary hook used by the Phase 4 verifier to score the specific
        nodes returned by KG-RAG retrieval.

        Parameters
        ----------
        names
            Entity names (as surfaced by retrieval) to look up.

        Returns
        -------
        List[Dict[str, Any]]
            Matching entity rows (normalised). May be empty.
        """
        if not names:
            return []
        cypher = f"""
            MATCH (e:{self._label})
            WHERE toLower(e.{self._name_prop}) IN $names
            RETURN {self._entity_return_clause('e')}
        """
        params = {
            "entity_label": self.config.schema.entity_label,
            "names": [n.lower() for n in names],
        }
        return [self._normalise_entity(r) for r in self.run_read(cypher, **params)]

    def count_entities(self) -> int:
        """Return the total number of entity nodes in the graph."""
        cypher = f"MATCH (e:{self._label}) RETURN count(e) AS c"
        rows = self.run_read(cypher)
        return rows[0]["c"] if rows else 0

    def get_supersession_edges(
        self, names: Optional[List[str]] = None
    ) -> List[Dict[str, str]]:
        """Return ``(old -> new)`` supersession pairs declared via relationships.

        A node is considered superseded when it is the source of a
        ``SUPERSEDED_BY`` relationship (type configurable) pointing at a newer
        entity. Used by the Phase 2 validity check.

        Parameters
        ----------
        names
            Optional case-insensitive filter on the *old* node's name.

        Returns
        -------
        List[Dict[str, str]]
            Dicts with keys ``old_element_id``, ``old_name``, ``new_element_id``
            and ``new_name``.
        """
        rel = _escape_label(self.config.schema.supersedes_relationship)
        where = ""
        params: Dict[str, Any] = {}
        if names:
            where = f"WHERE toLower(old.{self._name_prop}) IN $names"
            params["names"] = [n.lower() for n in names]
        cypher = f"""
            MATCH (old:{self._label})-[:{rel}]->(new:{self._label})
            {where}
            RETURN elementId(old) AS old_element_id,
                   old.{self._name_prop} AS old_name,
                   elementId(new) AS new_element_id,
                   new.{self._name_prop} AS new_name
        """
        return self.run_read(cypher, **params)


if __name__ == "__main__":  # pragma: no cover - manual entry point
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Inspect or migrate the KG (Phase 1).")
    parser.add_argument(
        "--migrate",
        action="store_true",
        help="Apply Phase 1 temporal metadata + indexes (writes to the graph).",
    )
    args = parser.parse_args()

    with Neo4jClient.from_config() as client:
        if not client.verify_connectivity():
            raise SystemExit("Could not connect to Neo4j - check kg_agent/config.py")

        print(f"Connected. Entity nodes: {client.count_entities()}")

        if args.migrate:
            print("Applying Phase 1 migration...")
            print(client.run_phase1_migration())
        else:
            sample = client.get_all_entities(limit=3)
            print("Sample entities (run with --migrate to add temporal metadata):")
            for row in sample:
                print(
                    f"  - {row['name']!r} types={row['types']} "
                    f"created_at={row['created_at']} source_type={row['source_type']} "
                    f"trust_score={row['trust_score']}"
                )
