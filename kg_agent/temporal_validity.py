"""Phase 2 - Temporal validity check.

Classifies entity nodes into one of four statuses:

* ``VALID``      - fresh, not superseded, no conflicting twin.
* ``OUTDATED``   - older than the configured threshold (or flagged manually).
* ``SUPERSEDED`` - a newer node explicitly replaces it.
* ``CONFLICTED`` - another node makes a competing claim under the same name.

The module is split into **pure functions** (operate on plain dicts, no
database, trivially unit-testable) and a thin **DB-backed entry point**
(:func:`generate_validity_report`) that fetches the data and delegates to the
pure layer. This satisfies the "each phase independently testable" constraint:
you can exercise every classification rule without a running Neo4j.

Status precedence (highest wins): SUPERSEDED > CONFLICTED > OUTDATED > VALID.

Run directly for a read-only report against the live graph::

    python -m kg_agent.temporal_validity
    python -m kg_agent.temporal_validity --threshold-days 7 --names "Pod,Agv"
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from kg_agent.config import Config, get_config
from kg_agent.neo4j_client import Neo4jClient, escape_label

logger = logging.getLogger(__name__)


class ValidityStatus(str, Enum):
    """Possible temporal-validity verdicts for a node."""

    VALID = "VALID"
    OUTDATED = "OUTDATED"
    SUPERSEDED = "SUPERSEDED"
    CONFLICTED = "CONFLICTED"


@dataclass
class NodeValidity:
    """Per-node temporal-validity result.

    Attributes
    ----------
    element_id
        The Neo4j ``elementId`` of the node.
    entity_id
        The node's domain id (the ``id`` property).
    name
        The node's display name.
    status
        The final :class:`ValidityStatus`.
    reasons
        Human-readable explanations for the assigned status.
    age_days
        Age of the node in days against its freshness reference, or ``None`` if
        no timestamp is available.
    is_outdated_flag
        Value of the manual ``is_outdated`` override on the node.
    superseded_by
        Name of the newer node, when ``status`` is ``SUPERSEDED``.
    conflicts_with
        Names of competing nodes, when ``status`` is ``CONFLICTED``.
    """

    element_id: str
    entity_id: Optional[str]
    name: Optional[str]
    status: ValidityStatus
    reasons: List[str] = field(default_factory=list)
    age_days: Optional[float] = None
    is_outdated_flag: bool = False
    superseded_by: Optional[str] = None
    conflicts_with: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dictionary (status rendered as its value)."""
        data = asdict(self)
        data["status"] = self.status.value
        return data


# --------------------------------------------------------------------------- #
# Pure logic (no database) - unit-testable in isolation
# --------------------------------------------------------------------------- #
def _utcnow() -> datetime:
    """Return the current timezone-aware UTC time (wrapped for testability)."""
    return datetime.now(timezone.utc)


_TIMESTAMP_FIELDS = ("validated_at", "updated_at", "created_at")


def freshness_reference(entity: Dict[str, Any]) -> Optional[datetime]:
    """Return the timestamp used to judge a node's freshness.

    The most recent of ``validated_at``, ``updated_at`` and ``created_at`` is
    used, so re-validating or updating a node refreshes it.

    A timestamp property that is *absent* (or null) is correct to skip: the node
    is simply treated as timeless (VALID by default). But a property that is
    *present with the wrong type* - e.g. an ISO string instead of a real Neo4j
    temporal value, which the driver would hand back as ``str`` - is different:
    silently skipping it makes a node that carries a real (mis-stored) timestamp
    look timeless, hiding a data problem. Those are logged as an explicit
    warning naming the node, field, value and actual type, so a schema mismatch
    in an upstream pipeline surfaces instead of passing unnoticed.

    Parameters
    ----------
    entity
        A normalised entity dict (Neo4j temporal types already converted to
        Python ``datetime`` by the driver/read layer).

    Returns
    -------
    Optional[datetime]
        The reference timestamp, or ``None`` if the node has no usable
        timestamp.
    """
    timestamps: List[datetime] = []
    for name in _TIMESTAMP_FIELDS:
        value = entity.get(name)
        if value is None:
            continue  # genuinely absent/null - treat as timeless, no warning
        if isinstance(value, datetime):
            timestamps.append(value)
            continue
        # Present but not a datetime after normalisation: don't drop it silently.
        node_id = entity.get("name") or entity.get("id") or entity.get("element_id") or "<unknown>"
        logger.warning(
            "Node %r has %s=%r of type %s, not a datetime after driver "
            "normalisation; ignoring it for temporal freshness (node would be "
            "treated as timeless). Store timestamps as a Neo4j temporal type "
            "(e.g. datetime()), not an ISO string.",
            node_id,
            name,
            value,
            type(value).__name__,
        )
    return max(timestamps) if timestamps else None


def compute_age_days(entity: Dict[str, Any], now: Optional[datetime] = None) -> Optional[float]:
    """Compute a node's age in days against its freshness reference.

    Parameters
    ----------
    entity
        A normalised entity dict.
    now
        Reference "current" time (defaults to :func:`_utcnow`). Injectable for
        deterministic tests.

    Returns
    -------
    Optional[float]
        Age in fractional days, or ``None`` if the node has no timestamp.
    """
    now = now or _utcnow()
    reference = freshness_reference(entity)
    if reference is None:
        return None
    delta = now - reference
    return delta.total_seconds() / 86400.0


def is_outdated(
    entity: Dict[str, Any], threshold_days: int, now: Optional[datetime] = None
) -> bool:
    """Return ``True`` if a node is outdated by age or by manual flag.

    A node is outdated when its ``is_outdated`` property is explicitly ``True``,
    or when its age exceeds ``threshold_days``.
    """
    if entity.get("is_outdated") is True:
        return True
    age = compute_age_days(entity, now)
    return age is not None and age > threshold_days


def find_conflicting_ids(
    entities: List[Dict[str, Any]], require_different_description: bool = False
) -> Dict[str, List[str]]:
    """Detect nodes that conflict by sharing a name with a competing claim.

    Two or more entities sharing a case-insensitive name are treated as a
    conflict (the same concept asserted by separate nodes). When
    ``require_different_description`` is set, a name clash only counts if the
    nodes' ``description`` values differ - distinguishing a genuine
    contradiction from a harmless duplicate.

    Parameters
    ----------
    entities
        Normalised entity dicts.
    require_different_description
        Whether differing descriptions are required to flag a conflict.

    Returns
    -------
    Dict[str, List[str]]
        Maps each conflicted node's ``element_id`` to the list of competing
        node *names*.
    """
    by_name: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for entity in entities:
        name = (entity.get("name") or "").strip().lower()
        if name:
            by_name[name].append(entity)

    conflicts: Dict[str, List[str]] = {}
    for group in by_name.values():
        if len(group) < 2:
            continue
        if require_different_description:
            descriptions = {(e.get("description") or "").strip() for e in group}
            if len(descriptions) < 2:
                continue
        for entity in group:
            others = [e["name"] for e in group if e["element_id"] != entity["element_id"]]
            conflicts[entity["element_id"]] = others
    return conflicts


def classify_entity(
    entity: Dict[str, Any],
    threshold_days: int,
    superseded_map: Dict[str, str],
    conflict_map: Dict[str, List[str]],
    now: Optional[datetime] = None,
    contradiction_map: Optional[Dict[str, List[str]]] = None,
) -> NodeValidity:
    """Classify a single entity into a :class:`NodeValidity`.

    Applies the precedence SUPERSEDED > CONFLICTED > OUTDATED > VALID.

    Two independent things can produce CONFLICTED, and they are not
    interchangeable. ``conflict_map`` is the structural signal: two nodes assert
    the same name, so one of them is probably wrong. ``contradiction_map`` is the
    semantic signal: an upstream classifier read two claims and judged them
    incompatible. The second is far stronger evidence and says *what* the
    disagreement is, so its reason text carries the classifier's own summary
    rather than a list of colliding names.

    Parameters
    ----------
    entity
        A normalised entity dict.
    threshold_days
        Age threshold for the OUTDATED rule.
    superseded_map
        Maps ``element_id`` -> superseding node name.
    conflict_map
        Maps ``element_id`` -> competing node names.
    now
        Injectable current time for deterministic tests.

    Returns
    -------
    NodeValidity
    """
    element_id = entity["element_id"]
    age = compute_age_days(entity, now)
    reasons: List[str] = []

    outdated = is_outdated(entity, threshold_days, now)

    result = NodeValidity(
        element_id=element_id,
        entity_id=entity.get("id"),
        name=entity.get("name"),
        status=ValidityStatus.VALID,
        age_days=round(age, 2) if age is not None else None,
        is_outdated_flag=bool(entity.get("is_outdated")),
    )

    # Precedence: supersession is the strongest signal.
    if element_id in superseded_map:
        result.status = ValidityStatus.SUPERSEDED
        result.superseded_by = superseded_map[element_id]
        result.reasons.append(f"Superseded by newer node '{superseded_map[element_id]}'.")
        return result

    if contradiction_map and element_id in contradiction_map:
        result.status = ValidityStatus.CONFLICTED
        result.conflicts_with = contradiction_map[element_id]
        result.reasons.append(
            "Contradiction flagged by the upstream classifier: "
            + "; ".join(contradiction_map[element_id])
        )
        return result

    if element_id in conflict_map:
        result.status = ValidityStatus.CONFLICTED
        result.conflicts_with = conflict_map[element_id]
        result.reasons.append(
            "Competing claim(s) under the same name: "
            + ", ".join(repr(n) for n in conflict_map[element_id])
            + "."
        )
        return result

    if outdated:
        result.status = ValidityStatus.OUTDATED
        if entity.get("is_outdated") is True:
            result.reasons.append("Manually flagged as outdated (is_outdated=true).")
        if age is not None and age > threshold_days:
            result.reasons.append(
                f"Older than threshold: age {age:.1f}d > {threshold_days}d."
            )
        return result

    if age is None:
        result.reasons.append("No timestamp available; treated as VALID by default.")
    else:
        result.reasons.append(f"Fresh: age {age:.1f}d <= {threshold_days}d.")
    return result


def find_pipeline_contradictions(
    client: Neo4jClient,
    config: Config,
    entities: List[Dict[str, Any]],
) -> Dict[str, List[str]]:
    """Find contradictions an upstream classifier already recorded in the graph.

    Walks ``(:Chunk)-[:MENTIONS]->(:Description)-[:HAS_CONTRADICTION]->(:Contradiction)``
    and attributes each contradiction to the entities mentioned by the *same*
    chunk. Sharing a chunk is what makes the attribution tight: the claim and the
    entity were written about together.

    Do NOT be tempted to reach these through ``(:Topic)-[:HAS_TYPE]->(:Type)
    -[:HAS_DESCRIPTION]->(:Description)`` instead. That path exists and looks
    equivalent, but ``Type`` is a coarse bucket: measured on the live graph, one
    contradiction reached 672 topics that way versus 6-12 through chunks. Using
    it would mark almost every answer CONFLICTED and make the status worthless.

    Returns
    -------
    Dict[str, List[str]]
        Maps ``element_id`` -> the contradiction summaries touching that node.
        Empty when the graph carries no ``:Contradiction`` layer at all, so
        graphs without one behave exactly as before.
    """
    names = [e["name"] for e in entities if e.get("name")]
    if not names:
        return {}
    entity_label = escape_label(config.schema.entity_label)
    name_prop = config.schema.entity_name_property
    chunk_label = escape_label(config.retrieval.chunk_label)
    cypher = f"""
        MATCH (c:{chunk_label})-[:MENTIONS]->(d:Description)-[:HAS_CONTRADICTION]->(k:Contradiction)
        MATCH (c)-[:MENTIONS]->(e:{entity_label})
        WHERE toLower(e.{name_prop}) IN $names
        RETURN elementId(e) AS element_id,
               collect(DISTINCT coalesce(k.summary, k.id)) AS summaries
    """
    try:
        rows = client.run_read(cypher, names=[n.lower() for n in names])
    except Exception:  # pragma: no cover - graph simply has no such layer
        logger.debug("No :Contradiction layer reachable; skipping.", exc_info=True)
        return {}
    return {r["element_id"]: r["summaries"] for r in rows if r.get("summaries")}


def build_report(
    entities: List[Dict[str, Any]],
    superseded_map: Dict[str, str],
    threshold_days: int,
    require_different_description: bool = False,
    now: Optional[datetime] = None,
    contradiction_map: Optional[Dict[str, List[str]]] = None,
) -> List[NodeValidity]:
    """Classify a list of entities into validity results (pure, no DB).

    Parameters
    ----------
    entities
        Normalised entity dicts.
    superseded_map
        Maps ``element_id`` -> superseding node name.
    threshold_days
        Age threshold for OUTDATED.
    require_different_description
        Passed through to :func:`find_conflicting_ids`.
    now
        Injectable current time.

    Returns
    -------
    List[NodeValidity]
    """
    now = now or _utcnow()
    conflict_map = find_conflicting_ids(entities, require_different_description)
    return [
        classify_entity(
            e, threshold_days, superseded_map, conflict_map, now,
            contradiction_map=contradiction_map,
        )
        for e in entities
    ]


def summarise(report: List[NodeValidity]) -> Dict[str, int]:
    """Return a ``{status: count}`` summary of a validity report."""
    counts = {status.value: 0 for status in ValidityStatus}
    for item in report:
        counts[item.status.value] += 1
    return counts


# --------------------------------------------------------------------------- #
# DB-backed entry points
# --------------------------------------------------------------------------- #
def generate_validity_report(
    client: Neo4jClient,
    config: Optional[Config] = None,
    names: Optional[List[str]] = None,
    threshold_days: Optional[int] = None,
    now: Optional[datetime] = None,
) -> List[NodeValidity]:
    """Run the Phase 2 validity check against the graph.

    Parameters
    ----------
    client
        A connected :class:`~kg_agent.neo4j_client.Neo4jClient`.
    config
        Configuration (defaults to ``client.config``).
    names
        Optional subset of entity names to check (case-insensitive). When
        ``None``, the whole graph is evaluated. The Phase 4 verifier passes the
        names returned by retrieval here.
    threshold_days
        Override for the OUTDATED threshold (defaults to config value).
    now
        Injectable current time for deterministic tests.

    Returns
    -------
    List[NodeValidity]
        One result per evaluated node.
    """
    config = config or client.config
    threshold = (
        threshold_days
        if threshold_days is not None
        else config.temporal_validity.outdated_threshold_days
    )

    # `names is None` -> evaluate the whole graph. An explicit empty list means
    # "these (zero) retrieved nodes" and must NOT fall back to the whole graph.
    if names is None:
        entities = client.get_all_entities()
    else:
        entities = client.get_entities_by_names(names)

    # Build the supersession map from explicit SUPERSEDED_BY edges and from any
    # node-level `superseded_by` property, scoped to the evaluated nodes.
    evaluated_ids: Set[str] = {e["element_id"] for e in entities}
    superseded_map: Dict[str, str] = {}
    for edge in client.get_supersession_edges(names=names):
        if edge["old_element_id"] in evaluated_ids:
            superseded_map[edge["old_element_id"]] = edge["new_name"]
    for entity in entities:
        if entity.get("superseded_by") and entity["element_id"] not in superseded_map:
            superseded_map[entity["element_id"]] = str(entity["superseded_by"])

    return build_report(
        entities,
        superseded_map=superseded_map,
        threshold_days=threshold,
        require_different_description=config.temporal_validity.conflict_requires_different_description,
        now=now,
        contradiction_map=find_pipeline_contradictions(client, config, entities),
    )


def check_retrieved_nodes(
    client: Neo4jClient,
    names: List[str],
    config: Optional[Config] = None,
) -> List[NodeValidity]:
    """Validity check scoped to a retrieval result set (Phase 4 helper).

    Thin wrapper over :func:`generate_validity_report` that always filters by
    the given node names.
    """
    return generate_validity_report(client, config=config, names=names)


if __name__ == "__main__":  # pragma: no cover - manual entry point
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Phase 2 temporal validity report.")
    parser.add_argument(
        "--threshold-days",
        type=int,
        default=None,
        help="Override the OUTDATED age threshold (default: config value).",
    )
    parser.add_argument(
        "--names",
        type=str,
        default=None,
        help="Comma-separated entity names to restrict the check to.",
    )
    parser.add_argument(
        "--show",
        type=str,
        default="non-valid",
        choices=["all", "non-valid"],
        help="Print every node or only the non-VALID ones (default).",
    )
    args = parser.parse_args()

    cfg = get_config()
    names = [n.strip() for n in args.names.split(",")] if args.names else None

    with Neo4jClient.from_config(cfg) as client:
        if not client.verify_connectivity():
            raise SystemExit("Could not connect to Neo4j - check kg_agent/config.py")

        report = generate_validity_report(
            client, cfg, names=names, threshold_days=args.threshold_days
        )
        counts = summarise(report)
        threshold = args.threshold_days or cfg.temporal_validity.outdated_threshold_days

        print(f"\nTemporal validity report ({len(report)} nodes, threshold={threshold}d)")
        print(f"  Summary: {counts}\n")

        to_show = report if args.show == "all" else [r for r in report if r.status != ValidityStatus.VALID]
        if not to_show:
            print("  All evaluated nodes are VALID.")
        for item in to_show:
            print(f"  [{item.status.value:<10}] {item.name!r} (age={item.age_days}d)")
            for reason in item.reasons:
                print(f"               - {reason}")
