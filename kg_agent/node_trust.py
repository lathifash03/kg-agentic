"""Phase 3 - Node trust scoring.

Computes a per-node trust score and writes it back to the graph::

    trust_score = confidence_score x source_weight x recency_factor

* ``confidence_score`` - the node's stored confidence (Phase 1 default 0.5).
* ``source_weight``    - provenance weight from config
  (paper=1.0, meeting=0.8, discussion=0.6, auto_extracted=0.4).
* ``recency_factor``   - exponential half-life decay with a floor::

      recency = max(recency_floor, 0.5 ** (age_days / half_life_days))

All three factors lie in ``[0, 1]`` so ``trust_score`` does too.

Like Phase 2 this is split into **pure functions** (no database, unit-testable)
and thin **DB-backed** entry points. Age is reused from
:mod:`kg_agent.temporal_validity` so freshness semantics stay consistent
(``max(validated_at, updated_at, created_at)``).

Run directly::

    python -m kg_agent.node_trust                 # compute + preview (read-only)
    python -m kg_agent.node_trust --store         # compute and write trust_score back
    python -m kg_agent.node_trust --names "Pod,Agv" --store
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from kg_agent.config import Config, get_config
from kg_agent.neo4j_client import Neo4jClient
from kg_agent.temporal_validity import _utcnow, compute_age_days

logger = logging.getLogger(__name__)


@dataclass
class TrustScore:
    """Per-node trust score with its decomposed factors (for explainability).

    Attributes
    ----------
    element_id, entity_id, name
        Node identity.
    confidence_score
        The confidence factor used.
    source_type, source_weight
        Provenance and its mapped weight.
    age_days, recency_factor
        Node age and the decayed recency multiplier.
    trust_score
        The final product ``confidence x source_weight x recency``.
    """

    element_id: str
    entity_id: Optional[str]
    name: Optional[str]
    confidence_score: float
    source_type: Optional[str]
    source_weight: float
    age_days: Optional[float]
    recency_factor: float
    trust_score: float

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dictionary of the score and its factors."""
        return asdict(self)


# --------------------------------------------------------------------------- #
# Pure logic (no database) - unit-testable in isolation
# --------------------------------------------------------------------------- #
def source_weight(source_type: Optional[str], config: Config) -> float:
    """Map a node's ``source_type`` to its provenance weight.

    Parameters
    ----------
    source_type
        The node's provenance label (e.g. ``"paper"``).
    config
        Configuration carrying ``trust.source_weights`` and the fallback
        ``trust.default_source_weight``.

    Returns
    -------
    float
        The weight for known sources, else the configured default.
    """
    if source_type is None:
        return config.trust.default_source_weight
    return config.trust.source_weights.get(source_type, config.trust.default_source_weight)


def recency_factor(age_days: Optional[float], config: Config) -> float:
    """Exponential half-life recency multiplier, floored.

    ``recency = max(floor, 0.5 ** (age_days / half_life_days))``.

    Parameters
    ----------
    age_days
        Node age in days. ``None`` (no timestamp) yields a neutral ``1.0`` so
        trust is not penalised for missing provenance data.
    config
        Configuration carrying ``trust.recency_half_life_days`` and
        ``trust.recency_floor``.

    Returns
    -------
    float
        A multiplier in ``[recency_floor, 1.0]``.
    """
    if age_days is None:
        return 1.0
    age = max(age_days, 0.0)
    decayed = 0.5 ** (age / config.trust.recency_half_life_days)
    return max(config.trust.recency_floor, decayed)


def compute_trust(
    entity: Dict[str, Any], config: Config, now: Optional[datetime] = None
) -> TrustScore:
    """Compute the trust score for a single (normalised) entity dict.

    Missing ``confidence_score`` falls back to the Phase 1 default so newly
    ingested nodes still receive a score.

    Parameters
    ----------
    entity
        A normalised entity dict (see ``Neo4jClient.get_all_entities``).
    config
        Configuration object.
    now
        Injectable current time for deterministic tests.

    Returns
    -------
    TrustScore
    """
    now = now or _utcnow()

    confidence = entity.get("confidence_score")
    if confidence is None:
        confidence = config.temporal_defaults.default_confidence_score

    s_type = entity.get("source_type")
    s_weight = source_weight(s_type, config)

    age = compute_age_days(entity, now)
    r_factor = recency_factor(age, config)

    score = float(confidence) * s_weight * r_factor

    return TrustScore(
        element_id=entity["element_id"],
        entity_id=entity.get("id"),
        name=entity.get("name"),
        confidence_score=round(float(confidence), 4),
        source_type=s_type,
        source_weight=round(s_weight, 4),
        age_days=round(age, 2) if age is not None else None,
        recency_factor=round(r_factor, 4),
        trust_score=round(score, 4),
    )


def score_entities_pure(
    entities: List[Dict[str, Any]], config: Config, now: Optional[datetime] = None
) -> List[TrustScore]:
    """Compute trust scores for a list of entity dicts (pure, no DB)."""
    now = now or _utcnow()
    return [compute_trust(e, config, now) for e in entities]


def summarise_scores(scores: List[TrustScore]) -> Dict[str, Any]:
    """Return min/mean/max and a coarse distribution of a list of trust scores."""
    if not scores:
        return {"count": 0, "min": None, "mean": None, "max": None, "buckets": {}}
    values = [s.trust_score for s in scores]
    buckets = {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}
    for v in values:
        if v < 0.2:
            buckets["0.0-0.2"] += 1
        elif v < 0.4:
            buckets["0.2-0.4"] += 1
        elif v < 0.6:
            buckets["0.4-0.6"] += 1
        elif v < 0.8:
            buckets["0.6-0.8"] += 1
        else:
            buckets["0.8-1.0"] += 1
    return {
        "count": len(values),
        "min": round(min(values), 4),
        "mean": round(sum(values) / len(values), 4),
        "max": round(max(values), 4),
        "buckets": buckets,
    }


# --------------------------------------------------------------------------- #
# DB-backed entry points
# --------------------------------------------------------------------------- #
def score_entities(
    client: Neo4jClient,
    config: Optional[Config] = None,
    names: Optional[List[str]] = None,
    now: Optional[datetime] = None,
) -> List[TrustScore]:
    """Fetch entities and compute their trust scores (does not write back).

    Parameters
    ----------
    client
        A connected :class:`~kg_agent.neo4j_client.Neo4jClient`.
    config
        Configuration (defaults to ``client.config``).
    names
        Optional case-insensitive subset of entity names. The Phase 4 verifier
        passes retrieved node names here.
    now
        Injectable current time.

    Returns
    -------
    List[TrustScore]
    """
    config = config or client.config
    # `names is None` -> whole graph; an explicit empty list -> zero entities
    # (must not silently fall back to scoring every node).
    if names is None:
        entities = client.get_all_entities()
    else:
        entities = client.get_entities_by_names(names)
    return score_entities_pure(entities, config, now)


def store_trust_scores(client: Neo4jClient, scores: List[TrustScore]) -> int:
    """Write computed trust scores back onto their nodes.

    Sets the configured trust-score property and a ``trust_scored_at``
    timestamp on each node, matched by ``elementId``. Batched via ``UNWIND``.

    Parameters
    ----------
    client
        A connected client.
    scores
        Scores produced by :func:`score_entities` / :func:`score_entities_pure`.

    Returns
    -------
    int
        The number of nodes updated.
    """
    if not scores:
        return 0
    prop = client.config.trust.trust_score_property
    rows = [{"element_id": s.element_id, "trust_score": s.trust_score} for s in scores]
    cypher = f"""
        UNWIND $rows AS row
        MATCH (e) WHERE elementId(e) = row.element_id
        SET e.{prop} = row.trust_score,
            e.trust_scored_at = datetime()
        RETURN count(e) AS updated
    """
    result = client.run_write(cypher, rows=rows)
    updated = result["records"][0]["updated"] if result["records"] else 0
    logger.info("Phase 3: stored trust scores on %s nodes", updated)
    return updated


def compute_and_store(
    client: Neo4jClient,
    config: Optional[Config] = None,
    names: Optional[List[str]] = None,
    now: Optional[datetime] = None,
) -> List[TrustScore]:
    """Compute trust scores and persist them back to the graph in one call."""
    scores = score_entities(client, config=config, names=names, now=now)
    store_trust_scores(client, scores)
    return scores


if __name__ == "__main__":  # pragma: no cover - manual entry point
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Phase 3 node trust scoring.")
    parser.add_argument("--names", type=str, default=None, help="Comma-separated entity names.")
    parser.add_argument("--store", action="store_true", help="Write trust_score back to nodes.")
    parser.add_argument("--top", type=int, default=5, help="How many top/bottom nodes to show.")
    args = parser.parse_args()

    cfg = get_config()
    names = [n.strip() for n in args.names.split(",")] if args.names else None

    with Neo4jClient.from_config(cfg) as client:
        if not client.verify_connectivity():
            raise SystemExit("Could not connect to Neo4j - check kg_agent/config.py")

        scores = (
            compute_and_store(client, cfg, names=names)
            if args.store
            else score_entities(client, cfg, names=names)
        )
        scores.sort(key=lambda s: s.trust_score, reverse=True)

        action = "computed + STORED" if args.store else "computed (read-only)"
        print(f"\nTrust scores {action} for {len(scores)} nodes")
        print(f"  Summary: {summarise_scores(scores)}\n")

        print(f"  Top {args.top}:")
        for s in scores[: args.top]:
            print(
                f"    {s.trust_score:.3f}  {s.name!r:<28} "
                f"(conf={s.confidence_score} x src[{s.source_type}]={s.source_weight} "
                f"x rec={s.recency_factor})"
            )
        print(f"  Bottom {args.top}:")
        for s in scores[-args.top :]:
            print(
                f"    {s.trust_score:.3f}  {s.name!r:<28} "
                f"(conf={s.confidence_score} x src[{s.source_type}]={s.source_weight} "
                f"x rec={s.recency_factor})"
            )
