"""Derive Phase 1 provenance for a paper corpus from evidence in the graph.

Phase 1's migration stamps the SAME constants onto every node, so trust
collapses to a single value (0.5 x 0.4 x 1.0 = 0.2) and the trust gate stops
discriminating: it rejects everything equally, which is indistinguishable from
having no gate at all. This module fills the same two properties from evidence
the graph already holds, so the formula

    trust = confidence_score x source_weight x recency_factor

is fed real inputs instead of defaults. The formula itself is untouched.

What each factor is taken to mean here
--------------------------------------
``source_type`` describes where the SOURCE came from. Every chunk in this
corpus carries ``source_kind='pdf'`` and belongs to a peer-reviewed paper, so
``paper`` is the honest label at the document level.

``confidence_score`` describes confidence in the EXTRACTION - a separate
question, which is why the formula multiplies the two rather than folding them
together. The pipeline never recorded an extraction confidence, so this is a
**derived proxy** built from how much corroborating evidence each entity has:

* how many distinct chunks mention it (the dominant term),
* whether the classifier assigned it a type,
* whether it participates in the concept hierarchy (subtopics, relations).

**This is a proxy and must be reported as one.** It does not measure whether an
extraction is correct; it measures how well-attested it is in the corpus. An
entity carried by twenty-five passages across two papers is more load-bearing
than one appearing once, and an answer resting only on one-off extractions
deserves a lower trust score - which is exactly the judgement the gate exists to
make. But a confidently-wrong extraction repeated often would score high, and
nothing here catches that.

``recency_factor`` is deliberately left neutral. It would need ``created_at``,
and the only dates available are publication years (1978-2021). The decay is
configured with a 90-day half-life, sensible for meeting notes and operational
decisions but not for published literature: applying it would floor every node
at ``recency_floor`` and destroy the very variance this module exists to create.
A literature corpus needs a half-life in years, or no recency term at all.

Run directly (read-only by default)::

    python -m kg_agent.provenance                 # preview the distribution
    python -m kg_agent.provenance --store         # WRITE to the graph
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from kg_agent.config import Config, get_config
from kg_agent.neo4j_client import Neo4jClient, escape_label

logger = logging.getLogger(__name__)


@dataclass
class DerivedProvenance:
    """Per-entity provenance derived from graph evidence, with its inputs kept.

    The component fields are retained so a score can be explained rather than
    asserted - the same reason :class:`~kg_agent.node_trust.TrustScore` keeps
    its factors.
    """

    element_id: str
    name: Optional[str]
    chunks: int
    documents: int
    has_type: bool
    has_subtopic: bool
    has_relation: bool
    support_score: float
    structure_score: float
    confidence_score: float
    source_type: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def support_score(chunks: int, saturation: int) -> float:
    """How well-attested an entity is, from its distinct mention count.

    Logarithmic and saturating: the step from one mention to three says far
    more than the step from twenty to twenty-three, and without a ceiling a
    single hub entity would flatten everything else by comparison.
    """
    if chunks <= 0:
        return 0.0
    return min(1.0, math.log1p(chunks) / math.log1p(max(1, saturation)))


def structure_score(has_type: bool, has_subtopic: bool, has_relation: bool) -> float:
    """Fraction of the structural signals an entity participates in.

    An entity the pipeline could classify and connect is one it understood;
    an isolated node with no type and no relations is closer to a loose string.
    """
    signals = (has_type, has_subtopic, has_relation)
    return sum(1 for s in signals if s) / len(signals)


def derive_confidence(
    chunks: int,
    has_type: bool,
    has_subtopic: bool,
    has_relation: bool,
    config: Config,
) -> Dict[str, float]:
    """Blend the evidence terms into a confidence in ``[floor, 1.0]``.

    Floored rather than allowed to reach zero: an entity that was extracted at
    all is weak evidence, not the absence of evidence, and a zero would wipe
    out the whole product regardless of the other factors.
    """
    p = config.provenance
    sup = support_score(chunks, p.support_saturation)
    struct = structure_score(has_type, has_subtopic, has_relation)
    total = p.weight_support + p.weight_structure
    blended = (
        (p.weight_support * sup + p.weight_structure * struct) / total
        if total > 0
        else sup
    )
    return {
        "support_score": round(sup, 4),
        "structure_score": round(struct, 4),
        "confidence_score": round(max(p.confidence_floor, min(1.0, blended)), 4),
    }


def derive_for_graph(
    client: Neo4jClient, config: Optional[Config] = None
) -> List[DerivedProvenance]:
    """Read evidence for every entity and compute its provenance (no writes)."""
    config = config or client.config
    label = escape_label(config.schema.entity_label)
    chunk_label = escape_label(config.retrieval.chunk_label)
    name_prop = config.schema.entity_name_property
    source_prop = config.retrieval.chunk_source_property

    doc_expr = f"count(DISTINCT ch.`{source_prop}`)" if source_prop else "0"
    cypher = f"""
        MATCH (e:{label})
        OPTIONAL MATCH (ch:{chunk_label})-[:MENTIONS]->(e)
        OPTIONAL MATCH (e)-[:HAS_TYPE]->(ty)
        OPTIONAL MATCH (e)-[:HAS_SUBTOPIC]-(sub)
        OPTIONAL MATCH (e)-[:RELATES_TO]-(rel)
        RETURN elementId(e) AS element_id,
               e.{name_prop} AS name,
               count(DISTINCT ch) AS chunks,
               {doc_expr} AS documents,
               count(DISTINCT ty) > 0 AS has_type,
               count(DISTINCT sub) > 0 AS has_subtopic,
               count(DISTINCT rel) > 0 AS has_relation
    """
    out: List[DerivedProvenance] = []
    for row in client.run_read(cypher):
        scores = derive_confidence(
            row["chunks"], row["has_type"], row["has_subtopic"],
            row["has_relation"], config,
        )
        out.append(
            DerivedProvenance(
                element_id=row["element_id"],
                name=row["name"],
                chunks=row["chunks"],
                documents=row["documents"],
                has_type=row["has_type"],
                has_subtopic=row["has_subtopic"],
                has_relation=row["has_relation"],
                source_type=config.provenance.source_type,
                **scores,
            )
        )
    return out


def store_provenance(client: Neo4jClient, rows: List[DerivedProvenance]) -> int:
    """Write derived ``confidence_score`` and ``source_type`` onto the entities.

    Overwrites both properties deliberately: a constant stamped by the Phase 1
    default is precisely what this replaces, so ``coalesce`` would preserve the
    problem. ``created_at`` is never touched - see the module docstring on why
    the recency term is left neutral.
    """
    if not rows:
        return 0
    payload = [
        {
            "element_id": r.element_id,
            "confidence_score": r.confidence_score,
            "source_type": r.source_type,
        }
        for r in rows
    ]
    result = client.run_write(
        """
        UNWIND $rows AS row
        MATCH (e) WHERE elementId(e) = row.element_id
        SET e.confidence_score = row.confidence_score,
            e.source_type      = row.source_type,
            e.provenance_derived_at = datetime(),
            e.provenance_method = 'kg_agent.provenance'
        RETURN count(e) AS updated
        """,
        rows=payload,
    )
    updated = result["records"][0]["updated"] if result["records"] else 0
    logger.info("Derived provenance written to %s entities", updated)
    return updated


def summarise(rows: List[DerivedProvenance]) -> Dict[str, Any]:
    """Distribution of the derived confidence, for judging it before storing."""
    if not rows:
        return {"count": 0}
    values = sorted(r.confidence_score for r in rows)
    buckets: Dict[str, int] = {}
    for v in values:
        key = f"{min(int(v * 5) / 5, 0.8):.1f}-{min(int(v * 5) / 5 + 0.2, 1.0):.1f}"
        buckets[key] = buckets.get(key, 0) + 1
    return {
        "count": len(values),
        "min": values[0],
        "median": values[len(values) // 2],
        "mean": round(sum(values) / len(values), 4),
        "max": values[-1],
        "distinct_values": len(set(values)),
        "buckets": dict(sorted(buckets.items())),
    }


if __name__ == "__main__":  # pragma: no cover - manual entry point
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Derive provenance from graph evidence.")
    parser.add_argument("--store", action="store_true",
                        help="WRITE confidence_score/source_type back to the graph.")
    parser.add_argument("--top", type=int, default=8, help="How many examples to show.")
    args = parser.parse_args()

    cfg = get_config()
    with Neo4jClient.from_config(cfg) as client:
        if not client.verify_connectivity():
            raise SystemExit("Could not connect to Neo4j - check kg_agent/config.py")
        rows = derive_for_graph(client, cfg)
        rows.sort(key=lambda r: -r.confidence_score)

        print(f"\nDerived provenance for {len(rows)} entities (source_type="
              f"{cfg.provenance.source_type!r})")
        print(f"  {summarise(rows)}\n")
        print(f"  Strongest {args.top}:")
        for r in rows[: args.top]:
            print(f"    conf={r.confidence_score:.3f}  {str(r.name)[:38]:<38} "
                  f"chunks={r.chunks:<3} type={int(r.has_type)} sub={int(r.has_subtopic)} "
                  f"rel={int(r.has_relation)}")
        print(f"  Weakest {args.top}:")
        for r in rows[-args.top:]:
            print(f"    conf={r.confidence_score:.3f}  {str(r.name)[:38]:<38} "
                  f"chunks={r.chunks:<3} type={int(r.has_type)} sub={int(r.has_subtopic)} "
                  f"rel={int(r.has_relation)}")

        if args.store:
            print(f"\nWriting to {cfg.neo4j.uri} ...")
            print(f"  updated {store_provenance(client, rows)} entities")
        else:
            print("\n(read-only preview - pass --store to write)")
