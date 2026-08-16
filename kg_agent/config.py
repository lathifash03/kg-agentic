"""Centralised, environment-overridable configuration for the KG-agentic layer.

Every threshold, weight and connection parameter used across Phases 1-4 lives
here so that *nothing* is hard-coded inside the logic modules. Values are read
from environment variables (a local ``.env`` file is loaded automatically when
``python-dotenv`` is installed) and fall back to sensible defaults that match
the live Neo4j instance shipped with this repository.

Usage
-----
    from kg_agent.config import get_config

    cfg = get_config()                  # read once from the environment
    cfg.temporal_validity.outdated_threshold_days = 7   # or override in code

The defaults below were verified against the running database:
the entity nodes carry the label ``__Entity__`` and expose ``id`` / ``name``
properties, while the structural plumbing relationships are ``MENTIONS``,
``NEXT`` and ``PART_OF``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:  # optional dependency - never required for the module to import
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is a convenience only
    pass


# --------------------------------------------------------------------------- #
# Small typed environment readers
# --------------------------------------------------------------------------- #
def _env_str(key: str, default: str) -> str:
    """Return the string value of ``key`` from the environment or ``default``."""
    value = os.getenv(key)
    return value if value not in (None, "") else default


def _env_int(key: str, default: int) -> int:
    """Return the int value of ``key`` from the environment or ``default``."""
    value = os.getenv(key)
    return int(value) if value not in (None, "") else default


def _env_float(key: str, default: float) -> float:
    """Return the float value of ``key`` from the environment or ``default``."""
    value = os.getenv(key)
    return float(value) if value not in (None, "") else default


def _env_bool(key: str, default: bool) -> bool:
    """Return the bool value of ``key`` (``1/true/yes/on``) or ``default``."""
    value = os.getenv(key)
    if value in (None, ""):
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# --------------------------------------------------------------------------- #
# Connection
# --------------------------------------------------------------------------- #
@dataclass
class Neo4jConfig:
    """Connection parameters for the backing Neo4j database."""

    uri: str = field(default_factory=lambda: _env_str("NEO4J_URI", "bolt://localhost:7687"))
    username: str = field(default_factory=lambda: _env_str("NEO4J_USERNAME", "neo4j"))
    password: str = field(default_factory=lambda: _env_str("NEO4J_PASSWORD", "password123"))
    database: str = field(default_factory=lambda: _env_str("NEO4J_DATABASE", "neo4j"))


# --------------------------------------------------------------------------- #
# Graph schema mapping (how *this* graph names things)
# --------------------------------------------------------------------------- #
@dataclass
class GraphSchemaConfig:
    """Maps abstract concepts ("an entity", "its name") onto the real labels
    and property keys used by the DylanTartarini1996 pipeline.

    Keeping these configurable means the agentic layer can be pointed at a
    differently-shaped graph without touching any query logic.
    """

    entity_label: str = field(default_factory=lambda: _env_str("KG_ENTITY_LABEL", "__Entity__"))
    entity_name_property: str = field(default_factory=lambda: _env_str("KG_ENTITY_NAME_PROP", "name"))
    entity_id_property: str = field(default_factory=lambda: _env_str("KG_ENTITY_ID_PROP", "id"))

    # Relationship type used to model "node A was superseded by node B".
    supersedes_relationship: str = field(
        default_factory=lambda: _env_str("KG_SUPERSEDES_REL", "SUPERSEDED_BY")
    )

    # Plumbing relationships that must NOT receive temporal validity metadata.
    structural_relationship_types: List[str] = field(
        default_factory=lambda: _env_str(
            "KG_STRUCTURAL_RELS", "MENTIONS,NEXT,PART_OF"
        ).split(",")
    )

    # When True, temporal metadata is only written to relationships whose both
    # endpoints are entities (the semantic graph). When False, every
    # non-structural relationship is annotated.
    temporal_relationships_entity_only: bool = field(
        default_factory=lambda: _env_bool("KG_TEMPORAL_RELS_ENTITY_ONLY", True)
    )


# --------------------------------------------------------------------------- #
# Phase 1 - default values written by the temporal-metadata migration
# --------------------------------------------------------------------------- #
@dataclass
class TemporalDefaults:
    """Default property values stamped onto nodes/relationships by Phase 1.

    These are only applied where a value is *missing*, so re-running the
    migration is idempotent and never clobbers curated data.
    """

    # Entity nodes in this graph were produced by the LLM extraction pipeline,
    # so the honest default provenance is "auto_extracted" (lowest trust).
    default_source_type: str = field(
        default_factory=lambda: _env_str("KG_DEFAULT_SOURCE_TYPE", "auto_extracted")
    )
    default_confidence_score: float = field(
        default_factory=lambda: _env_float("KG_DEFAULT_CONFIDENCE", 0.5)
    )
    default_created_by: str = field(
        default_factory=lambda: _env_str("KG_DEFAULT_CREATED_BY", "graph_miner")
    )
    default_relationship_confidence: float = field(
        default_factory=lambda: _env_float("KG_DEFAULT_REL_CONFIDENCE", 0.5)
    )
    # Stamp created_at/updated_at with the migration time. The timestamps are
    # honest ("first seen by the temporal layer at ..."); set to False to leave
    # them null and populate later from a real provenance source.
    set_created_at_to_now: bool = field(
        default_factory=lambda: _env_bool("KG_SET_CREATED_AT_NOW", True)
    )


# --------------------------------------------------------------------------- #
# Phase 2 - temporal validity check
# --------------------------------------------------------------------------- #
@dataclass
class TemporalValidityConfig:
    """Thresholds governing the VALID / OUTDATED / SUPERSEDED / CONFLICTED check."""

    # A node is OUTDATED once its freshness reference is older than this.
    outdated_threshold_days: int = field(
        default_factory=lambda: _env_int("KG_OUTDATED_THRESHOLD_DAYS", 30)
    )
    # Two entities sharing a (case-insensitive) name are treated as CONFLICTED.
    # When True, a name clash only counts as a conflict if the `description`
    # values also differ (a genuine contradiction rather than a duplicate).
    conflict_requires_different_description: bool = field(
        default_factory=lambda: _env_bool("KG_CONFLICT_REQUIRES_DIFF_DESC", False)
    )


# --------------------------------------------------------------------------- #
# Phase 3 - node trust scoring
# --------------------------------------------------------------------------- #
@dataclass
class TrustScoringConfig:
    """Weights and decay parameters for ``trust = confidence x source x recency``."""

    source_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "paper": 1.0,
            "meeting": 0.8,
            "discussion": 0.6,
            "auto_extracted": 0.4,
        }
    )
    # Weight applied when a node's source_type is unknown / unmapped.
    default_source_weight: float = field(
        default_factory=lambda: _env_float("KG_DEFAULT_SOURCE_WEIGHT", 0.4)
    )
    # Exponential decay: recency_factor = 0.5 ** (age_days / half_life_days).
    recency_half_life_days: float = field(
        default_factory=lambda: _env_float("KG_RECENCY_HALF_LIFE_DAYS", 90.0)
    )
    # Recency never decays below this floor, so old-but-valid facts keep signal.
    recency_floor: float = field(
        default_factory=lambda: _env_float("KG_RECENCY_FLOOR", 0.1)
    )
    # Property name under which the computed score is written back to each node.
    trust_score_property: str = field(
        default_factory=lambda: _env_str("KG_TRUST_SCORE_PROP", "trust_score")
    )


# --------------------------------------------------------------------------- #
# Phase 4 - LLM + agentic verifier
# --------------------------------------------------------------------------- #
# Default model id for the Groq provider. Exported because the Ollama clients
# need to recognise it: it is a *Groq* model name, so seeing it while talking to
# an Ollama server means KG_LLM_MODEL was never set for this provider.
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"


@dataclass
class LLMConfig:
    """LLM settings for answer generation + the faithfulness check (Phase 4).

    ``provider`` selects the client: ``"groq"`` (default, cloud) or ``"ollama"``
    (local, any model served by the Ollama instance at ``OLLAMA_URL``).
    """

    provider: str = field(default_factory=lambda: _env_str("KG_LLM_PROVIDER", "groq"))
    # For Groq: llama3-70b-8192 is being retired; llama-3.3-70b-versatile is the
    # current 70B model. The Ollama provider has no default model - set
    # KG_LLM_MODEL to a model that server actually serves (the clients raise a
    # message listing the available ones if you don't).
    model: str = field(default_factory=lambda: _env_str("KG_LLM_MODEL", DEFAULT_GROQ_MODEL))
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("GROQ_API_KEY"))
    temperature: float = field(default_factory=lambda: _env_float("KG_LLM_TEMPERATURE", 0.0))
    # Shared output-length cap applied identically to every real provider so a
    # benchmark measures the model, not differing default token budgets.
    max_tokens: int = field(default_factory=lambda: _env_int("KG_LLM_MAX_TOKENS", 512))
    # Per-call timeout (seconds) - this is per LLM call, not per HTTP request.
    # One /query makes up to 2 calls per attempt (answer + judge) x (max_retries
    # + 1) attempts, so the worst case a caller can observe is roughly
    #   timeout * 2 * (max_retries + 1).
    # Keep that product under the HTTP client's own timeout. Nothing cancels an
    # in-flight call when the caller disconnects (the endpoint is a sync `def`
    # running urllib in the threadpool), so a value larger than the client's
    # budget means the server keeps burning a worker and an Ollama slot on a
    # request nobody is waiting for any more. 120s leaves ~20x headroom over a
    # measured GPU call and still covers a CPU fallback of a 7-8B model.
    request_timeout: int = field(default_factory=lambda: _env_int("KG_LLM_TIMEOUT", 120))


@dataclass
class JudgeConfig:
    """Optional *separate* LLM for the faithfulness judge (Phase 4).

    Some models never emit a bounded answer suitable for a JSON verdict - e.g.
    thinking-only models that reason past any token budget and leave
    ``message.content`` empty. This lets the faithfulness check run on a plain
    instruct model while the main provider still handles tool-calling and answer
    generation.

    Backward compatible: with ``provider=""`` (the default) no separate judge is
    built and the faithfulness check reuses the main LLM client, exactly as
    before. Set ``KG_JUDGE_PROVIDER`` (``ollama`` | ``groq`` | ``mock``) plus
    ``KG_JUDGE_MODEL`` to enable it; ``KG_JUDGE_OLLAMA_URL`` defaults to the same
    ``OLLAMA_URL`` as everything else, so a local instruct model can judge while
    the main model lives on a remote endpoint.
    """

    provider: str = field(default_factory=lambda: _env_str("KG_JUDGE_PROVIDER", ""))
    model: str = field(default_factory=lambda: _env_str("KG_JUDGE_MODEL", ""))
    ollama_url: str = field(
        default_factory=lambda: _env_str(
            "KG_JUDGE_OLLAMA_URL", _env_str("OLLAMA_URL", "http://localhost:11434")
        )
    )

    @property
    def enabled(self) -> bool:
        """True when a separate judge client should be built."""
        return bool(self.provider.strip())


@dataclass
class RetrievalConfig:
    """Settings for the default KG-RAG retriever used by the Phase 4 demo.

    The retriever is hybrid: it uses Neo4j vector search when the embedding
    model is reachable, and falls back to keyword search otherwise.
    """

    top_k: int = field(default_factory=lambda: _env_int("KG_RETRIEVAL_TOP_K", 5))
    # Cap on entities surfaced as sources, ranked by how many retrieved chunks
    # mention them. Stops broad/bibliography chunks from flooding sources_used.
    max_sources: int = field(default_factory=lambda: _env_int("KG_MAX_SOURCES", 15))
    vector_index_name: str = field(default_factory=lambda: _env_str("KG_VECTOR_INDEX", "vector"))
    # Must match the model the chunks were embedded with (mxbai-embed-large here).
    embed_model: str = field(default_factory=lambda: _env_str("KG_EMBED_MODEL", "mxbai-embed-large"))
    ollama_url: str = field(default_factory=lambda: _env_str("OLLAMA_URL", "http://localhost:11434"))
    # Dedicated Ollama host for EMBEDDINGS. Empty -> reuse OLLAMA_URL. Lets the
    # embed model (e.g. mxbai-embed-large) live on a different server from the
    # chat model - the query MUST be embedded with the same model the chunks
    # were, or vector search returns garbage.
    embed_url: str = field(default_factory=lambda: _env_str("KG_EMBED_URL", ""))
    # Drop vector hits whose similarity score is below this (0.0 = keep all).
    # Neo4j cosine scores are in [0, 1]; raise this to stop an out-of-scope query
    # from dragging in the top-k of loosely-related chunks.
    vector_min_score: float = field(default_factory=lambda: _env_float("KG_VECTOR_MIN_SCORE", 0.0))
    prefer_vector: bool = field(default_factory=lambda: _env_bool("KG_PREFER_VECTOR", True))

    @property
    def effective_embed_url(self) -> str:
        """Ollama URL used for embeddings: KG_EMBED_URL if set, else OLLAMA_URL."""
        return self.embed_url or self.ollama_url

    # -- Text-bearing node schema (how "a chunk" and "chunk mentions entity" are
    # -- named in *this* graph). Defaults reproduce the original hardcoded
    # -- ``(:Chunk {text})-[:MENTIONS]->(:Entity)`` shape exactly, so existing
    # -- deployments are unaffected.
    #
    # Some graphs (e.g. a concept-extraction KG with no document-chunk layer)
    # attach text to entities through a different label/relationship, or even a
    # multi-hop path (Topic -[:HAS_TYPE]-> Type -[:HAS_DESCRIPTION]-> Description
    # in one real case). ``chunk_to_entity_pattern`` is a raw Cypher pattern
    # fragment using the fixed variable names ``c`` (the text-bearing node,
    # pre-filtered by ``chunk_label``) and ``e`` (the entity node); it is
    # rendered with ``.format(chunk_label=..., entity_label=...)`` (both already
    # backtick-escaped) before being spliced into a query. Cypher pattern
    # matching does not care which side is already bound, so the *same*
    # rendered pattern is reused verbatim across vector retrieval, keyword
    # retrieval and the caller-supplied-names lookup.
    chunk_label: str = field(default_factory=lambda: _env_str("KG_CHUNK_LABEL", "Chunk"))
    chunk_text_property: str = field(
        default_factory=lambda: _env_str("KG_CHUNK_TEXT_PROP", "text")
    )
    chunk_to_entity_pattern: str = field(
        default_factory=lambda: _env_str(
            "KG_CHUNK_TO_ENTITY_PATTERN", "(c:{chunk_label})-[:MENTIONS]->(e:{entity_label})"
        )
    )

    # -- Document provenance -------------------------------------------------
    # Property on the chunk node naming the document it came from (e.g.
    # ``filename`` in a multi-paper corpus). When set, retrieval reports which
    # documents an answer was drawn from, and which documents each entity was
    # mentioned by - the latter exposes entities whose names collide across
    # documents (a generic "Table 3" topic shared by four papers), which would
    # otherwise silently mix sources. Empty (the default) disables the feature
    # and leaves every query identical to before.
    chunk_source_property: str = field(
        default_factory=lambda: _env_str("KG_CHUNK_SOURCE_PROP", "")
    )


@dataclass
class VerifierConfig:
    """Gates, retry policy and confidence blend for the Phase 4 loop."""

    min_trust_score: float = field(default_factory=lambda: _env_float("KG_MIN_TRUST_SCORE", 0.4))
    min_faithfulness: float = field(default_factory=lambda: _env_float("KG_MIN_FAITHFULNESS", 0.7))
    max_retries: int = field(default_factory=lambda: _env_int("KG_MAX_RETRIES", 1))
    # Statuses that disqualify a retrieved node from being treated as reliable.
    failing_statuses: List[str] = field(
        default_factory=lambda: _env_str(
            "KG_FAILING_STATUSES", "OUTDATED,SUPERSEDED,CONFLICTED"
        ).split(",")
    )
    # Retrieval strategies tried in order across retries. Supported values:
    # "vector", "keyword", "expanded" (higher-k vector/keyword).
    retry_strategies: List[str] = field(
        default_factory=lambda: _env_str("KG_RETRY_STRATEGIES", "vector,keyword,expanded").split(",")
    )
    # Overall confidence = weighted blend of these three signals (auto-normalised).
    weight_faithfulness: float = field(default_factory=lambda: _env_float("KG_W_FAITHFULNESS", 0.5))
    weight_trust: float = field(default_factory=lambda: _env_float("KG_W_TRUST", 0.2))
    weight_validity: float = field(default_factory=lambda: _env_float("KG_W_VALIDITY", 0.3))


# --------------------------------------------------------------------------- #
# Phase 5 - native tool-calling orchestrator
# --------------------------------------------------------------------------- #
@dataclass
class OrchestratorConfig:
    """Settings for the LLM-driven tool-selection layer.

    The orchestrator lets a tool-calling model pick *which* registry tool to
    run; it never replaces the verification gates inside
    :meth:`AgenticVerifier.verify`. It is opt-in: with ``mode="off"`` (the
    default) every existing code path behaves exactly as before.
    """

    # "native" -> route through the Ollama tool-calling loop; "off" -> disabled.
    mode: str = field(default_factory=lambda: _env_str("KG_ORCHESTRATOR", "off"))
    # Hard ceiling on executed tool calls per request, counted per individual
    # call (a single assistant turn may request several in parallel).
    max_tool_calls: int = field(
        default_factory=lambda: _env_int("KG_ORCHESTRATOR_MAX_CALLS", 4)
    )

    @property
    def enabled(self) -> bool:
        """True when the orchestrator should handle queries."""
        return self.mode.strip().lower() == "native"


# --------------------------------------------------------------------------- #
# Write guard
# --------------------------------------------------------------------------- #
@dataclass
class SafetyConfig:
    """Global guard on tool-driven writes to the graph.

    Enforced inside :func:`kg_agent.tools.call_tool`, which is the single choke
    point every tool-driven write passes through - the HTTP API, and the
    orchestrator whether it was reached from the API or the CLI. Guarding there
    rather than in the API means a new route or a new caller cannot reopen the
    hole by forgetting to check.

    Defaults to **True** because the API has no authentication and is routinely
    pointed at a graph somebody else owns. ``KG_ORCHESTRATOR=off`` is not a
    substitute: it only stops the LLM from *choosing* a tool, and never blocked
    a direct ``POST /tools/ingest_meeting`` - nor ``POST /query`` with
    ``{"agentic": true}``, which turns the orchestrator on per-request.

    Not covered, deliberately: ``kg_agent.cli --setup``. That is an explicit
    operator action on a machine the operator already controls, not a request
    arriving over a port.
    """

    read_only: bool = field(default_factory=lambda: _env_bool("KG_READ_ONLY", True))


# --------------------------------------------------------------------------- #
# Operational audit trail
# --------------------------------------------------------------------------- #
@dataclass
class AuditConfig:
    """JSONL audit trail for tool calls that change the graph.

    Retention defaults to 30 days rather than the few days a debug log would
    warrant: this is an operational record of who changed a shared knowledge
    graph, which is the kind of question asked weeks later, not the same day.

    Never stores payload content - see :mod:`kg_agent.audit_log`.
    """

    enabled: bool = field(default_factory=lambda: _env_bool("KG_AUDIT_LOG", True))
    directory: str = field(default_factory=lambda: _env_str("KG_AUDIT_LOG_DIR", "logs"))
    retention_days: int = field(
        default_factory=lambda: _env_int("KG_AUDIT_RETENTION_DAYS", 30)
    )


# --------------------------------------------------------------------------- #
# Root config
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    """Aggregate configuration object passed through every phase."""

    neo4j: Neo4jConfig = field(default_factory=Neo4jConfig)
    schema: GraphSchemaConfig = field(default_factory=GraphSchemaConfig)
    temporal_defaults: TemporalDefaults = field(default_factory=TemporalDefaults)
    temporal_validity: TemporalValidityConfig = field(default_factory=TemporalValidityConfig)
    trust: TrustScoringConfig = field(default_factory=TrustScoringConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    verifier: VerifierConfig = field(default_factory=VerifierConfig)
    orchestrator: OrchestratorConfig = field(default_factory=OrchestratorConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)


def get_config() -> Config:
    """Build a :class:`Config` from the current environment.

    Returns
    -------
    Config
        A fully-populated configuration object. Call this once near the entry
        point and thread the result through the other modules.
    """
    return Config()


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    import json
    from dataclasses import asdict

    print(json.dumps(asdict(get_config()), indent=2, default=str))
