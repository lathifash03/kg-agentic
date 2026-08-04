"""Phase 4 - Agentic verification layer.

Wraps KG-RAG answering with a verification loop:

1. **Retrieve** entities + supporting chunks for the query (default hybrid
   retriever, or caller-supplied nodes).
2. **Generate** a draft answer grounded in the retrieved context.
3. **Temporal validity** check on the retrieved nodes (Phase 2).
4. **Trust scoring** of the retrieved nodes (Phase 3).
5. **Faithfulness** check - does the answer follow from the sources? (Groq LLM,
   with a deterministic mock fallback when no ``GROQ_API_KEY`` is present).
6. **Decide**: if faithfulness/trust/validity gates fail and retries remain,
   re-retrieve with a different strategy and regenerate; otherwise attach a
   disclaimer.

Output (:class:`VerifiedAnswer`) always carries: ``answer``, ``trust_score``,
``temporal_validity_status``, ``sources_used`` plus an explanation, the
faithfulness score and an overall confidence.

Run directly::

    python -m kg_agent.agentic_verifier --query "What is an RMFS?"
"""

from __future__ import annotations

import json
import logging
import re
import socket
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Protocol

from neo4j.exceptions import ClientError

from kg_agent.config import DEFAULT_GROQ_MODEL, Config, get_config
from kg_agent.neo4j_client import Neo4jClient, escape_label
from kg_agent.node_trust import score_entities, summarise_scores
from kg_agent.temporal_validity import (
    NodeValidity,
    ValidityStatus,
    generate_validity_report,
    summarise,
)

logger = logging.getLogger(__name__)


# =========================================================================== #
# LLM clients
# =========================================================================== #
class LLMClient(Protocol):
    """Minimal chat interface used by the verifier."""

    name: str

    def complete(self, system: str, user: str) -> str:
        """Return the model's text completion for a system+user prompt."""
        ...


class GroqLLMClient:
    """Groq-backed LLM client (used when ``GROQ_API_KEY`` is configured)."""

    def __init__(self, config: Config, *, model: Optional[str] = None) -> None:
        from groq import Groq  # imported lazily so the dep is optional

        self._client = Groq(api_key=config.llm.api_key)
        self._model = model or config.llm.model
        self._temperature = config.llm.temperature
        self._max_tokens = config.llm.max_tokens
        self.name = f"groq:{self._model}"

    def complete(self, system: str, user: str) -> str:
        """Call Groq chat completions and return the assistant message text."""
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""


def list_ollama_models(ollama_url: str, timeout: int = 5) -> List[str]:
    """Return the model names served by an Ollama instance, or ``[]``.

    Used only to make error messages actionable, so every failure mode
    (server down, timeout, unexpected payload) degrades to an empty list
    rather than raising.
    """
    try:
        with urllib.request.urlopen(f"{ollama_url}/api/tags", timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
    except Exception:  # pragma: no cover - purely advisory
        return []
    return sorted(m["name"] for m in data.get("models", []) if m.get("name"))


def resolve_ollama_model(config: Config) -> str:
    """Return the Ollama model id to use, or raise a message that names the fix.

    ``KG_LLM_MODEL`` defaults to a *Groq* model id, so if that value is still in
    place while we are pointed at an Ollama server, the model was never
    configured for this provider. Rather than silently substituting some model
    that may not be pulled - which surfaces later as a confusing HTTP 404 - fail
    immediately and list what the server actually serves.
    """
    model = (config.llm.model or "").strip()
    if model and model != DEFAULT_GROQ_MODEL:
        return model

    url = config.retrieval.ollama_url
    available = list_ollama_models(url)
    if available:
        hint = f"Models available at {url}: {', '.join(available)}."
    else:
        hint = f"Could not reach {url}/api/tags to list the available models."
    raise RuntimeError(
        "KG_LLM_MODEL is not set for the Ollama provider (it still holds the "
        f"Groq default {DEFAULT_GROQ_MODEL!r}). Set it to a model served by "
        f"your Ollama instance. {hint}"
    )


class OllamaLLMClient:
    """Local Ollama chat client via ``POST /api/chat``.

    Reuses the same Ollama server already configured for embeddings
    (``config.retrieval.ollama_url``) rather than hardcoding a URL. Model,
    temperature and max-tokens all come from config.
    """

    def __init__(
        self, config: Config, *, model: Optional[str] = None, ollama_url: Optional[str] = None
    ) -> None:
        # Explicit model/url let this client be reused for a separate judge on a
        # different model or endpoint; otherwise fall back to the main config.
        url = ollama_url or config.retrieval.ollama_url
        self._url = f"{url}/api/chat"
        self._model = model or resolve_ollama_model(config)
        self._temperature = config.llm.temperature
        self._max_tokens = config.llm.max_tokens
        self._timeout = config.llm.request_timeout
        self.name = f"ollama:{self._model}"

    def complete(self, system: str, user: str) -> str:
        """Call the local Ollama chat endpoint and return the message text."""
        payload = json.dumps(
            {
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                "options": {
                    "temperature": self._temperature,
                    "num_predict": self._max_tokens,  # Ollama's max-tokens knob
                },
            }
        ).encode()
        req = urllib.request.Request(
            self._url, data=payload, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                data = json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode(errors="ignore")
            except Exception:  # pragma: no cover - defensive
                pass
            if exc.code == 404 or "not found" in body.lower():
                raise RuntimeError(
                    f"Ollama model '{self._model}' is not available. "
                    f"Pull it first:  ollama pull {self._model}"
                ) from exc
            raise RuntimeError(f"Ollama request failed (HTTP {exc.code}): {body}") from exc
        except OSError as exc:
            # A read timeout (slow CPU inference) surfaces as socket.timeout, which
            # is an OSError; distinguish it from a genuinely unreachable server so
            # the message tells the user what actually went wrong.
            reason = getattr(exc, "reason", None)
            if isinstance(exc, (socket.timeout, TimeoutError)) or isinstance(
                reason, (socket.timeout, TimeoutError)
            ):
                raise RuntimeError(
                    f"Ollama request timed out after {self._timeout}s for model "
                    f"'{self._model}'. The model is likely slow on CPU - raise the "
                    f"limit with KG_LLM_TIMEOUT or lower KG_LLM_MAX_TOKENS."
                ) from exc
            raise RuntimeError(
                f"Cannot reach Ollama at {self._url}. Is the server running? "
                f"Start it with  ollama serve  and pull the model:  ollama pull {self._model}"
            ) from exc
        return (data.get("message") or {}).get("content", "")


class MockLLMClient:
    """Deterministic offline stand-in for Groq.

    Lets the full loop run and be tested without an API key. Answer generation
    extractively echoes the context; faithfulness is estimated from lexical
    overlap between the answer and the sources. It is **not** a quality model -
    it exists so the pipeline is exercisable offline.
    """

    name = "mock"

    def complete(self, system: str, user: str) -> str:
        """Return a templated answer or a JSON faithfulness verdict."""
        if "faithfulness" in system.lower() or "faithfulness" in user.lower():
            answer, context = _split_answer_context(user)
            score = _lexical_overlap(answer, context)
            return json.dumps(
                {
                    "faithfulness": round(score, 3),
                    "verdict": "supported" if score >= 0.5 else "weakly supported",
                    "unsupported_claims": [],
                }
            )
        # Answer-generation branch: stitch the most relevant context sentence.
        context = _extract_context_block(user)
        snippet = context.strip().split("\n")[0][:300] if context.strip() else ""
        if not snippet:
            return "The retrieved context does not contain enough information to answer."
        return f"Based on the retrieved sources: {snippet}"


def get_llm_client(config: Config) -> LLMClient:
    """Select the LLM client from config.

    - ``provider="ollama"`` -> :class:`OllamaLLMClient` (local; requires
      ``KG_LLM_MODEL``).
    - ``provider="groq"`` with an API key -> :class:`GroqLLMClient`.
    - otherwise -> deterministic :class:`MockLLMClient` (offline).

    The default provider is unchanged: with no env vars set this still resolves
    to Groq (when ``GROQ_API_KEY`` is present) or the mock (when it is not).
    """
    provider = config.llm.provider
    if provider == "ollama":
        client = OllamaLLMClient(config)
        logger.info("Using Ollama LLM client: %s", client.name)
        return client
    if provider == "groq":
        if config.llm.api_key:
            try:
                return GroqLLMClient(config)
            except Exception as exc:  # pragma: no cover - import/network dependent
                logger.warning("Falling back to MockLLMClient (Groq init failed: %s)", exc)
        else:
            logger.warning("GROQ_API_KEY not set - using deterministic MockLLMClient.")
    else:
        logger.warning("Unknown KG_LLM_PROVIDER=%r - using deterministic MockLLMClient.", provider)
    return MockLLMClient()


def get_judge_client(config: Config) -> Optional[LLMClient]:
    """Build the *separate* faithfulness-judge client, or ``None``.

    Returns ``None`` when no judge is configured (``KG_JUDGE_PROVIDER`` empty),
    in which case the verifier reuses its main LLM client - the original
    behaviour. Otherwise builds a client for the judge provider/model, which may
    differ from the main one (e.g. a plain instruct model to judge answers that
    a thinking-only main model generates).
    """
    judge = config.judge
    if not judge.enabled:
        return None

    provider = judge.provider.strip().lower()
    if provider == "mock":
        logger.info("Faithfulness judge: deterministic MockLLMClient.")
        return MockLLMClient()
    if provider == "ollama":
        if not judge.model:
            raise RuntimeError(
                "KG_JUDGE_PROVIDER=ollama requires KG_JUDGE_MODEL to be set."
            )
        client = OllamaLLMClient(config, model=judge.model, ollama_url=judge.ollama_url)
        logger.info("Faithfulness judge: %s (%s)", client.name, judge.ollama_url)
        return client
    if provider == "groq":
        if not judge.model:
            raise RuntimeError(
                "KG_JUDGE_PROVIDER=groq requires KG_JUDGE_MODEL to be set."
            )
        if not config.llm.api_key:
            raise RuntimeError(
                "KG_JUDGE_PROVIDER=groq requires GROQ_API_KEY to be set."
            )
        client = GroqLLMClient(config, model=judge.model)
        logger.info("Faithfulness judge: %s", client.name)
        return client
    raise RuntimeError(
        f"Unknown KG_JUDGE_PROVIDER={judge.provider!r} "
        "(expected 'ollama', 'groq' or 'mock')."
    )


def _split_answer_context(user: str) -> tuple:
    """Split a faithfulness prompt into (answer, context) best-effort."""
    answer = ""
    context = ""
    m = re.search(r"ANSWER:\s*(.*?)\s*SOURCES:", user, re.DOTALL)
    if m:
        answer = m.group(1)
    m = re.search(r"SOURCES:\s*(.*)", user, re.DOTALL)
    if m:
        context = m.group(1)
    return answer, context


def _extract_context_block(user: str) -> str:
    """Pull the CONTEXT block out of an answer-generation prompt."""
    m = re.search(r"CONTEXT:\s*(.*)", user, re.DOTALL)
    return m.group(1) if m else user


def _lexical_overlap(answer: str, context: str) -> float:
    """Fraction of answer tokens that also appear in the context (0-1)."""
    a = set(re.findall(r"\w+", answer.lower()))
    c = set(re.findall(r"\w+", context.lower()))
    if not a:
        return 0.0
    return len(a & c) / len(a)


# =========================================================================== #
# Retrieval
# =========================================================================== #
@dataclass
class RetrievedContext:
    """Result of a retrieval call.

    Attributes
    ----------
    node_names
        Entity names surfaced by retrieval (deduplicated).
    chunks
        Supporting chunk texts used as grounding context.
    strategy
        Which retrieval strategy produced this result.
    """

    node_names: List[str]
    chunks: List[str]
    strategy: str

    def context_text(self, max_chars: int = 4000) -> str:
        """Concatenate the supporting chunks into a single context string."""
        joined = "\n---\n".join(self.chunks)
        return joined[:max_chars]


def _embed_query_ollama(query: str, config: Config) -> Optional[List[float]]:
    """Embed ``query`` via the local ollama server, or return ``None`` on failure.

    Uses the model the chunks were embedded with so vectors are comparable.
    """
    url = f"{config.retrieval.ollama_url}/api/embeddings"
    payload = json.dumps({"model": config.retrieval.embed_model, "prompt": query}).encode()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        return data.get("embedding")
    except (OSError, ValueError) as exc:
        # OSError covers URLError, ConnectionError and socket.timeout (which is
        # not a TimeoutError on Python 3.9); ValueError covers bad JSON.
        logger.warning("Vector embedding unavailable (%s); will use keyword retrieval.", exc)
        return None


def _expand_rows_to_context(
    rows: List[Dict[str, Any]], strategy: str, max_sources: Optional[int] = None
) -> RetrievedContext:
    """Turn ``{text, entity_names}`` rows into a deduplicated RetrievedContext.

    Entities are ranked by how many retrieved chunks mention them (a proxy for
    relevance to the query) and capped at ``max_sources`` so broad chunks do not
    flood the source list. Chunk order (relevance) is used to break ties.
    """
    chunks: List[str] = []
    counts: Dict[str, int] = {}
    first_seen: Dict[str, int] = {}
    display: Dict[str, str] = {}
    order = 0
    for row in rows:
        if row.get("text"):
            chunks.append(row["text"])
        for n in row.get("entity_names") or []:
            if not n:
                continue
            key = n.lower()
            counts[key] = counts.get(key, 0) + 1
            if key not in first_seen:
                first_seen[key] = order
                display[key] = n
                order += 1
    # Rank: most-mentioned first, then earliest retrieved (most relevant chunk).
    ranked = sorted(counts.keys(), key=lambda k: (-counts[k], first_seen[k]))
    if max_sources is not None:
        ranked = ranked[:max_sources]
    names = [display[k] for k in ranked]
    return RetrievedContext(node_names=names, chunks=chunks, strategy=strategy)


def _chunk_to_entity_pattern(client: Neo4jClient, config: Config) -> str:
    """Render the configured chunk->entity Cypher pattern for this client/config.

    Both placeholders are already-escaped labels by the time they reach the
    template, so the result is safe to splice directly into a query string.
    """
    return config.retrieval.chunk_to_entity_pattern.format(
        chunk_label=escape_label(config.retrieval.chunk_label),
        entity_label=client._label,
    )


def retrieve_vector(client: Neo4jClient, config: Config, query: str, k: int) -> Optional[RetrievedContext]:
    """Vector retrieval over the configured chunk embedding index, expanded to
    entities via the configured chunk-to-entity pattern (``MENTIONS`` by
    default; may be a longer path for differently-shaped graphs).

    Returns ``None`` if the query cannot be embedded (model unavailable).
    """
    vec = _embed_query_ollama(query, config)
    if vec is None:
        return None
    # The chunk->entity path can be sparse, so fetch a wider set of chunks to
    # actually surface entity sources, then keep only the top-k for the answer
    # context to avoid diluting it.
    fetch_k = max(k * 3, 12)
    pattern = _chunk_to_entity_pattern(client, config)
    chunk_label = escape_label(config.retrieval.chunk_label)
    text_prop = config.retrieval.chunk_text_property
    cypher = f"""
        CALL db.index.vector.queryNodes($index, $k, $vec) YIELD node AS c, score
        WHERE c:{chunk_label}
        OPTIONAL MATCH {pattern}
        RETURN c.{text_prop} AS text, score, collect(DISTINCT e.{client._name_prop}) AS entity_names
        ORDER BY score DESC
    """
    try:
        rows = client.run_read(cypher, index=config.retrieval.vector_index_name, k=fetch_k, vec=vec)
    except ClientError as exc:
        # The configured vector index may not exist (or not cover the chunk
        # label) on this graph - a config/shape mismatch, not a transient
        # error. Fall back to keyword retrieval instead of crashing, mirroring
        # the "embedding unavailable" path above.
        logger.warning(
            "Vector index %r unusable (%s); will use keyword retrieval.",
            config.retrieval.vector_index_name,
            getattr(exc, "message", None) or exc,
        )
        return None
    ctx = _expand_rows_to_context(rows, "vector", config.retrieval.max_sources)
    ctx.chunks = ctx.chunks[:k]
    return ctx


# Common words filtered out before keyword matching so terms like "what"/"is"
# do not match every chunk (including title/TOC pages).
_STOPWORDS = {
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "the", "and", "for", "are", "was", "were", "this", "that", "these", "those",
    "with", "from", "into", "about", "does", "did", "can", "could", "would",
    "should", "has", "have", "had", "you", "your", "its", "their", "they",
}


def _query_terms(query: str) -> List[str]:
    """Tokenise a query into meaningful lowercased terms (stopwords removed)."""
    terms = [t for t in re.findall(r"\w+", query.lower()) if len(t) > 2 and t not in _STOPWORDS]
    return terms or [t for t in re.findall(r"\w+", query.lower()) if len(t) > 2]


def retrieve_keyword(client: Neo4jClient, config: Config, query: str, k: int) -> RetrievedContext:
    """Keyword retrieval: rank entity-bearing chunks by query-term hits.

    Dependency-free fallback that always works against the live graph. Only
    chunks that actually reach an entity via the configured chunk-to-entity
    pattern are considered (skips title/TOC pages or orphan text nodes), and
    ties are broken by ``elementId`` for deterministic results.
    """
    terms = _query_terms(query)
    pattern = _chunk_to_entity_pattern(client, config)
    chunk_label = escape_label(config.retrieval.chunk_label)
    text_prop = config.retrieval.chunk_text_property
    cypher = f"""
        MATCH (c:{chunk_label})
        WHERE any(t IN $terms WHERE toLower(c.{text_prop}) CONTAINS t)
          AND EXISTS {{ {pattern} }}
        WITH c, size([t IN $terms WHERE toLower(c.{text_prop}) CONTAINS t]) AS hits
        ORDER BY hits DESC, elementId(c)
        LIMIT $k
        MATCH {pattern}
        RETURN c.{text_prop} AS text, collect(DISTINCT e.{client._name_prop}) AS entity_names
    """
    rows = client.run_read(cypher, terms=terms, k=k)
    return _expand_rows_to_context(rows, "keyword", config.retrieval.max_sources)


def retrieve(
    client: Neo4jClient, config: Config, query: str, strategy: str, k: int
) -> RetrievedContext:
    """Dispatch to a retrieval strategy, falling back vector -> keyword.

    Parameters
    ----------
    strategy
        ``"vector"``, ``"keyword"`` or ``"expanded"`` (keyword/vector with a
        larger ``k``).
    """
    if strategy == "expanded":
        k = k * 2
        strategy = "vector" if config.retrieval.prefer_vector else "keyword"

    if strategy == "vector" and config.retrieval.prefer_vector:
        ctx = retrieve_vector(client, config, query, k)
        if ctx is not None and ctx.chunks:
            return ctx
        # graceful fallback when the embedding model is not available
        return retrieve_keyword(client, config, query, k)
    return retrieve_keyword(client, config, query, k)


# =========================================================================== #
# Prompts
# =========================================================================== #
_ANSWER_SYSTEM = (
    "You are a careful assistant answering questions about a research knowledge "
    "graph. Use ONLY the provided context. If the context is insufficient, say "
    "so explicitly. Be concise and do not invent facts."
)

_FAITHFULNESS_SYSTEM = (
    "You are a strict faithfulness judge. Decide whether the ANSWER is fully "
    "supported by the SOURCES. Respond with ONLY a JSON object of the form "
    '{"faithfulness": <float 0-1>, "verdict": "<short>", '
    '"unsupported_claims": ["..."]}. faithfulness is the fraction of the '
    "answer's claims that are supported by the sources."
)

# Returned by _generate_answer instead of a blank string when the main LLM's
# completion is empty (observed live with qwen3-vl:4b: a "thinking" model can
# spend its whole token budget reasoning and never emit bounded content).
# _check_faithfulness matches on this exact string to short-circuit the judge
# call rather than scoring emptiness as if it were a real claim.
_EMPTY_ANSWER_MESSAGE = (
    "The model did not produce an answer for this query (empty response)."
)


# =========================================================================== #
# Verifier
# =========================================================================== #
@dataclass
class VerifiedAnswer:
    """Final output of the agentic verification loop.

    Attributes
    ----------
    query
        The original user query.
    answer
        The (possibly disclaimered) verified answer.
    trust_score
        Mean trust score of the sources actually used.
    temporal_validity_status
        Aggregate status: ``VALID`` if all used sources are valid, otherwise the
        most severe status among them.
    sources_used
        Per-source detail (name, status, trust, used flag).
    overall_confidence
        Weighted blend of faithfulness, trust and validity in ``[0, 1]``.
    faithfulness
        The faithfulness score from the Step 3 check.
    passed
        Whether all gates passed without needing a disclaimer.
    retries
        How many re-retrievals were performed.
    strategy
        The retrieval strategy that produced the final answer.
    explanation
        Human-readable summary of the verification decision.
    disclaimer
        A non-empty caveat when gates failed, else ``""``.
    """

    query: str
    answer: str
    trust_score: float
    temporal_validity_status: str
    sources_used: List[Dict[str, Any]]
    overall_confidence: float
    faithfulness: float
    passed: bool
    retries: int
    strategy: str
    explanation: str
    disclaimer: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dictionary of the verified answer."""
        return asdict(self)


# Severity order for aggregating per-node statuses into one verdict.
_STATUS_SEVERITY = {
    ValidityStatus.VALID.value: 0,
    ValidityStatus.OUTDATED.value: 1,
    ValidityStatus.CONFLICTED.value: 2,
    ValidityStatus.SUPERSEDED.value: 3,
}


class AgenticVerifier:
    """Runs the Phase 4 verification loop over a KG-RAG answer."""

    def __init__(
        self,
        client: Neo4jClient,
        config: Optional[Config] = None,
        llm: Optional[LLMClient] = None,
        judge: Optional[LLMClient] = None,
    ) -> None:
        self.client = client
        self.config = config or client.config
        self.llm = llm or get_llm_client(self.config)
        # A separate faithfulness judge when configured (e.g. a plain instruct
        # model to grade answers a thinking-only main model produces); otherwise
        # the judge is the main client, exactly as before.
        self.judge = judge or get_judge_client(self.config) or self.llm

    # -- LLM steps -------------------------------------------------------- #
    def _generate_answer(self, query: str, context: str) -> str:
        """Generate a grounded draft answer from the query and context.

        Never returns a blank string. Some "thinking" models (e.g.
        ``qwen3-vl:4b``) can spend their whole token budget on internal
        reasoning and emit no bounded content at all - this was already known
        to break the faithfulness judge, and was found live to affect main
        answer generation too, on a real multi-source query against Nabhyla's
        graph. A blank answer must never look like a legitimate empty-context
        case, so it gets its own explicit sentinel.
        """
        if not context.strip():
            return "No supporting context was retrieved for this query."
        user = f"QUESTION: {query}\n\nCONTEXT:\n{context}"
        raw = self.llm.complete(_ANSWER_SYSTEM, user).strip()
        return raw or _EMPTY_ANSWER_MESSAGE

    def _check_faithfulness(self, answer: str, context: str) -> Dict[str, Any]:
        """Run the Step 3 faithfulness check, returning a parsed result dict.

        An empty/sentinel answer is never sent to the judge - there is no
        real content to check faithfulness against, and asking anyway risks a
        misleadingly non-zero score for literally nothing (observed live:
        hermes3:3b scored an empty answer 0.70, clearing the 0.7 gate).
        """
        if not answer.strip() or answer.strip() == _EMPTY_ANSWER_MESSAGE:
            return {"faithfulness": 0.0, "verdict": "empty_answer", "unsupported_claims": []}
        user = f"ANSWER: {answer}\n\nSOURCES: {context}"
        raw = self.judge.complete(_FAITHFULNESS_SYSTEM, user)
        return _parse_faithfulness(raw)

    # -- aggregation helpers --------------------------------------------- #
    def _aggregate_status(self, report: List[NodeValidity]) -> str:
        """Reduce per-node validity to a single worst-case status string."""
        if not report:
            return ValidityStatus.VALID.value
        worst = max(report, key=lambda r: _STATUS_SEVERITY[r.status.value])
        return worst.status.value

    def _overall_confidence(
        self, faithfulness: float, mean_trust: float, valid_fraction: float
    ) -> float:
        """Blend the three signals into a single confidence using config weights."""
        w = self.config.verifier
        total = w.weight_faithfulness + w.weight_trust + w.weight_validity
        if total <= 0:
            return round(faithfulness, 4)
        blended = (
            w.weight_faithfulness * faithfulness
            + w.weight_trust * mean_trust
            + w.weight_validity * valid_fraction
        ) / total
        return round(blended, 4)

    # -- main loop -------------------------------------------------------- #
    def verify(
        self, query: str, node_names: Optional[List[str]] = None
    ) -> VerifiedAnswer:
        """Run the full verification loop for ``query``.

        Parameters
        ----------
        query
            The user's question.
        node_names
            Optional pre-retrieved entity names (from an upstream KG-RAG
            retriever). When provided, retrieval is skipped on the first pass
            and these nodes are used directly.

        Returns
        -------
        VerifiedAnswer
        """
        cfg = self.config
        strategies = cfg.verifier.retry_strategies or ["keyword"]
        max_attempts = cfg.verifier.max_retries + 1

        best: Optional[VerifiedAnswer] = None

        for attempt in range(max_attempts):
            strategy = strategies[min(attempt, len(strategies) - 1)]

            # 1. Retrieve (or use caller-supplied nodes on the first attempt).
            if node_names and attempt == 0:
                names = node_names
                context = self._context_for_names(names)
                strategy = "caller-supplied"
            else:
                ctx = retrieve(self.client, cfg, query, strategy, cfg.retrieval.top_k)
                names = ctx.node_names
                context = ctx.context_text()

            # 2. Generate a grounded answer.
            answer = self._generate_answer(query, context)

            # 3. Temporal validity on the retrieved nodes (Phase 2).
            report = generate_validity_report(self.client, cfg, names=names)
            status_counts = summarise(report)
            valid_fraction = (
                status_counts[ValidityStatus.VALID.value] / len(report) if report else 1.0
            )

            # 4. Trust scoring on the retrieved nodes (Phase 3).
            scores = score_entities(self.client, cfg, names=names)
            trust_summary = summarise_scores(scores)
            mean_trust = trust_summary["mean"] or 0.0

            # 5. Faithfulness check (Phase 4 step 3).
            faith = self._check_faithfulness(answer, context)
            faithfulness = float(faith.get("faithfulness", 0.0))

            # Gates.
            faithfulness_ok = faithfulness >= cfg.verifier.min_faithfulness
            trust_ok = mean_trust >= cfg.verifier.min_trust_score
            failing = {
                r.name
                for r in report
                if r.status.value in cfg.verifier.failing_statuses
            }
            validity_ok = len(failing) == 0
            passed = faithfulness_ok and trust_ok and validity_ok

            sources_used = self._build_sources(report, scores, names)
            overall = self._overall_confidence(faithfulness, mean_trust, valid_fraction)
            explanation = self._explain(
                strategy, faithfulness_ok, trust_ok, validity_ok,
                faithfulness, mean_trust, failing,
            )

            candidate = VerifiedAnswer(
                query=query,
                answer=answer,
                trust_score=round(mean_trust, 4),
                temporal_validity_status=self._aggregate_status(report),
                sources_used=sources_used,
                overall_confidence=overall,
                faithfulness=round(faithfulness, 4),
                passed=passed,
                retries=attempt,
                strategy=strategy,
                explanation=explanation,
            )

            # Keep the most confident attempt seen so far.
            if best is None or candidate.overall_confidence > best.overall_confidence:
                best = candidate

            if passed:
                logger.info("Verification passed on attempt %s (%s).", attempt, strategy)
                return candidate

            # A retry only helps if the failing gate is something *retrieval* can
            # change (faithfulness or validity). A trust-only failure cannot be
            # fixed by re-retrieving the same low-trust graph, so stop early.
            retrieval_can_help = (not faithfulness_ok) or (not validity_ok)
            if not retrieval_can_help or attempt == max_attempts - 1:
                logger.info(
                    "Attempt %s did not pass (faith=%.2f trust=%.2f failing=%d); "
                    "no retry would help - finalising with disclaimer.",
                    attempt, faithfulness, mean_trust, len(failing),
                )
                break

            logger.info(
                "Attempt %s failed a fixable gate (faith=%.2f failing=%d); "
                "retrying with next strategy.",
                attempt, faithfulness, len(failing),
            )

        # All attempts exhausted: return the best, with a disclaimer.
        assert best is not None
        best.disclaimer = self._disclaimer(best)
        best.answer = f"{best.answer}\n\n[!] {best.disclaimer}"
        return best

    # -- builders --------------------------------------------------------- #
    def _context_for_names(self, names: List[str]) -> str:
        """Fetch supporting chunk text for caller-supplied entity names."""
        pattern = _chunk_to_entity_pattern(self.client, self.config)
        text_prop = self.config.retrieval.chunk_text_property
        cypher = f"""
            MATCH {pattern}
            WHERE toLower(e.{self.client._name_prop}) IN $names
            RETURN DISTINCT c.{text_prop} AS text
            LIMIT $k
        """
        rows = self.client.run_read(
            cypher,
            names=[n.lower() for n in names],
            k=self.config.retrieval.top_k,
        )
        return "\n---\n".join(r["text"] for r in rows if r.get("text"))[:4000]

    def _build_sources(
        self,
        report: List[NodeValidity],
        scores: List[Any],
        names: List[str],
    ) -> List[Dict[str, Any]]:
        """Assemble per-source detail combining validity and trust."""
        trust_by_id = {s.element_id: s.trust_score for s in scores}
        out: List[Dict[str, Any]] = []
        for r in report:
            out.append(
                {
                    "name": r.name,
                    "temporal_status": r.status.value,
                    "trust_score": trust_by_id.get(r.element_id),
                    "age_days": r.age_days,
                    "used": r.status.value not in self.config.verifier.failing_statuses,
                    "reasons": r.reasons,
                }
            )
        return out

    def _explain(
        self, strategy, faithfulness_ok, trust_ok, validity_ok,
        faithfulness, mean_trust, failing,
    ) -> str:
        """Compose a human-readable verification summary."""
        parts = [f"Retrieval strategy: {strategy}."]
        parts.append(
            f"Faithfulness {faithfulness:.2f} "
            f"({'>=' if faithfulness_ok else '<'} {self.config.verifier.min_faithfulness})."
        )
        parts.append(
            f"Mean source trust {mean_trust:.2f} "
            f"({'>=' if trust_ok else '<'} {self.config.verifier.min_trust_score})."
        )
        if validity_ok:
            parts.append("All sources passed the temporal validity check.")
        else:
            parts.append(f"Sources failing validity: {', '.join(sorted(map(str, failing)))}.")
        return " ".join(parts)

    def _disclaimer(self, answer: VerifiedAnswer) -> str:
        """Build a disclaimer explaining which gate(s) failed."""
        issues = []
        if answer.faithfulness < self.config.verifier.min_faithfulness:
            issues.append("the answer may not be fully grounded in the sources")
        if answer.trust_score < self.config.verifier.min_trust_score:
            issues.append("the supporting nodes have low trust scores")
        if answer.temporal_validity_status != ValidityStatus.VALID.value:
            issues.append(
                f"some sources are {answer.temporal_validity_status.lower()}"
            )
        reason = "; ".join(issues) if issues else "verification gates were not met"
        return f"Unverified: {reason}. Treat this answer with caution."


def _parse_faithfulness(raw: str) -> Dict[str, Any]:
    """Best-effort parse of the faithfulness judge's JSON response."""
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            data["faithfulness"] = max(0.0, min(1.0, float(data.get("faithfulness", 0.0))))
            return data
        except (ValueError, TypeError):
            pass
    return {"faithfulness": 0.0, "verdict": "unparseable", "unsupported_claims": []}


if __name__ == "__main__":  # pragma: no cover - manual entry point
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Phase 4 agentic verification.")
    parser.add_argument("--query", type=str, required=True, help="User question.")
    parser.add_argument("--names", type=str, default=None, help="Optional pre-retrieved node names.")
    args = parser.parse_args()

    cfg = get_config()
    names = [n.strip() for n in args.names.split(",")] if args.names else None

    with Neo4jClient.from_config(cfg) as client:
        if not client.verify_connectivity():
            raise SystemExit("Could not connect to Neo4j - check kg_agent/config.py")
        verifier = AgenticVerifier(client, cfg)
        result = verifier.verify(args.query, node_names=names)
        print(json.dumps(result.to_dict(), indent=2, default=str))
