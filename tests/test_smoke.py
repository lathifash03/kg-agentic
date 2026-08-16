"""Smoke test: semua modul bisa di-import dan registry tools konsisten.

Jalankan: python -m pytest tests/ -q   (atau cukup: python tests/test_smoke.py)
Butuh ``pytest`` (pip install pytest); tidak butuh Neo4j/LLM — model dan tool
dipalsukan, jadi test Phase 5 di bawah deterministik sepenuhnya.
"""

import importlib


MODULES = [
    "kg_agent.config",
    "kg_agent.neo4j_client",
    "kg_agent.temporal_validity",
    "kg_agent.node_trust",
    "kg_agent.agentic_verifier",
    "kg_agent.tools",
    "kg_agent.cli",
    "kg_agent.orchestrator",
]


def test_imports():
    for mod in MODULES:
        importlib.import_module(mod)


def test_tool_registry():
    from kg_agent.tools import TOOLS, tool_specs

    specs = tool_specs()
    assert {s["name"] for s in specs} == set(TOOLS)
    for spec in specs:
        assert spec["description"]
        assert spec["parameters"]["type"] == "object"


def test_every_tool_declares_whether_it_writes():
    """A tool added without a `writes` flag must not silently pass as safe."""
    from kg_agent.tools import TOOLS, tool_writes

    for name in TOOLS:
        assert "writes" in TOOLS[name], f"{name} tidak menyatakan flag 'writes'"
        assert isinstance(tool_writes(name), bool)
    assert tool_writes("ingest_meeting") is True
    assert tool_writes("answer_question") is False


def test_tool_specs_exclude_the_writes_flag():
    """tool_specs() feeds Ollama's function schema - no extra keys may leak."""
    from kg_agent.tools import tool_specs

    for spec in tool_specs():
        assert set(spec) == {"name", "description", "parameters"}


def test_writes_are_blocked_by_default():
    """Default posture is read-only: the API has no auth and is often pointed
    at somebody else's graph.
    """
    from kg_agent.config import SafetyConfig

    assert SafetyConfig().read_only is True


def test_call_tool_refuses_write_tool_when_read_only():
    """The guard lives at the dispatcher, so it holds for the API and for the
    orchestrator reached from either the API or the CLI.
    """
    import pytest

    from kg_agent.config import get_config
    from kg_agent.tools import call_tool

    cfg = get_config()
    cfg.safety.read_only = True
    cfg.audit.enabled = False  # guard under test, not the audit trail

    with pytest.raises(PermissionError, match="KG_READ_ONLY"):
        call_tool("ingest_meeting", None, cfg, {"title": "standup"})


def test_call_tool_allows_read_tool_when_read_only():
    """Read-only must not break the path Rio actually uses."""
    from kg_agent.config import get_config
    from kg_agent.tools import TOOLS, call_tool

    cfg = get_config()
    cfg.safety.read_only = True
    seen = {}

    def fake_stats(client, config):
        seen["called"] = True
        return {"entities": 1}

    original = TOOLS["kg_stats"]["fn"]
    TOOLS["kg_stats"]["fn"] = fake_stats
    try:
        assert call_tool("kg_stats", None, cfg) == {"entities": 1}
    finally:
        TOOLS["kg_stats"]["fn"] = original
    assert seen["called"]


def test_call_tool_allows_write_tool_when_writes_enabled():
    """Turning the guard off restores the original behaviour exactly."""
    from kg_agent.config import get_config
    from kg_agent.tools import TOOLS, call_tool

    cfg = get_config()
    cfg.safety.read_only = False
    cfg.audit.enabled = False  # otherwise this writes into the repo's logs/

    original = TOOLS["ingest_meeting"]["fn"]
    TOOLS["ingest_meeting"]["fn"] = lambda client, config, **kw: {"ok": True, **kw}
    try:
        assert call_tool("ingest_meeting", None, cfg, {"title": "standup"}) == {
            "ok": True,
            "title": "standup",
        }
    finally:
        TOOLS["ingest_meeting"]["fn"] = original


def test_config_defaults():
    from kg_agent.config import get_config

    cfg = get_config()
    assert cfg.trust.source_weights["meeting"] == 0.8
    assert cfg.verifier.max_retries >= 0


def test_orchestrator_disabled_by_default():
    """The orchestrator must be opt-in: existing paths stay untouched."""
    from kg_agent.config import OrchestratorConfig

    # Built directly rather than via get_config() so a developer's own
    # KG_ORCHESTRATOR=native in .env cannot mask a regression in the default.
    assert OrchestratorConfig(mode="off").enabled is False
    assert OrchestratorConfig(mode="native").enabled is True
    assert OrchestratorConfig(mode="off").max_tool_calls >= 1


# --------------------------------------------------------------------------- #
# Phase 5 - tool-call validation (deterministic, no LLM or Neo4j involved)
# --------------------------------------------------------------------------- #
def _rejects(name, arguments):
    """Return the validation error for a call, asserting that it is rejected."""
    from kg_agent.orchestrator import ToolValidationError, validate_tool_call

    try:
        validate_tool_call(name, arguments)
    except ToolValidationError as exc:
        return str(exc)
    raise AssertionError(f"validate_tool_call({name!r}, {arguments!r}) should have raised")


def test_validator_rejects_hallucinated_tool_name():
    """A tool outside the registry must never reach call_tool."""
    error = _rejects("delete_everything", {"target": "graph"})
    assert "Unknown tool" in error
    assert "answer_question" in error  # the message lists the real options


def test_validator_rejects_wrong_argument_type():
    error = _rejects("answer_question", {"query": 42})
    assert "'query' must be a string, got int" in error


def test_validator_rejects_wrong_array_item_type():
    """Element types matter - mixed lists would reach Neo4j as bad data."""
    error = _rejects("ingest_meeting", {"title": "Sync", "participants": ["Alice", 7]})
    assert "participants[1]" in error


def test_validator_rejects_missing_and_empty_required_args():
    assert "missing required argument 'query'" in _rejects("answer_question", {})
    assert "must not be empty" in _rejects("answer_question", {"query": "   "})


def test_validator_rejects_unknown_argument():
    error = _rejects("answer_question", {"query": "hi", "temperature": 0.9})
    assert "'temperature'" in error


def test_validator_rejects_non_object_arguments():
    assert "must be a JSON object" in _rejects("kg_stats", ["not", "an", "object"])
    assert "not valid JSON" in _rejects("answer_question", "{broken")


def test_validator_accepts_valid_calls():
    from kg_agent.orchestrator import validate_tool_call

    assert validate_tool_call("answer_question", {"query": "What is an RMFS?"}) == {
        "query": "What is an RMFS?"
    }
    assert validate_tool_call("kg_stats", None) == {}  # no-arg tool
    # Arguments arriving as a JSON string (non-Ollama servers) are decoded.
    assert validate_tool_call("answer_question", '{"query": "hi"}') == {"query": "hi"}


# --------------------------------------------------------------------------- #
# Phase 5 - orchestration loop, driven by a scripted fake model
# --------------------------------------------------------------------------- #
class _FakeLLM:
    """Replays a scripted list of assistant messages; records what it was sent."""

    model = "fake-model"

    def __init__(self, messages):
        self._scripted = list(messages)
        self.received = []

    def chat(self, messages, tools=None):
        self.received.append(list(messages))
        if not self._scripted:
            return {"content": "done", "tool_calls": []}
        return self._scripted.pop(0)


def _tool_call(name, arguments):
    return {"function": {"name": name, "arguments": arguments}}


def _orchestrator(monkeypatch, scripted, tool_result=None):
    """Build an orchestrator with a fake LLM and a stubbed tool executor."""
    from kg_agent import orchestrator as orch
    from kg_agent.config import get_config

    calls = []

    def fake_call_tool(name, client, cfg, arguments):
        calls.append((name, arguments))
        if isinstance(tool_result, Exception):
            raise tool_result
        return tool_result if tool_result is not None else {"stub": name}

    monkeypatch.setattr(orch, "call_tool", fake_call_tool)
    llm = _FakeLLM(scripted)
    agent = orch.NativeToolOrchestrator(client=None, config=get_config(), llm=llm)
    return agent, calls, llm


VERIFIED = {
    "query": "q",
    "answer": "RMFS is a warehouse system.\n\n[!] Low trust in sources.",
    "trust_score": 0.31,
    "temporal_validity_status": "OUTDATED",
    "faithfulness": 0.62,
    "overall_confidence": 0.48,
    "passed": False,
    "disclaimer": "Low trust in sources.",
    "sources_used": [],
    "retries": 1,
    "strategy": "keyword",
    "explanation": "gates failed",
}


def test_orchestrator_attaches_verified_answer_unmodified(monkeypatch):
    """Gate fields must survive the orchestrator byte-for-byte."""
    scripted = [
        {"content": "", "tool_calls": [_tool_call("answer_question", {"query": "q"})]},
        {"content": "It's a warehouse thing, and it's totally reliable!", "tool_calls": []},
    ]
    agent, calls, _ = _orchestrator(monkeypatch, scripted, tool_result=VERIFIED)
    result = agent.run("q")

    assert result.verified_answer == VERIFIED  # not edited, not rescaled
    # The gated wording (with its disclaimer) is what surfaces - NOT the model's
    # cheerful paraphrase, which dropped the caveat.
    assert result.response == VERIFIED["answer"]
    assert "totally reliable" in result.model_narration
    assert result.tools_used == ["answer_question"]
    assert calls == [("answer_question", {"query": "q"})]


def test_orchestrator_omits_gate_fields_for_non_verifying_tools(monkeypatch):
    """kg_stats has no verification result - none may be fabricated."""
    scripted = [
        {"content": "", "tool_calls": [_tool_call("kg_stats", {})]},
        {"content": "The graph holds 3 entities.", "tool_calls": []},
    ]
    agent, _, _ = _orchestrator(monkeypatch, scripted, tool_result={"entities": 3})
    result = agent.run("how big is the graph?")

    assert result.verified_answer is None
    assert result.tool_results == [{"tool": "kg_stats", "result": {"entities": 3}}]
    assert result.response == "The graph holds 3 entities."


def test_orchestrator_reprompts_once_then_stops(monkeypatch):
    """One re-prompt on a bad call; a second failure ends the run."""
    scripted = [
        {"content": "", "tool_calls": [_tool_call("answer_question", {"query": 42})]},
        {"content": "", "tool_calls": [_tool_call("nonexistent_tool", {})]},
    ]
    agent, calls, llm = _orchestrator(monkeypatch, scripted)
    result = agent.run("q")

    assert calls == []  # nothing invalid was ever executed
    assert result.ok is False
    assert result.stopped_reason.startswith("validation_failed_twice")
    assert [e["tool_name"] for e in result.trace] == ["answer_question", "nonexistent_tool"]
    assert all(e["validation_error"] for e in result.trace)
    # The first error was fed back to the model as a tool-result message.
    fed_back = [m for m in llm.received[-1] if m.get("role") == "tool"]
    assert fed_back and "must be a string" in fed_back[0]["content"]


def test_orchestrator_enforces_call_cap(monkeypatch):
    """The cap counts individual calls, not turns."""
    turn = {
        "content": "",
        "tool_calls": [_tool_call("kg_stats", {}) for _ in range(3)],
    }
    agent, calls, _ = _orchestrator(monkeypatch, [turn, turn, turn])
    agent.max_tool_calls = 4
    result = agent.run("q")

    assert len(calls) == 4
    assert result.stopped_reason == "tool_call_cap_reached: 4"
    assert result.ok is False


def test_orchestrator_executes_parallel_calls_in_order(monkeypatch):
    """Several calls in one assistant message each get their own tool message."""
    scripted = [
        {
            "content": "",
            "tool_calls": [
                _tool_call("kg_stats", {}),
                _tool_call("answer_question", {"query": "q"}),
            ],
        },
        {"content": "both done", "tool_calls": []},
    ]
    agent, calls, llm = _orchestrator(monkeypatch, scripted, tool_result={"ok": 1})
    result = agent.run("q")

    assert [c[0] for c in calls] == ["kg_stats", "answer_question"]
    tool_messages = [m for m in llm.received[-1] if m.get("role") == "tool"]
    assert [m["tool_name"] for m in tool_messages] == ["kg_stats", "answer_question"]
    assert len(result.trace) == 2


def test_orchestrator_keeps_content_sent_alongside_tool_calls(monkeypatch):
    """content and tool_calls arriving together must not break the loop."""
    scripted = [
        {
            "content": "Let me look that up.",
            "tool_calls": [_tool_call("kg_stats", {})],
        },
        {"content": "", "tool_calls": []},
    ]
    agent, calls, llm = _orchestrator(monkeypatch, scripted, tool_result={"entities": 1})
    result = agent.run("q")

    assert calls == [("kg_stats", {})]
    assistant = [m for m in llm.received[-1] if m.get("role") == "assistant"]
    assert assistant[0]["content"] == "Let me look that up."
    # An empty closing turn is a failure, but the earlier prose is still kept.
    assert result.model_narration == "Let me look that up."


def test_orchestrator_rejects_corrupted_content(monkeypatch):
    """Leaked control tokens mean the reply was mis-parsed - never answer with it."""
    scripted = [{"content": '<tool_call>{"name": "kg_stats"}', "tool_calls": []}]
    agent, calls, _ = _orchestrator(monkeypatch, scripted)
    result = agent.run("q")

    assert calls == []
    assert result.ok is False
    assert "corrupted_response" in result.stopped_reason


def test_orchestrator_survives_tool_exception(monkeypatch):
    """A raising tool is traced and reported, not propagated."""
    scripted = [{"content": "", "tool_calls": [_tool_call("kg_stats", {})]}]
    agent, _, _ = _orchestrator(monkeypatch, scripted, tool_result=RuntimeError("neo4j down"))
    result = agent.run("q")

    assert result.ok is False
    assert "neo4j down" in result.stopped_reason
    assert result.trace[0]["result_or_error"] == "RuntimeError: neo4j down"


def test_trace_is_always_returned(monkeypatch):
    """Every run returns a trace, success or failure."""
    scripted = [{"content": "", "tool_calls": [_tool_call("bogus", {})]}]
    agent, _, _ = _orchestrator(monkeypatch, scripted)
    result = agent.run("q")

    entry = result.trace[0]
    assert set(entry) == {
        "step",
        "tool_name",
        "arguments",
        "validation_error",
        "result_or_error",
        "timestamp",
    }


def test_judge_disabled_by_default_reuses_main_llm():
    """With no KG_JUDGE_* set, no separate judge is built (backward compatible)."""
    from kg_agent.agentic_verifier import get_judge_client
    from kg_agent.config import JudgeConfig, get_config

    cfg = get_config()
    cfg.judge = JudgeConfig(provider="", model="")
    assert cfg.judge.enabled is False
    assert get_judge_client(cfg) is None


def test_judge_mock_provider_builds_mock_client():
    from kg_agent.agentic_verifier import MockLLMClient, get_judge_client
    from kg_agent.config import JudgeConfig, get_config

    cfg = get_config()
    cfg.judge = JudgeConfig(provider="mock", model="")
    judge = get_judge_client(cfg)
    assert isinstance(judge, MockLLMClient)


def test_judge_ollama_uses_separate_model_and_url():
    """A separate judge points at its own model/url, distinct from the main LLM."""
    from kg_agent.agentic_verifier import get_judge_client
    from kg_agent.config import JudgeConfig, get_config

    cfg = get_config()
    cfg.judge = JudgeConfig(
        provider="ollama", model="hermes3:8b", ollama_url="http://localhost:11434"
    )
    judge = get_judge_client(cfg)  # no network: model given, so no /api/tags lookup
    assert judge.name == "ollama:hermes3:8b"
    assert judge._url == "http://localhost:11434/api/chat"


def test_judge_ollama_requires_model():
    from kg_agent.agentic_verifier import get_judge_client
    from kg_agent.config import JudgeConfig, get_config

    cfg = get_config()
    cfg.judge = JudgeConfig(provider="ollama", model="")
    try:
        get_judge_client(cfg)
    except RuntimeError as exc:
        assert "KG_JUDGE_MODEL" in str(exc)
    else:
        raise AssertionError("get_judge_client should have raised")


def test_judge_unknown_provider_raises():
    from kg_agent.agentic_verifier import get_judge_client
    from kg_agent.config import JudgeConfig, get_config

    cfg = get_config()
    cfg.judge = JudgeConfig(provider="banana", model="x")
    try:
        get_judge_client(cfg)
    except RuntimeError as exc:
        assert "Unknown KG_JUDGE_PROVIDER" in str(exc)
    else:
        raise AssertionError("get_judge_client should have raised")


# --------------------------------------------------------------------------- #
# Phase 2 - freshness_reference must not silently swallow wrong-typed timestamps
# --------------------------------------------------------------------------- #
def test_freshness_reference_warns_on_wrong_type_timestamp(caplog):
    """A present-but-wrong-type timestamp (e.g. ISO string) logs a warning."""
    import logging

    from kg_agent.temporal_validity import freshness_reference

    entity = {"name": "Legacy Node", "created_at": "2025-01-01T00:00:00+00:00"}
    with caplog.at_level(logging.WARNING, logger="kg_agent.temporal_validity"):
        result = freshness_reference(entity)

    assert result is None  # the bad value is not used as a timestamp
    assert len(caplog.records) == 1
    msg = caplog.records[0].getMessage()
    assert "Legacy Node" in msg  # names the node
    assert "created_at" in msg  # names the field
    assert "str" in msg  # names the actual type


def test_freshness_reference_silent_when_timestamp_absent(caplog):
    """A genuinely missing timestamp is treated as timeless WITHOUT a warning."""
    import logging

    from kg_agent.temporal_validity import freshness_reference

    with caplog.at_level(logging.WARNING, logger="kg_agent.temporal_validity"):
        # No timestamp keys at all, plus an explicit-null one.
        assert freshness_reference({"name": "Fresh Node"}) is None
        assert freshness_reference({"name": "Null Node", "created_at": None}) is None

    assert caplog.records == []  # absence is normal, not worth warning about


def test_freshness_reference_uses_valid_datetime_without_warning(caplog):
    """A correct datetime is used, and mixing in a bad sibling still warns once."""
    import logging
    from datetime import datetime, timezone

    from kg_agent.temporal_validity import freshness_reference

    good = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with caplog.at_level(logging.WARNING, logger="kg_agent.temporal_validity"):
        # updated_at is valid; created_at is a bad string -> use the good one,
        # but still warn about the bad sibling rather than hide it.
        result = freshness_reference(
            {"name": "Mixed", "updated_at": good, "created_at": "2020-01-01"}
        )

    assert result == good
    assert len(caplog.records) == 1
    assert "created_at" in caplog.records[0].getMessage()


def test_ollama_model_guard_rejects_groq_default():
    """Ollama provider with KG_LLM_MODEL unset must fail loudly, not default."""
    from kg_agent.agentic_verifier import resolve_ollama_model
    from kg_agent.config import DEFAULT_GROQ_MODEL, get_config

    cfg = get_config()
    cfg.llm.model = DEFAULT_GROQ_MODEL
    # Unroutable address so the advisory /api/tags lookup fails fast.
    cfg.retrieval.ollama_url = "http://127.0.0.1:1"
    try:
        resolve_ollama_model(cfg)
    except RuntimeError as exc:
        assert "KG_LLM_MODEL is not set" in str(exc)
        assert "hermes3" not in str(exc)  # the stale default is gone for good
    else:
        raise AssertionError("resolve_ollama_model should have raised")

    cfg.llm.model = "qwen3-vl:4b"
    assert resolve_ollama_model(cfg) == "qwen3-vl:4b"


# --------------------------------------------------------------------------- #
# Configurable chunk/entity schema (adapting the retriever to a differently-
# shaped graph, e.g. Nabhyla's Topic/Type/Description concept graph, without
# touching any live data - these tests only check the rendered Cypher pattern).
# --------------------------------------------------------------------------- #
def test_default_chunk_pattern_matches_original_hardcoded_shape(monkeypatch):
    """With no overrides, the rendered pattern is exactly the old literal Cypher.

    Explicitly clears the relevant env vars first: RetrievalConfig's fields
    read straight from os.environ via _env_str, so this test would otherwise
    silently assert on whatever profile happens to be active in a real
    developer's .env (e.g. the Nabhyla profile) instead of the true default -
    exactly the kind of environment-dependent flakiness that bit this suite
    once already.
    """
    for key in (
        "KG_CHUNK_LABEL",
        "KG_CHUNK_TEXT_PROP",
        "KG_CHUNK_TO_ENTITY_PATTERN",
        "KG_ENTITY_LABEL",
    ):
        monkeypatch.delenv(key, raising=False)

    from kg_agent.agentic_verifier import _chunk_to_entity_pattern
    from kg_agent.config import RetrievalConfig
    from kg_agent.neo4j_client import Neo4jClient, escape_label

    retrieval = RetrievalConfig()
    assert retrieval.chunk_label == "Chunk"
    assert retrieval.chunk_text_property == "text"

    class _Cfg:
        pass

    cfg = _Cfg()
    cfg.retrieval = retrieval

    client = Neo4jClient.__new__(Neo4jClient)  # no live connection needed
    client._label = escape_label("__Entity__")

    pattern = _chunk_to_entity_pattern(client, cfg)
    assert pattern == "(c:`Chunk`)-[:MENTIONS]->(e:`__Entity__`)"


def test_chunk_pattern_renders_nabhyla_two_hop_shape():
    """A differently-shaped graph (no Chunk/MENTIONS at all) is expressible
    purely through config - no code changes needed per graph.

    This mirrors Nabhyla's real schema recon: entities are ``Topic`` nodes,
    and the text-bearing node (``Description``) reaches them two hops away,
    through an intermediate ``Type`` node, via ``HAS_DESCRIPTION``/``HAS_TYPE``
    (not ``MENTIONS``).
    """
    from kg_agent.agentic_verifier import _chunk_to_entity_pattern
    from kg_agent.config import RetrievalConfig, get_config
    from kg_agent.neo4j_client import Neo4jClient

    cfg = get_config()
    cfg.retrieval = RetrievalConfig(
        chunk_label="Description",
        chunk_text_property="text",
        chunk_to_entity_pattern=(
            "(c:{chunk_label})<-[:HAS_DESCRIPTION]-(:Type)<-[:HAS_TYPE]-(e:{entity_label})"
        ),
    )

    client = Neo4jClient.__new__(Neo4jClient)
    client._label = "`Topic`"

    pattern = _chunk_to_entity_pattern(client, cfg)
    assert pattern == (
        "(c:`Description`)<-[:HAS_DESCRIPTION]-(:Type)<-[:HAS_TYPE]-(e:`Topic`)"
    )
    # Both node variables the rest of the retriever depends on (`c`, `e`) are
    # still present, whatever the path between them looks like.
    assert "(c:" in pattern
    assert "(e:" in pattern


def test_escape_label_is_public_and_backward_compatible():
    """escape_label replaces the old _escape_label; the alias must still work."""
    from kg_agent.neo4j_client import _escape_label, escape_label

    assert escape_label is _escape_label
    assert escape_label("Weird`Label") == "`Weird``Label`"


class _StubNabhylaClient:
    """Fakes just enough of Neo4jClient to exercise retrieval end to end,
    returning canned rows shaped like a real Topic/Type/Description query
    would - without any live database.
    """

    _label = "`Topic`"
    _name_prop = "name"

    def __init__(self, canned_rows):
        self._canned = canned_rows
        self.queries = []

    def run_read(self, cypher, **params):
        self.queries.append((cypher, params))
        return self._canned


def _nabhyla_config():
    from kg_agent.config import RetrievalConfig, get_config

    cfg = get_config()
    cfg.retrieval = RetrievalConfig(
        chunk_label="Description",
        chunk_text_property="text",
        chunk_to_entity_pattern=(
            "(c:{chunk_label})<-[:HAS_DESCRIPTION]-(:Type)<-[:HAS_TYPE]-(e:{entity_label})"
        ),
    )
    return cfg


def test_retrieve_keyword_against_nabhyla_shaped_rows():
    """retrieve_keyword issues Topic/Type/Description Cypher (no Chunk/MENTIONS
    anywhere) and correctly parses the returned rows into a RetrievedContext.
    """
    from kg_agent.agentic_verifier import retrieve_keyword

    canned = [
        {
            "text": "PNP (Pick-and-Pass) is an order picking method where "
            "totes move through a sequence of fixed zones.",
            "entity_names": ["Pnp"],
        }
    ]
    client = _StubNabhylaClient(canned)
    cfg = _nabhyla_config()

    ctx = retrieve_keyword(client, cfg, "what is the pick and pass method?", k=5)

    assert ctx.node_names == ["Pnp"]
    assert ctx.chunks == [canned[0]["text"]]

    all_cypher = " ".join(q[0] for q in client.queries)
    assert "Description" in all_cypher
    assert "HAS_DESCRIPTION" in all_cypher
    assert "HAS_TYPE" in all_cypher
    assert "Chunk" not in all_cypher
    assert "MENTIONS" not in all_cypher


def test_retrieve_falls_back_to_keyword_when_vector_index_missing(monkeypatch):
    """A missing/unusable vector index must degrade to keyword retrieval, not
    crash. Regression: with a working embed model but no vector index (a common
    shape mismatch, e.g. an index on :Chunk when the graph has :Description),
    the ClientError from db.index.vector.queryNodes used to propagate and kill
    the whole query.
    """
    from neo4j.exceptions import ClientError

    from kg_agent import agentic_verifier as av

    # Pretend embedding succeeds so we actually reach the vector index query.
    monkeypatch.setattr(av, "_embed_query_ollama", lambda query, config: [0.1, 0.2, 0.3])

    canned = [{"text": "PNP is a pick-and-pass order picking method.", "entity_names": ["Pnp"]}]

    class _IndexMissingClient:
        _label = "`Topic`"
        _name_prop = "name"

        def __init__(self):
            self.queries = []

        def run_read(self, cypher, **params):
            self.queries.append(cypher)
            if "queryNodes" in cypher:  # the vector-index call
                raise ClientError("There is no such vector schema index: vector")
            return canned  # the keyword-fallback query

    client = _IndexMissingClient()
    cfg = _nabhyla_config()
    cfg.retrieval.prefer_vector = True

    ctx = av.retrieve(client, cfg, "what is pnp?", "vector", k=5)

    assert ctx.node_names == ["Pnp"]  # keyword result surfaced, no crash
    assert ctx.chunks == [canned[0]["text"]]
    assert any("queryNodes" in q for q in client.queries)  # vector WAS attempted
    assert any("HAS_DESCRIPTION" in q for q in client.queries)  # then keyword ran


def test_retrieve_vector_filters_by_configured_chunk_label(monkeypatch):
    """The vector-index call must filter results to the configured chunk_label.

    Regression found live against Nabhyla's real graph: her vector index
    ``vector`` covers ``:Chunk``, but the Topic/Description profile's
    ``chunk_label`` is ``Description``. Without a label filter right after
    ``db.index.vector.queryNodes``, the call returns real ``:Chunk`` nodes
    (which happen to share the ``text`` property name) with zero entity
    attribution - an answer that *looks* like it worked (non-empty text) but
    silently grounds itself in the wrong layer of the graph (sources_used
    empty, trust_score 0.0). The fix filters the vector hits to the
    configured chunk label so a mismatch falls through to keyword retrieval
    instead, via the existing "no chunks -> fallback" path in ``retrieve()``.
    """
    from kg_agent import agentic_verifier as av

    monkeypatch.setattr(av, "_embed_query_ollama", lambda query, config: [0.1, 0.2, 0.3])

    class _RecordingClient:
        _label = "`Topic`"
        _name_prop = "name"

        def __init__(self, rows):
            self._rows = rows
            self.queries = []

        def run_read(self, cypher, **params):
            self.queries.append(cypher)
            return self._rows

    client = _RecordingClient([])
    cfg = _nabhyla_config()

    av.retrieve_vector(client, cfg, "what is pnp?", k=5)

    cypher = client.queries[0]
    assert "queryNodes" in cypher
    assert "WHERE c:`Description`" in cypher


def test_parse_faithfulness_handles_braceless_json():
    """Regression: hermes3 emits brace-less JSON; the old parser dropped it to 0.0."""
    from kg_agent.agentic_verifier import _parse_faithfulness

    # full object
    assert _parse_faithfulness('{"faithfulness": 0.9, "verdict": "ok"}')["faithfulness"] == 0.9
    # brace-LESS (the real hermes3 quirk) - must NOT collapse to 0.0
    braceless = '"faithfulness": 0.8, "verdict": "Supported", "unsupported_claims": []'
    assert _parse_faithfulness(braceless)["faithfulness"] == 0.8
    # number embedded in prose
    assert _parse_faithfulness('Here is my verdict: "faithfulness": 0.7 done')["faithfulness"] == 0.7
    # genuinely unparseable -> 0.0
    assert _parse_faithfulness("the answer looks fine to me")["faithfulness"] == 0.0


def test_effective_embed_url_falls_back_to_ollama_url():
    """KG_EMBED_URL overrides the embed host; empty reuses OLLAMA_URL."""
    from kg_agent.config import RetrievalConfig

    assert RetrievalConfig(ollama_url="http://main:11434", embed_url="").effective_embed_url == (
        "http://main:11434"
    )
    assert RetrievalConfig(
        ollama_url="http://main:11434", embed_url="http://embed:11434"
    ).effective_embed_url == "http://embed:11434"


def test_similarity_threshold_suppresses_keyword_fallback(monkeypatch):
    """With a threshold set, an empty vector result must NOT flood back via keyword."""
    from kg_agent import agentic_verifier as av
    from kg_agent.agentic_verifier import RetrievedContext
    from kg_agent.config import get_config

    cfg = get_config()
    cfg.retrieval.prefer_vector = True
    empty_vec = RetrievedContext(node_names=[], chunks=[], strategy="vector")
    monkeypatch.setattr(av, "retrieve_vector", lambda *a, **k: empty_vec)
    calls = {"kw": 0}

    def fake_keyword(*a, **k):
        calls["kw"] += 1
        return RetrievedContext(node_names=["X"], chunks=["kw"], strategy="keyword")

    monkeypatch.setattr(av, "retrieve_keyword", fake_keyword)

    # threshold OFF -> empty vector degrades to keyword (old behaviour)
    cfg.retrieval.vector_min_score = 0.0
    r0 = av.retrieve(None, cfg, "q", "vector", 5)
    assert r0.strategy == "keyword" and calls["kw"] == 1

    # threshold ON -> empty vector is honoured; keyword NOT called again
    cfg.retrieval.vector_min_score = 0.6
    r1 = av.retrieve(None, cfg, "q", "vector", 5)
    assert r1.strategy == "vector" and r1.chunks == [] and calls["kw"] == 1


def test_context_for_names_against_nabhyla_shaped_rows():
    """AgenticVerifier._context_for_names uses the same configurable pattern
    for the caller-supplied-names path (used on a fixed first retrieval).
    """
    from kg_agent.agentic_verifier import AgenticVerifier

    canned = [{"text": "Throughput improved 18% over the FCFS baseline."}]
    client = _StubNabhylaClient(canned)
    cfg = _nabhyla_config()

    verifier = AgenticVerifier.__new__(AgenticVerifier)  # skip __init__ (no LLM needed)
    verifier.client = client
    verifier.config = cfg

    text = verifier._context_for_names(["Throughput"])

    assert canned[0]["text"] in text
    cypher, params = client.queries[-1]
    assert "HAS_TYPE" in cypher
    assert "HAS_DESCRIPTION" in cypher
    assert params["names"] == ["throughput"]


# --------------------------------------------------------------------------- #
# Empty-answer handling: a "thinking" main LLM (qwen3-vl:4b) can spend its
# whole token budget reasoning and emit no bounded content at all. Found live
# against Nabhyla's real graph: the answer was blank, yet the faithfulness
# judge scored it 0.70 - clearing the gate for literally nothing.
# --------------------------------------------------------------------------- #
class _FakeCompletionLLM:
    """A minimal LLMClient stand-in that returns a scripted completion."""

    name = "fake"

    def __init__(self, response):
        self._response = response
        self.calls = []

    def complete(self, system, user):
        self.calls.append((system, user))
        return self._response


def _bare_verifier(llm=None, judge=None):
    from kg_agent.agentic_verifier import AgenticVerifier
    from kg_agent.config import get_config

    verifier = AgenticVerifier.__new__(AgenticVerifier)  # skip __init__: no live LLM/DB
    verifier.config = get_config()
    verifier.llm = llm
    verifier.judge = judge if judge is not None else llm
    return verifier


def test_generate_answer_never_returns_blank_string():
    """An empty completion becomes an explicit sentinel, never ''."""
    from kg_agent.agentic_verifier import _EMPTY_ANSWER_MESSAGE

    verifier = _bare_verifier(llm=_FakeCompletionLLM("   "))  # whitespace-only
    answer = verifier._generate_answer("what is pnp?", "some real context")

    assert answer == _EMPTY_ANSWER_MESSAGE
    assert answer.strip() != ""


def test_generate_answer_passes_through_real_content():
    """A normal, non-empty completion is returned unmodified (stripped)."""
    verifier = _bare_verifier(llm=_FakeCompletionLLM("  PNP is pick-and-pass.  "))
    answer = verifier._generate_answer("what is pnp?", "some real context")

    assert answer == "PNP is pick-and-pass."


def test_check_faithfulness_short_circuits_on_empty_answer():
    """An empty/sentinel answer must never reach the judge - regression for
    the live finding: hermes3:3b scored a blank answer 0.70 (>= the 0.7
    gate), which would have silently passed verification for no content.
    """
    from kg_agent.agentic_verifier import _EMPTY_ANSWER_MESSAGE

    judge = _FakeCompletionLLM('{"faithfulness": 0.7, "verdict": "supported"}')
    verifier = _bare_verifier(llm=None, judge=judge)

    result = verifier._check_faithfulness(_EMPTY_ANSWER_MESSAGE, "some real context")

    assert result["faithfulness"] == 0.0
    assert result["verdict"] == "empty_answer"
    assert judge.calls == []  # the judge was never actually asked


def test_check_faithfulness_still_calls_judge_for_real_answers():
    """Non-empty answers are unaffected - still sent to the judge as before."""
    judge = _FakeCompletionLLM('{"faithfulness": 0.9, "verdict": "supported"}')
    verifier = _bare_verifier(llm=None, judge=judge)

    result = verifier._check_faithfulness("PNP is pick-and-pass.", "some real context")

    assert result["faithfulness"] == 0.9
    assert len(judge.calls) == 1


def _chunk_source_config(source_prop=""):
    """Config for a Chunk/MENTIONS corpus, optionally reporting document origin."""
    from kg_agent.config import RetrievalConfig, get_config

    cfg = get_config()
    cfg.retrieval = RetrievalConfig(
        chunk_label="Chunk",
        chunk_text_property="text",
        chunk_to_entity_pattern="(c:{chunk_label})-[:MENTIONS]->(e:{entity_label})",
        chunk_source_property=source_prop,
    )
    return cfg


def test_retrieval_omits_document_fields_when_source_prop_unset():
    """Default config is unchanged: no source property, no document reporting,
    and the Cypher must not reference a source column at all.
    """
    from kg_agent.agentic_verifier import retrieve_keyword

    canned = [{"text": "SCI improves performance.", "entity_names": ["Sci"]}]
    client = _StubNabhylaClient(canned)

    ctx = retrieve_keyword(client, _chunk_source_config(), "supply chain integration", k=5)

    assert ctx.documents == []
    assert ctx.entity_documents == {}
    assert "null AS source_doc" in client.queries[0][0]


def test_retrieval_reports_document_origin_when_source_prop_set():
    """With a source property configured, retrieval ranks the documents the
    chunks came from (most-cited first) and records which documents mention
    each entity.
    """
    from kg_agent.agentic_verifier import retrieve_keyword

    canned = [
        {"text": "Neely: 12 approaches.", "entity_names": ["Servitization"], "source_doc": "Neely_2008"},
        {"text": "Brax: configuration.", "entity_names": ["Servitization"], "source_doc": "Brax_2021"},
        {"text": "Brax: paradox.", "entity_names": ["Paradox"], "source_doc": "Brax_2021"},
    ]
    client = _StubNabhylaClient(canned)

    ctx = retrieve_keyword(client, _chunk_source_config("filename"), "servitization", k=5)

    assert "c.`filename` AS source_doc" in client.queries[0][0]
    # Brax contributed two chunks, Neely one -> Brax ranks first.
    assert ctx.documents == [
        {"name": "Brax_2021", "chunks": 2},
        {"name": "Neely_2008", "chunks": 1},
    ]
    # "Servitization" is mentioned by both papers - the collision is visible.
    assert ctx.entity_documents["Servitization"] == ["Neely_2008", "Brax_2021"]
    assert ctx.entity_documents["Paradox"] == ["Brax_2021"]


def test_build_sources_attaches_documents_per_entity():
    """Per-entity document provenance reaches sources_used, so an answer that
    silently mixed two papers can be detected from the output alone.
    """
    from kg_agent.temporal_validity import NodeValidity, ValidityStatus

    verifier = _bare_verifier(llm=None, judge=None)
    report = [
        NodeValidity(
            element_id="1", entity_id="t3", name="Table 3",
            status=ValidityStatus.VALID, age_days=None, reasons=[],
        )
    ]

    sources = verifier._build_sources(
        report, scores=[], names=["Table 3"],
        entity_documents={"Table 3": ["Neely_2008", "Franke_1978"]},
    )

    assert sources[0]["documents"] == ["Neely_2008", "Franke_1978"]


def test_build_sources_omits_documents_when_absent():
    """Without provenance the key is left out entirely rather than set empty."""
    from kg_agent.temporal_validity import NodeValidity, ValidityStatus

    verifier = _bare_verifier(llm=None, judge=None)
    report = [
        NodeValidity(
            element_id="1", entity_id="sci", name="Sci",
            status=ValidityStatus.VALID, age_days=None, reasons=[],
        )
    ]

    sources = verifier._build_sources(report, scores=[], names=["Sci"])

    assert "documents" not in sources[0]


def _loop_verifier(monkeypatch, faithfulness, retrieved):
    """A verifier whose retrieval, validity and trust steps are stubbed out, so
    ``verify()`` can be driven end to end without a database or an LLM.

    ``retrieved`` collects the strategy of every retrieval actually performed,
    which is the ground truth the ``retries`` field is checked against.
    """
    import kg_agent.agentic_verifier as av

    def fake_retrieve(client, config, query, strategy, k):
        retrieved.append(strategy)
        return av.RetrievedContext(node_names=["Servitization"], chunks=["ctx"], strategy=strategy)

    monkeypatch.setattr(av, "retrieve", fake_retrieve)
    monkeypatch.setattr(av, "generate_validity_report", lambda *a, **k: [])
    monkeypatch.setattr(av, "score_entities", lambda *a, **k: [])

    verifier = _bare_verifier(llm=_FakeCompletionLLM("Some answer."))
    verifier.judge = _FakeCompletionLLM('{"faithfulness": %s, "verdict": "x"}' % faithfulness)
    verifier.client = None
    # Pin the loop's policy so the test does not depend on the ambient .env.
    verifier.config.verifier.max_retries = 1
    verifier.config.verifier.min_faithfulness = 0.7
    verifier.config.verifier.min_trust_score = 0.4
    return verifier


def test_retries_counts_attempts_run_not_the_winning_attempt(monkeypatch):
    """A confidence tie keeps attempt 0 as the answer, but the retry that
    really happened must still be reported.

    Reproduces the live "How many servitization strategies did Neely identify?"
    case: both attempts scored faithfulness 0.0, so the argmax (a strict ``>``)
    kept the first, and its attempt index hid the second retrieval.
    """
    retrieved = []
    verifier = _loop_verifier(monkeypatch, faithfulness=0.0, retrieved=retrieved)

    result = verifier.verify("how many servitization strategies did Neely identify?")

    assert len(retrieved) == 2, "a retry really happened"
    assert result.passed is False
    # Attempt 0 still wins the tie and supplies the answer...
    assert result.strategy == retrieved[0]
    # ...but the retry is no longer hidden.
    assert result.retries == 1


def test_retries_is_zero_when_the_first_attempt_passes(monkeypatch):
    """The passing path returns on attempt 0 with no retry and no disclaimer -
    the counter must not drift now that it is tracked separately.
    """
    retrieved = []
    verifier = _loop_verifier(monkeypatch, faithfulness=0.95, retrieved=retrieved)
    verifier.config.verifier.min_trust_score = 0.0  # let the trust gate pass

    result = verifier.verify("what are the dimensions of supply chain integration?")

    assert len(retrieved) == 1
    assert result.passed is True
    assert result.retries == 0
    assert result.disclaimer == ""



# --------------------------------------------------------------------------- #
# Audit trail for write tools
# --------------------------------------------------------------------------- #
TRANSCRIPT = (
    "Budi said the Q3 revenue target is confidential and must not leave the room. "
    "Siti raised a concern about the vendor contract."
)


def _audit_cfg(tmpdir, read_only):
    from kg_agent.config import get_config

    cfg = get_config()
    cfg.safety.read_only = read_only
    cfg.audit.enabled = True
    cfg.audit.directory = str(tmpdir)
    return cfg


def _audit_lines(tmpdir):
    import json
    import pathlib as _p

    path = _p.Path(tmpdir) / "ingest_meeting.jsonl"
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def test_audit_records_a_rejected_write(tmp_path):
    """A refused attempt is the one trace that somebody tried to write at all,
    so it must be logged even though nothing reached the graph.
    """
    import pytest

    from kg_agent.tools import call_tool

    cfg = _audit_cfg(tmp_path, read_only=True)
    with pytest.raises(PermissionError):
        call_tool("ingest_meeting", None, cfg,
                  {"title": "Sync", "notes": TRANSCRIPT}, caller="10.0.0.7")

    lines = _audit_lines(tmp_path)
    assert len(lines) == 1
    entry = lines[0]
    assert entry["status"] == "rejected_403"
    assert entry["caller"] == "10.0.0.7"
    assert entry["input_size_chars"] == len("Sync") + len(TRANSCRIPT)
    assert entry["duration_ms"] >= 0
    assert entry["request_id"]
    assert entry["error"]


def test_audit_never_writes_transcript_content(tmp_path):
    """The whole payload of this endpoint is meeting content. None of it may
    reach a log file that is kept for a month and read over SSH.
    """
    import pytest

    from kg_agent.tools import call_tool

    cfg = _audit_cfg(tmp_path, read_only=True)
    with pytest.raises(PermissionError):
        call_tool("ingest_meeting", None, cfg,
                  {"title": "Sync", "notes": TRANSCRIPT,
                   "participants": ["Budi", "Siti"]})

    raw = (tmp_path / "ingest_meeting.jsonl").read_text()
    assert TRANSCRIPT not in raw
    assert "confidential" not in raw
    assert "vendor contract" not in raw
    # Size is recorded, content is not.
    assert str(len("Sync") + len(TRANSCRIPT) + len("Budi") + len("Siti")) in raw


def test_audit_records_a_successful_write_with_counts(tmp_path):
    from kg_agent.tools import TOOLS, call_tool

    cfg = _audit_cfg(tmp_path, read_only=False)
    original = TOOLS["ingest_meeting"]["fn"]
    TOOLS["ingest_meeting"]["fn"] = lambda client, config, **kw: {
        "meeting_id": "4:abc:12",
        "counters": {"nodes_created": 3, "relationships_created": 2},
    }
    try:
        call_tool("ingest_meeting", None, cfg, {"title": "Sync", "notes": TRANSCRIPT})
    finally:
        TOOLS["ingest_meeting"]["fn"] = original

    entry = _audit_lines(tmp_path)[0]
    assert entry["status"] == "success"
    assert entry["meeting_id"] == "4:abc:12"
    assert entry["nodes_created"] == 3
    assert entry["relationships_created"] == 2
    assert entry["error"] is None


def test_audit_records_a_failed_write(tmp_path):
    import pytest

    from kg_agent.tools import TOOLS, call_tool

    cfg = _audit_cfg(tmp_path, read_only=False)

    def boom(client, config, **kw):
        raise RuntimeError("neo4j unreachable")

    original = TOOLS["ingest_meeting"]["fn"]
    TOOLS["ingest_meeting"]["fn"] = boom
    try:
        with pytest.raises(RuntimeError):
            call_tool("ingest_meeting", None, cfg, {"title": "Sync"})
    finally:
        TOOLS["ingest_meeting"]["fn"] = original

    entry = _audit_lines(tmp_path)[0]
    assert entry["status"] == "failed"
    assert "neo4j unreachable" in entry["error"]


def test_audit_ignores_read_only_tools(tmp_path):
    """Read tools would bury the file under query traffic and change nothing."""
    from kg_agent.tools import TOOLS, call_tool

    cfg = _audit_cfg(tmp_path, read_only=True)
    original = TOOLS["kg_stats"]["fn"]
    TOOLS["kg_stats"]["fn"] = lambda client, config: {"entities": 1}
    try:
        call_tool("kg_stats", None, cfg)
    finally:
        TOOLS["kg_stats"]["fn"] = original

    assert not (tmp_path / "kg_stats.jsonl").exists()


def test_ingest_meeting_counters_cover_linked_entities():
    """Counters must total every write, not just the Meeting node - otherwise
    the audit trail under-reports what a call actually created.
    """
    from kg_agent.config import get_config
    from kg_agent.tools import ingest_meeting

    class _CountingClient:
        def __init__(self):
            self.calls = 0

        def run_write(self, cypher, **params):
            self.calls += 1
            if self.calls == 1:
                return {"records": [{"meeting_id": "4:abc:1"}], "nodes_created": 1,
                        "relationships_created": 0, "properties_set": 5}
            return {"records": [], "nodes_created": 1,
                    "relationships_created": 1, "properties_set": 3}

    out = ingest_meeting(_CountingClient(), get_config(), title="Sync",
                         entities=["Alpha", "Beta"])
    assert out["meeting_id"] == "4:abc:1"
    assert out["counters"]["nodes_created"] == 3
    assert out["counters"]["relationships_created"] == 2



def _query_lines(tmpdir):
    import json
    import pathlib as _p

    path = _p.Path(tmpdir) / "answer_question.jsonl"
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


VERIFIED_ANSWER = {
    "answer": "Levitt and List found the described pattern was fictional. " * 4,
    "passed": False,
    "faithfulness": 0.7,
    "trust_score": 0.2,
    "overall_confidence": 0.69,
    "temporal_validity_status": "VALID",
    "strategy": "keyword",
    "retries": 1,
    "sources_used": [{"name": "Hawthorne Effect"}, {"name": "Productivity"}],
    "documents_used": [{"name": "Levitt_2009", "chunks": 12}],
}


def test_query_audit_records_gate_outcome_and_retrieval(tmp_path):
    """Latency alone cannot explain a bad answer; the gate outcome and which
    documents were cited are what make production behaviour reviewable.
    """
    from kg_agent.tools import TOOLS, call_tool

    cfg = _audit_cfg(tmp_path, read_only=True)
    original = TOOLS["answer_question"]["fn"]
    TOOLS["answer_question"]["fn"] = lambda client, config, **kw: VERIFIED_ANSWER
    try:
        call_tool("answer_question", None, cfg,
                  {"query": "What did Levitt conclude?"}, caller="10.0.0.9")
    finally:
        TOOLS["answer_question"]["fn"] = original

    entry = _query_lines(tmp_path)[0]
    assert entry["status"] == "success"
    assert entry["caller"] == "10.0.0.9"
    assert entry["duration_ms"] >= 0
    assert entry["passed"] is False
    assert entry["faithfulness"] == 0.7
    assert entry["strategy"] == "keyword"
    assert entry["retries"] == 1
    assert entry["n_sources"] == 2
    assert entry["documents"] == ["Levitt_2009"]
    assert entry["query"] == "What did Levitt conclude?"


def test_query_audit_stores_answer_length_not_answer_text(tmp_path):
    """The answer is reproducible by re-running; storing it only bloats a log
    that is kept for a month.
    """
    from kg_agent.tools import TOOLS, call_tool

    cfg = _audit_cfg(tmp_path, read_only=True)
    original = TOOLS["answer_question"]["fn"]
    TOOLS["answer_question"]["fn"] = lambda client, config, **kw: VERIFIED_ANSWER
    try:
        call_tool("answer_question", None, cfg, {"query": "q"})
    finally:
        TOOLS["answer_question"]["fn"] = original

    raw = (tmp_path / "answer_question.jsonl").read_text()
    assert "fictional" not in raw
    assert _query_lines(tmp_path)[0]["answer_chars"] == len(VERIFIED_ANSWER["answer"])


def test_query_text_can_be_switched_off(tmp_path):
    from kg_agent.tools import TOOLS, call_tool

    cfg = _audit_cfg(tmp_path, read_only=True)
    cfg.audit.log_query_text = False
    original = TOOLS["answer_question"]["fn"]
    TOOLS["answer_question"]["fn"] = lambda client, config, **kw: VERIFIED_ANSWER
    try:
        call_tool("answer_question", None, cfg, {"query": "a secret question"})
    finally:
        TOOLS["answer_question"]["fn"] = original

    raw = (tmp_path / "answer_question.jsonl").read_text()
    assert "secret question" not in raw
    assert "query" not in _query_lines(tmp_path)[0]
    # Size is still recorded, so volume stays analysable.
    assert _query_lines(tmp_path)[0]["input_size_chars"] == len("a secret question")


def test_query_logging_can_be_disabled_without_affecting_writes(tmp_path):
    """Query traffic dwarfs writes, so an operator may want only the write
    trail - and turning queries off must not silently drop the write trail.
    """
    import pytest

    from kg_agent.tools import TOOLS, call_tool

    cfg = _audit_cfg(tmp_path, read_only=True)
    cfg.audit.log_queries = False
    original = TOOLS["answer_question"]["fn"]
    TOOLS["answer_question"]["fn"] = lambda client, config, **kw: VERIFIED_ANSWER
    try:
        call_tool("answer_question", None, cfg, {"query": "q"})
    finally:
        TOOLS["answer_question"]["fn"] = original
    assert not (tmp_path / "answer_question.jsonl").exists()

    with pytest.raises(PermissionError):
        call_tool("ingest_meeting", None, cfg, {"title": "Sync"})
    assert len(_audit_lines(tmp_path)) == 1


def test_query_audit_records_a_failure(tmp_path):
    import pytest

    from kg_agent.tools import TOOLS, call_tool

    cfg = _audit_cfg(tmp_path, read_only=True)

    def boom(client, config, **kw):
        raise RuntimeError("ollama unreachable")

    original = TOOLS["answer_question"]["fn"]
    TOOLS["answer_question"]["fn"] = boom
    try:
        with pytest.raises(RuntimeError):
            call_tool("answer_question", None, cfg, {"query": "q"})
    finally:
        TOOLS["answer_question"]["fn"] = original

    entry = _query_lines(tmp_path)[0]
    assert entry["status"] == "failed"
    assert "ollama unreachable" in entry["error"]


def test_stats_tool_still_not_audited(tmp_path):
    """Only writes and question answering are worth a persistent record."""
    from kg_agent.audit_log import should_audit

    assert should_audit("ingest_meeting", True) is True
    assert should_audit("answer_question", False) is True
    assert should_audit("answer_question", False, log_queries=False) is False
    assert should_audit("kg_stats", False) is False



def _log_dir_snapshot():
    """Sizes of any audit files present, taken at import time - before any test
    runs - so a developer's own logs from running the API locally are not
    mistaken for test output.
    """
    import pathlib as _p

    stray = _p.Path("logs")
    if not stray.exists():
        return {}
    return {f.name: f.stat().st_size for f in stray.glob("*.jsonl")}


_LOG_DIR_AT_IMPORT = _log_dir_snapshot()


def test_test_suite_does_not_write_into_the_repo_log_dir():
    """Audit defaults to ./logs, so a test using an unmodified config would
    quietly append to the working tree. Every audited call in this file must
    either disable the audit or point it at a tmp_path.

    Compares against a snapshot taken at import time rather than asserting the
    directory is empty: running the API locally legitimately leaves files there.
    """
    after = _log_dir_snapshot()
    grew = {
        name: (_LOG_DIR_AT_IMPORT.get(name, 0), size)
        for name, size in after.items()
        if size != _LOG_DIR_AT_IMPORT.get(name, 0)
    }
    assert not grew, (
        f"logs/ changed during the test run ({grew}) - a test is using the "
        "default audit directory instead of tmp_path"
    )



def test_no_context_answer_scores_zero_faithfulness_without_asking_the_judge():
    """A lookup that found nothing must not outrank one that succeeded.

    "No supporting context was retrieved" is trivially consistent with having
    no sources, so a judge scores it 1.0. Blended with a full validity term
    (nothing failed, because nothing was retrieved) that put a failed query at
    0.80 confidence against 0.69 for a correctly answered one.
    """
    from kg_agent.agentic_verifier import _NO_CONTEXT_MESSAGE

    judge = _FakeCompletionLLM('{"faithfulness": 1.0, "verdict": "supported"}')
    verifier = _bare_verifier(llm=None, judge=judge)

    result = verifier._check_faithfulness(_NO_CONTEXT_MESSAGE, "")

    assert result["faithfulness"] == 0.0
    assert result["verdict"] == "no_context"
    assert judge.calls == [], "the judge must never see a no-context sentinel"


def test_empty_context_produces_the_no_context_sentinel():
    """_generate_answer and _check_faithfulness must agree on the exact string,
    or the short-circuit silently stops matching.
    """
    from kg_agent.agentic_verifier import _NO_CONTEXT_MESSAGE

    verifier = _bare_verifier(llm=_FakeCompletionLLM("unused"))
    assert verifier._generate_answer("anything", "   ") == _NO_CONTEXT_MESSAGE



# --------------------------------------------------------------------------- #
# Derived provenance - feeds Phase 3 trust with real inputs
# --------------------------------------------------------------------------- #
def test_support_score_saturates_and_is_monotonic():
    """More mentions never lower the score, and a hub entity cannot run away
    with a value that flattens everything else by comparison.
    """
    from kg_agent.provenance import support_score

    scores = [support_score(n, 8) for n in (0, 1, 2, 3, 8, 25, 500)]
    assert scores == sorted(scores)
    assert scores[0] == 0.0
    assert scores[-1] == 1.0
    assert support_score(8, 8) == 1.0


def test_derived_confidence_separates_well_attested_from_one_off():
    """The whole point: a load-bearing concept must outrank a passing mention."""
    from kg_agent.config import get_config
    from kg_agent.provenance import derive_confidence

    cfg = get_config()
    strong = derive_confidence(25, True, True, True, cfg)["confidence_score"]
    weak = derive_confidence(1, False, False, False, cfg)["confidence_score"]

    assert strong == 1.0
    assert weak < 0.4 < strong
    assert weak >= cfg.provenance.confidence_floor


def test_derived_confidence_never_reaches_zero():
    """A zero would wipe out the trust product regardless of the other factors,
    and an entity that was extracted at all is weak evidence, not none.
    """
    from kg_agent.config import get_config
    from kg_agent.provenance import derive_confidence

    cfg = get_config()
    assert derive_confidence(0, False, False, False, cfg)["confidence_score"] >= (
        cfg.provenance.confidence_floor
    )


def test_derived_provenance_actually_varies():
    """Guards the failure this module exists to fix: a single constant across
    every node makes the trust gate indistinguishable from no gate.
    """
    from kg_agent.config import get_config
    from kg_agent.provenance import derive_confidence

    cfg = get_config()
    combos = [
        (1, False, False, False), (1, True, False, False), (1, True, True, False),
        (3, True, True, True), (12, True, True, True), (25, True, True, True),
    ]
    values = {derive_confidence(*c, cfg)["confidence_score"] for c in combos}
    assert len(values) >= 5, f"expected a spread, got {sorted(values)}"


def test_derived_confidence_feeds_the_unchanged_trust_formula():
    """The formula is untouched - only its inputs stop being defaults."""
    from kg_agent.config import get_config
    from kg_agent.node_trust import compute_trust
    from kg_agent.provenance import derive_confidence

    cfg = get_config()
    conf = derive_confidence(25, True, True, True, cfg)["confidence_score"]
    entity = {
        "element_id": "1", "name": "Servitization",
        "confidence_score": conf, "source_type": "paper",
    }
    score = compute_trust(entity, cfg)
    # paper weight 1.0, no timestamp so recency is neutral 1.0
    assert score.source_weight == 1.0
    assert score.recency_factor == 1.0
    assert score.trust_score == round(conf * 1.0 * 1.0, 4)
    assert score.trust_score >= cfg.verifier.min_trust_score


def test_provenance_summary_reports_the_spread():
    from kg_agent.provenance import DerivedProvenance, summarise

    def row(conf):
        return DerivedProvenance(
            element_id="x", name="n", chunks=1, documents=1, has_type=True,
            has_subtopic=False, has_relation=False, support_score=0.0,
            structure_score=0.0, confidence_score=conf, source_type="paper",
        )

    out = summarise([row(0.22), row(0.45), row(0.9)])
    assert out["count"] == 3
    assert out["min"] == 0.22
    assert out["max"] == 0.9
    assert out["distinct_values"] == 3


if __name__ == "__main__":
    import pytest as _pytest

    raise SystemExit(_pytest.main([__file__, "-q"]))
