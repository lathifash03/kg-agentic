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


if __name__ == "__main__":
    import pytest as _pytest

    raise SystemExit(_pytest.main([__file__, "-q"]))
