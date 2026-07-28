"""Phase 5 - native Ollama tool-calling orchestration.

Lets a tool-calling LLM decide *which* registry tool answers a request, then
executes that decision here, in Python. The model chooses; it never verifies.
When it picks ``answer_question`` the call goes through
:meth:`AgenticVerifier.verify` exactly as before, and the resulting
:class:`VerifiedAnswer` is attached to the response **unmodified** - the
trust/temporal/faithfulness gates are not bypassed, softened or re-summarised
away.

How it works
------------
``POST {OLLAMA_URL}/api/chat`` with the registry exported as Ollama's native
``tools`` parameter. Ollama parses the model's tool-call syntax server-side and
returns ``message.tool_calls`` as structured JSON, so no XML/regex scraping
happens here. Every requested call is checked against the registry whitelist
*and* its JSON schema before anything executes.

Enable with ``KG_ORCHESTRATOR=native`` (default ``off``), and point
``KG_LLM_MODEL`` at a model whose ``/api/show`` capabilities include ``tools``.

Run directly::

    KG_ORCHESTRATOR=native python -m kg_agent.orchestrator --query "..."
"""

from __future__ import annotations

import json
import logging
import socket
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from kg_agent.agentic_verifier import resolve_ollama_model
from kg_agent.config import Config, get_config
from kg_agent.neo4j_client import Neo4jClient
from kg_agent.tools import TOOLS, call_tool, tool_specs

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are the kg-agent orchestrator. You answer questions using a Neo4j knowledge graph, and you have tools that reach it.

Rules:
1. Any question about what the knowledge graph contains - facts, decisions, \
entities, meetings, "what did we decide", "who worked on X" - MUST be answered \
by calling `answer_question`. Never answer such a question from your own \
knowledge: only `answer_question` applies the trust, temporal-validity and \
faithfulness checks, and an answer that skips it is unverified and unusable.
2. Use `kg_stats` for questions about the size or shape of the graph itself \
(how many entities, meetings or relationships it holds).
3. Use `ingest_meeting` to record a meeting into the graph.
4. Only answer directly, without a tool, when the request has nothing to do \
with the knowledge graph.
5. Pass arguments exactly as each tool's schema declares them.

After a tool returns, report its result faithfully. Never contradict, inflate or
drop the caveats in a verified answer."""

# Raw control tokens that must never survive into assistant content. Ollama's
# built-in parser strips them, so their presence means the response was parsed
# wrongly and the text cannot be trusted as an answer (see ollama#14493).
_CONTROL_TOKENS = ("<tool_call>", "</tool_call>", "<think>", "</think>", "<|im_start|>")

_JSON_TYPES = {
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "array": list,
    "object": dict,
}


class ToolValidationError(Exception):
    """A requested tool call is not safe to execute."""


# --------------------------------------------------------------------------- #
# Validation - runs before anything is executed
# --------------------------------------------------------------------------- #
def _type_error(name: str, value: Any, expected: str) -> Optional[str]:
    """Return an error string if ``value`` does not match a JSON-schema type."""
    py_type = _JSON_TYPES.get(expected)
    if py_type is None:  # unknown/unspecified type - nothing to enforce
        return None
    # bool is a subclass of int in Python; a boolean is not a valid number here.
    if expected in ("number", "integer") and isinstance(value, bool):
        return f"{name!r} must be a {expected}, got boolean"
    if not isinstance(value, py_type):
        return f"{name!r} must be a {expected}, got {type(value).__name__}"
    return None


def validate_tool_call(name: str, arguments: Any) -> Dict[str, Any]:
    """Validate a model-proposed tool call against the registry and its schema.

    Returns the normalised arguments dict, or raises
    :class:`ToolValidationError` with a message written to be fed back to the
    model verbatim.

    Enforces, in order: the tool exists in :data:`kg_agent.tools.TOOLS`
    (whitelist - a hallucinated name never reaches ``call_tool``), the arguments
    are an object, no undeclared properties are present, every declared property
    matches its JSON-schema type (including ``array`` item types), and every
    required property is present and non-empty.
    """
    if name not in TOOLS:
        raise ToolValidationError(
            f"Unknown tool {name!r}. Available tools: {', '.join(sorted(TOOLS))}. "
            "Call one of these, or answer directly without a tool."
        )

    # Ollama returns arguments already decoded; other servers send a JSON string.
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError as exc:
            raise ToolValidationError(
                f"Arguments for {name!r} are not valid JSON: {exc}"
            ) from exc
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        raise ToolValidationError(
            f"Arguments for {name!r} must be a JSON object, got "
            f"{type(arguments).__name__}."
        )

    schema = TOOLS[name]["parameters"]
    properties: Dict[str, Any] = schema.get("properties", {})
    required: List[str] = schema.get("required", [])
    errors: List[str] = []

    unknown = sorted(set(arguments) - set(properties))
    if unknown:
        errors.append(
            f"unexpected argument(s) {', '.join(repr(u) for u in unknown)}; "
            f"{name!r} accepts only {', '.join(sorted(properties)) or '(no arguments)'}"
        )

    for key, value in arguments.items():
        spec = properties.get(key)
        if spec is None:
            continue  # already reported as unknown
        err = _type_error(key, value, spec.get("type", ""))
        if err:
            errors.append(err)
            continue
        # Element types matter: ["Alice", 3] would reach Neo4j as mixed data.
        if spec.get("type") == "array" and "items" in spec:
            item_type = spec["items"].get("type", "")
            for i, item in enumerate(value):
                item_err = _type_error(f"{key}[{i}]", item, item_type)
                if item_err:
                    errors.append(item_err)

    for key in required:
        if key not in arguments:
            errors.append(f"missing required argument {key!r}")
        elif isinstance(arguments[key], str) and not arguments[key].strip():
            # Schema-valid but useless: answer_question with query="" would run
            # the whole retrieval pipeline against nothing.
            errors.append(f"required argument {key!r} must not be empty")

    if errors:
        raise ToolValidationError(
            f"Invalid call to {name!r}: " + "; ".join(errors) + ". "
            f"Expected schema: {json.dumps(schema)}"
        )
    return arguments


# --------------------------------------------------------------------------- #
# Minimal tool-calling client
# --------------------------------------------------------------------------- #
def ollama_tool_specs() -> List[Dict[str, Any]]:
    """Registry specs in Ollama's ``{type, function}`` tool shape."""
    return [{"type": "function", "function": spec} for spec in tool_specs()]


class OllamaToolClient:
    """``POST /api/chat`` with native ``tools`` support.

    Deliberately separate from
    :class:`~kg_agent.agentic_verifier.OllamaLLMClient`: that client is a plain
    ``complete(system, user) -> str`` helper with no ``tools`` parameter and no
    ``tool_calls`` parsing, so it cannot drive this loop.

    Uses ``urllib`` rather than the ``ollama`` pip package to match the
    dependency-light style of the rest of the package - the surface we need is
    one POST and one JSON payload, so a new runtime dependency (and a second way
    to configure a URL) would buy nothing here.
    """

    def __init__(self, config: Config) -> None:
        self._url = f"{config.retrieval.ollama_url}/api/chat"
        self._model = resolve_ollama_model(config)
        self._temperature = config.llm.temperature
        self._timeout = config.llm.request_timeout
        self.name = f"ollama-tools:{self._model}"

    @property
    def model(self) -> str:
        """The resolved Ollama model id."""
        return self._model

    def chat(
        self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Send a message list plus tool specs; return the assistant message."""
        payload: Dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self._temperature},
        }
        if tools:
            payload["tools"] = tools

        req = urllib.request.Request(
            self._url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
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
                    f"Ollama model {self._model!r} is not available at {self._url}. "
                    f"Pull it first:  ollama pull {self._model}"
                ) from exc
            raise RuntimeError(f"Ollama request failed (HTTP {exc.code}): {body}") from exc
        except OSError as exc:
            reason = getattr(exc, "reason", None)
            if isinstance(exc, (socket.timeout, TimeoutError)) or isinstance(
                reason, (socket.timeout, TimeoutError)
            ):
                raise RuntimeError(
                    f"Ollama request timed out after {self._timeout}s for model "
                    f"{self._model!r}. Raise KG_LLM_TIMEOUT if the model is slow."
                ) from exc
            raise RuntimeError(
                f"Cannot reach Ollama at {self._url}. Is the server running?"
            ) from exc
        return data.get("message") or {}


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #
@dataclass
class OrchestrationResult:
    """Outcome of one orchestrated request.

    Attributes
    ----------
    query
        The original user request.
    response
        The text surfaced to the caller. When ``answer_question`` was the
        terminal tool this is the *verified* answer (disclaimer included),
        never the model's re-summary of it.
    verified_answer
        The full :class:`VerifiedAnswer` dict - trust_score,
        temporal_validity_status, faithfulness, passed, disclaimer and the rest
        - exactly as the verifier produced it, or ``None`` when the run did not
        end in ``answer_question``. Never synthesised.
    tool_results
        Raw results of every executed tool, in call order.
    tools_used
        Names of the executed tools. Empty means the model answered from its own
        knowledge with no verification behind it.
    trace
        Every step attempted, including rejected calls. Always populated.
    stopped_reason
        Why the loop ended.
    ok
        False when the run ended in a validation failure, a corrupted response
        or a tool error.
    model_narration
        The model's own closing prose. Kept separate from ``response`` so a
        verified answer's wording and caveats are never replaced by it.
    """

    query: str
    response: str
    verified_answer: Optional[Dict[str, Any]] = None
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)
    trace: List[Dict[str, Any]] = field(default_factory=list)
    stopped_reason: str = ""
    ok: bool = True
    model: str = ""
    model_narration: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dictionary of the result."""
        return asdict(self)


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
class NativeToolOrchestrator:
    """Drives the model -> validate -> execute -> feed-back loop.

    The model only ever chooses a tool name and its arguments. Execution,
    validation and verification stay in Python, so no model output can skip the
    Phase 4 gates.
    """

    def __init__(
        self,
        client: Neo4jClient,
        config: Optional[Config] = None,
        llm: Optional[OllamaToolClient] = None,
    ) -> None:
        self.client = client
        self.config = config or get_config()
        self.llm = llm or OllamaToolClient(self.config)
        self.max_tool_calls = self.config.orchestrator.max_tool_calls

    # -- trace helper ------------------------------------------------------- #
    @staticmethod
    def _entry(
        step: int,
        tool_name: Optional[str],
        arguments: Any = None,
        validation_error: Optional[str] = None,
        result_or_error: Any = None,
    ) -> Dict[str, Any]:
        """Build one trace record."""
        return {
            "step": step,
            "tool_name": tool_name,
            "arguments": arguments,
            "validation_error": validation_error,
            "result_or_error": result_or_error,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def run(self, query: str) -> OrchestrationResult:
        """Answer ``query`` by letting the model pick tools, and execute them."""
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]
        result = OrchestrationResult(query=query, response="", model=self.llm.model)

        step = 0
        executed = 0
        validation_failures = 0
        last_content = ""
        # One turn per executed call, plus the closing turn and a margin for the
        # single permitted re-prompt. Bounds the loop independently of the model.
        max_turns = self.max_tool_calls + 3

        for _ in range(max_turns):
            step += 1
            try:
                message = self.llm.chat(messages, ollama_tool_specs())
            except RuntimeError as exc:
                result.trace.append(self._entry(step, None, result_or_error=str(exc)))
                result.ok = False
                result.stopped_reason = f"llm_error: {exc}"
                result.response = last_content
                return result

            content = (message.get("content") or "").strip()
            tool_calls = message.get("tool_calls") or []

            # Corruption guard: raw control tokens mean the response was parsed
            # wrongly, so the text is not trustworthy as an answer.
            leaked = [tok for tok in _CONTROL_TOKENS if tok in content]
            if leaked:
                result.trace.append(
                    self._entry(
                        step,
                        None,
                        result_or_error=f"control tokens in content: {leaked}",
                    )
                )
                result.ok = False
                result.stopped_reason = (
                    f"corrupted_response: assistant content leaked {leaked}"
                )
                return result

            # Content and tool_calls can both be non-empty; keep the prose either
            # way rather than assuming one excludes the other. Recorded as we go
            # so an early return still reports what the model actually said.
            if content:
                last_content = content
                result.model_narration = content

            if not tool_calls:
                if not content:
                    result.trace.append(
                        self._entry(step, None, result_or_error="empty response")
                    )
                    result.ok = False
                    result.stopped_reason = "model returned neither tool calls nor content"
                    result.response = ""
                    return result
                result.stopped_reason = "model returned a final answer"
                break

            # Echo the assistant turn back. `thinking` is deliberately dropped:
            # Qwen3 guidance is not to replay reasoning into later turns.
            assistant_msg: Dict[str, Any] = {"role": "assistant", "content": content}
            assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)

            stop = False
            for call in tool_calls:
                function = call.get("function") or {}
                name = function.get("name") or ""
                raw_args = function.get("arguments")

                if executed >= self.max_tool_calls:
                    result.trace.append(
                        self._entry(
                            step,
                            name,
                            raw_args,
                            result_or_error=(
                                f"skipped: tool-call cap of {self.max_tool_calls} reached"
                            ),
                        )
                    )
                    result.ok = False
                    result.stopped_reason = (
                        f"tool_call_cap_reached: {self.max_tool_calls}"
                    )
                    stop = True
                    break

                try:
                    arguments = validate_tool_call(name, raw_args)
                except ToolValidationError as exc:
                    validation_failures += 1
                    result.trace.append(
                        self._entry(step, name, raw_args, validation_error=str(exc))
                    )
                    logger.warning("Rejected tool call %r: %s", name, exc)
                    # Re-prompt once with the error, then give up.
                    if validation_failures > 1:
                        result.ok = False
                        result.stopped_reason = (
                            "validation_failed_twice: the model could not produce a "
                            f"valid tool call ({exc})"
                        )
                        stop = True
                        break
                    messages.append(
                        {
                            "role": "tool",
                            "tool_name": name or "unknown",
                            "content": json.dumps({"error": str(exc)}, ensure_ascii=False),
                        }
                    )
                    continue

                try:
                    tool_result = call_tool(name, self.client, self.config, arguments)
                except Exception as exc:  # tool failures must not kill the loop
                    logger.exception("Tool %r raised", name)
                    result.trace.append(
                        self._entry(
                            step,
                            name,
                            arguments,
                            result_or_error=f"{type(exc).__name__}: {exc}",
                        )
                    )
                    result.ok = False
                    result.stopped_reason = f"tool_error in {name!r}: {exc}"
                    stop = True
                    break

                executed += 1
                result.tools_used.append(name)
                result.tool_results.append({"tool": name, "result": tool_result})
                result.trace.append(
                    self._entry(step, name, arguments, result_or_error=tool_result)
                )
                # The verifier's own output is attached as-is; the orchestrator
                # never edits, rescales or re-summarises the gate fields.
                if name == "answer_question":
                    result.verified_answer = tool_result

                messages.append(
                    {
                        "role": "tool",
                        "tool_name": name,
                        "content": json.dumps(tool_result, ensure_ascii=False, default=str),
                    }
                )

            if stop:
                break
        else:
            result.stopped_reason = result.stopped_reason or "max_turns_reached"
            result.ok = False

        # A verified answer wins over the model's paraphrase of it, so the
        # gated wording - including any disclaimer - is what the caller sees.
        if result.verified_answer and result.verified_answer.get("answer"):
            result.response = result.verified_answer["answer"]
        else:
            result.response = last_content

        if not result.tools_used and result.ok:
            logger.warning(
                "Orchestrator answered %r without calling any tool - the response "
                "carries no verification.",
                query,
            )
        return result


def run_orchestrated(
    client: Neo4jClient, config: Config, query: str
) -> OrchestrationResult:
    """Convenience wrapper: build an orchestrator and run one query."""
    return NativeToolOrchestrator(client, config).run(query)


def main() -> None:  # pragma: no cover - manual entry point
    """Run one orchestrated query from the command line."""
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Native Ollama tool-calling orchestrator.")
    parser.add_argument("--query", required=True, help="Question or instruction.")
    args = parser.parse_args()

    cfg = get_config()
    with Neo4jClient.from_config(cfg) as client:
        result = run_orchestrated(client, cfg, args.query)
    print(json.dumps(result.to_dict(), indent=2, default=str))


if __name__ == "__main__":  # pragma: no cover
    main()
