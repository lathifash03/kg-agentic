"""Operational audit trail: one JSON line per write-tool call.

Separate from the ordinary ``logging`` output, which is debug material and gets
truncated, reformatted and thrown away. This is an operational record - who
asked to change the graph, when, how long it took, and whether it was allowed -
so it is written as machine-readable JSONL and kept for a retention window
measured in weeks rather than until the next restart.

Two kinds of call are audited, each to its own file: tools that WRITE to the
graph, and question answering. Everything else (plain statistics lookups) is
skipped - it changes nothing and answers nothing worth reconstructing later.

The two differ in what is worth recording, so each contributes its own fields
on top of the shared ones: a write reports how many nodes it created, a
question reports the gate outcome, retrieval strategy and which documents were
cited. Sharing one module keeps rotation, failure handling and line format
identical across both.

**Write payloads are never recorded.** Meeting transcripts are the whole point
of the write path and the one thing that must not leak into a log file that is
read over SSH and kept for a month. Only sizes and counts are stored, and there
is no passthrough of arbitrary payload keys that could carry text in later.

Question text is the deliberate exception: it is the request itself rather than
confidential source material, and without it the log cannot answer "why did
this query return nothing", which is the main reason to keep it. Set
``KG_AUDIT_QUERY_TEXT=false`` to drop it. Generated answers are never stored -
only their length, since the text is reproducible by re-running the query.

Files land one per tool, so ``ingest_meeting`` writes ``ingest_meeting.jsonl``.
Rotation is daily with ``retention_days`` files kept.

Failures here never propagate: an audit sink that takes the API down with it
when a disk fills is worse than a missing line.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import pathlib
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# One dedicated logger per tool, built once. Guarded because FastAPI serves
# requests from a thread pool and two concurrent first-calls would otherwise
# attach two handlers to the same file.
_LOCK = threading.Lock()
_LOGGERS: Dict[str, logging.Logger] = {}

# Keeps a stray newline or a huge exception string from breaking one-line-per-
# call parsing downstream.
_MAX_ERROR_CHARS = 300


def _sink(tool: str, directory: str, retention_days: int) -> Optional[logging.Logger]:
    """Return the JSONL logger for ``tool``, creating it on first use."""
    key = f"{directory}::{tool}"
    existing = _LOGGERS.get(key)
    if existing is not None:
        return existing
    with _LOCK:
        existing = _LOGGERS.get(key)
        if existing is not None:
            return existing
        try:
            path = pathlib.Path(directory).expanduser()
            path.mkdir(parents=True, exist_ok=True)
            handler = logging.handlers.TimedRotatingFileHandler(
                path / f"{tool}.jsonl",
                when="midnight",
                backupCount=max(1, retention_days),
                encoding="utf-8",
                utc=True,
            )
            # The record IS the line - no level, no timestamp prefix, because
            # the JSON already carries its own and a prefix would break parsing.
            handler.setFormatter(logging.Formatter("%(message)s"))
            sink = logging.getLogger(f"kg_agent.audit.{tool}")
            sink.setLevel(logging.INFO)
            sink.propagate = False  # never duplicated into the console logs
            sink.handlers = [handler]
            _LOGGERS[key] = sink
            return sink
        except Exception as exc:  # pragma: no cover - disk/permission dependent
            logger.warning("Audit log unavailable for %r: %s", tool, exc)
            return None


def payload_size_chars(arguments: Optional[Dict[str, Any]]) -> int:
    """Total characters across string values in the payload.

    Measures size, never content. For ``ingest_meeting`` this is dominated by
    the transcript in ``notes``; short scalars like the title contribute a
    negligible amount. Strings nested one level inside lists (participants,
    entity names) are counted too.
    """
    total = 0
    for value in (arguments or {}).values():
        if isinstance(value, str):
            total += len(value)
        elif isinstance(value, (list, tuple)):
            total += sum(len(v) for v in value if isinstance(v, str))
    return total


def _summarise_write(result: Any) -> Dict[str, Any]:
    """Mutation counts from a write tool, tolerating any result shape."""
    counters = result.get("counters") if isinstance(result, dict) else None
    if not isinstance(counters, dict):
        return {"nodes_created": None, "relationships_created": None}
    return {
        "nodes_created": counters.get("nodes_created"),
        "relationships_created": counters.get("relationships_created"),
    }


def _summarise_answer(result: Any) -> Dict[str, Any]:
    """Gate outcome and retrieval shape from a verified answer.

    These are the fields that make production behaviour answerable after the
    fact: how often retrieval retries, which papers actually get cited, how
    faithfulness is distributed, and whether an empty result came from a query
    that matched nothing or from a graph that holds nothing.
    """
    if not isinstance(result, dict):
        return {}
    documents = result.get("documents_used") or []
    sources = result.get("sources_used") or []
    return {
        "passed": result.get("passed"),
        "faithfulness": result.get("faithfulness"),
        "trust_score": result.get("trust_score"),
        "overall_confidence": result.get("overall_confidence"),
        "temporal_validity_status": result.get("temporal_validity_status"),
        "strategy": result.get("strategy"),
        "retries": result.get("retries"),
        "n_sources": len(sources),
        "n_documents": len(documents),
        "documents": [d.get("name") for d in documents if isinstance(d, dict)],
        "answer_chars": len(result.get("answer") or ""),
    }


# A tool is audited when it writes, or when it appears here. `kg_stats` has
# neither, so it stays out of the log rather than adding noise.
_SUMMARISERS = {
    "ingest_meeting": _summarise_write,
    "answer_question": _summarise_answer,
}


def should_audit(tool: str, writes: bool, *, log_queries: bool = True) -> bool:
    """True when calls to ``tool`` belong in the audit trail."""
    if writes:
        return True
    if tool == "answer_question":
        return log_queries
    return tool in _SUMMARISERS


def record_tool_call(
    *,
    tool: str,
    request_id: str,
    status: str,
    duration_ms: float,
    arguments: Optional[Dict[str, Any]],
    result: Any = None,
    error: Optional[str] = None,
    caller: Optional[str] = None,
    directory: str = "logs",
    retention_days: int = 30,
    include_query_text: bool = True,
) -> Optional[Dict[str, Any]]:
    """Append one audit line. Returns the entry written, or ``None``.

    ``status`` is ``success``, ``failed`` or ``rejected_403``. Never raises.
    """
    try:
        args = arguments or {}
        meeting_id = args.get("meeting_id")
        if meeting_id is None and isinstance(result, dict):
            meeting_id = result.get("meeting_id")
        entry = {
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tool": tool,
            "caller": caller or "local",
            "status": status,
            "duration_ms": round(duration_ms, 1),
            "input_size_chars": payload_size_chars(args),
        }
        if meeting_id is not None:
            entry["meeting_id"] = meeting_id
        elif tool == "ingest_meeting":
            entry["meeting_id"] = None
        # The question itself, opt-out. Never the generated answer.
        if tool == "answer_question" and include_query_text:
            entry["query"] = args.get("query")
        summarise = _SUMMARISERS.get(tool)
        if summarise is not None:
            entry.update(summarise(result))
        entry["error"] = error[:_MAX_ERROR_CHARS] if error else None
        sink = _sink(tool, directory, retention_days)
        if sink is not None:
            sink.info(json.dumps(entry, ensure_ascii=False, default=str))
        return entry
    except Exception as exc:  # pragma: no cover - must never break a request
        logger.warning("Failed to write audit line for %r: %s", tool, exc)
        return None
