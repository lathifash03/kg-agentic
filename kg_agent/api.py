"""FastAPI service — endpoint supaya agent ini bisa dipakai dari luar
(termasuk digabungkan ke knowledge graph milik tim lain).

Jalankan lokal::

    uvicorn kg_agent.api:app --reload

Atau lewat Docker (lihat docker-compose.yml).

Endpoint
--------
GET  /health          -> status koneksi Neo4j
GET  /tools           -> daftar tool + JSON schema (untuk function-calling/MCP)
POST /tools/{name}    -> panggil tool dengan arguments JSON
POST /query           -> shortcut untuk tool answer_question; dengan
                         ``{"agentic": true}`` (atau KG_ORCHESTRATOR=native)
                         LLM yang memilih tool dan respons memuat ``tool_trace``
POST /setup           -> Phase 1 migration + Phase 3 trust scoring (idempotent)

Untuk menunjuk KG milik teman: cukup ganti NEO4J_URI / NEO4J_USERNAME /
NEO4J_PASSWORD dan (bila skema grafnya beda) variabel KG_ENTITY_LABEL,
KG_ENTITY_NAME_PROP, dst. di environment — tidak perlu mengubah kode.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from kg_agent.config import get_config
from kg_agent.neo4j_client import Neo4jClient
from kg_agent.node_trust import compute_and_store, summarise_scores
from kg_agent.orchestrator import run_orchestrated
from kg_agent.tools import call_tool, tool_specs

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# App lifecycle: satu koneksi Neo4j dipakai bersama
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = get_config()
    client = Neo4jClient.from_config(cfg)
    app.state.cfg = cfg
    app.state.client = client
    try:
        yield
    finally:
        client.close()


app = FastAPI(
    title="kg-agent",
    description="Agentic verification layer di atas knowledge graph Neo4j.",
    version="0.1.0",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Pertanyaan untuk KG.")
    agentic: bool = Field(
        default=False,
        description="Bila true, LLM yang memilih tool lewat orchestrator "
        "(Phase 5) dan respons memuat `tool_trace`. Gate verifikasi tetap sama.",
    )


class ToolCallRequest(BaseModel):
    arguments: Optional[Dict[str, Any]] = Field(default=None)


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/health")
def health() -> Dict[str, Any]:
    ok = app.state.client.verify_connectivity()
    return {"status": "ok" if ok else "degraded", "neo4j_connected": ok}


@app.get("/tools")
def list_tools() -> Dict[str, Any]:
    return {"tools": tool_specs()}


@app.post("/tools/{name}")
def invoke_tool(name: str, body: ToolCallRequest) -> Dict[str, Any]:
    try:
        result = call_tool(name, app.state.client, app.state.cfg, body.arguments)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except TypeError as exc:  # argumen tidak cocok dengan signature tool
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"tool": name, "result": result}


@app.post("/query")
def query(body: QueryRequest) -> Dict[str, Any]:
    cfg = app.state.cfg
    if not (body.agentic or cfg.orchestrator.enabled):
        return call_tool("answer_question", app.state.client, cfg, {"query": body.query})

    try:
        result = run_orchestrated(app.state.client, cfg, body.query)
    except RuntimeError as exc:  # unreachable Ollama / KG_LLM_MODEL not set
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    common = {
        "tool_trace": result.trace,
        "tools_used": result.tools_used,
        "stopped_reason": result.stopped_reason,
        "ok": result.ok,
        "model": result.model,
    }
    # When the run ended in answer_question, keep the response shape identical to
    # the non-agentic path (the full VerifiedAnswer) and just add the trace.
    if result.verified_answer:
        return {**result.verified_answer, **common}
    return {
        "query": result.query,
        "response": result.response,
        "tool_results": result.tool_results,
        **common,
    }


@app.post("/setup")
def setup() -> Dict[str, Any]:
    """Phase 1 (temporal metadata) + Phase 3 (trust scoring). Idempotent."""
    if not app.state.client.verify_connectivity():
        raise HTTPException(status_code=503, detail="Neo4j tidak terhubung.")
    migration = app.state.client.run_phase1_migration()
    scores = compute_and_store(app.state.client, app.state.cfg)
    return {"migration": migration, "trust_scoring": summarise_scores(scores)}
