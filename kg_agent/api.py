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
POST /query           -> shortcut untuk tool answer_question; saat
                         KG_ORCHESTRATOR=native, LLM yang memilih tool dan
                         respons memuat ``tool_trace``. ``{"agentic": true}``
                         hanya boleh dipakai bila server sudah menyalakannya —
                         kalau tidak, 403.
                         Dua opsi tambahan, keduanya default MATI: body
                         ``{"allow_ungrounded": true}`` mengisi
                         ``ungrounded_answer`` (jawaban tanpa sumber, di field
                         terpisah — ``answer``/``passed``/
                         ``overall_confidence``/``sources_used`` tidak ikut
                         berubah), dan query param ``?spoken=true`` mengisi
                         ``answer_spoken`` (ringkasan untuk TTS; satu
                         panggilan LLM ekstra).
POST /setup           -> Phase 1 migration + Phase 3 trust scoring (idempotent)

Untuk menunjuk KG milik teman: cukup ganti NEO4J_URI / NEO4J_USERNAME /
NEO4J_PASSWORD dan (bila skema grafnya beda) variabel KG_ENTITY_LABEL,
KG_ENTITY_NAME_PROP, dst. di environment — tidak perlu mengubah kode.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from kg_agent.config import get_config
from kg_agent.neo4j_client import Neo4jClient
from kg_agent.node_trust import compute_and_store, summarise_scores
from kg_agent.orchestrator import run_orchestrated
from kg_agent.tools import call_tool, tool_specs, tool_writes

logger = logging.getLogger(__name__)


def _refuse_if_read_only(what: str) -> None:
    """Raise 403 when writes are disabled and ``what`` would write.

    Only needed for write paths that do NOT go through ``call_tool`` (i.e.
    ``/setup``); tool invocations are guarded at the dispatcher itself and
    surface here as :class:`PermissionError`.
    """
    if app.state.cfg.safety.read_only:
        raise HTTPException(
            status_code=403,
            detail=(
                f"Graph dalam mode read-only, {what} ditolak. "
                "Set KG_READ_ONLY=false untuk mengizinkan tulis - "
                "pastikan dulu ada izin tulis ke graph tujuan."
            ),
        )


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
    allow_ungrounded: bool = Field(
        default=False,
        description="Bila true DAN retrieval kosong, respons memuat "
        "`ungrounded_answer`: jawaban dari pengetahuan umum model, TANPA "
        "sumber KG. Disimpan di field sendiri — `answer` tetap penolakan, "
        "`passed` tetap false, `overall_confidence` tetap 0.0, `sources_used` "
        "tetap kosong. Default false: jawaban tanpa sumber harus diminta.",
    )
    agentic: bool = Field(
        default=False,
        description="Bila true, LLM yang memilih tool lewat orchestrator "
        "(Phase 5) dan respons memuat `tool_trace`. Gate verifikasi tetap sama. "
        "Ditolak 403 bila server menjalankan KG_ORCHESTRATOR=off.",
    )


class ToolCallRequest(BaseModel):
    arguments: Optional[Dict[str, Any]] = Field(default=None)


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/health")
def health() -> Dict[str, Any]:
    ok = app.state.client.verify_connectivity()
    return {
        "status": "ok" if ok else "degraded",
        "neo4j_connected": ok,
        # Surfaced so a caller can tell a refusal-by-policy from a real failure
        # before sending anything.
        "read_only": app.state.cfg.safety.read_only,
    }


@app.get("/tools")
def list_tools() -> Dict[str, Any]:
    # `writes` is annotated here rather than inside tool_specs() so the schema
    # sent to the LLM stays a clean function definition.
    tools = [{**spec, "writes": tool_writes(spec["name"])} for spec in tool_specs()]
    return {"tools": tools, "read_only": app.state.cfg.safety.read_only}


def _caller(request: Request) -> str:
    """Best-effort caller identity for the audit trail.

    Prefers the forwarded client address when a reverse proxy is in front,
    since otherwise every request would be attributed to the proxy itself.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


@app.post("/tools/{name}")
def invoke_tool(name: str, body: ToolCallRequest, request: Request) -> Dict[str, Any]:
    try:
        result = call_tool(
            name, app.state.client, app.state.cfg, body.arguments,
            caller=_caller(request),
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:  # write tool while KG_READ_ONLY is on
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except TypeError as exc:  # argumen tidak cocok dengan signature tool
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"tool": name, "result": result}


@app.post("/query")
def query(
    body: QueryRequest, request: Request, spoken: bool = False
) -> Dict[str, Any]:
    cfg = app.state.cfg
    # `agentic` may only narrow what the server already allows, never widen it.
    # KG_ORCHESTRATOR=off is an operator decision (the orchestrator can reach
    # every registry tool), so a request body must not be able to switch it back
    # on - otherwise the server's own setting is advisory.
    if body.agentic and not cfg.orchestrator.enabled:
        raise HTTPException(
            status_code=403,
            detail=(
                "Orchestrator dimatikan di server (KG_ORCHESTRATOR=off); "
                "`agentic: true` tidak bisa menyalakannya per-request. "
                "Kirim tanpa flag itu untuk jalur terverifikasi biasa."
            ),
        )
    if not cfg.orchestrator.enabled:
        return call_tool(
            "answer_question", app.state.client, cfg,
            {
                "query": body.query,
                "allow_ungrounded": body.allow_ungrounded,
                "spoken": spoken,
            },
            caller=_caller(request),
        )

    # Jalur orchestrator: argumen tool dipilih LLM, jadi kedua opsi ini tidak
    # bisa diteruskan. Menerimanya diam-diam berarti membalas 200 tanpa field
    # yang diminta - caller menyimpulkan "tidak ada jawaban ungrounded", padahal
    # flag-nya memang tak pernah sampai ke verifier. Tolak terang-terangan.
    if body.allow_ungrounded or spoken:
        raise HTTPException(
            status_code=400,
            detail=(
                "`allow_ungrounded` dan `spoken` hanya berlaku di jalur "
                "terverifikasi langsung; server ini menjalankan "
                "KG_ORCHESTRATOR=native, di mana argumen tool dipilih LLM. "
                "Pakai POST /tools/answer_question untuk memakai keduanya."
            ),
        )

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

    # No verified answer means the gates never ran. Returning that as 200 would
    # hand the caller an ungated completion in a *different* response shape than
    # the verified path - a client reading `answer` sees the field simply
    # missing, with nothing in the status code to signal it. Fail loudly instead,
    # so /query has exactly one successful shape.
    logger.warning(
        "Orchestrated /query produced no verified answer (%s)", result.stopped_reason
    )
    raise HTTPException(
        status_code=502,
        detail={
            "error": "no_verified_answer",
            "query": result.query,
            "response": result.response,
            "tool_results": result.tool_results,
            **common,
        },
    )


@app.post("/setup")
def setup() -> Dict[str, Any]:
    """Phase 1 (temporal metadata) + Phase 3 (trust scoring). Idempotent.

    Menulis ke SETIAP node entity, jadi ditolak saat API read-only.
    """
    _refuse_if_read_only("/setup menjalankan migrasi yang menulis ke seluruh node")
    if not app.state.client.verify_connectivity():
        raise HTTPException(status_code=503, detail="Neo4j tidak terhubung.")
    migration = app.state.client.run_phase1_migration()
    scores = compute_and_store(app.state.client, app.state.cfg)
    return {"migration": migration, "trust_scoring": summarise_scores(scores)}
