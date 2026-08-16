"""Agent tools registry.

Setiap tool dideskripsikan dengan spec ala function-calling (name, description,
parameters JSON-schema) supaya nanti mudah dipasang ke LLM function-calling,
MCP server, atau dipanggil lewat endpoint ``POST /tools/{name}``.

Tools yang tersedia:

- ``answer_question``  : jawab pertanyaan lewat AgenticVerifier (retrieval + gates).
- ``ingest_meeting``   : masukkan hasil meeting ke KG sebagai node ``Meeting``
                         dengan ``source_type='meeting'`` (bobot trust 0.8) dan
                         hubungkan ke entitas yang disebut.
- ``kg_stats``         : statistik ringkas graph (jumlah entitas, meeting, dst).

Menambah tool baru: tulis fungsi, lalu daftarkan di ``TOOLS`` di bawah.
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from kg_agent.agentic_verifier import AgenticVerifier
from kg_agent.audit_log import record_tool_call, should_audit
from kg_agent.config import Config
from kg_agent.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Tool implementations
# --------------------------------------------------------------------------- #
def answer_question(client: Neo4jClient, cfg: Config, *, query: str) -> Dict[str, Any]:
    """Jawab pertanyaan terhadap KG dengan verifikasi penuh (Phase 4)."""
    verifier = AgenticVerifier(client, cfg)
    return verifier.verify(query).to_dict()


def ingest_meeting(
    client: Neo4jClient,
    cfg: Config,
    *,
    title: str,
    date: Optional[str] = None,
    participants: Optional[List[str]] = None,
    notes: str = "",
    entities: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Simpan sebuah meeting ke KG dan tautkan ke entitas yang disebut.

    Parameters
    ----------
    title:
        Judul meeting (dipakai sebagai id unik bersama ``date``).
    date:
        ISO date/datetime meeting; default = sekarang (UTC).
    participants:
        Daftar nama peserta.
    notes:
        Ringkasan / catatan meeting.
    entities:
        Nama entitas KG yang dibahas. Entitas yang belum ada akan dibuat
        (MERGE) dengan ``source_type='meeting'`` sehingga langsung mendapat
        bobot trust "meeting" saat scoring Phase 3 dijalankan ulang.
    """
    now = datetime.now(timezone.utc).isoformat()
    meeting_date = date or now
    participants = participants or []
    entities = entities or []

    label = cfg.schema.entity_label
    name_prop = cfg.schema.entity_name_property

    result = client.run_write(
        """
        MERGE (m:Meeting {title: $title, date: $date})
        ON CREATE SET m.created_at = $now, m.created_by = 'kg_agent.tools.ingest_meeting'
        SET m.participants = $participants,
            m.notes = $notes,
            m.updated_at = $now,
            m.source_type = 'meeting'
        RETURN elementId(m) AS meeting_id
        """,
        title=title,
        date=meeting_date,
        participants=participants,
        notes=notes,
        now=now,
    )

    # Counters from the meeting write only cover the Meeting node. Each linked
    # entity is a separate write, so totals must be accumulated - otherwise the
    # audit trail reports 1 node created for a meeting that created twenty.
    totals = {
        "nodes_created": result.get("nodes_created", 0),
        "relationships_created": result.get("relationships_created", 0),
        "properties_set": result.get("properties_set", 0),
    }
    meeting_id = (result.get("records") or [{}])[0].get("meeting_id")

    linked: List[str] = []
    for name in entities:
        entity_result = client.run_write(
            f"""
            MATCH (m:Meeting {{title: $title, date: $date}})
            MERGE (e:`{label}` {{{name_prop}: $name}})
            ON CREATE SET e.created_at = $now,
                          e.created_by = 'kg_agent.tools.ingest_meeting',
                          e.source_type = 'meeting',
                          e.confidence_score = $confidence
            SET e.updated_at = $now
            MERGE (e)-[r:DISCUSSED_IN]->(m)
            ON CREATE SET r.created_at = $now
            """,
            title=title,
            date=meeting_date,
            name=name,
            now=now,
            confidence=cfg.temporal_defaults.default_confidence_score,
        )
        for key in totals:
            totals[key] += entity_result.get(key, 0)
        linked.append(name)

    logger.info("Meeting %r ingested, %d entities linked.", title, len(linked))
    return {
        "meeting": {"title": title, "date": meeting_date, "participants": participants},
        "meeting_id": meeting_id,
        "entities_linked": linked,
        "counters": totals,
        "note": "Jalankan ulang trust scoring (POST /setup atau --setup) agar "
        "bobot source_type='meeting' diperhitungkan.",
    }


def kg_stats(client: Neo4jClient, cfg: Config) -> Dict[str, Any]:
    """Statistik ringkas isi knowledge graph."""
    label = cfg.schema.entity_label
    rows = client.run_read(
        f"""
        CALL () {{
            MATCH (e:`{label}`) RETURN count(e) AS entities
        }}
        CALL () {{
            MATCH (m:Meeting) RETURN count(m) AS meetings
        }}
        CALL () {{
            MATCH ()-[r]->() RETURN count(r) AS relationships
        }}
        RETURN entities, meetings, relationships
        """
    )
    return rows[0] if rows else {"entities": 0, "meetings": 0, "relationships": 0}


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
ToolFn = Callable[..., Dict[str, Any]]


# Setiap entri WAJIB punya ``writes``: True kalau tool-nya menyentuh graph
# dengan operasi tulis. Dipakai API untuk menolak tool tulis saat read-only,
# jadi tool baru yang lupa menandainya akan gagal keras di ``tool_writes``
# ketimbang diam-diam lolos sebagai read-only.
TOOLS: Dict[str, Dict[str, Any]] = {
    "answer_question": {
        "fn": answer_question,
        "writes": False,
        "description": "Jawab pertanyaan terhadap knowledge graph dengan "
        "verifikasi trust/temporal/faithfulness.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Pertanyaan."}},
            "required": ["query"],
        },
    },
    "ingest_meeting": {
        "fn": ingest_meeting,
        "writes": True,
        "description": "Simpan hasil meeting (judul, tanggal, peserta, catatan, "
        "entitas yang dibahas) ke knowledge graph.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "date": {"type": "string", "description": "ISO date, opsional."},
                "participants": {"type": "array", "items": {"type": "string"}},
                "notes": {"type": "string"},
                "entities": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["title"],
        },
    },
    "kg_stats": {
        "fn": kg_stats,
        "writes": False,
        "description": "Statistik ringkas knowledge graph.",
        "parameters": {"type": "object", "properties": {}},
    },
}


def tool_specs() -> List[Dict[str, Any]]:
    """Spec semua tool tanpa fungsi Python-nya (aman untuk JSON).

    Sengaja TIDAK memuat ``writes``: hasil fungsi ini dikirim apa adanya
    sebagai schema ``function`` ke Ollama (lihat ``ollama_tool_specs``), jadi
    field non-standar tidak boleh bocor ke sana. Konsumen yang butuh tahu
    sifat tulis sebuah tool memanggil :func:`tool_writes`.
    """
    return [
        {"name": name, "description": t["description"], "parameters": t["parameters"]}
        for name, t in TOOLS.items()
    ]


def tool_writes(name: str) -> bool:
    """True kalau tool ``name`` menulis ke graph.

    Raises
    ------
    KeyError
        Kalau tool tidak dikenal, atau terdaftar tanpa flag ``writes`` - lebih
        baik gagal keras daripada menganggap tool tak bertanda sebagai aman.
    """
    if name not in TOOLS:
        raise KeyError(f"Unknown tool: {name!r}. Available: {sorted(TOOLS)}")
    if "writes" not in TOOLS[name]:
        raise KeyError(f"Tool {name!r} tidak menyatakan flag 'writes'.")
    return bool(TOOLS[name]["writes"])


def call_tool(
    name: str,
    client: Neo4jClient,
    cfg: Config,
    arguments: Optional[Dict[str, Any]] = None,
    *,
    caller: Optional[str] = None,
) -> Dict[str, Any]:
    """Dispatch pemanggilan tool berdasarkan nama.

    Titik sempit tunggal untuk semua eksekusi tool - API, orchestrator lewat
    API, maupun orchestrator lewat CLI. Guard read-only DAN audit trail
    ditegakkan di sini, bukan di lapisan HTTP, supaya route baru atau pemanggil
    baru tidak bisa melewatkan keduanya karena lupa.

Tool yang menulis DAN tool tanya-jawab dicatat satu baris JSON per
    panggilan - termasuk yang DITOLAK - ke ``logs/<nama-tool>.jsonl``.
    Percobaan yang ditolak justru yang paling perlu terekam: itu satu-satunya
    jejak bahwa ada yang mencoba menulis ke graph. Isi payload tulis tidak
    pernah tercatat, hanya ukuran dan hitungan.

    Parameters
    ----------
    caller
        Identitas pemanggil (IP untuk request HTTP). ``None`` untuk pemakaian
        lokal lewat CLI.

    Raises
    ------
    KeyError
        Tool tidak dikenal.
    PermissionError
        Tool menulis sementara ``KG_READ_ONLY`` aktif.
    """
    if name not in TOOLS:
        raise KeyError(f"Unknown tool: {name!r}. Available: {sorted(TOOLS)}")

    writes = tool_writes(name)
    audit = cfg.audit.enabled and should_audit(
        name, writes, log_queries=cfg.audit.log_queries
    )
    request_id = str(uuid.uuid4()) if audit else ""
    started = time.perf_counter()

    def _audit(status: str, result: Any = None, error: Optional[str] = None) -> None:
        if not audit:
            return
        record_tool_call(
            tool=name,
            request_id=request_id,
            status=status,
            duration_ms=(time.perf_counter() - started) * 1000,
            arguments=arguments,
            result=result,
            error=error,
            caller=caller,
            directory=cfg.audit.directory,
            retention_days=cfg.audit.retention_days,
            include_query_text=cfg.audit.log_query_text,
        )

    if writes and cfg.safety.read_only:
        message = (
            f"Tool {name!r} menulis ke graph, sedangkan KG_READ_ONLY aktif. "
            "Set KG_READ_ONLY=false untuk mengizinkan - pastikan dulu ada izin "
            "tulis ke graph tujuan."
        )
        _audit("rejected_403", error=message)
        raise PermissionError(message)

    try:
        result = TOOLS[name]["fn"](client, cfg, **(arguments or {}))
    except Exception as exc:
        _audit("failed", error=f"{type(exc).__name__}: {exc}")
        raise
    _audit("success", result=result)
    return result
