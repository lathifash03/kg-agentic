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
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from kg_agent.agentic_verifier import AgenticVerifier
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

    linked: List[str] = []
    for name in entities:
        client.run_write(
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
        linked.append(name)

    logger.info("Meeting %r ingested, %d entities linked.", title, len(linked))
    return {
        "meeting": {"title": title, "date": meeting_date, "participants": participants},
        "entities_linked": linked,
        "counters": result,
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


TOOLS: Dict[str, Dict[str, Any]] = {
    "answer_question": {
        "fn": answer_question,
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
        "description": "Statistik ringkas knowledge graph.",
        "parameters": {"type": "object", "properties": {}},
    },
}


def tool_specs() -> List[Dict[str, Any]]:
    """Spec semua tool tanpa fungsi Python-nya (aman untuk JSON)."""
    return [
        {"name": name, "description": t["description"], "parameters": t["parameters"]}
        for name, t in TOOLS.items()
    ]


def call_tool(
    name: str, client: Neo4jClient, cfg: Config, arguments: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Dispatch pemanggilan tool berdasarkan nama."""
    if name not in TOOLS:
        raise KeyError(f"Unknown tool: {name!r}. Available: {sorted(TOOLS)}")
    return TOOLS[name]["fn"](client, cfg, **(arguments or {}))
