# kg-agent

Agentic verification layer di atas knowledge graph Neo4j — versi mandiri dari
modul `kg_agentic` yang awalnya dikembangkan di atas repo
[knowledge-graph (DylanTartarini1996)](https://github.com/DylanTartarini1996/knowledge-graphs).
Tanpa Streamlit: dipakai lewat terminal (CLI) atau HTTP API, dan siap Docker.

## Apa yang dilakukan agent ini

Agent menjawab pertanyaan terhadap sebuah knowledge graph yang **sudah ada**
di Neo4j, dengan empat lapis verifikasi:

1. **Temporal metadata** — stempel `created_at`, `source_type`, `confidence_score` pada node/relasi (idempotent).
2. **Temporal validity** — deteksi node VALID / OUTDATED / SUPERSEDED / CONFLICTED.
3. **Node trust scoring** — `trust = confidence × bobot_sumber × recency` (paper 1.0, meeting 0.8, discussion 0.6, auto_extracted 0.4).
4. **Agentic verifier** — retrieval (vector → keyword → expanded), generate jawaban, cek faithfulness, retry bila gagal melewati gate.

> Catatan penting: repo ini **tidak** berisi pipeline ingestion dokumen→graph.
> Ia mengasumsikan KG sudah terisi (mis. oleh pipeline temanmu). Untuk memakai
> KG lain, cukup atur environment — lihat bagian *Menghubungkan ke KG lain*.

## Struktur

```
kg_agent/
  config.py             # semua threshold & koneksi, override via env
  neo4j_client.py       # koneksi + migrasi Phase 1
  temporal_validity.py  # Phase 2
  node_trust.py         # Phase 3
  agentic_verifier.py   # Phase 4 (retrieval, LLM, gates)
  evaluation.py         # evaluasi offline (RAGAS, opsional)
  tools.py              # registry tools agent (answer_question, ingest_meeting, kg_stats)
  cli.py                # testing dari terminal
  api.py                # FastAPI endpoint
```

## Setup lokal

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # lalu isi kredensial Neo4j + GROQ_API_KEY
```

## Testing di terminal

```bash
# pertama kali: migrasi metadata + trust scoring, lalu jawab query default
python -m kg_agent.cli --setup

# query sendiri
python -m kg_agent.cli --query "Apa itu RMFS?"

# output JSON
python -m kg_agent.cli --query "..." --json
```

Tanpa Neo4j/LLM sungguhan, set `KG_LLM_PROVIDER=mock` untuk menguji alurnya.

## API

```bash
uvicorn kg_agent.api:app --reload
```

| Method | Path           | Fungsi                                        |
|--------|----------------|-----------------------------------------------|
| GET    | /health        | status koneksi Neo4j                          |
| GET    | /tools         | daftar tool + JSON schema (function-calling)  |
| POST   | /tools/{name}  | panggil tool: `{"arguments": {...}}`          |
| POST   | /query         | `{"query": "..."}` → jawaban terverifikasi    |
| POST   | /setup         | Phase 1 + Phase 3 (idempotent)                |

Contoh:

```bash
curl -X POST localhost:8000/query -H 'Content-Type: application/json' \
  -d '{"query": "What is a RMFS?"}'

curl -X POST localhost:8000/tools/ingest_meeting -H 'Content-Type: application/json' \
  -d '{"arguments": {"title": "Sprint sync", "participants": ["Thifa"], "notes": "...", "entities": ["RMFS"]}}'
```

## Docker

```bash
cp .env.example .env
docker compose up --build
# API di http://localhost:8000, Neo4j browser di http://localhost:7474
```

CLI di dalam container:

```bash
docker compose run --rm agent python -m kg_agent.cli --query "..." --setup
```

## Menghubungkan ke KG lain (mis. punya temanmu)

Tidak perlu mengubah kode — atur di `.env`:

```
NEO4J_URI=neo4j+s://<instance>.databases.neo4j.io
NEO4J_USERNAME=...
NEO4J_PASSWORD=...
# kalau skema grafnya beda:
KG_ENTITY_LABEL=Entity
KG_ENTITY_NAME_PROP=title
KG_STRUCTURAL_RELS=MENTIONS,NEXT
```

Lalu hapus override `NEO4J_URI` di `docker-compose.yml` (service `neo4j` juga
boleh dihapus sekalian kalau tidak dipakai).

## Menambah tool baru untuk agent

Tulis fungsi di `kg_agent/tools.py` dengan signature
`fn(client, cfg, *, ...params) -> dict`, lalu daftarkan di dict `TOOLS`.
Tool otomatis muncul di `GET /tools` dan bisa dipanggil via
`POST /tools/{name}` — siap dipetakan ke LLM function-calling / MCP.

## Lisensi & atribusi

Konsep pipeline KG mengikuti repo GPL-3.0 milik DylanTartarini1996; modul
`kg_agent` merupakan lapisan verifikasi yang dikembangkan terpisah. Jika kamu
mendistribusikan proyek ini bersama kode turunan dari repo tersebut, gunakan
lisensi yang kompatibel (GPL-3.0).
