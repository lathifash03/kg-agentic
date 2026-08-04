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

Opsional (Phase 5): **orchestrator tool-calling** yang membiarkan LLM lokal
memilih *tool mana* yang dipanggil. Lapisan ini tidak menggantikan gate di atas
— lihat [Orchestrator agentic](#orchestrator-agentic-phase-5).

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
  orchestrator.py       # Phase 5 (LLM memilih tool via Ollama native tool-calling)
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

# biarkan LLM yang memilih tool (Phase 5); --json ikut memuat tool-call trace
python -m kg_agent.cli --query "..." --agentic
```

Tanpa Neo4j/LLM sungguhan, set `KG_LLM_PROVIDER=mock` untuk menguji alurnya.

Jalankan test: `pip install pytest && python -m pytest tests/ -q` (tidak butuh
Neo4j maupun LLM — model dan tool dipalsukan).

## API

```bash
uvicorn kg_agent.api:app --reload
```

| Method | Path           | Fungsi                                        |
|--------|----------------|-----------------------------------------------|
| GET    | /health        | status koneksi Neo4j                          |
| GET    | /tools         | daftar tool + JSON schema (function-calling)  |
| POST   | /tools/{name}  | panggil tool: `{"arguments": {...}}`          |
| POST   | /query         | `{"query": "..."}` → jawaban terverifikasi; `{"agentic": true}` → LLM memilih tool + `tool_trace` |
| POST   | /setup         | Phase 1 + Phase 3 (idempotent)                |

Contoh:

```bash
curl -X POST localhost:8000/query -H 'Content-Type: application/json' \
  -d '{"query": "What is a RMFS?"}'

# mode agentic: respons sama seperti di atas + tool_trace/tools_used
curl -X POST localhost:8000/query -H 'Content-Type: application/json' \
  -d '{"query": "What is a RMFS?", "agentic": true}'

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

## Troubleshooting

### Neo4j `docker compose up` gagal: `JAVA_HOME is not defined` / `pthread_create failed (EPERM)`

Di Docker Desktop versi lama (mis. server 20.10.x), image resmi `neo4j:5`
crash saat start. Gejalanya di `docker logs`:

```
... chmod: changing permissions of '.../java': Operation not permitted
Error: JAVA_HOME is not defined correctly. We cannot execute /opt/java/openjdk/bin/java
```

Kalau dijalankan langsung, JVM-nya menunjukkan akar masalahnya:

```
pthread_create failed (EPERM) ... Failed to create worker thread
```

Penyebabnya **bukan** JAVA_HOME, melainkan profil **seccomp** default Docker
lama yang memblokir syscall `clone3` yang dipakai JDK baru untuk membuat
thread — JVM gagal membuat thread GC lalu keluar dengan pesan yang menyesatkan.

**Solusi:** jalankan Neo4j secara manual dengan seccomp dimatikan, bukan lewat
`docker compose up` (parameter/port/auth sama seperti service `neo4j` di
`docker-compose.yml`):

```bash
docker run -d --name kg-neo4j \
  --security-opt seccomp=unconfined \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password123 \
  neo4j:5

# tunggu siap:
until docker exec kg-neo4j cypher-shell -u neo4j -p password123 'RETURN 1' >/dev/null 2>&1; do sleep 2; done

# ... pakai kg-agent seperti biasa (NEO4J_URI=bolt://localhost:7687) ...

# selesai:
docker rm -f kg-neo4j
```

**Meng-upgrade Docker Desktop** ke versi lebih baru (profil seccomp yang
sudah mengizinkan `clone3`) kemungkinan besar menghilangkan kebutuhan
workaround ini — sesudah upgrade, `docker compose up` semestinya jalan normal.

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

Kalau graf tujuan **tidak** punya lapisan `(:Chunk {text})-[:MENTIONS]->(entity)`
sama sekali — mis. graph ekstraksi-konsep yang menaruh teks di label lain,
lewat relasi lain, atau bahkan dua hop jauhnya — retrieval tetap bisa
dikonfigurasi tanpa mengubah kode lewat tiga variabel tambahan:

```
KG_CHUNK_LABEL=Description          # label node yang berisi teks
KG_CHUNK_TEXT_PROP=text             # properti teksnya
# pola Cypher mentah dari node teks (c) ke node entity (e); boleh multi-hop:
KG_CHUNK_TO_ENTITY_PATTERN=(c:{chunk_label})<-[:HAS_DESCRIPTION]-(:Type)<-[:HAS_TYPE]-(e:{entity_label})
```

Contoh nyata (skema KG Nabhyla: Topic/Type/Description) ada di `.env.example`
dan `docs/NABHYLA_CONNECT.md` — graph itu ternyata punya dua lapisan sekaligus
(satu lapisan `Chunk/MENTIONS` generik, satu lapisan konsep Topic/Description),
dan profil ini sengaja menunjuk ke lapisan konsepnya, bukan berarti graph itu
tidak punya `Chunk`/`MENTIONS` sama sekali. Untuk mencoba profil semacam ini
tanpa menyentuh graph orang lain, seed dulu bentuk yang sama di Neo4j lokal
dengan `tests/seed_nabhyla_shape.py`.

## Orchestrator agentic (Phase 5)

Secara default agent selalu memanggil `answer_question`. Dengan orchestrator
aktif, sebuah LLM lokal menerima registry tool lewat parameter `tools` milik
Ollama dan **memilih sendiri** tool mana yang dipakai — misalnya `kg_stats`
untuk "seberapa besar graph-nya?" dan `answer_question` untuk pertanyaan isi.

**Yang tidak berubah:** LLM hanya memilih *tool mana*. Ia tidak pernah
menggantikan gate verifikasi. Saat tool terakhir adalah `answer_question`,
seluruh `VerifiedAnswer` (trust_score, temporal_validity_status, faithfulness,
passed, disclaimer, …) dikembalikan apa adanya, dan teks jawaban yang
ditampilkan adalah teks bergate itu — bukan parafrase model (yang bisa saja
menghilangkan disclaimer). Bila loop berakhir di `kg_stats`/`ingest_meeting`,
tidak ada field gate yang dikarang: hasil tool itu sendiri yang dikembalikan.

Konfigurasi:

| Env | Default | Fungsi |
|-----|---------|--------|
| `KG_ORCHESTRATOR` | `off` | `native` untuk mengaktifkan; `off` = perilaku lama persis |
| `KG_ORCHESTRATOR_MAX_CALLS` | `4` | batas keras tool call yang dieksekusi per request |
| `OLLAMA_URL` | `http://localhost:11434` | server Ollama |
| `KG_LLM_MODEL` | — | **wajib** untuk provider ollama; model harus punya kapabilitas `tools` |
| `KG_JUDGE_PROVIDER` / `KG_JUDGE_MODEL` | kosong | juri faithfulness terpisah (lihat di bawah); kosong = pakai model utama |

```bash
export OLLAMA_URL=http://100.118.203.111:11434
export KG_LLM_PROVIDER=ollama
export KG_LLM_MODEL=qwen3-vl:4b
export KG_ORCHESTRATOR=native

python -m kg_agent.cli --query "How big is the graph?" --json
```

Model apa pun bisa dipakai asalkan Ollama mengiklankan kapabilitas `tools`:

```bash
curl $OLLAMA_URL/api/show -d '{"model":"qwen3-vl:4b"}' | jq .capabilities
# ["completion","vision","tools","thinking"]
```

Pengaman di `orchestrator.py`:

- **Whitelist + JSON schema** — setiap tool call divalidasi (nama ada di
  registry, tipe argumen cocok, argumen wajib terisi, tidak ada argumen asing)
  *sebelum* dieksekusi, jadi nama tool halusinasi tidak pernah sampai ke
  `call_tool`.
- **Satu kali re-prompt** — kegagalan validasi dikirim balik ke model sebagai
  pesan tool; kalau gagal lagi, loop berhenti dengan alasan eksplisit.
- **Batas 4 call** — dihitung per call individual (satu giliran bisa meminta
  beberapa call paralel).
- **Trace lengkap** — `[{step, tool_name, arguments, validation_error,
  result_or_error, timestamp}]` selalu dikembalikan, sukses maupun gagal.
- **Deteksi korupsi** — bila token kontrol (`<tool_call>`, `<think>`) bocor ke
  teks jawaban, respons ditolak alih-alih diteruskan sebagai jawaban.

Catatan: kalau model menjawab tanpa memanggil tool sama sekali, `tools_used`
kosong dan jawaban itu **tidak melewati verifikasi** — orchestrator mencatat
peringatan, dan field gate tidak diisi.

### Juri faithfulness terpisah

Gate faithfulness memakai LLM sebagai juri (mengembalikan verdict JSON). Model
**thinking-only** seperti `qwen3-vl:4b` bermasalah di sini: ia bernalar melebihi
budget token dan tidak pernah mengeluarkan `content` final, sehingga verdict
selalu gagal di-parse → faithfulness `0.0` → `passed` selalu `false`. (Tool
selection, answer generation, trust, dan temporal tetap normal — hanya panggilan
juri ini yang terdampak.)

Solusi: arahkan juri ke model instruct biasa lewat env, sementara model utama
tetap mengurus tool-calling + answer generation:

```bash
KG_JUDGE_PROVIDER=ollama         # ollama | groq | mock ; kosong = pakai model utama
KG_JUDGE_MODEL=hermes3:8b        # model instruct non-thinking
KG_JUDGE_OLLAMA_URL=http://localhost:11434   # default: sama dgn OLLAMA_URL
```

`KG_JUDGE_OLLAMA_URL` default mengikuti `OLLAMA_URL`, jadi juri bisa berjalan di
Ollama lokal walau model utama ada di endpoint remote. Kosongkan
`KG_JUDGE_PROVIDER` untuk kembali ke perilaku lama (juri = model utama).

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
