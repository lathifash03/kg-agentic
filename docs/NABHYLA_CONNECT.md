# Runbook: menghubungkan kg-agent ke KG Nabhyla

Checklist yang dijalankan **saat KG Nabhyla sudah terisi data**. Sampai saat itu
tiba, jangan jalankan langkah menulis apa pun ke graph Nabhyla.

> **Status per recon terakhir (read-only):** endpoint Tailscale
> `bolt://100.95.227.48:7687` **reachable**, kredensial `neo4j/password123`
> valid, tapi database `neo4j` **masih kosong** (0 node, 0 relasi). Yang sudah
> ada hanya sebuah **vector index `vector`** pada `:Chunk(embedding)` —
> **1024 dimensi, COSINE** — jadi bentuknya disiapkan untuk pipeline yang sama
> dengan kg-agent, tinggal menunggu ingestion. Label entity, properti nama, dan
> relasi MENTIONS belum bisa dipastikan karena belum ada data.

## Aturan tetap
- **Read-only dulu** sampai langkah 5. Pakai driver + `execute_read` (atau
  cypher-shell query biasa). Jangan `kg_agent.cli --setup`,
  `run_phase1_migration`, atau `compute_and_store` sebelum izin Nabhyla.
- Ini KG **bersama** milik Nabhyla, bukan sandbox lokal.

## Langkah

### 1. Recon ulang (read-only) — script sudah ada
```bash
# script inspeksi read-only (raw driver, execute_read, tanpa kode kg_agent):
python inspect_nabhyla.py bolt://100.95.227.48:7687
# (dan follow-up databases/constraints/indexes:)
python inspect_nabhyla2.py bolt://100.95.227.48:7687
```
Script ada di scratchpad sesi kerja; kalau hilang, isinya sederhana: connect
dengan `neo4j.GraphDatabase.driver`, `session(default_access_mode=READ_ACCESS)`,
lalu `db.labels()`, `db.relationshipTypes()`, hitung node per label, sample
`(c)-[:MENTIONS]->(e)`, dan cek properti temporal. **Semua `execute_read`.**

### 2. Tentukan bentuk graph sebenarnya
Dari hasil recon, catat:
- **Nama label entity** yang paling banyak count-nya (default kg-agent:
  `__Entity__`).
- **Properti nama/judul** entity (default: `name`).
- Apakah relasi **`MENTIONS`** ada dan benar menghubungkan node teks (`:Chunk`)
  ke node entity — bukan ke node lain.
- Apakah node teks memang berlabel `:Chunk` dengan properti `text` + `embedding`.

### 3. Catat override `.env` yang diperlukan (JANGAN diterapkan di sini)
Kalau berbeda dari default kg-agent, cukup **catat** nilainya untuk didiskusikan
— jangan langsung ubah `.env`:
| Env | Default kg-agent | Nilai di KG Nabhyla | Perlu override? |
|-----|------------------|---------------------|-----------------|
| `KG_ENTITY_LABEL` | `__Entity__` | _(isi dari langkah 2)_ | |
| `KG_ENTITY_NAME_PROP` | `name` | | |
| `KG_STRUCTURAL_RELS` | `MENTIONS,NEXT,PART_OF` | | |
| `KG_VECTOR_INDEX` | `vector` | `vector` (sudah cocok) | tidak |

Untuk menunjuk endpoint Nabhyla:
```
NEO4J_URI=bolt://100.95.227.48:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password123
```

### 4. Konfirmasi dimensi embedding tetap 1024
```bash
# read-only:
#   SHOW INDEXES YIELD name, type, options WHERE type='VECTOR'
#   -> options.indexConfig['vector.dimensions'] harus 1024
```
`mxbai-embed-large` (default `KG_EMBED_MODEL`) menghasilkan **1024** dim. Kalau
index/embedding Nabhyla berbeda dimensinya, `db.index.vector.queryNodes` akan
gagal dan retrieval jatuh ke keyword — samakan `KG_EMBED_MODEL` dengan model
yang dipakai Nabhyla, atau sepakati satu model bersama.

### 5. MINTA IZIN Nabhyla sebelum menulis
`kg_agent.cli --setup` menjalankan **Phase 1 migration** (menstempel
`created_at`/`updated_at`/`source_type`/`confidence_score` ke node & relasi)
dan **Phase 3** (menulis `trust_score`). Ini **menulis ke graph Nabhyla**.
Dapatkan persetujuan eksplisit dulu. Idempotent dan hanya mengisi nilai yang
kosong (`coalesce`), tapi tetap sebuah perubahan pada data bersama.

Kalau perlu, uji dulu terhadap salinan (dump/restore ke Neo4j lokal) sebelum
menyentuh graph asli.

### 6. Sesudah izin + `--setup`: query uji ala Step 3
Jalankan (dengan `KG_ORCHESTRATOR=native` atau `--agentic`) dua query:
- Satu yang **harus `passed: true`** (pertanyaan yang jawabannya didukung
  sumber segar & tepercaya).
- Satu yang **harus kena gate** (mis. entity yang sudah OUTDATED, atau
  pertanyaan tanpa konteks pendukung → disclaimer).

Konfirmasi field gate (`trust_score`, `temporal_validity_status`,
`faithfulness`, `passed`, `disclaimer`) terisi nilai nyata dari data Nabhyla,
lalu laporkan hasilnya.

## Catatan
- Juri faithfulness butuh model non-thinking (lihat README bagian *Juri
  faithfulness terpisah*): mis. `KG_JUDGE_PROVIDER=ollama`,
  `KG_JUDGE_MODEL=hermes3:3b`, `KG_JUDGE_OLLAMA_URL=http://localhost:11434`,
  sementara model utama (`qwen3-vl:4b`) mengurus tool-calling + answer
  generation.
- Recon sebelumnya sempat gagal ke `bolt://192.168.0.185:7687` — itu alamat LAN
  yang tidak reachable dari mesin ini (`No route to host`). Gunakan alamat
  **Tailscale** `100.95.227.48`, bukan LAN.
