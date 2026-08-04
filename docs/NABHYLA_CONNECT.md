# Runbook: menghubungkan kg-agent ke KG Nabhyla

Checklist menghubungkan kg-agent ke KG Nabhyla. Langkah 1-4 (recon + adaptasi
kode + verifikasi live read-only) **sudah selesai**. Langkah 5-6 (permission +
write) **belum** — jangan jalankan apa pun yang menulis ke graph Nabhyla
sebelum izin eksplisit.

> **Status terkini:** endpoint Tailscale `bolt://100.95.227.48:7687`
> reachable, kredensial `neo4j/password123` valid, graph **berisi 335 node**.
> Profil skema di bawah (Topic/Type/Description) sudah **dikonfirmasi** oleh
> pemilik repo sebagai bentuk KG yang dimaksud — sudah diuji end-to-end
> (offline + lokal + read-only langsung ke graph asli) dan menghasilkan
> jawaban ter-grounding dari 15 Topic nyata.

## Koneksi LIVE vs salinan CLONE (penting — sering bikin bingung)

Ada **dua cara** kg-agent berhubungan dengan graph Nabhyla. Keduanya pakai kode
yang sama; bedanya cuma `NEO4J_URI`.

| | **Koneksi LIVE** | **Salinan CLONE (lokal)** |
|---|---|---|
| `NEO4J_URI` | `bolt://100.95.227.48:7687` (endpoint Nabhyla) | `bolt://localhost:7687` (Neo4j lokalmu) |
| Data | **selalu terkini** — dibaca langsung dari graph Nabhyla saat query | **potret satu waktu** — beku sejak di-clone |
| Ikut perubahan KG Nabhyla? | **Ya, otomatis.** Nabhyla rebuild/tambah data → query berikutnya langsung lihat data baru. Tanpa clone ulang, tanpa ubah kode. | **Tidak.** Harus **clone ulang** kalau mau data terbaru. |
| Untuk apa | **Cara pakai sebenarnya**: menjawab pertanyaan terhadap graph Nabhyla (read-only aman). | **Eval/benchmark saja**: karena eval MENULIS data uji sintetis (node low-trust/temporal palsu) yang haram ditulis ke graph bersama. |
| Menulis? | Jangan (butuh izin) — lihat aturan di bawah. | Bebas — ini sandbox disposable milikmu. |

**Konsekuensi praktis:**
- **Menjawab pertanyaan live** → cukup `.env` menunjuk endpoint Nabhyla + servernya
  hidup. Saat KG-nya berubah, jawaban otomatis mengikuti data baru.
- **Eval ulang setelah KG Nabhyla berubah** → **clone ulang** ke lokal (satu
  perintah, ~detik), lalu jalankan toolkit lagi. Clone lama TIDAK ikut berubah.
- **Kalau skema KG Nabhyla berubah saat rebuild** (label/relasi beda dari
  Topic/Type/Description) → jalankan ulang recon read-only (langkah 1 di bawah)
  dan sesuaikan profil `.env` (`KG_ENTITY_LABEL`, `KG_CHUNK_TO_ENTITY_PATTERN`,
  dst.) sebelum dipakai.
- **Cara clone** (baca live read-only → tulis ke lokal): baca semua node+relasi
  dari sumber lewat `execute_read`, buat ulang di lokal (lihat skrip
  `clone_graph.py` di scratchpad sesi eval; ~335 node butuh beberapa detik,
  dengan retry karena endpoint Nabhyla kadang naik-turun).

> Trust rendah tanpa Phase 1: jawaban live terhadap graph Nabhyla tanpa
> menjalankan Phase 1 (`--setup`, MENULIS) akan berskor trust ~0.2 → banyak
> disclaimer. Menaikkannya butuh izin menulis. Di eval, trust ~0.5 dicapai
> dengan `source_type=paper` **di salinan lokal**, bukan di graph Nabhyla.

## Aturan tetap
- **Read-only dulu** sampai langkah 5. `python -m kg_agent.cli --query "..."`
  (tanpa `--setup`) **aman** dipakai langsung terhadap endpoint Nabhyla — jalur
  ini hanya `MATCH`/`RETURN`, tidak pernah menulis (`compute_and_store`/
  `run_phase1_migration` hanya dipanggil dari `--setup` / `POST /setup`; sudah
  diaudit ulang kodenya untuk memastikan ini).
- **`ingest_meeting` adalah tool MENULIS** (MERGE node `Meeting` + entity baru).
  Jangan pernah panggil tool ini (langsung atau lewat orchestrator) dengan
  endpoint Nabhyla sebagai target, tanpa izin eksplisit.
- Jangan `kg_agent.cli --setup`, `run_phase1_migration`, atau
  `compute_and_store` sebelum izin Nabhyla — ini KG **bersama**, bukan sandbox
  lokal.

## Langkah

### 1-2. Recon + bentuk graph sebenarnya — SELESAI

Graph Nabhyla ternyata berisi **dua lapisan sekaligus** (recon pertama, lewat
query yang sengaja exclude `Chunk`/`Document`, sempat membuat lapisan pertama
di bawah ini tidak kelihatan):

| Lapisan | Label | Jumlah | Keterangan |
|---------|-------|--------|------------|
| RAG generik (bukan yang dipakai) | `Chunk` | 106 | punya `text` + `embedding` (1024 dim, index vektor `vector`); terhubung ke `Topic` lewat `MENTIONS` (111 edge). Ini kelihatan seperti output pipeline chunking+embedding generik, **bukan** hasil ekstraksi konsep yang dimaksud. |
| **KG yang dipakai (dikonfirmasi)** | `Topic` | 94 | `name`/`id` identik — ini "entity" |
| " | `Type` | 9 | kategori tetap: Problem/Existing Research/Research Goal/Result/Experiment/dst |
| " | `Description` | 99 | `text` (isi konten sebenarnya), TIDAK punya `embedding` |
| " | `Agent` | 23 | penulis/pembimbing |
| " | `Role` | 2 | co-advisor, defense committee member |
| " | `Source` | 1 (per dokumen) | judul dokumen sumber |

Relasi lapisan KG: `HAS_TYPE` (Topic→Type), `HAS_SUBTOPIC` (Topic→Topic,
hierarki), `HAS_DESCRIPTION` (Type→Description), `HAS_SOURCE` (Topic→Source),
`ROLE_IN_PAPER` (Agent→Role), `RELATES_TO` (Topic→Topic), `WRITES_ABOUT`
(Agent→Topic). Teks (`Description.text`) tidak menempel langsung ke `Topic` —
jaraknya dua hop lewat `Type`:
`Topic-[:HAS_TYPE]->Type-[:HAS_DESCRIPTION]->Description`.

Juga belum ada `created_at`/`source_type`/`confidence_score`/`trust_score` di
label manapun (wajar — Phase 1 belum pernah jalan di sini).

**Ini bukan sekadar beda nama label** — asumsi keras retriever lama
(`(:Chunk {text})-[:MENTIONS]->(:Entity)`, satu hop) tidak punya analog
langsung untuk lapisan Topic/Description. Karena itu kg-agent sendiri **sudah
diubah** supaya path chunk→entity bisa dikonfigurasi lewat env, termasuk path
multi-hop:
- `KG_CHUNK_LABEL` — label node berisi teks (default `Chunk`; Nabhyla: `Description`)
- `KG_CHUNK_TEXT_PROP` — properti teksnya (default `text`; Nabhyla: `text`, sudah cocok)
- `KG_CHUNK_TO_ENTITY_PATTERN` — pola Cypher mentah dari `c` (chunk) ke `e`
  (entity), pakai placeholder `{chunk_label}`/`{entity_label}`. Default
  `(c:{chunk_label})-[:MENTIONS]->(e:{entity_label})`.

### 3. Override `.env` untuk skema Nabhyla — CANONICAL

Ini profil yang dikonfirmasi dipakai (juga ada di `.env.example`):

| Env | Default kg-agent | Nilai untuk KG Nabhyla |
|-----|------------------|------------------------|
| `KG_ENTITY_LABEL` | `__Entity__` | `Topic` |
| `KG_ENTITY_NAME_PROP` | `name` | `name` (sudah cocok) |
| `KG_ENTITY_ID_PROP` | `id` | `id` (sudah cocok) |
| `KG_STRUCTURAL_RELS` | `MENTIONS,NEXT,PART_OF` | `HAS_TYPE,HAS_SUBTOPIC,HAS_DESCRIPTION,HAS_SOURCE,ROLE_IN_PAPER,RELATES_TO,WRITES_ABOUT` (lihat catatan di bawah) |
| `KG_CHUNK_LABEL` | `Chunk` | `Description` |
| `KG_CHUNK_TEXT_PROP` | `text` | `text` (sudah cocok) |
| `KG_CHUNK_TO_ENTITY_PATTERN` | `(c:{chunk_label})-[:MENTIONS]->(e:{entity_label})` | `(c:{chunk_label})<-[:HAS_DESCRIPTION]-(:Type)<-[:HAS_TYPE]-(e:{entity_label})` |
| `KG_VECTOR_INDEX` | `vector` | `vector` — index ini milik lapisan `:Chunk`, BUKAN `:Description`. Sudah aman dibiarkan default: `retrieve_vector` memfilter hasil ke `KG_CHUNK_LABEL`, jadi index yang tidak cocok otomatis diabaikan (jatuh ke keyword), bukan bocor jadi jawaban salah-atribusi (bug ini sempat ditemukan &ditutup — lihat bagian "Bug yang ditemukan" di bawah). |

Untuk menunjuk endpoint Nabhyla (read-only-safe untuk `--query` biasa):
```
NEO4J_URI=bolt://100.95.227.48:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password123
```

Catatan soal `KG_STRUCTURAL_RELS`: daftar di atas menandai **semua** relasi
Nabhyla sebagai "plumbing" (tidak diberi metadata temporal walau Phase 1
nanti dijalankan). Ini pilihan yang disengaja: graph Nabhyla adalah **peta
konsep statis dari satu dokumen** (hierarki topik + klasifikasi + deskripsi),
bukan graph klaim/fakta yang berubah dari waktu ke waktu (rapat demi rapat,
dokumen demi dokumen) seperti yang diasumsikan Phase 2
(VALID/OUTDATED/SUPERSEDED/CONFLICTED). Kalau nanti Nabhyla mengingest
dokumen kedua dan ada topik yang saling menggantikan/bertentangan, ini perlu
ditinjau ulang.

### 4. Dimensi embedding — SELESAI dicek, hasilnya: vector tidak berlaku untuk profil ini
`:Chunk` (106 node) punya `text` + `embedding` (1024 dim, index `vector`,
COSINE) — tapi ini lapisan yang **bukan** dipakai. `:Description` (99 node,
semua punya `text`) **tidak** punya `embedding` sama sekali. Konsekuensinya:
retrieval untuk profil Topic/Description **selalu keyword**, tidak pernah
vector — ini otomatis (lihat bug fix di bawah), bukan sesuatu yang perlu
dikonfigurasi manual (`KG_PREFER_VECTOR` boleh dibiarkan default `true`).

Kalau nanti Nabhyla ingin retrieval vektor untuk lapisan KG (bukan Chunk),
`Description` (atau `Topic`) perlu diberi `embedding` sendiri — ini pekerjaan
terpisah (butuh menulis embedding baru ke graph-nya, jadi juga butuh izin).

### Bug yang ditemukan & diperbaiki (selama verifikasi live)
Dua bug ditemukan saat menguji profil ini terhadap Neo4j sungguhan (bukan
cuma test offline), keduanya sudah diperbaiki di `agentic_verifier.py`:
1. **Crash saat vector index tidak ada** (ketemu di uji lokal): index vektor
   `vector` tidak ada di Neo4j lokal (fixture `seed_nabhyla_shape.py` tidak
   membuatnya) → `ClientError` dari `db.index.vector.queryNodes` dulunya
   menjalar sampai bikin CLI crash. Sekarang ditangkap, fallback ke keyword.
2. **Salah-atribusi diam-diam** (ketemu saat uji live ke graph Nabhyla asli):
   index `vector` memang ADA di sana, tapi meng-index `:Chunk`, bukan
   `:Description`. Tanpa filter label, `db.index.vector.queryNodes`
   mengembalikan node `:Chunk` asli (yang kebetulan juga punya properti
   `text`) dengan nol entity ter-atribusi — jawabannya kelihatan berhasil
   (teks terisi) tapi `sources_used` kosong dan `trust_score` 0.0, padahal
   isinya sebenarnya ditarik dari lapisan Chunk yang bukan dimaksud. Sekarang
   `retrieve_vector` memfilter hasil vector-index ke `KG_CHUNK_LABEL` yang
   dikonfigurasi, jadi index yang salah lapisan otomatis diabaikan dan jatuh
   ke keyword retrieval (yang sudah terbukti bekerja: 15 Topic nyata
   ter-retrieve untuk query "What is PNP in order picking?").

### Bug kedua: jawaban kosong dari model utama disamarkan sebagai lolos gate
Ditemukan live (query nyata `--query "What is PNP in order picking?"` dengan
`KG_LLM_PROVIDER=ollama`, `KG_LLM_MODEL=qwen3-vl:4b`, terhadap 15 sumber real
dari graph Nabhyla): jawaban yang dihasilkan **kosong** (`qwen3-vl:4b`
menghabiskan token budget-nya untuk "berpikir" tanpa pernah mengeluarkan
konten final — persis limitasi yang sudah didiagnosis untuk juri faithfulness,
ternyata juga terjadi di generasi jawaban utama pada query nyata, bukan cuma
di skenario juri). Yang lebih serius: juri (`hermes3:3b`) memberi skor
faithfulness **0.70** untuk jawaban kosong itu — pas lolos gate 0.7, seolah
jawaban kosong itu "didukung penuh oleh sumber". Sudah diperbaiki:
`_generate_answer` sekarang tidak pernah mengembalikan string kosong (diganti
sentinel eksplisit), dan `_check_faithfulness` memaksa skor 0.0 untuk jawaban
kosong/sentinel tanpa memanggil juri sama sekali (4 test regresi baru).

Catatan jujur: perbaikan ini membuat gate berperilaku benar (jawaban kosong
sekarang otomatis gagal, bukan lolos), tapi TIDAK memperbaiki akar masalah
kenapa `qwen3-vl:4b` menghasilkan jawaban kosong untuk query nyata. Kalau ini
sering terjadi, pertimbangkan salah satu: naikkan `KG_LLM_MAX_TOKENS` banyak
(belum tentu cukup — sudah diuji sampai 4000 untuk kasus juri dan tetap
kosong), atau pakai model instruct biasa (non-thinking) untuk generasi jawaban
juga, mirip pola `KG_JUDGE_*` yang sudah ada untuk juri.

### 5. MINTA IZIN Nabhyla sebelum menulis — BELUM DIMINTA
`kg_agent.cli --setup` menjalankan **Phase 1 migration** (menstempel
`created_at`/`updated_at`/`source_type`/`confidence_score` ke node & relasi)
dan **Phase 3** (menulis `trust_score`). Ini **menulis ke graph Nabhyla**.
Dapatkan persetujuan eksplisit dulu. Idempotent dan hanya mengisi nilai yang
kosong (`coalesce`), tapi tetap sebuah perubahan pada data bersama.

**Konsekuensi kalau belum diizinkan/dijalankan** (sudah diverifikasi live):
setiap `Topic` yang belum punya `confidence_score`/`source_type` otomatis
dapat `confidence=0.5 (default) x source_weight=0.4 (default) x recency=1.0
(tanpa created_at)` = **trust_score 0.2** — di bawah `KG_MIN_TRUST_SCORE`
default (0.4). Confirmed lewat query nyata: `trust_score: 0.2`,
`passed: false`, tapi jawabannya sendiri relevan dan ter-grounding
(`faithfulness: 0.929`). Artinya **semua** jawaban dari data Nabhyla akan
kena gate trust dan dapat disclaimer sampai Phase 1 jalan — bukan karena
isinya salah, murni karena metadata provenance belum ada.

### 6. Sesudah izin + `--setup`: query uji ala Step 3
Query read-only sudah pernah dicoba (`"What is PNP in order picking?"` →
15 sources, faithfulness 0.929, trust 0.2/gagal gate trust seperti diprediksi).
Sesudah Phase 1 benar-benar jalan (dengan izin), ulangi dengan
`KG_ORCHESTRATOR=native` atau `--agentic`, dan konfirmasi:
- Satu query yang **harus `passed: true`** sekarang (trust score sudah wajar).
- Satu yang **masih harus kena gate** (mis. topik dengan confidence rendah,
  atau pertanyaan tanpa konteks pendukung).

## Testing lokal (tidak menyentuh graph Nabhyla)
`tests/seed_nabhyla_shape.py` membuat potongan kecil graph dengan bentuk sama
persis (Topic/Type/Description/Agent/Source) di Neo4j **lokal** milikmu
sendiri (skrip menolak berjalan kalau `NEO4J_URI` mengandung `100.95.227.48`,
sebagai pengaman tambahan karena skrip ini MENULIS). Sudah dipakai untuk
memvalidasi profil ini end-to-end (dan menemukan bug #1 di atas) sebelum
menyentuh data asli:
```bash
# .env mengarah ke Neo4j lokal (docker run ... lihat README Troubleshooting)
# + baris KG_ENTITY_LABEL/KG_CHUNK_*/KG_STRUCTURAL_RELS dari tabel langkah 3
python tests/seed_nabhyla_shape.py
python -m kg_agent.cli --query "What is PNP?"
python tests/seed_nabhyla_shape.py --clear
```

## Catatan
- Juri faithfulness butuh model non-thinking (lihat README bagian *Juri
  faithfulness terpisah*): mis. `KG_JUDGE_PROVIDER=ollama`,
  `KG_JUDGE_MODEL=hermes3:3b`, `KG_JUDGE_OLLAMA_URL=http://localhost:11434`,
  sementara model utama (`qwen3-vl:4b`) mengurus tool-calling + answer
  generation.
- Recon sebelumnya sempat gagal ke `bolt://192.168.0.185:7687` — itu alamat LAN
  yang tidak reachable dari mesin ini (`No route to host`). Gunakan alamat
  **Tailscale** `100.95.227.48`, bukan LAN.
