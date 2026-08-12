# kg-agent — Ringkasan Keseluruhan Project

**Dibuat:** 11 Agustus 2026
**Repo:** `kg-agent` (branch `main`)
**Status ringkas:** Phase 1–5 selesai · 47 test lulus · evaluasi paper-suite L1+L2 selesai dijalankan · satu blocker terbuka (Phase 1 butuh izin)

---

## 1. Apa ini

Lapisan **verifikasi agentic** di atas knowledge graph Neo4j. Bukan pipeline ingestion — KG diasumsikan sudah terisi. Agent menjawab pertanyaan terhadap KG, lalu menilai jawabannya sendiri lewat empat gate berlapis:

```
Phase 1  Temporal metadata   created_at / source_type / confidence_score (idempotent)
Phase 2  Temporal validity   VALID / OUTDATED / SUPERSEDED / CONFLICTED
Phase 3  Node trust          trust = confidence x bobot_sumber x recency
Phase 4  Agentic verifier    retrieval -> generate -> faithfulness gate -> retry
Phase 5  Orchestrator        LLM memilih tool (opsional, tidak menggantikan gate)
```

Bobot sumber: `paper 1.0 · meeting 0.8 · discussion 0.6 · auto_extracted 0.4`.

Versi mandiri dari modul `kg_agentic` (asalnya di atas repo knowledge-graph milik DylanTartarini1996), tanpa Streamlit. Dipakai lewat CLI, HTTP API (FastAPI), dan Docker.

---

## 2. Timeline pengembangan

8 commit, 17 Juli – 10 Agustus 2026.

| Commit | Tanggal | Isi |
|---|---|---|
| `d511d8a` | 17 Jul | Initial — port standalone, Phase 1–4, CLI, API, Docker (3.146 baris) |
| `099ea36` | 28 Jul | Phase 5 orchestrator tool-calling + juri faithfulness terpisah (1.705 baris) |
| `52c2b92` | 4 Agu | Skema retrieval configurable + vector fallback + toolkit eval E1 (2.338 baris) |
| `41c47f5` | 7 Agu | Eval ronde 2 — profil Chunk/vector, judge-comparison toolkit |
| `17f2489` | 10 Agu | Full eval E1/E2/E3 + ablation |
| `997a8e8` | 10 Agu | Artefak deploy produksi (compose API-only + template env) |
| `df60bff` | 10 Agu | Arahkan config ke Neo4j hasil rebuild |
| `350d858` | 10 Agu | Answer-gen → hermes3 (qwen3-vl mengosongkan jawaban di konteks panjang) |

**Belum di-commit:** atribusi paper (`KG_CHUNK_SOURCE_PROP`) + suite evaluasi baru `eval/paper_suite/`.

---

## 3. Komponen inti

| Modul | Baris | Fungsi |
|---|---|---|
| `kg_agent/agentic_verifier.py` | 958 | Phase 4 — retrieval (vector→keyword→expanded), generate, gate, retry |
| `kg_agent/orchestrator.py` | 558 | Phase 5 — LLM memilih tool, whitelist + validasi schema, cap 4 call, trace |
| `kg_agent/temporal_validity.py` | 481 | Phase 2 |
| `kg_agent/neo4j_client.py` | 454 | Koneksi + migrasi Phase 1 |
| `kg_agent/config.py` | 409 | Semua threshold & skema, override via env |
| `kg_agent/node_trust.py` | 332 | Phase 3 |
| `kg_agent/tools.py` | 199 | Registry: `answer_question`, `ingest_meeting`, `kg_stats` |
| `kg_agent/api.py` + `cli.py` | 291 | 5 endpoint + CLI |

**Test: 47 lulus** (~0,3 detik, tanpa Neo4j maupun LLM — model dan tool dipalsukan).

### Antarmuka

```bash
python -m kg_agent.cli --setup                  # Phase 1 + Phase 3 (MENULIS)
python -m kg_agent.cli --query "..." --json     # read-only
python -m kg_agent.cli --query "..." --agentic  # LLM memilih tool
```

| Method | Path | Fungsi |
|---|---|---|
| GET | `/health` | status koneksi Neo4j |
| GET | `/tools` | daftar tool + JSON schema |
| POST | `/tools/{name}` | panggil tool |
| POST | `/query` | jawaban terverifikasi (`{"agentic": true}` → tool_trace) |
| POST | `/setup` | Phase 1 + Phase 3 |

### Catatan desain Phase 5

LLM **hanya memilih tool mana**, tidak pernah menggantikan gate. Kalau tool terakhir adalah `answer_question`, seluruh `VerifiedAnswer` dikembalikan apa adanya — teks yang ditampilkan adalah teks bergate, bukan parafrase model (yang bisa saja menghilangkan disclaimer). Kalau loop berakhir di `kg_stats`/`ingest_meeting`, tidak ada field gate yang dikarang.

Pengaman: whitelist + validasi JSON-schema sebelum eksekusi, satu kali re-prompt saat validasi gagal, batas 4 call per request, trace lengkap, dan deteksi kebocoran token kontrol (`<tool_call>`, `<think>`).

---

## 4. Dua knowledge graph yang berbeda

Ini sumber kebingungan paling besar, jadi dipisahkan eksplisit.

| | **Graph lama** (tesis, ~Jul–7 Agu) | **Graph sekarang** (rebuild 10 Agu) |
|---|---|---|
| Endpoint | `bolt://100.95.227.48:7687` | `bolt://100.110.179.78:7687` |
| Node | 335 | **4.551** |
| Dokumen | 1 tesis | **8 paper** |
| Chunk | 106 | 816 (semua ber-embedding 1024-dim) |
| Topic | 94 | 1.782 |
| Description | 99 | 1.818 (322 ber-embedding) |

**Konsekuensi kritis: seluruh evaluasi lama sudah tidak bisa direproduksi.** Snapshot fingerprint, node injeksi sintetis, dan ground truth 20 pertanyaan di `eval/eval_toolkit/` semuanya menunjuk ke graph yang sudah tidak ada.

### Struktur graph sekarang

```
Label:  Description 1818 · Topic 1782 · Chunk 816 · Agent 98 · Type 25 · Source 8 · Document 8 · Role 4
Relasi: MENTIONS 7224 · HAS_TYPE 1598 · HAS_DESCRIPTION 1449 · HAS_SUBTOPIC 1028
        PART_OF 816 · NEXT 808 · RELATES_TO 36 · HAS_SOURCE 22 · ROLE_IN_PAPER 22
Index vektor: `vector` (Chunk) · `description_vector` (Description, hanya 322/1818 ter-embed)
Embedding: mxbai-embed-large, chunk_size 1000, overlap 100
```

Atribusi per-paper **hanya andal lewat lapisan Chunk** (`Chunk.filename`, 816/816 terisi). Lewat Topic praktis tidak ada — `HAS_SOURCE` hanya menghubungkan 21 dari 1.782 Topic.

---

## 5. Korpus 8 paper

| ID | Paper | Topik | Chunk |
|---|---|---|---|
| P1 | Corbett, Montes-Sancho & Kirsch (2005) — *The Financial Impact of ISO 9000 Certification in the US* | ISO 9000 → financial performance | 86 |
| P2 | Martínez-Costa et al. (2009) — *ISO 9000/1994, ISO 9001/2000 and TQM* | ISO → performance & TQM | 120 |
| P3 | Franke & Kaul (1978) — *The Hawthorne Experiments: First Statistical Interpretation* | relay assembly experiment | 71 |
| P4 | Levitt & List (2009) — *Was there Really a Hawthorne Effect at the Hawthorne Plant?* | reanalisis illumination experiments | 42 |
| P5 | Brax et al. (2021) — *Explaining the servitization paradox* | servitization–performance | 129 |
| P6 | Neely (2008) — *Exploring the Financial Consequences of the Servitization of Manufacturing* | servitization & konsekuensi finansial | 93 |
| P7 | Flynn, Huo & Zhao (2010) — *The impact of supply chain integration on performance* | SCI & performance | 137 |
| P8 | Zhao, Feng & Wang (2015) — *Is more supply chain integration always beneficial?* | SCI, efek nonlinear | 138 |

> **Catatan identitas:** file bernama `Kohtamaki_2021_Servitization` berisi paper **Brax et al. (2021)**, bukan paper dengan Kohtamäki sebagai penulis utama. Selalu identifikasi paper lewat **judul**, bukan nama file.

**Kenapa korpus ini kuat untuk evaluasi:** membentuk empat pasangan **same-topic / different-author** yang temuannya berbeda — P1↔P2, P3↔P4, P5↔P6, P7↔P8. Karena topik tiap pasangan identik, kemiripan semantik saja tidak cukup untuk mendarat di paper yang benar. Inilah yang diuji.

---

## 6. Evaluasi lama (E1/E2/E3) — hasil dan masalahnya

Dijalankan terhadap **clone lokal** graph tesis dengan fault injection sintetis, N=20.

- **E1 Gate correctness** — profil final (Chunk/vector + parse-fix): **false-pass 0%, false-block 0%**
- **E2 Judge reliability** — `hermes3:3b` **Cohen's κ = 0.833** (akurasi 0.92) vs proxy leksikal κ = 0.167 → membenarkan pemakaian juri LLM alih-alih heuristik leksikal
- **E3 Ablation** — trust OFF → false-pass 57% · temporal OFF → 7% · **semua gate OFF → 100%**. Trust adalah pertahanan utama; tanpa verifikasi apa pun seluruh jawaban buruk lolos

### Masalah yang belum diperbaiki

Angka di `eval/eval_toolkit/EVALUATION_SUMMARY.md` **tidak cocok dengan artefak JSON dari run yang sama**:

| Klaim di ringkasan | Isi `results_*.json` |
|---|---|
| Final: 19/20 = 95%, kategori c 4/5 | **17/20 = 85%**, kategori c **2/5** |
| "satu-satunya error: c04" | error = **c02, c04, c05** |
| Desc/Type: 85% | **75%** |
| Chunk/vector: 80%, false-block 15% | **65%**, false-block 3/6 = **50%** |

Yang **tetap cocok** justru dua metrik headline-nya: false-pass 0% dan false-block 0% pada profil final. Jadi kesimpulannya tidak runtuh, tapi angka akurasi overall dan per-kategori harus dikoreksi ke angka artefak sebelum dipakai untuk tulisan tesis.

---

## 7. Evaluasi baru — paper suite

Suite baru di `eval/paper_suite/`, menggantikan toolkit lama.

### Pergeseran metodologi

Toolkit lama menilai **keputusan gate**. Suite ini menilai **provenance retrieval** lebih dulu — karena gate tidak bisa dinilai secara bermakna selama retrieval mendarat di paper yang salah.

| Metrik | Arti |
|---|---|
| `retrieval_hit` | minimal satu paper yang diharapkan menyumbang chunk (syarat perlu) |
| `top_paper_correct` | paper penyumbang chunk terbanyak adalah yang diharapkan (uji ketat) |
| `chunk_precision` | proporsi chunk dari paper yang diharapkan |
| `coverage` | proporsi paper yang diharapkan yang benar-benar tersentuh — **metrik utama untuk multi-source** |
| `evidence_present` | cek leksikal; **sinyal skrining untuk review manusia, bukan vonis benar/salah** |
| `cross_paper_entities` | Topic yang namanya diklaim >1 paper |

Cek evidence punya tiga mode: `expected_evidence` (semua harus muncul), `expected_evidence_any` (salah satu cukup — untuk klaim yang boleh diparafrase), `forbidden_evidence` (tidak boleh muncul — kriteria negatif untuk item faithfulness dan insufficient-evidence).

### Prasyarat

```bash
KG_CHUNK_SOURCE_PROP=filename     # WAJIB, tanpa ini semua metrik provenance kosong
KG_ENTITY_LABEL=Topic
KG_CHUNK_LABEL=Chunk
KG_CHUNK_TO_ENTITY_PATTERN=(c:{chunk_label})-[:MENTIONS]->(e:{entity_label})
KG_EMBED_MODEL=mxbai-embed-large
KG_LLM_MODEL=hermes3:3b           # non-thinking
```

Ollama yang punya `hermes3:3b` + `mxbai-embed-large` adalah **localhost**, bukan `100.118.203.111` (server itu hanya punya keluarga qwen3).

---

## 8. Hasil Level 1 — direct retrieval (D01–D08)

```
OK [D01] want=P1  got=P1:12,P2:3   prec=0.80  evid=Y
XX [D02] want=P1  got=P1:9, P2:6   prec=0.60  evid=n   <- gagal nyata
OK [D03] want=P7  got=P7:5         prec=1.00  evid=Y
OK [D04] want=P7  got=P7:13,P8:2   prec=0.87  evid=Y
OK [D05] want=P3  got=P3:5         prec=1.00  evid=Y
OK [D06] want=P4  got=P4:12,P3:3   prec=0.80  evid=Y
XX [D07] want=P6  got=P5:12,P6:3   prec=0.20  evid=n   <- gagal nyata
OK [D08] want=P5  got=P5:5         prec=1.00  evid=Y
```

| Metrik | Nilai |
|---|---|
| retrieval_hit_rate | **1.00** |
| top_paper_accuracy | **0.875** (7/8) |
| mean_chunk_precision | 0.783 |
| mean_coverage | 1.00 |
| evidence_present_rate | 0.75 |
| forbidden_claims | **0** |
| mean_faithfulness | 0.694 |
| gate_passed_rate | 0.00 *(lihat §11)* |

### Adjudikasi

**D02 — kegagalan nyata (dalam-paper).** Paper benar (P1 di atas), tapi jawaban hanya mendeskripsikan *metode* (event study, 1987–1997, SIC 2000–3999) dan tidak pernah menyebut ROA/ROS. Diverifikasi ke graph: **31 chunk Corbett mendefinisikan ROA/ROS/Tobin's Q** — datanya ada, tapi 9 chunk yang ter-retrieve bukan yang itu. Ini kegagalan *relevansi chunk di dalam paper yang benar*.

**D06 — bukan kegagalan agent, kegagalan kriteria.** Jawabannya substantif benar (*"existing descriptions of supposedly remarkable data patterns were entirely fictional... did not find any statistically significant changes"*). Cek awal menuntut kata harfiah "little"+"evidence", padahal parafrase itu sah. Harness diperbaiki (`expected_evidence_any` + `forbidden_evidence`), setelah itu D06 lolos.

**D07 — kegagalan nyata (antar-paper).** Retrieval mendarat di P5:12 vs P6:3, padahal "12 separate approaches" ada di chunk Neely.

---

## 9. Hasil Level 2 — multi-source, comparative, multi-hop

```
OK [M01] want=P1+P2   got=P1:10,P2:5              prec=1.00  cov=1.00
OK [M02] want=P3+P4   got=P4:4, P3:1              prec=1.00  cov=1.00
XX [M03] want=P5+P6   got=P5:3, P8:2              prec=0.60  cov=0.50  <- P6 hilang
OK [M04] want=P7+P8   got=P7:11,P8:4              prec=1.00  cov=1.00
OK [C01] want=P1+P2   got=P2:9, P1:6              prec=1.00  cov=1.00
OK [C02] want=P3+P4   got=P3:8, P4:7              prec=1.00  cov=1.00
OK [C03] want=P5+P6   got=P5:14,P6:1              prec=1.00  cov=1.00  (timpang 14:1)
OK [C04] want=P7+P8   got=P7:8, P8:7              prec=1.00  cov=1.00
XX [C05] want=6 paper got=P7:5,P8:4,P5:2,P2:2,P3:1,P1:1  prec=0.93  cov=0.83  <- P6 hilang
OK [H01] want=P7      got=P7:14,P8:1              prec=0.93  cov=1.00
OK [H02] want=P3      got=P3:4, P4:1              prec=0.80  cov=1.00
OK [H03] want=P5      got=P5:14,P6:1              prec=0.93  cov=1.00
```

| Metrik | Nilai |
|---|---|
| retrieval_hit_rate | **1.00** |
| top_paper_accuracy | **1.00** |
| mean_chunk_precision | **0.933** |
| mean_coverage | **0.944** |
| evidence_present_rate | 1.00 |
| forbidden_claims | **0** |
| mean_faithfulness | 0.764 |

L2 justru **lebih baik** dari L1. Pertanyaan komparatif menyebut kedua author sekaligus, sehingga query embedding berada di tengah kedua paper dan menarik keduanya — persis yang diinginkan. C02 (P3:8, P4:7) dan C04 (P7:8, P8:7) hampir seimbang sempurna.

> **Catatan tentang harness:** penanda `OK`/`XX` per baris awalnya hanya melihat `top_paper_correct` + evidence, belum memasukkan `coverage`, sehingga M03 sempat tercetak `OK` padahal coverage-nya 0.5. Metrik agregat `mean_coverage` tetap benar. Tabel di atas sudah dikoreksi manual.

---

## 10. Temuan utama: P6 (Neely) kelaparan retrieval

Agregat 20 pertanyaan (L1 + L2):

| Paper | Total chunk ter-retrieve | Muncul di berapa pertanyaan |
|---|---|---|
| P7 Flynn | 56 | 6 |
| P5 Brax | 50 | 6 |
| P1 Corbett | 38 | 5 |
| P2 Martinez | 25 | 5 |
| P4 Levitt | 24 | 4 |
| P3 Franke | 22 | 6 |
| P8 Zhao | 20 | 6 |
| **P6 Neely** | **5** | **3** |

**Setiap kegagalan retrieval di seluruh suite melibatkan P6:** D07 (paper salah), M03 (hilang), C05 (hilang), C03 (1 chunk vs 14), H03 (1 chunk).

### Diagnosis akar penyebab

**Bukan struktur graph.** Neely justru punya densitas MENTIONS **tertinggi** dari semua paper (3,7 rata-rata; 91 dari 93 chunk terhubung ke Topic), panjang chunk normal (823 karakter).

**Penyebabnya kompresi ruang embedding.** Skor similarity mentah untuk query D07:

```
top-40 chunk, rentang skor: 0.8387 (#1) ... 0.8151 (#24)   -> spread hanya 0.024
  #1  0.8387  Kohtamaki (P5)
  #2  0.8320  Neely (P6)     <- chunk yang benar ada di peringkat 2!
  #3  0.8291  Neely (P6)
  #4-#11      Kohtamaki x8
  #12 0.8196  Neely (P6)
  #13-#15     Kohtamaki x3
```

Rantai sebabnya:

1. `mxbai-embed-large` memberi skor nyaris identik untuk semua chunk setopik — spread 0,024 di 40 chunk teratas. Embedding **tidak bisa membedakan author atau spesifisitas**.
2. Threshold `KG_VECTOR_MIN_SCORE=0.78` jauh di bawah semuanya, jadi tidak menyaring apa pun di sini.
3. `fetch_k = max(top_k × 3, 12)` = **15**. Jendela 15 teratas lalu terisi oleh paper yang punya **lebih banyak chunk** — Brax (129) mengalahkan Neely (93).
4. Hasilnya persis `P5:12, P6:3` yang teramati.

Jadi chunk Neely yang benar **peringkat #2**, tapi konteks final (top_k=5) tetap didominasi Brax karena volume.

### Implikasi perbaikan

Masalahnya bukan retrieval yang salah total — `retrieval_hit_rate` 1.00 menunjukkan paper yang benar **selalu** ikut ter-retrieve, hanya kalah porsi. Jadi perbaikan yang proporsional:

- **Diversifikasi per-dokumen** (MMR atau kuota chunk maksimum per dokumen dalam jendela fetch) — paling langsung mengatasi dominasi volume
- **Filter/boost per-paper saat query menyebut author** — menangani D07 secara spesifik
- **Naikkan `fetch_k`** — murah, tapi hanya menggeser masalah
- **Hybrid keyword+vector** — nama author punya daya pisah leksikal yang tidak dimiliki embedding

Bukan penggantian arsitektur retrieval.

---

## 11. Blocker utama: Phase 1 belum jalan

Graph 8-paper belum pernah menerima Phase 1 — `trust_score`, `created_at`, dan `source_type` **kosong di semua 1.782 Topic**.

Akibatnya tiap Topic mendapat `0,5 (confidence default) × 0,4 (source_weight default) × 1,0 (recency)` = **trust 0,2**, di bawah `KG_MIN_TRUST_SCORE` 0,4.

**Artinya `passed` selalu false dan setiap jawaban kena disclaimer — terlepas benar atau salahnya isi.** Terkonfirmasi di seluruh 20 item (gate_passed_rate 0,00; mean_trust 0,200 persis).

Metrik retrieval, faithfulness, dan evidence **tidak terpengaruh** karena tidak bergantung pada trust — jadi semua hasil di §8–§10 tetap valid.

`--setup` / `POST /setup` menjalankan Phase 1 + Phase 3 yang **MENULIS** ke graph. Ini KG bersama, jadi butuh izin eksplisit pemiliknya. Seluruh operasi yang sudah dijalankan bersifat read-only.

---

## 12. Bug yang ditemukan lewat evaluasi

| # | Bug | Perbaikan | Status |
|---|---|---|---|
| 1 | Crash saat vector index tidak ada | Tangkap `ClientError`, fallback ke keyword | fixed |
| 2 | Salah-atribusi diam-diam — index meng-index `:Chunk` tapi profil menunjuk `:Description`; jawaban tampak sukses tapi `sources_used` kosong, trust 0.0 | `retrieve_vector` memfilter hasil ke `KG_CHUNK_LABEL` | fixed |
| 3 | Atribusi via `Type` singleton — satu Description ter-atribusi ke ~27 Topic → membanjiri gate trust → **false-pass 14%** | Pindah ke lapisan `Chunk/MENTIONS` | fixed |
| 4 | Jawaban kosong lolos gate — `qwen3-vl:4b` habis token untuk "thinking", juri memberi faithfulness **0,70** (pas lolos) untuk jawaban kosong | Sentinel eksplisit + short-circuit 0.0 tanpa memanggil juri | fixed |
| 5 | Parser faithfulness menolak JSON tak-berkurung | Parser toleran (full JSON / tak-berkurung / regex) — ini yang menghapus false-block | fixed |
| 6 | Tabrakan nama Topic antar-paper — `Table 3`, `Figure 1`, `Servitization` diklaim beberapa paper sekaligus, `sources_used` mencampur sumber tanpa jejak | Terdeteksi otomatis via `cross_paper_entities` | terdeteksi |

Bug #6 muncul di D04, D07, M04, C02, C04, C05, H01, H03 — 8 dari 20 pertanyaan. `Table 3` sendiri diklaim oleh 4 paper berbeda.

---

## 13. Fitur baru sesi ini: atribusi paper

Menambahkan `KG_CHUNK_SOURCE_PROP` (default kosong → perilaku lama persis tidak berubah). Saat diset, agent melaporkan:

- `documents_used` — `{name, chunks}` per dokumen, terurut paling banyak menyumbang
- `sources_used[].documents` — tiap entity dilacak ke dokumen asalnya

Tanpa ini, kolom "Expected source: P1" pada test suite tidak bisa di-score otomatis — agent hanya melaporkan nama Topic (`Table 3`, `Servitization Level`) tanpa jejak paper.

Perubahan di `kg_agent/config.py`, `kg_agent/agentic_verifier.py`, plus 4 test regresi di `tests/test_smoke.py` (43 → 47 test).

---

## 14. Deploy

- `docker-compose.prod.yml` — **API-only**, Neo4j & Ollama sebagai service eksternal, healthcheck, `restart: unless-stopped`
- `.env.docker.example` — API di container
- `.env.server.example` — API co-located di server dev `citi-cygnus` (100.118.203.111)

**Bug config, belum diperbaiki:** kedua template menunjuk `hermes3:3b` + `mxbai-embed-large` ke `100.118.203.111`, padahal server itu hanya punya `qwen3-embedding:0.6b`, `qwen3:4b`, `qwen3-vl:4b`, `qwen3:8b`. Deploy dengan config itu **akan gagal**. Ollama yang benar ada di localhost.

---

## 15. Status & pekerjaan terbuka

### Selesai
- Phase 1–5 lengkap, CLI/API/Docker
- 47 test lulus
- Atribusi paper (`KG_CHUNK_SOURCE_PROP`)
- Suite evaluasi baru: L1 (8 item) + L2 (12 item), keduanya dijalankan penuh read-only
- Diagnosis akar penyebab kelaparan retrieval P6

### Terbuka

| Item | Catatan |
|---|---|
| **Izin + Phase 1 di KG live** | blocker utama — tanpa ini gate tidak bisa dievaluasi |
| L3 temporal & trust (T01–T04, TR01) | **butuh menulis** metadata sintetis → harus di clone lokal, jangan di graph bersama |
| L3 insufficient/faithfulness/retry (IE01–IE05, F01–F03, RG01–RG02) | read-only, ground truth belum ditulis |
| Perbaiki retrieval P6 | diversifikasi per-dokumen atau hybrid keyword+vector |
| Koreksi angka `EVALUATION_SUMMARY.md` | lihat §6 |
| Perbaiki penanda OK/XX agar memasukkan `coverage` | lihat §9 |
| Perbaiki config deploy | lihat §14 |
| Adjudikasi manusia untuk item yang lolos cek leksikal | `evidence_present` hanya skrining |
| Inter-rater manusia E2 | lembar anotasi 137 baris siap, belum diisi 2 annotator |
| Keputusan KG target untuk `ingest_meeting` | tool ini MENULIS |

### Keterbatasan yang perlu disebut jujur di tulisan

- N kecil: 20 pertanyaan (8 L1 + 12 L2), 12 pasang juri di E2 — angka indikatif
- Kategori c/d di eval lama memakai **fault injection sintetis** — praktik standar untuk menguji mekanisme deteksi, tapi wajib dilabeli eksplisit dan tidak boleh disajikan sebagai distribusi data alami
- `evidence_present` adalah cek leksikal, bukan penilaian kebenaran
- Jawaban LLM tidak deterministik — D06 memberi teks berbeda antar-run meski kesimpulannya sama
- Evaluasi lama dan baru berjalan di **graph yang berbeda**; angkanya tidak boleh dibandingkan langsung

---

## 16. Artefak

```
kg_agent/                       kode inti (3.632 baris)
tests/test_smoke.py             47 test
eval/eval_toolkit/              evaluasi lama (E1/E2/E3) - graph tesis, TIDAK reproducible lagi
eval/paper_suite/
  papers.json                   peta P1-P8 -> filename
  ground_truth/
    level1_direct.jsonl         D01-D08
    level2_multisource.jsonl    M01-M04, C01-C05, H01-H03
  results/
    level1.json                 hasil L1 + jawaban penuh tiap item
    level2.json                 hasil L2 + jawaban penuh tiap item
  scripts/run_suite.py          runner + scoring
  README.md
docs/NABHYLA_CONNECT.md         runbook koneksi KG + aturan read-only
docs/PROJECT_SUMMARY.md         dokumen ini
```
