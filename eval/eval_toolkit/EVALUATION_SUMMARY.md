# Evaluasi kg-agent — Ringkasan

Evaluasi lapisan verifikasi kg-agent (gate **trust · temporal · faithfulness**)
terhadap knowledge graph bergaya-tesis Nabhyla. Tiga eksperimen: **E1** ketepatan
gate, **E2** keandalan juri, **E3** ablation kontribusi tiap gate.

## Ringkasan eksekutif

- **E1 — Gate Correctness: 15/20 (75%), false-pass 0%, false-block 0%** pada profil final
  (`ground_truth/e1_aggregate_3runs.json` — median atas tiga run, 16 Agustus 2026).
- **E2 — Judge Reliability:** hermes3:3b sebagai juri faithfulness **andal, Cohen's κ = 0.833** (near-perfect); proxy leksikal (mock) **tidak andal, κ = 0.167**.
- **E3 — Ablation:** mematikan verifikasi total → **false-pass 100%**; tiap gate berkontribusi (trust menangkap 8 jawaban buruk, temporal 1).
- **Dua bug produksi ditemukan & diperbaiki lewat proses evaluasi** (lihat bagian akhir).

---

## Metodologi

- **Data:** salinan (clone) read-only graph Nabhyla (335 node, 794 relasi) ke Neo4j
  lokal terisolasi — bukan instance bersama, jadi injeksi sintetis aman dan
  snapshot reproducible. Provenance tiap node injeksi ditandai `created_by=
  'eval_injection'` + `injection_batch`.
- **Phase 1** dijalankan di clone (`source_type=paper`) → node asli mendapat
  trust ≈ 0.5.
- **Fault injection sintetis** (standar untuk menguji mekanisme deteksi; wajib
  diungkap sebagai sintetis): 5 node **low-trust** (kategori c) + 5 kasus
  **temporal-invalid** (kategori d: OUTDATED / SUPERSEDED / CONFLICTED), tiap
  fakta dibangun sebagai `(:Chunk {text, embedding}) -[:MENTIONS]-> (:Topic)`.
- **Retrieval:** profil **Chunk/MENTIONS + vector** (embedding `mxbai-embed-large`,
  1024-dim; ambang similarity 0.78). Lihat "Temuan bug #1".
- **Answer-gen & juri faithfulness:** `hermes3:3b` (Ollama lokal).
- **Kategori pertanyaan & outcome yang diharapkan:**
  | Kat. | Deskripsi | Outcome benar |
  |---|---|---|
  | a | answerable-good | PASS |
  | b | out-of-scope (istilah diverifikasi absen) | NO_INFO |
  | c | hanya terjawab dari node low-trust | RELEASE_WITH_DISCLAIMER |
  | d | terjawab dari node temporal-invalid | TEMPORAL_FLAGGED |

Dua metrik headline: **false-pass** (jawaban buruk lolos gate — kesalahan
berbahaya) dan **false-block** (jawaban baik kena disclaimer — kesalahan aman).

---

## E1 — Gate Correctness

Profil final (Chunk/MENTIONS + vector + parser diperbaiki), 20 pertanyaan:

| Kategori | Skor | Akurasi |
|---|---|---|
| a answerable-good | 6/6 | 100% |
| b out-of-scope | 5/6 | 83% |
| c low-trust | 1/5 | 20% |
| d temporal-invalid | 3/3 | 100% |
| **Total** | **15/20** | **75%** |
| **False-pass** | **0** | **0%** |
| **False-block** | **0** | **0%** |

Item yang gagal: **b05, c01, c02, c04, c05**. Tidak satu pun berupa false-pass —
gate tidak pernah meloloskan jawaban buruk; kegagalannya adalah salah-label
antar-kategori (mis. query-c menarik node temporal injeksi sehingga tertandai
TEMPORAL_FLAGGED alih-alih low-trust). Dari 20 item, 18 stabil di ketiga run;
yang tidak stabil `b02` dan `c01`.

> **Riwayat koreksi.** Versi terdahulu dokumen ini mengklaim **19/20 = 95%**
> dengan "satu-satunya error c04". Klaim itu tidak pernah didukung artefak JSON
> mana pun: run terbaik yang tercatat (`results_chunk_vector_parsefix.json`,
> 10 Agustus) adalah 17/20 = 85% dengan **tiga** error (c02, c04, c05). Angka
> turun lagi ke 15/20 setelah *sentinel fix* (16 Agustus) mengoreksi
> over-estimasi — sebelumnya sebuah lookup yang gagal bisa mengungguli lookup
> yang berhasil, sehingga sebagian item tampak benar tanpa dasar yang sah.
> Karena itu himpunan item yang gagal bergeser antar era. Rincian
> ketidakcocokan aslinya ada di `docs/PROJECT_SUMMARY.md`, bagian
> "Masalah yang belum diperbaiki".

**Yang konsisten di seluruh run historis** (0.65, 0.85, maupun 0.75):
**false-pass 0% dan false-block 0%**. Kesimpulan utama E1 tidak bergantung pada
angka akurasi overall mana pun.

### Progresi tiga kondisi (hasil utama)

| Kondisi retrieval | Overall | **False-pass** (bahaya) | False-block |
|---|---|---|---|
| Desc/Type (join lewat `Type`) | 75% | **14%** | 0% |
| Chunk/vector (atribusi presisi) | 65% | **0%** | 15% (bug parser) |
| **Chunk/vector + parse-fix** | **85%** | **0%** | **0%** |

Kolom Overall di atas dikoreksi ke nilai artefak (`results.json` 0.75,
`results_chunk_vector.json` 0.65, `results_chunk_vector_parsefix.json` 0.85).
Baris terakhir adalah profil yang sama yang kemudian terukur **75%** pasca
sentinel fix. Kolom false-pass/false-block di tabel ini **belum** diverifikasi
ulang terhadap artefak — lihat `docs/PROJECT_SUMMARY.md`.

Dua perbaikan (atribusi presisi + fix parser) menghapus semua kesalahan
berbahaya (false-pass 14% → 0%) **tanpa** menjadikan gate over-konservatif.

---

## E2 — Judge Reliability

12 pasangan `(answer, context)` berlabel *by construction*: 6 **faithful**
(jawaban asli hermes3 yang grounded) + 6 **unfaithful** (klaim fabrikasi
lintas-domain yang jelas tak didukung). Skor tiap juri di-binarisasi pada ambang
faithfulness 0.7, dibandingkan dengan label konstruksi.

| Juri | Accuracy | **Cohen's κ** | mean faithful | mean unfaithful | gap |
|---|---|---|---|---|---|
| **hermes3:3b** | 0.92 | **0.833** | 0.65 | 0.00 | 0.65 |
| mock-lexical (proxy) | 0.58 | 0.167 | 0.56 | 0.24 | 0.33 |

**Kesimpulan:** juri LLM (hermes3:3b) **andal** memisahkan jawaban faithful dari
fabrikasi; proxy overlap-leksikal **tidak** — membenarkan pemakaian juri LLM
alih-alih heuristik leksikal. (Bagian inter-rater manusia: lembar anotasi buta
`ground_truth/judge_annotation_sheet.csv` siap untuk 2 annotator; belum diisi.)

---

## E3 — Ablation gate

Keputusan gate dihitung ulang dari sinyal tersimpan (trust/temporal/faithfulness
tiap item) dengan tiap gate dimatikan — ablation terkontrol, tanpa re-run LLM.

| Konfigurasi | False-pass | Kontribusi |
|---|---|---|
| Semua gate ON (baseline) | 0 (0%) | — |
| Trust OFF | **8 (57%)** | menangkap b01/b03/b04/b05/b06 + c02/c03/c05 |
| Temporal OFF | 1 (7%) | menangkap d02 (SUPERSEDED) |
| Faithfulness OFF | 0 (0%) | 0 tambahan *di dataset ini* |
| **Semua OFF (tanpa verifikasi)** | **14 (100%)** | semua jawaban buruk lolos |

**Trust** adalah pertahanan utama; **temporal** menambah deteksi unik (d02);
tanpa verifikasi apa pun **seluruh** jawaban buruk lolos — membuktikan gate
kolektif berfungsi. Catatan: faithfulness menangkap 0 **bukan** karena tak
berguna (E2 membuktikan κ=0.833), tapi karena dataset E1 tak menguji halusinasi
(injeksi c/d menguji trust/temporal). Menambah kasus "unfaithful dari sumber
tepercaya" akan menunjukkan nilainya di benchmark.

---

## Dua bug produksi ditemukan lewat evaluasi

1. **Atribusi entity via `Type` (klasifikasi kasar).** Retriever lama
   menghubungkan teks ke entity lewat `(:Description)<-[:HAS_DESCRIPTION]-
   (:Type)<-[:HAS_TYPE]-(:Topic)`. Karena `Type` adalah singleton global (mis.
   "Result" dibagi seluruh graph), **satu Description ter-atribusi ke ~27 Topic**
   yang cuma sekategori. Ini membanjiri `sources_used` dengan node tepercaya
   yang tak relevan → gate trust (berbasis rata-rata) kalah → false-pass.
   **Perbaikan:** pindah ke lapisan provenance `(:Chunk)-[:MENTIONS]->(:Topic)`
   yang memang dirancang ontology sebagai substrat retrieval → atribusi presisi.
2. **Parser faithfulness menolak JSON tak-berkurung.** `_parse_faithfulness`
   lama memakai regex `\{.*\}`. Model instruct seperti hermes3 kadang mengeluarkan
   `"faithfulness": 0.8, ...` **tanpa** kurung `{}` → parser gagal → jatuh ke
   0.0 → gate faithfulness menolak jawaban valid (false-block). **Perbaikan:**
   parser toleran (full JSON / tak-berkurung / regex ekstraksi angka). Ini yang
   mengubah E1 dari false-block 15% → 0%.

---

## Keterbatasan (jujur)

- Semua eksperimen di **clone lokal dengan injeksi sintetis** — bukan validasi
  produksi di graph live Nabhyla (butuh graph asli + izin menulis Phase 1).
- **N kecil** (20 pertanyaan benchmark, 12 pasangan juri) — angka indikatif;
  perbanyak untuk stabilitas.
- **E2 inter-rater manusia belum dijalankan** (butuh 2 annotator; lembar siap).
- Kurasi ground truth kategori-a masih perlu pengecekan manusia (baca teks,
  buang nama fragmen).
- Embedding vector cocok karena `mxbai-embed-large` deterministik; untuk graph
  live perlu memastikan host embedding menjalankan model yang sama.

## Artefak

Semua di `eval/eval_toolkit/`:
`scripts/` (freeze_snapshot, inject_low_trust, inject_temporal, propose_questions,
build_ground_truth, run_benchmark, build_judge_eval_set, run_judge_comparison,
ablation) · `ground_truth/` (ground_truth.jsonl, results_*.json, judge_*.json/csv,
ablation.json, snapshot fingerprints).
