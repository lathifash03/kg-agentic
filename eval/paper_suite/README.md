# Paper suite — evaluasi kg-agent terhadap KG 8-paper

Suite evaluasi untuk knowledge graph berisi **8 paper** (rebuild Agustus 2026,
`bolt://100.110.179.78:7687`). Menggantikan `eval/eval_toolkit/`, yang menilai
graph tesis 335-node yang **sudah tidak ada lagi** — snapshot fingerprint dan
node injeksinya tidak bisa direproduksi terhadap graph sekarang.

## Kenapa korpus ini bagus untuk diuji

8 paper membentuk **empat pasangan same-topic/different-author** yang temuannya
berbeda:

| Pasangan | Topik | Ketegangan |
|---|---|---|
| P1 Corbett ↔ P2 Martinez-Costa | ISO 9000/9001 | abnormal financial gain vs tidak outperform |
| P3 Franke & Kaul ↔ P4 Levitt & List | Hawthorne | managerial discipline vs sedikit bukti efek konvensional |
| P5 Brax ↔ P6 Neely | Servitization | penjelasan konfigurasional vs bukti finansial agregat |
| P7 Flynn ↔ P8 Zhao | Supply chain integration | asosiasi positif vs inverted-U |

Karena tiap pasangan **topiknya identik**, retrieval berbasis kemiripan semantik
saja tidak cukup untuk mendarat di paper yang benar. Itulah yang diukur suite ini.

> Catatan identitas paper: file `Kohtamaki_2021_Servitization` berisi paper
> **Brax et al. (2021)**, bukan paper dengan Kohtamäki sebagai penulis utama.
> Selalu identifikasi paper lewat **judul**, bukan nama file. Peta lengkap ada
> di `papers.json`.

## Yang diukur (berbeda dari toolkit lama)

Toolkit lama menilai **keputusan gate**. Suite ini menilai **provenance
retrieval** lebih dulu, karena D07 membuktikan gate tidak bisa dinilai selama
retrieval mengambil dari paper yang salah:

| Metrik | Arti |
|---|---|
| `retrieval_hit` | minimal satu paper yang diharapkan menyumbang chunk (syarat perlu) |
| `top_paper_correct` | paper yang paling banyak menyumbang chunk adalah yang diharapkan (uji ketat) |
| `chunk_precision` | proporsi chunk yang berasal dari paper yang diharapkan |
| `coverage` | proporsi paper yang diharapkan yang benar-benar tersentuh — metrik utama untuk M/C |
| `evidence_present` | cek leksikal kasar; **sinyal skrining untuk review manusia**, bukan vonis benar/salah |
| `cross_paper_entities` | Topic yang namanya diklaim >1 paper (mis. `Table 3`) |

## Prasyarat

```bash
KG_CHUNK_SOURCE_PROP=filename   # WAJIB - tanpa ini semua metrik provenance kosong
KG_ENTITY_LABEL=Topic
KG_CHUNK_LABEL=Chunk
KG_CHUNK_TO_ENTITY_PATTERN=(c:{chunk_label})-[:MENTIONS]->(e:{entity_label})
KG_EMBED_MODEL=mxbai-embed-large     # harus sama dengan model yang meng-embed chunk
KG_LLM_MODEL=hermes3:3b              # non-thinking; qwen3-vl mengosongkan jawaban
```

Ollama yang punya `hermes3:3b` + `mxbai-embed-large` adalah **localhost**, bukan
`100.118.203.111` (server itu hanya punya keluarga qwen3).

## Menjalankan

```bash
python eval/paper_suite/scripts/run_suite.py \
    --gt ground_truth/level1_direct.jsonl --out results/level1.json

python eval/paper_suite/scripts/run_suite.py --only D07   # satu item
```

Level 1 dan 2 **read-only** — aman terhadap graph bersama.

## Status per level

| Level | Isi | Status |
|---|---|---|
| L1 direct | D01–D08 | siap, read-only |
| L2 multi-source/comparative/multi-hop | M01–M04, C01–C05, H01–H03 | siap, read-only |
| L3 temporal/trust | T01–T04, TR01 | **butuh menulis** metadata sintetis → harus di clone lokal, jangan di graph bersama |
| L3 insufficient/faithfulness/retry | IE01–IE05, F01–F03, RG01–RG02 | read-only, belum ditulis |

## Peringatan penting: Phase 1 belum jalan

Graph ini belum pernah menerima Phase 1 — `trust_score`, `created_at`, dan
`source_type` kosong di **semua** 1.782 Topic. Akibatnya tiap Topic mendapat
`0.5 (confidence default) × 0.4 (source_weight default) × 1.0 (recency)` =
**trust 0.2**, di bawah `KG_MIN_TRUST_SCORE` 0.4. Jadi **`passed` selalu false
dan setiap jawaban kena disclaimer**, tanpa memandang benar atau tidaknya isi.

Konsekuensi untuk evaluasi: `gate_passed_rate` di L1/L2 **tidak informatif**
sampai Phase 1 dijalankan (butuh izin pemilik KG). Metrik retrieval, faithfulness,
dan evidence tetap valid karena tidak bergantung pada trust.
