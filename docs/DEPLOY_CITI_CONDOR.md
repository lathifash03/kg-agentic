# Migrasi kg-agent: citi-cygnus → citi-condor

Pindah ke `citi-condor` (100.122.56.39) supaya kg-agent satu server dengan Rio.

> **STATUS: SELESAI & LIVE per 2026-08-14.** kg-agent jalan sebagai systemd
> service `kg-agent` di condor, `0.0.0.0:8003`, `enabled` (start on boot).
> Terverifikasi dari luar via `http://100.122.56.39:8003` — lihat "Hasil" di bawah.

Kontrak API untuk Rio, aturan read-only, dan bentuk unit systemd tidak berubah:
lihat [DEPLOY_CITI_CYGNUS.md](DEPLOY_CITI_CYGNUS.md). Dokumen ini memuat yang
**berbeda** di condor + jejak deploy aktual.

## Neo4j — KG resmi ada di cygnus (bukan lagi laptop Wildan)

**Target final:** `bolt://100.118.203.111:7690` (user `sinyo`, read-only; browser
`http://100.118.203.111:7475`) — KG Wildan/Nabhyla sudah **dipindah dari laptop
Wildan ke server cygnus**. Sesama tailnet arya dengan condor, jadi terjangkau
tanpa urusan firewall. Kredensial di `~/kg-agent/.env`, bukan di sini. Ini yang
sudah dipakai service condor sejak deploy.

> Catatan historis: node lama `bolt://100.110.179.78:7687` (PC Windows
> `nigger`, `wildanfauzy4@`) **tidak dipakai lagi**. Dari condor host itu memang
> tak terjangkau di TCP (`tailscale ping` lolos, tapi TCP 7687/7474 di-drop
> firewall Windows — allow cygnus, belum condor). Karena KG sudah pindah ke
> cygnus, ini tidak perlu diperbaiki; dicatat hanya supaya tak ada yang mencoba
> mengarahkan condor ke IP itu lagi.

Introspeksi 2026-08-14:

    814 chunk · 1531 :Topic · 1739 MENTIONS · index `vector`
    c.embeddings_model = qwen3-embedding:0.6b   (COCOK — tak perlu mxbai)
    filename ada di 814 chunk (provenance/documents_used jalan)

Profil skema `.env.server.example` cocok apa adanya (label `Topic`, Chunk/
MENTIONS, `filename`). Kredensial Neo4j hanya ada di `~/kg-agent/.env` (mode 600),
tidak di repo.

## Kondisi condor per 2026-08-14

| Hal | Status |
|---|---|
| Ollama | 0.32.5, sehat. `hermes3:3b` (di-pull 2026-08-14), `qwen3-embedding:0.6b`, `qwen3:4b`, `qwen3-vl:4b` — 100% VRAM, ~195 tok/s |
| `hermes3:3b` | **DI-PULL & TERVERIFIKASI** — konten penuh 225 tok/s; juri faithfulness ter-parse 0.75 (bukan 0.0) |
| `qwen3-embedding:0.6b` | **TERVERIFIKASI** — `/api/embeddings` balik vektor 1024-dim; cocok dgn graph |
| Docker | 29.6.2, akses tanpa sudo — tersedia bila mau jalankan Neo4j dari dump di condor |
| Neo4j (kg-agent) | `bolt://100.118.203.111:7690`, terjangkau dari condor, `verify_connectivity: True` |
| Port 8003 | Dipakai service `kg-agent` (sebelumnya kosong; tak bentrok Supabase di 8080) |

## Hasil verifikasi (dari luar condor, via IP tailnet)

```
/health                    -> {"status":"ok","neo4j_connected":true,"read_only":true}
POST /tools/ingest_meeting -> HTTP 403   (read-only tegak)
POST /query                -> 1.9-2.5s, 15 sumber, faithfulness 0.75-0.95,
                              documents_used terisi (Kohtamaki_2021, Neely_2008)
```

## Kalau perlu deploy ulang / dari nol

Prasyarat 1 (hermes3:3b) sudah selesai lewat Ollama HTTP API dari jarak jauh.

## Prasyarat — kerjakan berurutan

### 1. `ollama pull hermes3:3b` — SUDAH SELESAI (2026-08-14)

Sudah di-pull ke condor lewat Ollama HTTP API (`POST /api/pull`, 24 dtk) dan
diverifikasi menghasilkan konten penuh (225 tok/s), juri faithfulness ter-parse
benar. Kalau perlu ulang:

```bash
ollama pull hermes3:3b     # ~2 GB
```

`KG_LLM_MODEL` dan `KG_JUDGE_MODEL` dua-duanya menunjuk `hermes3:3b`. Menggantinya
ke model yang sudah ada di condor akan **merusak pipeline, bukan sekadar
memperlambat**. Diuji langsung di condor dengan prompt sepele `"Say OK."`:

```
qwen3:4b       gen=64 tok @ 195 tok/s | content_len=0  | hit_cap=True
qwen3-vl:4b    gen=63 tok @ 180 tok/s | content_len=2  | hit_cap=False
```

`qwen3:4b` menghabiskan seluruh budget token untuk thinking dan mengembalikan
konten kosong — persis mode gagal yang sudah tercatat di
`kg_agent/agentic_verifier.py` (`_EMPTY_ANSWER_MESSAGE`). Di pipeline nyata
dengan `num_predict=512` hasilnya: jawaban kosong → faithfulness 0.0 → retry →
kosong lagi. `qwen3-vl:4b` lolos di prompt sepele ini, tapi sudah pernah
teramati mengembalikan konten kosong pada query multi-sumber sungguhan.

### 2. Pastikan condor bisa menjangkau Neo4j yang dipilih (TCP, bukan cuma ping)

```bash
# dari condor. Ganti host:port sesuai NEO4J_URI di .env.
timeout 8 bash -c '</dev/tcp/100.118.203.111/7690' && echo "bolt OK" || echo "exit=$?"
# exit 124 = timeout/di-drop firewall  ·  exit 1 = refused/tidak listen
```

**`tailscale ping` TIDAK cukup** untuk cek ini: ping lolos di layer WireGuard
sementara TCP tetap di-drop firewall host (persis yang terjadi pada Neo4j lama
`100.110.179.78` — lihat bagian "Neo4j" di atas). Uji TCP ke port bolt-nya.

## Deploy

```bash
# Kode ditransfer via tar-over-ssh (bukan git clone) supaya fix lokal yang
# BELUM ter-push (mis. KG_LLM_TIMEOUT) ikut. Dari mesin dev:
#   tar czf - --exclude=.git --exclude=.venv --exclude=.env --exclude=__pycache__ . \
#     | tailscale ssh citi@citi-condor 'mkdir -p ~/kg-agent && tar xzf - -C ~/kg-agent'
cd ~/kg-agent
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

cp .env.server.example .env
# lalu set NEO4J_URI/USERNAME/PASSWORD ke instance yang dipakai. Untuk deploy
# aktif: bolt://100.118.203.111:7690 (user & password dari catatan ops, JANGAN
# di repo). Tulis .env via stdin, jangan taruh password di argv — Rio ada di
# server ini: `umask 077; cat > ~/kg-agent/.env`.
# Sisanya sudah benar untuk condor apa adanya:
#   OLLAMA_URL=http://localhost:11434      KG_EMBED_URL=http://localhost:11434
#   KG_EMBED_MODEL=qwen3-embedding:0.6b    (cocok dgn graph)
#   KG_LLM_TIMEOUT=60      (plafon server 60x2x2 = 240 dtk < 300 dtk klien)
#   KG_READ_ONLY=true      (dipertahankan; /query jalan, hanya write yang 403)
```

Unit systemd yang terpasang: `/etc/systemd/system/kg-agent.service`, `User=citi`,
`WorkingDirectory=/home/citi/kg-agent`, `ExecStart=.venv/bin/uvicorn
kg_agent.api:app --host 0.0.0.0 --port 8003`, `Restart=always`, `enabled`.
Pemasangan butuh sudo (password citi lewat stdin `sudo -S`, bukan argv).

## Verifikasi setelah deploy

```bash
# 1. hidup dan terhubung
curl -s http://100.122.56.39:8003/health
# harap: {"status":"ok","neo4j_connected":true,"read_only":true}

# 2. read-only masih ditegakkan
curl -s -o /dev/null -w '%{http_code}\n' -X POST \
  http://100.122.56.39:8003/tools/ingest_meeting \
  -H 'Content-Type: application/json' -d '{"arguments":{"title":"tes"}}'
# harus 403

# 3. jawaban nyata + latensi
curl -s -w '\n%{time_total}s\n' -X POST http://100.122.56.39:8003/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"How many servitization strategies did Neely identify?"}'
# harap: answer terisi (BUKAN "did not produce an answer"), di bawah 10 detik
```

Kalau `answer` berisi `"The model did not produce an answer for this query"`,
`hermes3:3b` belum ter-pull dan konfigurasi jatuh ke model thinking — kembali ke
prasyarat 1.

## Jebakan embedding — baca sebelum pakai dump uji

`KG_EMBED_MODEL` **harus** sama dengan model yang dipakai saat chunk di-embed.
Korpus paper memakai `qwen3-embedding:0.6b`; snapshot graph tesis memakai
`mxbai-embed-large`. Keduanya 1024 dimensi, jadi salah tunjuk **tidak
memunculkan error sama sekali** — skor similarity ambruk jadi noise, retrieval
balik kosong, dan `/health` tetap hijau. Cek isinya dulu:

```cypher
MATCH (c:Chunk) RETURN DISTINCT c.embeddings_model, count(*)
```

Kalau hasilnya `mxbai-embed-large`, jalankan juga `ollama pull mxbai-embed-large`
di condor dan set `KG_EMBED_MODEL=mxbai-embed-large`.

## Menunggu: dump Neo4j untuk testing

Prosedur import sudah ada di
[eval/eval_toolkit/README.md](../eval/eval_toolkit/README.md) — Neo4j kedua di
port 7688 supaya instance korpus paper tidak tersentuh, lalu
`clone_graph.py --source file://<dump> --target bolt://localhost:7688`.

Begitu file dump-nya ada, yang perlu ditentukan: `c.embeddings_model` di dalamnya
(menentukan `KG_EMBED_MODEL` dan apakah perlu pull `mxbai-embed-large`), dan
apakah instance uji ini jalan di condor atau tetap di laptop.
