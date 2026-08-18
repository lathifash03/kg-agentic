# Deploy kg-agent API

Target saat ini: **citi-condor** (`100.122.56.39`), endpoint **`http://100.122.56.39:8003`**.
Port `8003` = "lab-brain-agent"; `8000` sering sudah dipakai lab-brain-backend.

Runbook ini tidak terikat satu mesin - ganti IP dan `APP_DIR` sesuai target.
Sebelumnya dokumen ini khusus citi-cygnus (`100.118.203.111`); langkahnya sama.

**Jalur yang disarankan: Docker.** Image sudah membawa dependensi, healthcheck,
dan `restart: unless-stopped`, serta berjalan sebagai non-root (`uid=10001`)
walaupun repo di-clone sebagai root. Jalur native (uvicorn + venv) tetap
didokumentasikan di bawah untuk server tanpa Docker - perhatikan bahwa venv
TIDAK bisa dipindah antar direktori (path absolut di shebang dan `pyvenv.cfg`),
jadi pindah lokasi berarti membuat ulang venv.

## Dua hal yang paling sering menggagalkan deploy ini

**1. Neo4j yang salah.** Sumber kebenaran adalah `bolt://100.118.203.111:7690`
(user `sinyo`) - 8 paper. Endpoint lama `bolt://100.110.179.78:7687` adalah
instance BERBEDA: korpus yang sama plus belasan transkrip meeting, terus
bertambah. Memakainya membuat jawaban bersumber dari graph yang salah tanpa
gejala apa pun. Verifikasi dengan menghitung **paper**, bukan total Document -
korpus juga memuat transkrip meeting dan Document fixture tanpa chunk:
`MATCH (d:Document) WHERE d.doc_type='paper' RETURN count(d)` -> harus **8**.

**2. Ollama tidak terjangkau dari container.** Di Docker Linux, Ollama yang
co-located hanya listen di `127.0.0.1` dan menolak koneksi dari container.
Gejalanya senyap: `/health` hijau, semua jawaban kosong.

| Ollama ada di | Berkas compose | `OLLAMA_URL` di `.env` |
|---|---|---|
| mesin yang sama (Linux) | `docker-compose.host.yml` | `http://localhost:11434` |
| mesin lain (tailscale) | `docker-compose.prod.yml` | `http://<ip>:11434` |

Selalu buktikan dengan `scripts/smoke_endpoint.py`, jangan berhenti di `/health`.

---


## Prasyarat (cek dulu — semua di luar kendali laptop)

1. **Akses terminal ke server target.** SSH dari laptop Lathifah DITOLAK ACL tailnet
   (`tailnet policy does not permit you to SSH to this node`). Opsi: minta Arya
   buka ACL Tailscale SSH untuk user ini, ATAU Arya yang menjalankan runbook ini,
   ATAU akses via console server langsung.
2. **ACL tailnet mengizinkan port 8003** dari mesin Rio → server. (ACL yang sama
   yang memblok SSH bisa juga membatasi port; pastikan `:8003` dibuka.)
3. **Ollama server** punya model: `hermes3:3b` (answer-gen + juri) dan
   `qwen3-embedding:0.6b` (embedding retrieval).
   - `qwen3-embedding:0.6b` sudah ada di citi-cygnus (server itu menyimpan
     keluarga qwen3), dan itu juga model yang meng-embed chunk di KG. **Di
     condor belum diverifikasi** — cek dulu, jangan diasumsikan.
   - `hermes3:3b` **belum ada** dan harus di-pull (langkah 4). Jangan diganti
     `qwen3:4b`/`qwen3:8b`/`qwen3-vl:4b` — keluarga "thinking" itu menghabiskan
     token budget untuk bernalar dan mengembalikan jawaban kosong pada konteks
     panjang; sudah diuji dan ditinggalkan (lihat commit `350d858`).
   - **JANGAN** pull `mxbai-embed-large` di sini. Runbook versi sebelumnya
     menyuruh begitu dan itu keliru: chunk di KG di-embed dengan qwen3, jadi
     memakai mxbai membuat retrieval balik kosong **tanpa error apa pun**
     (dua-duanya 1024 dim, index tidak pernah protes).
4. **Neo4j `100.118.203.111:7690` reachable DARI server** (bukan cuma dari laptop).
   Uji dengan `nc -z 100.118.203.111 7690` di server (langkah 5).
5. **Repo bisa di-clone di server.** `github.com/lathifash03/kg-agentic` — kalau
   private, siapkan Personal Access Token / deploy key, atau `git pull` kalau repo
   sudah ada di server.

---

## Langkah native/uvicorn (jalankan DI server target)

```bash
# 1. Ambil kode
cd ~ && git clone https://github.com/lathifash03/kg-agentic.git kg-agent && cd kg-agent
#    (kalau sudah ada:  cd ~/kg-agent && git pull)

# 2. venv + dependencies
python3 -m venv .venv
./.venv/bin/pip install -U pip
./.venv/bin/pip install -r requirements.txt

# 3. .env — profil server (Ollama=localhost, Neo4j via tailscale, hermes3)
cp .env.server.example .env
#    (tinjau isinya; NEO4J_URI harus bolt://100.118.203.111:7690, user sinyo)

# 4. Pastikan model ada di Ollama server
ollama pull hermes3:3b          # answer-gen + juri; belum ada di server ini
ollama list | grep qwen3-embedding   # embedding: harus SUDAH ada, jangan pull mxbai

# 5. Sanity: server bisa lihat Neo4j?
nc -z 100.118.203.111 7690 && echo "neo4j OK"

# 5b. Sanity embedding — ini yang kemarin gagal senyap. Model di .env HARUS
#     sama dengan yang tersimpan di chunk. Query berikut harus mengembalikan
#     tepat satu baris, berisi qwen3-embedding:0.6b.
./.venv/bin/python -c "
from kg_agent.config import get_config
from kg_agent.neo4j_client import Neo4jClient
cfg = get_config()
with Neo4jClient.from_config(cfg) as c:
    print('graph :', c.run_read('MATCH (c:Chunk) RETURN DISTINCT c.embeddings_model AS m, count(*) AS n'))
    print('.env  :', cfg.retrieval.embed_model)
"

# 6. Jalankan API di port 8003, tetap hidup (tmux — cara cepat)
tmux new -s kgagent -d './.venv/bin/uvicorn kg_agent.api:app --host 0.0.0.0 --port 8003'
sleep 5 && curl -s http://localhost:8003/health
#    harus: {"status":"ok","neo4j_connected":true}
```

## Verifikasi dari luar (laptop / mesin Rio)

```bash
curl http://100.122.56.39:8003/health
# {"status":"ok","neo4j_connected":true}  → endpoint hidup & bisa diberikan ke Rio

curl -X POST http://100.122.56.39:8003/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"What is supply chain integration?"}'
```

---

## Selalu-hidup lintas reboot (opsional, disarankan untuk meeting): systemd

Ganti `<user>` dengan username server. Buat `/etc/systemd/system/kg-agent.service`:

```ini
[Unit]
Description=kg-agent API (module 5 verifier)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/home/<user>/kg-agent
EnvironmentFile=/home/<user>/kg-agent/.env
ExecStart=/home/<user>/kg-agent/.venv/bin/uvicorn kg_agent.api:app --host 0.0.0.0 --port 8003
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now kg-agent
systemctl status kg-agent
curl -s http://localhost:8003/health
```

---

## Kontrak API untuk Rio

| Endpoint | Method | Body | Hasil |
|---|---|---|---|
| `/health` | GET | — | `{status, neo4j_connected, read_only}` |
| `/query` | POST | `{"query": "..."}` | `{answer, passed, trust_score, faithfulness, overall_confidence, sources_used[], documents_used[], temporal_validity_status, explanation, disclaimer, retries, strategy}` |
| `/tools` | GET | — | `{tools[], read_only}`; tiap tool punya flag `writes` |
| `/tools/{name}` | POST | `{"arguments": {...}}` | hasil tool; **403** kalau tool-nya menulis dan read-only aktif |
| `/setup` | POST | — | **403** saat read-only (memang begitu maunya) |

Yang perlu Rio tahu tentang bentuk jawabannya:

- **`passed` akan `false` untuk semua pertanyaan, dan itu bukan bug integrasi.**
  Node di graph tidak punya `source_type`/`confidence_score`, sehingga trust
  konstan 0.2 di bawah ambang 0.4. Field `answer` tetap terisi penuh dan
  `faithfulness` tetap bermakna — jangan perlakukan `passed=false` sebagai
  kegagalan panggilan.
- **`answer` sudah memuat disclaimer** di ujungnya (`\n\n[!] Unverified: ...`)
  saat gate tidak lolos. Kalau Rio mau merender disclaimer terpisah, pakai field
  `disclaimer` dan potong bagian itu dari `answer`.
- **Korpus lengkap 8 paper** (ISO, Hawthorne, servitization, supply chain
  integration). Satu cacat tersisa: P1 Corbett kehilangan chunk 50 dan 83
  (`Document.n_chunks` menyebut 86, isinya 84) - menunggu re-ingest.
- **Pertanyaan di luar korpus tidak dijawab "tidak tahu".** Diukur pada 5 item
  insufficient-evidence: agent hanya menolak di 1 dari 5, dan 3 karangan LOLOS
  gate. Jangan sajikan `answer` ke pengguna akhir tanpa menampilkan
  `documents_used` - kalau kosong, jawabannya tidak bersumber dari korpus.
- **Latensi terukur 0,5–7 detik**, bukan 40–180 detik seperti versi lama catatan
  ini. Angka lama berasal dari asumsi inferensi CPU yang **sudah tidak berlaku**
  di server ini: `/api/ps` menunjukkan `hermes3:3b` dilayani Ollama 100% di VRAM
  (`size_vram == size`), terukur ~4.900 tok/s prompt eval dan ~245 tok/s
  generasi. Jalur retry (`retries: 1`) pun selesai 3,5 detik.
  Sampling 2026-08-14, 5 query, model sudah warm, satu klien. **Konkurensi belum
  diuji** — Ollama menyerialisasi request, jadi wall time ikut naik seiring
  kedalaman antrean kalau Rio menembak paralel.
- **Timeout klien tetap 300 detik** — jangan diturunkan. Itu bukan perkiraan
  latensi, melainkan pasangan dari plafon sisi server: `KG_LLM_TIMEOUT=60`
  berlaku **per panggilan LLM**, dan satu `/query` memanggil LLM sampai 4x
  (2 per attempt x 2 attempt), jadi server dibatasi `60 x 2 x 2 = 240` detik —
  tepat di bawah 300. Kalau plafon server melewati budget klien, klien menyerah
  duluan sementara server terus membakar worker dan slot Ollama untuk request
  yang sudah tidak ditunggu siapa pun.
- Untuk UX: harapkan **di bawah 10 detik**, jangan bangun spinner 3 menit. 300
  detik itu batas aman untuk kasus patologis, bukan waktu tunggu normal.

## Mode read-only — jangan dimatikan tanpa izin tulis

`KG_READ_ONLY=true` (default) menolak setiap tool yang menulis dengan 403,
ditegakkan di `call_tool` — satu titik sempit yang dilewati API maupun
orchestrator. Verifikasi setelah deploy:

```bash
curl -s http://100.122.56.39:8003/health          # "read_only": true
curl -s -o /dev/null -w '%{http_code}\n' -X POST \
  http://100.122.56.39:8003/tools/ingest_meeting \
  -H 'Content-Type: application/json' -d '{"arguments":{"title":"tes"}}'
# harus 403
```

`KG_ORCHESTRATOR=off` **bukan** pengaman yang setara — flag itu hanya mencegah
LLM memilih tool. `POST /tools/ingest_meeting` langsung, dan `POST /query`
dengan `{"agentic": true}`, dua-duanya melewatinya. `KG_READ_ONLY` yang menutup
keduanya.

## Jalur Docker

Base image sekarang `python:3.11-slim-bullseye`, jadi bug seccomp/`clone3` yang
dulu menggagalkan build di Docker lama **sudah tidak berlaku** — build diuji
berhasil di Docker 20.10.2. Tidak perlu upgrade Docker.

Yang sudah diverifikasi end-to-end pada image ini (di macOS):

| | |
|---|---|
| `docker build` | lulus, image 172 MB |
| Container start + `/health` | `{"status":"ok","neo4j_connected":true,"read_only":true}` |
| `HEALTHCHECK` bawaan image | `healthy` |
| Berjalan sebagai non-root | `uid=10001(kgagent)` |
| Retrieval nyata | 2 dokumen / 7 sumber |
| Guard tulis | `ingest_meeting` → 403 |
| `smoke_endpoint.py` | `VERDICT: WORKING` |

```bash
cp .env.docker.example .env     # WAJIB tinjau: NEO4J_URI (harus :7690/sinyo), OLLAMA_URL
docker compose -f docker-compose.prod.yml up -d --build
docker compose -f docker-compose.prod.yml exec agent \
    python scripts/smoke_endpoint.py --url http://localhost:8000
```

### Pilih berkas compose sesuai letak Ollama

Sudah disiapkan dua berkas, jadi tidak perlu mengedit YAML di server:

| Ollama ada di | Perintah | `.env` |
|---|---|---|
| mesin yang sama (Linux) | `docker compose -f docker-compose.host.yml up -d --build` | ketiga URL Ollama = `http://localhost:11434` |
| mesin lain (tailscale) | `docker compose -f docker-compose.prod.yml up -d --build` | ketiga URL Ollama = `http://<ip>:11434` |

`docker-compose.host.yml` memakai `network_mode: host`, membind uvicorn langsung
ke 8003, dan **menimpa HEALTHCHECK bawaan image** — yang bawaan menembak
`localhost:8000`, sehingga tanpa penimpaan container akan selamanya `unhealthy`
padahal sehat.

Opsi ketiga (`OLLAMA_HOST=0.0.0.0` + bridge) memang bisa, tapi mengekspos Ollama
ke seluruh tailnet — hindari kecuali memang diinginkan.

Apa pun pilihannya, buktikan dengan `smoke_endpoint.py` — kalau Ollama tidak
terjangkau, `/health` tetap hijau sementara setiap jawaban balik kosong.

## Menambah endpoint baru untuk Rio

Tidak perlu menyentuh `api.py`. Tulis fungsi di `kg_agent/tools.py`, daftarkan
di dict `TOOLS` dengan `description`, `parameters`, dan **`writes`** (wajib —
`tool_writes` sengaja gagal keras kalau flag-nya lupa, supaya tool baru tidak
diam-diam lolos sebagai read-only). Tool itu langsung muncul di `GET /tools` dan
bisa dipanggil lewat `POST /tools/<nama>`, sekaligus otomatis tersedia untuk
orchestrator.
