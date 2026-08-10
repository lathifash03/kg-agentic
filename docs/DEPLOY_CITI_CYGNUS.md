# Deploy kg-agent API di server dev (citi-cygnus, 100.118.203.111)

Endpoint final untuk Rio (module 5): **`http://100.118.203.111:8003`**
(port `8003` = "lab-brain-agent"; port 8000 sudah dipakai lab-brain-backend.)

API dijalankan **native (uvicorn) co-located dengan Ollama** di server — bukan
Docker — supaya sederhana dan menghindari bug seccomp Docker lama. Ollama diakses
`localhost:11434`; Neo4j diakses lewat Tailscale.

---

## Prasyarat (cek dulu — semua di luar kendali laptop)

1. **Akses terminal ke citi-cygnus.** SSH dari laptop Lathifah DITOLAK ACL tailnet
   (`tailnet policy does not permit you to SSH to this node`). Opsi: minta Arya
   buka ACL Tailscale SSH untuk user ini, ATAU Arya yang menjalankan runbook ini,
   ATAU akses via console server langsung.
2. **ACL tailnet mengizinkan port 8003** dari mesin Rio → server. (ACL yang sama
   yang memblok SSH bisa juga membatasi port; pastikan `:8003` dibuka.)
3. **Ollama server** punya model: `hermes3:3b` (answer-gen + juri) dan
   `mxbai-embed-large` (embedding retrieval). Kalau belum → `ollama pull` (langkah 4).
4. **Neo4j `100.110.179.78:7687` reachable DARI server** (bukan cuma dari laptop).
   Uji dengan `nc -z 100.110.179.78 7687` di server (langkah 5).
5. **Repo bisa di-clone di server.** `github.com/lathifash03/kg-agentic` — kalau
   private, siapkan Personal Access Token / deploy key, atau `git pull` kalau repo
   sudah ada di server.

---

## Langkah (jalankan DI citi-cygnus)

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
#    (tinjau isinya; NEO4J_URI harus bolt://100.110.179.78:7687)

# 4. Pastikan model ada di Ollama server
ollama pull hermes3:3b
ollama pull mxbai-embed-large

# 5. Sanity: server bisa lihat Neo4j?
nc -z 100.110.179.78 7687 && echo "neo4j OK"

# 6. Jalankan API di port 8003, tetap hidup (tmux — cara cepat)
tmux new -s kgagent -d './.venv/bin/uvicorn kg_agent.api:app --host 0.0.0.0 --port 8003'
sleep 5 && curl -s http://localhost:8003/health
#    harus: {"status":"ok","neo4j_connected":true}
```

## Verifikasi dari luar (laptop / mesin Rio)

```bash
curl http://100.118.203.111:8003/health
# {"status":"ok","neo4j_connected":true}  → endpoint hidup & bisa diberikan ke Rio

curl -X POST http://100.118.203.111:8003/query \
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
| `/health` | GET | — | `{status, neo4j_connected}` |
| `/query` | POST | `{"query": "..."}` | `{answer, passed, trust_score, faithfulness, sources_used[], temporal_status}` |
| `/tools` | GET | — | daftar tool |
| `/tools/{name}` | POST | args tool | hasil tool spesifik |

## Catatan penting yang masih terbuka

- **Disclaimer "low trust":** graph `100.110.179.78` belum dijalankan Phase 1, jadi
  node tak punya `source_type`/`confidence` → trust ~0.2 → tiap jawaban `passed=false`
  dengan disclaimer. Untuk menghilangkan → jalankan Phase 1 (MENULIS ke graph) —
  butuh keputusan izin tulis dulu.
- **Menulis ke KG (`ingest_meeting`, `--setup`):** orchestrator `off` secara default,
  jadi API hanya membaca. Aktifkan hanya setelah izin tulis jelas.
