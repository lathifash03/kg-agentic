# Ground-Truth Toolkit — E1 (Gate Correctness)

Builds the labeled dataset for evaluating whether kg-agent's gates make the
RIGHT decision per question category. Labels describe the **expected gate
outcome**, not free-form "correct answers" — that is what makes this
tractable and auditable.

## Restoring the graph these results refer to

The thesis-style graph E1/E2/E3 were scored against **no longer exists on any
live server** — the shared instance was rebuilt into a different corpus. It
survives only as a JSON dump, which is what makes these numbers reproducible:

    eval/eval_toolkit/backups/local_eval_snapshot_2026-08-12.json
    355 nodes · 805 relationships · 116 embedded chunks (mxbai-embed-large)
    20 synthetic injection nodes · trust 0.115-0.858 · all 4 temporal statuses

That dump is **gitignored on purpose**: it carries full chunk text from a third
party's thesis, and this repository is public. Keep it out of the remote.

```bash
# A second Neo4j, so the paper-corpus clone on 7687 survives untouched.
# --security-opt seccomp=unconfined is REQUIRED on Docker < 23: the JVM calls
# clone3(), the default seccomp profile rejects it, and the container dies with
# the misleading "JAVA_HOME is not defined correctly".
docker run -d --name kg-neo4j-thesis --security-opt seccomp=unconfined \
    -p 7688:7687 -p 7475:7474 -e NEO4J_AUTH=neo4j/password123 neo4j:5

python eval/eval_toolkit/scripts/clone_graph.py \
    --source file://$(pwd)/eval/eval_toolkit/backups/local_eval_snapshot_2026-08-12.json \
    --target bolt://localhost:7688

set -a && source .env.thesis && set +a      # profile: port 7688 + mxbai
```

`.env.thesis` is also gitignored (local credentials); recreate it from the block
in this file's git history, or copy `.env.server.example` and change
`NEO4J_URI=bolt://localhost:7688` plus `KG_EMBED_MODEL=mxbai-embed-large`.

**The embedding model is the trap.** This graph is embedded with
`mxbai-embed-large`; the paper corpus uses `qwen3-embedding:0.6b`. Both are
1024-dimensional, so pointing the wrong one at it raises no error at all —
similarity scores just collapse to noise and retrieval returns nothing while
`/health` stays green. Verify with:

    MATCH (c:Chunk) RETURN DISTINCT c.embeddings_model, count(*)

## Prerequisites (in order)
1. Answer-generation model produces non-empty answers (fix the qwen3-vl
   empty-answer issue first: set answer-gen to hermes3 in .env).
2. Nabhyla's graph copied into YOUR local Neo4j (dump -> load). Do NOT run
   injections against her live instance.
3. Phase 1 migration run on YOUR copy (needs her OK only if run on her
   instance; on your copy it is your call — note it in the writeup).

## Run order
```bash
pip install neo4j
# point env at YOUR local snapshot copy
export NEO4J_URI=bolt://localhost:7687 NEO4J_PASSWORD=...

# STEP 1 — freeze + fingerprint (labels refer to THIS snapshot)
python scripts/freeze_snapshot.py

# STEP 2 — inject synthetic failure cases (categories c, d)
python scripts/inject_low_trust.py --dry-run   # review first
python scripts/inject_low_trust.py
python scripts/inject_temporal.py --dry-run
python scripts/inject_temporal.py

# STEP 3 — propose candidates for categories a, b from the real graph
python scripts/propose_questions.py --n-a 15 --n-b 15
#   -> curate ground_truth/candidates.jsonl (rephrase, drop weak ones)
#   -> merge curated items into ground_truth/ground_truth.jsonl
#   -> fill evidence_node_ids for injected items (printed at injection time,
#      or query: MATCH (n {injected:true}) RETURN elementId(n), n.name)

# RE-freeze AFTER injection so the fingerprint includes injected nodes:
python scripts/freeze_snapshot.py --out ground_truth/snapshot_post_injection.json
```

## Label schema (ground_truth.jsonl)
| field | meaning |
|---|---|
| id | a01/b01/c01/d01... category-prefixed |
| category | a_answerable_good / b_out_of_scope / c_low_trust / d_temporal_invalid |
| question | natural phrasing (curated, not template-y) |
| expected_gate_outcome | PASS / NO_INFO / RELEASE_WITH_DISCLAIMER / TEMPORAL_FLAGGED |
| expected_temporal_status | (d only) OUTDATED / SUPERSEDED / CONFLICTED |
| evidence_node_ids | auditability: which nodes justify this label |
| injected | true for c/d — synthetic injection MUST be disclosed in writeup |
| notes | verification evidence (e.g. "0 keyword matches for term X") |

## Scoring (for the benchmark runner)
Per item: compare agent JSON output to expected_gate_outcome:
- PASS            -> verified == true
- NO_INFO         -> answer states no info AND no fabricated content
- RELEASE_WITH_DISCLAIMER -> verified == false AND disclaimer present AND answer released
- TEMPORAL_FLAGGED -> temporal status of evidence nodes correctly detected
Report per-category precision/recall + the two headline rates:
**false-pass** (bad answer passed gates — the dangerous error) and
**false-block** (good answer needlessly disclaimed).

## Methodology notes for the thesis writeup (do not skip)
- Categories c/d use **synthetic fault injection** — standard practice for
  testing detection mechanisms; label it explicitly, never present injected
  items as natural data distribution.
- Category b terms are **verified absent** (keyword check recorded in
  `notes`) — this is what makes NO_INFO labels objective.
- NO_INFO scoring has a judgment component ("did it fabricate?"): decide
  the rubric BEFORE running (e.g. any claim about the term beyond "not
  found" counts as fabrication), and note who annotated.
- All labels refer to the frozen snapshot fingerprint(s) in this directory;
  never compare runs across different snapshots silently.
