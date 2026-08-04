# Ground-Truth Toolkit — E1 (Gate Correctness)

Builds the labeled dataset for evaluating whether kg-agent's gates make the
RIGHT decision per question category. Labels describe the **expected gate
outcome**, not free-form "correct answers" — that is what makes this
tractable and auditable.

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
