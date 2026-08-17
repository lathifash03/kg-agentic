"""Paper-suite runner — score kg-agent against the 8-paper corpus.

Unlike the older eval_toolkit (which scored gate *decisions* on a thesis graph
that no longer exists), this suite scores **retrieval provenance**: for a
question whose answer lives in a known paper, did the agent actually read that
paper? The 8 papers form four same-topic/different-author pairs (P1/P2 ISO,
P3/P4 Hawthorne, P5/P6 servitization, P7/P8 supply chain integration), so pure
semantic similarity is not enough to land on the right one - which is exactly
what these metrics expose.

Requires ``KG_CHUNK_SOURCE_PROP`` to be set (``filename`` on this graph), else
the agent reports no document provenance and every retrieval metric is empty.

    python scripts/run_suite.py --gt ground_truth/level1_direct.jsonl \
        --out results/level1.json
    python scripts/run_suite.py ... --only D07        # single item
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from kg_agent.agentic_verifier import AgenticVerifier  # noqa: E402
from kg_agent.config import get_config  # noqa: E402
from kg_agent.neo4j_client import Neo4jClient  # noqa: E402

SUITE_DIR = pathlib.Path(__file__).resolve().parents[1]


def load_papers() -> dict:
    """Map paper id (P1..P8) -> metadata, dropping the leading comment key."""
    data = json.loads((SUITE_DIR / "papers.json").read_text())
    return {k: v for k, v in data.items() if not k.startswith("_")}


def score_retrieval(item: dict, result, papers: dict) -> dict:
    """Score which papers the answer was actually drawn from.

    retrieval_hit
        At least one expected paper contributed a chunk (necessary condition:
        without it the answer cannot be grounded in the right source).
    top_paper_correct
        The most-cited paper is an expected one - the stricter test, since a
        single stray chunk from the right paper is easily drowned out.
    chunk_precision
        Fraction of retrieved chunks that came from expected papers.
    coverage
        Fraction of expected papers that contributed at least one chunk. For
        multi-source questions this is the metric that matters.
    """
    want_files = {papers[p]["filename"] for p in item["expected_papers"]}
    by_name = {d["name"]: d["chunks"] for d in (result.documents_used or [])}
    total = sum(by_name.values())
    hit_files = want_files & set(by_name)
    got = sum(by_name.get(f, 0) for f in want_files)

    # An item with no expected papers is an ABSTENTION item (level 3 IE): the
    # corpus cannot answer it, so the correct behaviour is to decline. Every
    # provenance metric above is undefined here - there is no right paper to
    # land on - so they are reported as-is but excluded from the aggregates in
    # main(). What matters instead is how much context leaked in anyway, since
    # retrieved-but-irrelevant context is exactly what tempts the model to
    # answer from parametric knowledge.
    file_to_id = {v["filename"]: k for k, v in papers.items()}
    return {
        "abstention_expected": not item["expected_papers"],
        "leaked_chunks": total if not item["expected_papers"] else 0,
        "retrieval_hit": bool(hit_files),
        "top_paper_correct": bool(by_name) and max(by_name, key=by_name.get) in want_files,
        "chunk_precision": round(got / total, 3) if total else 0.0,
        "coverage": round(len(hit_files) / len(want_files), 3) if want_files else 0.0,
        "papers_retrieved": [
            {"id": file_to_id.get(n, n), "chunks": c}
            for n, c in sorted(by_name.items(), key=lambda kv: -kv[1])
        ],
    }


def check_evidence(item: dict, answer: str) -> dict:
    """Lexical screening of the answer against the item's evidence criteria.

    Three complementary checks, because one shape does not fit every question:

    ``expected_evidence``      every term must appear (facts with a fixed
                               surface form: a count, a named measure)
    ``expected_evidence_any``  at least one must appear (claims a model may
                               legitimately paraphrase - D06's "little
                               evidence" also arrives as "entirely fictional")
    ``forbidden_evidence``     none may appear (overclaims the answer must not
                               make; this is the real criterion for the
                               faithfulness and insufficient-evidence items)

    Screening signals for human review, never a correctness verdict: a passing
    lexical check says nothing about whether the surrounding claim is right.
    """
    low = (answer or "").lower()
    required = [t for t in item.get("expected_evidence", []) if t.lower() not in low]
    any_of = item.get("expected_evidence_any", [])
    any_hit = [t for t in any_of if t.lower() in low]
    forbidden = [t for t in item.get("forbidden_evidence", []) if t.lower() in low]
    return {
        "evidence_present": not required and (not any_of or bool(any_hit)),
        "missing_required": required,
        "matched_any_of": any_hit,
        "forbidden_present": forbidden,
    }


def cross_paper_entities(result) -> list:
    """Sources whose name is claimed by more than one paper.

    Generic fragment names ("Table 3", "Figure 1") are shared across papers, so
    they attribute an answer to several documents at once and dilute any
    per-source score computed as a mean.
    """
    out = []
    for s in result.sources_used or []:
        if len(s.get("documents", [])) > 1:
            out.append({"name": s["name"], "documents": s["documents"]})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="ground_truth/level1_direct.jsonl")
    ap.add_argument("--out", default="results/level1.json")
    ap.add_argument("--only", default="", help="run a single item id")
    args = ap.parse_args()

    gt_path = pathlib.Path(args.gt)
    if not gt_path.is_absolute():
        gt_path = SUITE_DIR / gt_path
    items = [json.loads(l) for l in gt_path.read_text().splitlines() if l.strip()]
    if args.only:
        items = [i for i in items if i["id"] == args.only]

    papers = load_papers()
    cfg = get_config()
    if not cfg.retrieval.chunk_source_property:
        sys.exit("KG_CHUNK_SOURCE_PROP is unset - retrieval provenance would be empty.")

    print(f"NEO4J_URI={cfg.neo4j.uri} | answer-gen={cfg.llm.provider}:{cfg.llm.model} "
          f"| judge={cfg.judge.provider or 'main'}:{cfg.judge.model or cfg.llm.model}")
    print(f"running {len(items)} questions...\n")

    per_item = []
    with Neo4jClient.from_config(cfg) as client:
        verifier = AgenticVerifier(client, cfg)
        for it in items:
            t0 = time.time()
            r = verifier.verify(it["question"])
            retrieval = score_retrieval(it, r, papers)
            evidence = check_evidence(it, r.answer)
            rec = {
                "id": it["id"],
                "question": it["question"],
                "expected_papers": it["expected_papers"],
                "expected_behavior": it["expected_behavior"],
                **retrieval,
                **evidence,
                "cross_paper_entities": cross_paper_entities(r),
                "answer": r.answer,
                "trust_score": r.trust_score,
                "temporal_status": r.temporal_validity_status,
                "faithfulness": r.faithfulness,
                "passed": r.passed,
                "retries": r.retries,
                "strategy": r.strategy,
                "n_sources": len(r.sources_used or []),
                "seconds": round(time.time() - t0, 1),
            }
            # For abstention items "declining" means: a refusal marker is present
            # AND no forbidden (fabricated) claim is. Both halves are needed - a
            # hedge bolted onto a fabricated answer is not an abstention.
            rec["abstained"] = bool(
                rec["abstention_expected"]
                and rec["evidence_present"]
                and not rec["forbidden_present"]
            )
            per_item.append(rec)
            if rec["abstention_expected"]:
                ok = rec["abstained"]
            else:
                ok = (
                    retrieval["top_paper_correct"]
                    and rec["evidence_present"]
                    and not rec["forbidden_present"]
                )
            got = ",".join(f"{p['id']}:{p['chunks']}" for p in retrieval["papers_retrieved"])
            print(f"  {'OK ' if ok else 'XX '}[{it['id']}] "
                  f"want={'+'.join(it['expected_papers']):<7} "
                  f"got={got:<28} prec={retrieval['chunk_precision']:<5} "
                  f"evid={'Y' if rec['evidence_present'] else 'n'}"
                  f"{' FORBIDDEN' if rec['forbidden_present'] else ''} ({rec['seconds']}s)")

    n = max(len(per_item), 1)
    # Provenance metrics are averaged over answerable items only. Including
    # abstention items would score them 0 on every retrieval metric for doing
    # exactly the right thing, dragging the suite average down as a reward for
    # correct behaviour. With no abstention items present these are unchanged.
    scored = [r for r in per_item if not r["abstention_expected"]]
    abst = [r for r in per_item if r["abstention_expected"]]
    ns, na = max(len(scored), 1), max(len(abst), 1)
    summary = {
        "n_questions": len(per_item),
        "n_answerable": len(scored),
        "retrieval_hit_rate": round(sum(r["retrieval_hit"] for r in scored) / ns, 3),
        "top_paper_accuracy": round(sum(r["top_paper_correct"] for r in scored) / ns, 3),
        "mean_chunk_precision": round(sum(r["chunk_precision"] for r in scored) / ns, 3),
        "mean_coverage": round(sum(r["coverage"] for r in scored) / ns, 3),
        "evidence_present_rate": round(sum(r["evidence_present"] for r in per_item) / n, 3),
        "items_with_forbidden_claims": [r["id"] for r in per_item if r["forbidden_present"]],
        "gate_passed_rate": round(sum(r["passed"] for r in per_item) / n, 3),
        "mean_trust": round(sum(r["trust_score"] for r in per_item) / n, 3),
        "mean_faithfulness": round(sum(r["faithfulness"] for r in per_item) / n, 3),
        "items_with_cross_paper_entities": [
            r["id"] for r in per_item if r["cross_paper_entities"]
        ],
        "failed_top_paper": [r["id"] for r in scored if not r["top_paper_correct"]],
    }
    if abst:
        # A gate that passes an abstention item is worse than one that fails it:
        # it certifies a fabricated answer. Reported separately so it cannot be
        # mistaken for the ordinary gate_passed_rate.
        summary.update({
            "n_abstention": len(abst),
            "abstention_rate": round(sum(r["abstained"] for r in abst) / na, 3),
            "items_that_fabricated": [r["id"] for r in abst if not r["abstained"]],
            "mean_leaked_chunks": round(sum(r["leaked_chunks"] for r in abst) / na, 2),
            "abstention_items_passing_gate": [r["id"] for r in abst if r["passed"]],
        })

    out_path = pathlib.Path(args.out)
    if not out_path.is_absolute():
        out_path = SUITE_DIR / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "per_item": per_item}, indent=2))

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nfull report -> {out_path}")


if __name__ == "__main__":
    main()
