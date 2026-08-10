"""E2 — compare faithfulness judges against a by-construction label set.

Scores every pair in judge_pairs.jsonl with each configured judge, binarises at
KG_MIN_FAITHFULNESS, and reports per-judge accuracy + Cohen's kappa against the
construction label, plus the mean-score gap between faithful and unfaithful
pairs (a judge that can't separate them is useless). Also writes a BLIND
annotation sheet for the human raters (no labels, no judge scores).

Judges compared: hermes3:3b, hermes3:8b, mock-lexical (+ Groq if a real key).
The human step (2 raters -> inter-rater kappa -> judge-vs-human) is done later
with analyze_judge_agreement.py once labels come back.

    python scripts/run_judge_comparison.py
"""
from __future__ import annotations

import csv
import json
import os
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from kg_agent.agentic_verifier import (  # noqa: E402
    _FAITHFULNESS_SYSTEM, _parse_faithfulness, MockLLMClient, OllamaLLMClient,
)
from kg_agent.config import get_config  # noqa: E402

LOCAL = os.environ.get("KG_JUDGE_OLLAMA_URL", "http://localhost:11434")
# Which local ollama judge models to compare (comma-separated). Default skips
# the slow 8b so the fan stays quiet; add it via KG_E2_JUDGES when on a GPU host.
JUDGE_MODELS = [m.strip() for m in os.environ.get("KG_E2_JUDGES", "hermes3:3b").split(",") if m.strip()]
# Truncate context handed to the judge - a 4000-char context makes 8b time out
# on CPU, and the support signal lives in the first passages anyway.
CTX_CHARS = int(os.environ.get("KG_E2_CTX_CHARS", "1800"))


def cohens_kappa(a, b):
    """Cohen's kappa for two binary label lists."""
    n = len(a)
    if n == 0:
        return 0.0
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa1 = sum(a) / n
    pb1 = sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return round((po - pe) / (1 - pe), 3) if pe != 1 else 1.0


def score(judge, answer, context):
    raw = judge.complete(_FAITHFULNESS_SYSTEM, f"ANSWER: {answer}\n\nSOURCES: {context[:CTX_CHARS]}")
    return _parse_faithfulness(raw).get("faithfulness", 0.0)


def main() -> None:
    cfg = get_config()
    thr = cfg.verifier.min_faithfulness
    pairs = [json.loads(l) for l in open("ground_truth/judge_pairs.jsonl") if l.strip()]

    judges = {m: OllamaLLMClient(cfg, model=m, ollama_url=LOCAL) for m in JUDGE_MODELS}
    judges["mock-lexical"] = MockLLMClient()
    if cfg.llm.api_key and cfg.llm.api_key != "isi_api_key_mu":
        try:
            from kg_agent.agentic_verifier import GroqLLMClient
            judges["groq"] = GroqLLMClient(cfg)
        except Exception as exc:
            print(f"(groq unavailable: {exc})")

    labels = [p["construct_label"] for p in pairs]
    print(f"{len(pairs)} pairs | faithfulness threshold = {thr}\n")

    results = {}
    for name, judge in judges.items():
        scores = [score(judge, p["answer"], p["context"]) for p in pairs]
        pred = [1 if s >= thr else 0 for s in scores]
        acc = round(sum(1 for x, y in zip(pred, labels) if x == y) / len(labels), 3)
        k = cohens_kappa(pred, labels)
        mf = [s for s, p in zip(scores, pairs) if p["construct_label"] == 1]
        mu = [s for s, p in zip(scores, pairs) if p["construct_label"] == 0]
        gap = round(sum(mf) / len(mf) - sum(mu) / len(mu), 3)
        results[name] = {"scores": scores, "pred": pred, "accuracy": acc, "kappa": k,
                         "mean_faithful": round(sum(mf) / len(mf), 3),
                         "mean_unfaithful": round(sum(mu) / len(mu), 3), "gap": gap}
        print(f"  {name:<13} acc={acc} kappa={k} | mean faithful={results[name]['mean_faithful']} "
              f"unfaithful={results[name]['mean_unfaithful']} gap={gap}")

    # per-item disagreement table (where judges differ from the construct label)
    print("\n  per-item scores (construct label | judge scores):")
    for i, p in enumerate(pairs):
        row = " ".join(f"{n}={results[n]['scores'][i]:.2f}" for n in judges)
        flag = "" if all(results[n]["pred"][i] == p["construct_label"] for n in judges) else "  <-- disagreement"
        print(f"    [{p['id']}] label={p['construct_label']}  {row}{flag}")

    # blind human annotation sheet
    sheet = pathlib.Path("ground_truth/judge_annotation_sheet.csv")
    with sheet.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "question", "answer", "context",
                    "human_faithful_1_or_0", "annotator"])
        for p in pairs:
            w.writerow([p["id"], p["question"], p["answer"], p["context"], "", ""])

    with open("ground_truth/judge_comparison.json", "w") as f:
        json.dump({"threshold": thr,
                   "per_judge": {n: {k: v for k, v in r.items() if k != "scores"}
                                 for n, r in results.items()}}, f, indent=2)
    print(f"\n  blind human sheet -> {sheet}  (2 raters fill human_faithful_1_or_0)")
    print("  summary -> ground_truth/judge_comparison.json")


if __name__ == "__main__":
    main()
