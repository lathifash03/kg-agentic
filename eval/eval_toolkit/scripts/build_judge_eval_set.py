"""E2 helper — build a (answer, context) set for judging faithfulness.

Each question is retrieved + answered by the real pipeline, giving a FAITHFUL
pair (answer grounded in its own context). Pairing that same context with the
answer from a DIFFERENT question yields an UNFAITHFUL pair (the answer is about
another topic) - an automatic, bias-free way to get a known label.

Output: ground_truth/judge_pairs.jsonl  {id, question, context, answer,
construct_label (1=faithful, 0=unfaithful), pair_type}

Run with the SAME retrieval/answer-gen config as the benchmark.
"""
from __future__ import annotations

import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from kg_agent.agentic_verifier import AgenticVerifier, retrieve  # noqa: E402
from kg_agent.config import get_config  # noqa: E402
from kg_agent.neo4j_client import Neo4jClient  # noqa: E402

QUESTIONS = [
    "What is order to pod assignment?",
    "What is pile on and percentage of pods used?",
    "What is the two phase assignment method?",
    "What is order batching?",
    "What is the pod status cycle?",
    "What is throughput in the warehouse simulation?",
]


def main() -> None:
    cfg = get_config()
    print(f"answer-gen = {cfg.llm.provider}:{cfg.llm.model}\n")
    base = []
    with Neo4jClient.from_config(cfg) as client:
        v = AgenticVerifier(client, cfg)
        for q in QUESTIONS:
            ctx = retrieve(client, cfg, q, "vector", cfg.retrieval.top_k)
            context = ctx.context_text()
            answer = v._generate_answer(q, context)
            base.append({"question": q, "context": context, "answer": answer})
            print(f"  built: {q[:50]!r}  ctx={len(context)}ch answer={len(answer)}ch")

    pairs = []
    n = len(base)
    for i, b in enumerate(base):
        pairs.append({"id": f"f{i:02d}", "question": b["question"], "context": b["context"],
                      "answer": b["answer"], "construct_label": 1, "pair_type": "faithful"})
        # unfaithful: this context + an answer from a different question
        j = (i + 1) % n
        pairs.append({"id": f"u{i:02d}", "question": b["question"], "context": b["context"],
                      "answer": base[j]["answer"], "construct_label": 0,
                      "pair_type": f"unfaithful(answer from {base[j]['question'][:30]!r})"})

    out = pathlib.Path("ground_truth/judge_pairs.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"\nwrote {len(pairs)} pairs "
          f"({sum(p['construct_label'] for p in pairs)} faithful / "
          f"{sum(1 - p['construct_label'] for p in pairs)} unfaithful) -> {out}")


if __name__ == "__main__":
    main()
