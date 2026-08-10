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

    # Clearly-fabricated answers: specific claims NOT supported by ANY RMFS
    # thesis chunk (numbers/entities from unrelated domains). Sharper than an
    # on-domain answer-swap, which a judge rightly rates "supported" for
    # overlapping content.
    FABRICATED = [
        "The system was trained on 4.2 million ImageNet photographs and reached "
        "98.7% top-1 accuracy using a ResNet-50 backbone on 8 NVIDIA A100 GPUs.",
        "According to the sources, the optimal warehouse temperature is 4 degrees "
        "Celsius and forklifts must be recharged with hydrogen fuel cells every 90 minutes.",
        "The study concludes that Bitcoin mining difficulty doubled in Q3 and that "
        "the CRISPR-Cas9 protocol requires a 37-degree incubation for 12 hours.",
        "The results show the vaccine reached 91% efficacy across 43,000 trial "
        "participants over a six-month double-blind study.",
        "The paper reports that the Boeing 787 wing flex tolerance is 7.6 meters "
        "and that quarterly revenue grew 23% year over year to $4.1 billion.",
        "The authors recommend planting the wheat seeds 3 cm deep with 20 cm row "
        "spacing and irrigating twice weekly during the germination phase.",
    ]
    pairs = []
    for i, b in enumerate(base):
        pairs.append({"id": f"f{i:02d}", "question": b["question"], "context": b["context"],
                      "answer": b["answer"], "construct_label": 1, "pair_type": "faithful"})
        pairs.append({"id": f"u{i:02d}", "question": b["question"], "context": b["context"],
                      "answer": FABRICATED[i % len(FABRICATED)], "construct_label": 0,
                      "pair_type": "unfaithful(fabricated off-domain claims)"})

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
