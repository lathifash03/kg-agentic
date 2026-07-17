"""RAGAS evaluation metrics (offline benchmarking for the thesis).

The in-loop faithfulness check lives in :mod:`kg_agent.agentic_verifier` (a
fast Groq call). This module provides the heavier, standardised **RAGAS**
metrics for offline evaluation of a batch of answers - the numbers you would
report in a write-up.

RAGAS is optional: imports are guarded so the module loads even when RAGAS is
not installed, and :func:`evaluate_faithfulness` transparently falls back to a
lexical-overlap proxy (clearly flagged in the result) so a benchmark can still
be produced.

A sample is a dict::

    {"question": str, "answer": str, "contexts": List[str], "ground_truth": str?}
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from kg_agent.config import Config, get_config

logger = logging.getLogger(__name__)

try:  # RAGAS + its deps are optional
    from datasets import Dataset  # type: ignore
    from ragas import evaluate as _ragas_evaluate  # type: ignore
    from ragas.metrics import answer_relevancy, faithfulness  # type: ignore

    RAGAS_AVAILABLE = True
except Exception:  # pragma: no cover - depends on optional install
    RAGAS_AVAILABLE = False


def _lexical_faithfulness(sample: Dict[str, Any]) -> float:
    """Fraction of answer tokens supported by the concatenated contexts (0-1)."""
    import re

    answer = sample.get("answer", "")
    contexts = " ".join(sample.get("contexts", []))
    a = set(re.findall(r"\w+", answer.lower()))
    c = set(re.findall(r"\w+", contexts.lower()))
    if not a:
        return 0.0
    return len(a & c) / len(a)


def _build_ragas_llm(config: Config):
    """Wrap the configured Groq model as a RAGAS-compatible LLM, or return None."""
    if not config.llm.api_key:
        return None
    try:  # pragma: no cover - network/optional dependent
        from langchain_groq import ChatGroq
        from ragas.llms import LangchainLLMWrapper

        chat = ChatGroq(
            model=config.llm.model,
            temperature=config.llm.temperature,
            api_key=config.llm.api_key,
        )
        return LangchainLLMWrapper(chat)
    except Exception as exc:
        logger.warning("Could not build a RAGAS LLM from Groq config: %s", exc)
        return None


def evaluate_faithfulness(
    samples: List[Dict[str, Any]],
    config: Optional[Config] = None,
    metrics: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Evaluate a batch of answers with RAGAS faithfulness (with fallback).

    Parameters
    ----------
    samples
        List of ``{question, answer, contexts[, ground_truth]}`` dicts.
    config
        Configuration (defaults to ``get_config()``); supplies the Groq LLM.
    metrics
        Optional subset of metric names. ``"faithfulness"`` is always included;
        ``"answer_relevancy"`` is added when requested and RAGAS is available.

    Returns
    -------
    Dict[str, Any]
        ``{"backend": "ragas"|"lexical_fallback", "per_sample": [...],
        "mean_faithfulness": float, ...}``.
    """
    config = config or get_config()
    if not samples:
        return {"backend": "none", "per_sample": [], "mean_faithfulness": None}

    llm = _build_ragas_llm(config) if RAGAS_AVAILABLE else None

    if RAGAS_AVAILABLE and llm is not None:
        try:  # pragma: no cover - heavy/optional path
            dataset = Dataset.from_list(
                [
                    {
                        "question": s["question"],
                        "answer": s["answer"],
                        "contexts": s.get("contexts", []),
                        "ground_truth": s.get("ground_truth", ""),
                    }
                    for s in samples
                ]
            )
            chosen = [faithfulness]
            if metrics and "answer_relevancy" in metrics:
                chosen.append(answer_relevancy)
            result = _ragas_evaluate(dataset, metrics=chosen, llm=llm)
            df = result.to_pandas()
            scores = df["faithfulness"].fillna(0.0).tolist()
            return {
                "backend": "ragas",
                "per_sample": scores,
                "mean_faithfulness": round(sum(scores) / len(scores), 4),
            }
        except Exception as exc:
            logger.warning("RAGAS evaluation failed (%s); using lexical fallback.", exc)

    # Fallback: lexical overlap proxy.
    scores = [_lexical_faithfulness(s) for s in samples]
    return {
        "backend": "lexical_fallback",
        "per_sample": [round(x, 4) for x in scores],
        "mean_faithfulness": round(sum(scores) / len(scores), 4),
        "note": "RAGAS unavailable or no Groq key; lexical-overlap proxy used.",
    }


if __name__ == "__main__":  # pragma: no cover - manual entry point
    import json

    demo = [
        {
            "question": "What is an RMFS?",
            "answer": "An RMFS is a robotic mobile fulfillment system using AGVs and pods.",
            "contexts": [
                "Robotic Mobile Fulfillment Systems (RMFS) use autonomous guided "
                "vehicles (AGVs) to carry pods to picking stations."
            ],
        }
    ]
    print(f"RAGAS available: {RAGAS_AVAILABLE}")
    print(json.dumps(evaluate_faithfulness(demo), indent=2))
