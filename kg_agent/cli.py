"""Entry point - end-to-end demo of the KG-agentic verification stack.

Wires Phases 1-4 together::

    python -m kg_agent.cli                      # run the default demo query
    python -m kg_agent.cli --query "..."        # custom query
    python -m kg_agent.cli --setup              # run Phase 1 + Phase 3 first
    python -m kg_agent.cli --query "..." --json # machine-readable output
    python -m kg_agent.cli --query "..." --agentic  # let the LLM pick the tool
    python -m kg_agent.cli --query "..." --spoken   # also emit answer_spoken
    python -m kg_agent.cli --query "..." --allow-ungrounded  # see below

``--setup`` applies the Phase 1 temporal-metadata migration and computes/stores
Phase 3 trust scores before answering (idempotent; safe to repeat). Without it,
the demo assumes those have already been run.

``--spoken`` adds ``answer_spoken``: a 2-3 sentence, citation-free rendering of
the finished answer for text-to-speech. ``answer`` itself is never touched, and
the extra LLM call only happens when the flag is passed.

``--allow-ungrounded`` opts in to ``ungrounded_answer`` for questions the corpus
cannot answer: a general-knowledge completion kept in its own field, leaving
``answer`` as the refusal and ``passed``/``overall_confidence``/``sources_used``
untouched. Off by default - an unsourced answer has to be asked for.

``--agentic`` (or ``KG_ORCHESTRATOR=native``) routes the query through the
Phase 5 tool-calling orchestrator, which asks the LLM which tool to use instead
of always calling the verifier. The verification gates are unchanged either
way; with ``--json`` the tool-call trace is included in the output.
"""

from __future__ import annotations

import argparse
import json
import logging

from kg_agent.agentic_verifier import AgenticVerifier, VerifiedAnswer
from kg_agent.config import get_config
from kg_agent.neo4j_client import Neo4jClient
from kg_agent.node_trust import compute_and_store
from kg_agent.orchestrator import OrchestrationResult, run_orchestrated

DEFAULT_QUERY = "What is a Robotic Mobile Fulfillment System (RMFS)?"


def _print_human(result: VerifiedAnswer) -> None:
    """Pretty-print a :class:`VerifiedAnswer` for the terminal."""
    print("\n" + "=" * 72)
    print(f"QUERY: {result.query}")
    print("=" * 72)
    print(f"\nANSWER:\n{result.answer}\n")
    print("-" * 72)
    print(f"  trust_score (mean of sources) : {result.trust_score}")
    print(f"  temporal_validity_status      : {result.temporal_validity_status}")
    print(f"  faithfulness                  : {result.faithfulness}")
    print(f"  overall_confidence            : {result.overall_confidence}")
    print(f"  passed gates                  : {result.passed}")
    print(f"  retrieval strategy / retries  : {result.strategy} / {result.retries}")
    if result.answer_spoken is not None:
        print(f"\nANSWER (spoken form):\n{result.answer_spoken}\n")
        print("-" * 72)
    if result.ungrounded_answer is not None:
        print(
            "\nUNGROUNDED ANSWER (opt-in; NOT from the knowledge graph, not "
            f"gated, not counted in any score above):\n{result.ungrounded_answer}\n"
        )
        print("-" * 72)
    print(f"\n  explanation: {result.explanation}")
    print("\n  sources_used:")
    if not result.sources_used:
        print("    (none retrieved)")
    for s in result.sources_used:
        flag = "OK " if s["used"] else "DROP"
        print(
            f"    [{flag}] {s['name']!r:<30} status={s['temporal_status']:<10} "
            f"trust={s['trust_score']}"
        )
    print("=" * 72 + "\n")


def _print_orchestration(result: OrchestrationResult) -> None:
    """Pretty-print an :class:`OrchestrationResult` for the terminal."""
    print("\n" + "=" * 72)
    print(f"QUERY: {result.query}")
    print(f"MODEL: {result.model}   (agentic tool selection)")
    print("=" * 72)
    print(f"\nRESPONSE:\n{result.response}\n")
    print("-" * 72)
    print("  TOOL-CALL TRACE:")
    if not result.trace:
        print("    (no steps recorded)")
    for entry in result.trace:
        status = "REJECTED" if entry["validation_error"] else "ok"
        print(f"    [step {entry['step']}] {entry['tool_name']!r} -> {status}")
        print(f"        arguments: {json.dumps(entry['arguments'], default=str)}")
        if entry["validation_error"]:
            print(f"        validation_error: {entry['validation_error']}")
        else:
            preview = json.dumps(entry["result_or_error"], default=str)
            print(f"        result: {preview[:200]}")
    print(f"\n  tools_used     : {result.tools_used or '(none - answer NOT verified)'}")
    print(f"  stopped_reason : {result.stopped_reason}")
    print(f"  ok             : {result.ok}")

    if result.verified_answer:
        va = result.verified_answer
        print("\n  VERIFICATION GATES (from AgenticVerifier, unmodified):")
        print(f"    trust_score              : {va.get('trust_score')}")
        print(f"    temporal_validity_status : {va.get('temporal_validity_status')}")
        print(f"    faithfulness             : {va.get('faithfulness')}")
        print(f"    overall_confidence       : {va.get('overall_confidence')}")
        print(f"    passed gates             : {va.get('passed')}")
        print(f"    disclaimer               : {va.get('disclaimer') or '(none)'}")
    else:
        print("\n  (no verification result - the run did not end in answer_question)")
    print("=" * 72 + "\n")


def main() -> None:
    """Run the end-to-end verified-answer demo."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="KG-agentic verified-answer demo.")
    parser.add_argument("--query", type=str, default=DEFAULT_QUERY, help="Question to answer.")
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Run Phase 1 migration + Phase 3 trust scoring before answering.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    parser.add_argument(
        "--spoken",
        action="store_true",
        help="Also produce `answer_spoken`, a short TTS-friendly summary of the "
        "finished answer. Costs one extra LLM call; `answer` is unchanged.",
    )
    parser.add_argument(
        "--allow-ungrounded",
        action="store_true",
        help="When retrieval finds nothing, also produce `ungrounded_answer` "
        "from the model's general knowledge. Kept in its own field: `answer` "
        "stays the refusal and the gate verdict is unaffected.",
    )
    parser.add_argument(
        "--agentic",
        action="store_true",
        help="Let the LLM choose the tool via the Phase 5 orchestrator "
        "(same as KG_ORCHESTRATOR=native). Verification gates are unchanged.",
    )
    args = parser.parse_args()

    cfg = get_config()
    with Neo4jClient.from_config(cfg) as client:
        if not client.verify_connectivity():
            raise SystemExit("Could not connect to Neo4j - check kg_agent/config.py")

        if args.setup:
            logging.getLogger(__name__).info("Running Phase 1 migration + Phase 3 scoring...")
            client.run_phase1_migration()
            compute_and_store(client, cfg)

        if args.agentic or cfg.orchestrator.enabled:
            orchestrated = run_orchestrated(client, cfg, args.query)
            if args.json:
                print(json.dumps(orchestrated.to_dict(), indent=2, default=str))
            else:
                _print_orchestration(orchestrated)
            return

        verifier = AgenticVerifier(client, cfg)
        result = verifier.verify(
            args.query,
            allow_ungrounded=args.allow_ungrounded,
            spoken=args.spoken,
        )

        if args.json:
            print(json.dumps(result.to_dict(), indent=2, default=str))
        else:
            _print_human(result)


if __name__ == "__main__":
    main()
