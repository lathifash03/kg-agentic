"""Smoke test: semua modul bisa di-import dan registry tools konsisten.

Jalankan: python -m pytest tests/ -q   (atau cukup: python tests/test_smoke.py)
Tidak butuh Neo4j/LLM — hanya memastikan struktur paket sehat.
"""

import importlib


MODULES = [
    "kg_agent.config",
    "kg_agent.neo4j_client",
    "kg_agent.temporal_validity",
    "kg_agent.node_trust",
    "kg_agent.agentic_verifier",
    "kg_agent.tools",
    "kg_agent.cli",
]


def test_imports():
    for mod in MODULES:
        importlib.import_module(mod)


def test_tool_registry():
    from kg_agent.tools import TOOLS, tool_specs

    specs = tool_specs()
    assert {s["name"] for s in specs} == set(TOOLS)
    for spec in specs:
        assert spec["description"]
        assert spec["parameters"]["type"] == "object"


def test_config_defaults():
    from kg_agent.config import get_config

    cfg = get_config()
    assert cfg.trust.source_weights["meeting"] == 0.8
    assert cfg.verifier.max_retries >= 0


if __name__ == "__main__":
    test_imports()
    test_tool_registry()
    test_config_defaults()
    print("OK - all smoke tests passed")
