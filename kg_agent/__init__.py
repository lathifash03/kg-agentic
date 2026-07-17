"""KG-agentic verification layer.

A research add-on to the DylanTartarini1996 knowledge-graph pipeline that adds
temporal metadata, temporal-validity checking, node trust scoring and an
agentic verification loop on top of the existing Neo4j knowledge graph.

Phases
------
1. Temporal metadata migration   -> ``kg_agent.neo4j_client``
2. Temporal validity check        -> ``kg_agent.temporal_validity``
3. Node trust scoring             -> ``kg_agent.node_trust``
4. Agentic verification           -> ``kg_agent.agentic_verifier``

Evaluation (RAGAS, offline)       -> ``kg_agent.evaluation``
Entry point CLI                   -> ``kg_agent.cli``
"""

__all__ = [
    "config",
    "neo4j_client",
    "temporal_validity",
    "node_trust",
    "agentic_verifier",
    "evaluation",
]
