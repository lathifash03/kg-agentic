"""Minimal example: connect to Neo4j from code (no Neo4j Browser needed).

Run it::

    python examples/connect_neo4j.py
    python examples/connect_neo4j.py "MATCH (t:Topic) RETURN t.name AS name LIMIT 10"

Notes
-----
* Uses the Tailscale address ``100.95.227.48`` - reachable from this machine.
  Nabhyla's LAN address ``192.168.0.185`` only works from within her own
  network, so don't use it here.
* READ-ONLY: every query runs inside ``session.execute_read``, which the server
  rejects if it contains a write clause. This is deliberate - the graph is
  shared. To modify it you would use ``execute_write`` AND have permission.
* Requires the driver: ``pip install neo4j`` (already in this repo's .venv).
"""

from __future__ import annotations

import sys

from neo4j import GraphDatabase

# Override with env vars if you like, but these are the confirmed values.
URI = "bolt://100.95.227.48:7687"
AUTH = ("neo4j", "password123")

DEFAULT_QUERY = "MATCH (t:Topic) RETURN t.name AS name LIMIT 5"


def main() -> None:
    """Connect, run one read-only query, print the rows."""
    query = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUERY

    driver = GraphDatabase.driver(URI, auth=AUTH)
    try:
        driver.verify_connectivity()  # raises a clear error if unreachable
        print(f"connected: {URI}\n")

        with driver.session() as session:
            rows = session.execute_read(lambda tx: tx.run(query).data())

        if not rows:
            print("(no rows returned)")
        for row in rows:
            print(row)
        print(f"\n{len(rows)} row(s).")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
