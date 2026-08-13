"""Smoke-test a running kg-agent endpoint, from outside.

``/health`` only reports Neo4j connectivity. It says nothing about Ollama, the
embedding model, or whether retrieval returns anything - so a fully green
``/health`` sat on top of a deployment that answered 20/20 questions with zero
sources. This script exercises the path a caller actually uses, and fails on
the things ``/health`` cannot see.

Layers, cheapest first; a failure at any layer explains the ones after it::

    L1  /health           service up, Neo4j reachable
    L2  /tools            registry sane, write flags present
    L3  write guard       a writing tool is refused with 403
    L4  retrieval         an in-corpus question returns real documents  <-- the important one
    L5  generation        the answer is real text, not an empty-context sentinel
    L6  negative control  an out-of-corpus question returns NO documents

L6 matters as much as L4: without it, a retriever that matches everything
indiscriminately looks identical to one that works.

Pure stdlib - no pip install needed on the server or on a consumer's machine.

    python scripts/smoke_endpoint.py                              # localhost:8003
    python scripts/smoke_endpoint.py --url http://100.118.203.111:8003
    python scripts/smoke_endpoint.py --url ... --skip-slow        # L1-L3 only
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

# Answerable from the papers currently in the graph (Hawthorne / servitization).
IN_CORPUS = "What did Levitt and List conclude about the Hawthorne effect?"
# Deliberately far outside any paper in this corpus. If this returns documents,
# the similarity threshold is not filtering and every result is suspect.
OUT_OF_CORPUS = "What is the recommended oil viscosity for a diesel generator?"

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"
results: list = []


def record(layer: str, status: str, detail: str) -> None:
    results.append((layer, status, detail))
    mark = {PASS: "  ok  ", FAIL: " FAIL ", WARN: " warn "}[status]
    print(f"[{mark}] {layer:<22} {detail}", flush=True)


def request(url: str, payload=None, timeout: int = 30):
    """Return (status_code, parsed_body_or_text). Never raises for HTTP errors."""
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(url, data=data, headers=headers,
                                 method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode()
            try:
                return resp.status, json.loads(raw)
            except ValueError:
                return resp.status, raw
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode(errors="ignore")
        try:
            return exc.code, json.loads(raw)
        except ValueError:
            return exc.code, raw
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def tcp_open(base: str, timeout: float = 5.0) -> bool:
    """True if something accepts a TCP connection at ``base``'s host:port."""
    import socket
    from urllib.parse import urlparse

    u = urlparse(base)
    port = u.port or (443 if u.scheme == "https" else 80)
    try:
        with socket.create_connection((u.hostname, port), timeout=timeout):
            return True
    except OSError:
        return False


def l1_health(base: str) -> bool:
    # /health calls verify_connectivity(), which blocks on the Neo4j driver's
    # own TCP timeout. When Neo4j has no route that wait runs far past a
    # typical HTTP timeout, so a short one here reports "service down" for a
    # service that is up and about to answer "degraded". Probe the socket
    # first, then allow the slow path the time it actually needs.
    listening = tcp_open(base)
    code, body = request(f"{base}/health", timeout=90)
    if code is None:
        if listening:
            record("L1 health", FAIL,
                   f"port is open but /health did not answer within 90s - the "
                   f"Neo4j connectivity check is most likely hanging on an "
                   f"unroutable host. Detail: {body}")
        else:
            record("L1 health", FAIL, f"nothing listening on that host:port - {body}")
        return False
    if code != 200:
        record("L1 health", FAIL, f"HTTP {code}: {body}")
        return False
    if not isinstance(body, dict) or not body.get("neo4j_connected"):
        # The service itself is up; only the graph is missing. Record the
        # failure, but keep going - L2 and L3 never touch Neo4j, and the write
        # guard is exactly what you want to confirm while the graph is still
        # unreachable rather than after.
        record("L1 health", FAIL,
               f"service is up but Neo4j is NOT connected: {body}. "
               "Answers will have no sources. Continuing with the checks that "
               "do not need the graph.")
        return True
    ro = body.get("read_only")
    record("L1 health", PASS, f"up, neo4j connected, read_only={ro}")
    if ro is None:
        record("L1 read_only", WARN,
               "field absent - server runs code from before the write guard")
    elif ro is False:
        record("L1 read_only", WARN,
               "WRITES ARE ENABLED - intended only with explicit permission")
    return True


def l2_tools(base: str) -> bool:
    code, body = request(f"{base}/tools", timeout=10)
    if code != 200 or not isinstance(body, dict):
        record("L2 tools", FAIL, f"HTTP {code}: {body}")
        return False
    tools = body.get("tools") or []
    names = sorted(t.get("name") for t in tools)
    if "answer_question" not in names:
        record("L2 tools", FAIL, f"answer_question missing; got {names}")
        return False
    missing_flag = [t["name"] for t in tools if "writes" not in t]
    if missing_flag:
        record("L2 tools", WARN, f"no `writes` flag on {missing_flag} (older build)")
    else:
        record("L2 tools", PASS, f"{len(tools)} tools, all flagged: {names}")
    return True


def l3_write_guard(base: str) -> bool:
    """A writing tool must be refused. Uses a title that is obvious in the graph
    if the guard ever fails open."""
    code, body = request(
        f"{base}/tools/ingest_meeting",
        {"arguments": {"title": "SMOKE TEST - should never be written"}},
        timeout=30,
    )
    if code == 403:
        record("L3 write guard", PASS, "ingest_meeting refused with 403")
        return True
    if code == 404:
        record("L3 write guard", WARN, "tool absent - nothing to guard")
        return True
    record("L3 write guard", FAIL,
           f"expected 403, got HTTP {code} - A NODE MAY HAVE BEEN WRITTEN. {body}")
    return False


def l4_l5_retrieval(base: str, timeout: int) -> bool:
    t0 = time.time()
    code, body = request(f"{base}/query", {"query": IN_CORPUS}, timeout=timeout)
    secs = round(time.time() - t0, 1)
    if code is None:
        record("L4 retrieval", FAIL, f"no response after {secs}s - {body}")
        return False
    if code != 200 or not isinstance(body, dict):
        record("L4 retrieval", FAIL, f"HTTP {code} after {secs}s: {body}")
        return False

    docs = body.get("documents_used") or []
    srcs = body.get("sources_used") or []
    if not docs and not srcs:
        record("L4 retrieval", FAIL,
               f"ZERO sources after {secs}s - embedding model most likely does not "
               "match the one the chunks were embedded with (this fails silently: "
               "two different 1024-dim models produce no error, just noise scores)")
        ok = False
    else:
        got = ", ".join(f"{d['name'][:34]}:{d['chunks']}" for d in docs[:3])
        record("L4 retrieval", PASS, f"{len(docs)} docs / {len(srcs)} sources in {secs}s [{got}]")
        ok = True

    answer = (body.get("answer") or "").strip()
    empty_markers = ("No supporting context was retrieved",
                     "did not produce an answer")
    if not answer:
        record("L5 generation", FAIL, "answer is empty")
        ok = False
    elif any(m in answer for m in empty_markers):
        record("L5 generation", FAIL, f"sentinel answer: {answer[:90]!r}")
        ok = False
    else:
        record("L5 generation", PASS,
               f"{len(answer)} chars, faithfulness={body.get('faithfulness')}, "
               f"passed={body.get('passed')}")

    if body.get("passed") is False:
        record("L5 gate", WARN,
               "passed=false (expected while the graph has no provenance metadata; "
               "not an endpoint fault)")
    if secs > 180:
        record("L5 latency", WARN, f"{secs}s is slow - clients need a >300s timeout")
    return ok


def l6_negative(base: str, timeout: int) -> bool:
    """An out-of-corpus question must NOT come back with documents."""
    t0 = time.time()
    code, body = request(f"{base}/query", {"query": OUT_OF_CORPUS}, timeout=timeout)
    secs = round(time.time() - t0, 1)
    if code != 200 or not isinstance(body, dict):
        record("L6 negative ctl", WARN, f"HTTP {code} after {secs}s - inconclusive")
        return True
    docs = body.get("documents_used") or []
    if docs:
        record("L6 negative ctl", FAIL,
               f"out-of-corpus question returned {len(docs)} docs in {secs}s - the "
               "similarity threshold is not filtering; L4 passing may be meaningless")
        return False
    record("L6 negative ctl", PASS, f"correctly returned no documents ({secs}s)")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default="http://localhost:8003", help="base URL of the endpoint")
    ap.add_argument("--timeout", type=int, default=300, help="per-query timeout (seconds)")
    ap.add_argument("--skip-slow", action="store_true", help="L1-L3 only, no LLM calls")
    args = ap.parse_args()
    base = args.url.rstrip("/")

    print(f"\nkg-agent smoke test -> {base}\n" + "-" * 74)
    if not l1_health(base):
        print("\nVERDICT: DOWN - service not answering or Neo4j unreachable.")
        sys.exit(1)
    l2_tools(base)
    l3_write_guard(base)

    if args.skip_slow:
        print("\n(--skip-slow: retrieval and generation not tested)")
    else:
        print(f"\n  running 2 live queries, up to {args.timeout}s each...\n")
        l4_l5_retrieval(base, args.timeout)
        l6_negative(base, args.timeout)

    print("-" * 74)
    failed = [r for r in results if r[1] == FAIL]
    warned = [r for r in results if r[1] == WARN]
    if failed:
        print(f"VERDICT: NOT WORKING - {len(failed)} check(s) failed:")
        for layer, _, detail in failed:
            print(f"  - {layer}: {detail}")
        sys.exit(1)
    print(f"VERDICT: WORKING{f' ({len(warned)} warning(s) - read them)' if warned else ''}")
    sys.exit(0)


if __name__ == "__main__":
    main()
