"""Comprehensive test for the AtomMem upgrade package.

Runs:
  1. Each module's deterministic self-test (Δ2, Δ3, Δ4).
  2. Tool registration against a MockFastMCPApp (all 7 tools present).
  3. Tool invocation against an isolated temp DB (status, temporal profile
     round-trip) — no pollution of the production memory.db.
  4. A read-only smoke of atommem_graph_recall against the real memory.db.

LLM/embedder-dependent tools (extract_atomic_facts, verify_fact_before_store)
are exercised in the live module self-tests / run_live.py, not here, to keep
this suite fast and network-light. Exit code = number of failures.
"""

from __future__ import annotations

import os
import sqlite3
import sys
import tempfile

FAILURES = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global FAILURES
    if not cond:
        FAILURES += 1
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('' if cond else ' :: ' + detail)}")


class MockFastMCPApp:
    """Mirror of the repo's test harness for tool registration."""

    def __init__(self):
        self.tools = {}

    def tool(self):
        def decorator(func):
            self.tools[func.__name__] = func
            return func

        return decorator


def main() -> int:
    # --- 1. module self-tests ---------------------------------------------- #
    print("== module self-tests ==")
    from atommem.idf_keyword_graph import _selftest as idf_selftest
    from atommem.fact_verification import _selftest as fv_selftest
    from atommem.temporal_profile import _selftest as tp_selftest

    global FAILURES
    FAILURES += idf_selftest()
    FAILURES += fv_selftest()
    FAILURES += tp_selftest()

    # --- Δ1 extractor: JSON preamble parsing + fallback semantics ---------- #
    from atommem.atomic_facts import AtomicFactExtractor
    from atommem.llm_cli import _extract_json

    check(
        "extract_json: fenced JSON after Thinking preamble",
        _extract_json('Thinking...\nThe user wants JSON.\n```json\n{"a": 1}\n```')
        == {"a": 1},
    )
    check(
        "extract_json: bare array after prose preamble",
        _extract_json("Thinking...\nSome prose [not json]\n[1, 2, 3]") == [1, 2, 3],
    )

    class _EmptyLLM:  # LLM "available" but returns a deliberate empty extraction
        def call_json(self, system, user):
            return {"data": [], "_provider": "stub"}

    class _FailLLM:  # LLM path genuinely unavailable
        def call_json(self, system, user):
            return {"_unavailable": True}

    ex_empty = AtomicFactExtractor(llm=_EmptyLLM())
    out_empty = ex_empty.extract_structured(
        "Nothing interesting here.", session_time="2026-08-05"
    )
    check(
        "empty extraction -> no passthrough",
        out_empty == [] and ex_empty.last_extract_error is None,
        str(out_empty),
    )

    ex_fail = AtomicFactExtractor(llm=_FailLLM())
    out_fail = ex_fail.extract_structured(
        "Marc moved the ARC run to fedora.", session_time="2026-08-05"
    )
    check(
        "failed LLM -> honest passthrough flagged",
        len(out_fail) == 1
        and out_fail[0].get("_extracted") is False
        and out_fail[0]["fact"].startswith("Marc"),
        str(out_fail),
    )
    check(
        "failed LLM -> last_extract_error recorded",
        ex_fail.last_extract_error is not None,
        str(ex_fail.last_extract_error),
    )

    # --- 2. tool registration ---------------------------------------------- #
    print("\n== tool registration (temp db) ==")
    from atommem.tools import register_atommem_tools

    tmp_db = os.path.join(tempfile.gettempdir(), f"atommem_tools_test_{os.getpid()}.db")
    # Seed a minimal entities/observations schema so read tools have a table.
    conn = sqlite3.connect(tmp_db)
    conn.executescript(
        """
        CREATE TABLE entities (id INTEGER PRIMARY KEY, name TEXT, entity_type TEXT);
        CREATE TABLE observations (id INTEGER PRIMARY KEY, entity_id INTEGER, content TEXT);
        INSERT INTO entities (id, name, entity_type) VALUES (1, 'Storage RAID', 'infra');
        INSERT INTO observations (entity_id, content) VALUES (1, 'SSDRAID0 primary drive operational');
        INSERT INTO entities (id, name, entity_type) VALUES (2, 'Backup drive', 'infra');
        INSERT INTO observations (entity_id, content) VALUES (2, 'FILES drive remounted for backup storage');
        """
    )
    conn.commit()
    conn.close()

    app = MockFastMCPApp()
    register_atommem_tools(app, tmp_db)
    expected = {
        "extract_atomic_facts",
        "atommem_graph_recall",
        "atommem_keyword_neighbors",
        "verify_fact_before_store",
        "upsert_temporal_profile",
        "query_temporal_profile",
        "atommem_status",
    }
    check(
        "all 7 atommem tools registered",
        expected.issubset(set(app.tools)),
        f"missing={expected - set(app.tools)}",
    )

    # temporal_profiles table created by registration
    conn = sqlite3.connect(tmp_db)
    has_tp = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='temporal_profiles'"
    ).fetchone()
    conn.close()
    check("temporal_profiles table created on registration", has_tp is not None)

    # --- 3. invoke tools on temp db ---------------------------------------- #
    print("\n== tool invocation (temp db) ==")
    status = app.tools["atommem_status"]()
    check(
        "atommem_status returns delta map",
        "deltas" in status and len(status["deltas"]) == 4,
        str(status.get("deltas")),
    )

    up = app.tools["upsert_temporal_profile"](
        subject="TestSubject",
        content="TestSubject lives in Denver.",
        valid_from="2020",
        keywords=["location", "denver"],
        evidence=["T1"],
    )
    check("upsert_temporal_profile -> new", up.get("action") == "new", str(up))
    up2 = app.tools["upsert_temporal_profile"](
        subject="TestSubject",
        content="TestSubject lives in Austin.",
        valid_from="2023",
        keywords=["location", "austin", "denver"],
        evidence=["T9"],
    )
    check(
        "upsert state change -> update_current",
        up2.get("action") == "update_current",
        str(up2),
    )
    q_old = app.tools["query_temporal_profile"](
        subject="TestSubject", query_time="2021"
    )
    q_now = app.tools["query_temporal_profile"](subject="TestSubject")
    check(
        "query 2021 -> Denver (historical)",
        bool(q_old["profiles"]) and "Denver" in q_old["profiles"][0]["content"],
        str(q_old),
    )
    check(
        "query now -> Austin (current)",
        bool(q_now["profiles"]) and "Austin" in q_now["profiles"][0]["content"],
        str(q_now),
    )

    recall = app.tools["atommem_graph_recall"](query="backup storage drive", top_k=5)
    check(
        "graph_recall returns structure",
        "results" in recall and "query_keywords" in recall,
        str(list(recall.keys())),
    )

    if os.path.exists(tmp_db):
        os.remove(tmp_db)

    # --- 4. read-only smoke on real memory.db ------------------------------ #
    print("\n== real memory.db read-only smoke ==")
    real_db = os.path.expanduser("~/.claude/enhanced_memories/memory.db")
    if os.path.exists(real_db):
        app2 = MockFastMCPApp()
        register_atommem_tools(app2, real_db)
        recall = app2.tools["atommem_graph_recall"](
            query="storage drive raid backup", max_entities=600, top_k=5
        )
        check(
            "real graph_recall scanned entities",
            recall.get("entities_scanned", 0) > 0,
            f"scanned={recall.get('entities_scanned')}",
        )
        print(f"    query_keywords={recall.get('query_keywords')}")
        for r in recall.get("results", [])[:5]:
            print(f"    -> {r['name'][:50]!r} score={r['graph_score']:.6f}")
    else:
        print("  [SKIP] real memory.db not found")

    print(f"\n{'=' * 50}")
    print(
        f"{'ALL PASS' if FAILURES == 0 else str(FAILURES) + ' FAILURE(S)'} — atommem suite"
    )
    return FAILURES


if __name__ == "__main__":
    sys.exit(main())
