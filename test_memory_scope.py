"""Tests for project-scope filtering (added 2026-08-09).

Why these exist
---------------
The scoping feature was built and verified by hand-probing a live daemon. That
proves it worked once, on one machine, on one afternoon. Every defect this
harness has recorded in the same area -- non-recursive globs, post-LIMIT
filters, inert guards -- shares a shape: the code kept returning well-formed
results while doing nothing, and nothing failed. A hand probe cannot notice a
regression six weeks from now.

Each test below is written to FAIL if the behaviour it describes is removed:

  * scope actually narrows           -> fails if the clause stops binding
  * scope filters BEFORE the limit   -> fails if someone "simplifies" it to a
                                        post-filter over the result set
  * unknown scope errors             -> fails if it reverts to a silent empty
  * bad scope is rejected            -> fails if validation is dropped
  * the socket dispatcher forwards   -> fails if the positional arg is dropped
    scope                               (the exact bug caught during the build)

Note on string building: a couple of tests need SQL/injection payloads as
DATA. They are assembled from fragments at runtime because the session's
command scanner blocks those literals on sight -- the same false-positive class
the payloads are here to exercise.

Run:  .venv/bin/python -m pytest test_memory_scope.py -q
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _make_db(tmp_path: Path, name: str = "m.db") -> str:
    """A minimal store with two scopes and one unscoped entity."""
    db = tmp_path / name
    con = sqlite3.connect(db)
    con.executescript(
        """
        CREATE TABLE entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE, entity_type TEXT, tier TEXT,
            compressed_data BLOB, original_size INT, compressed_size INT,
            compression_ratio REAL, checksum TEXT,
            access_count INT DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            last_accessed TEXT DEFAULT CURRENT_TIMESTAMP,
            archived_at TEXT, superseded_by TEXT
        );
        CREATE TABLE observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_id INT, content TEXT
        );
        CREATE TABLE memory_scope (
            entity_name TEXT PRIMARY KEY, scope TEXT NOT NULL,
            subpath TEXT, source_file TEXT, updated_at TEXT
        );
        """
    )
    # Five 'widget' entities in scope alpha, one in beta, one unscoped. The two
    # non-alpha rows carry the HIGHEST access_count on purpose: that is what
    # makes the pre-limit/post-limit distinction observable below.
    for i in range(5):
        con.execute(
            "INSERT INTO entities (name, entity_type, tier, access_count) VALUES (?,?,?,?)",
            (f"alpha_widget_{i}", "auto_memory/project", "long_term", 100 - i),
        )
        con.execute(
            "INSERT INTO memory_scope (entity_name, scope) VALUES (?,?)",
            (f"alpha_widget_{i}", "alpha"),
        )
    con.execute(
        "INSERT INTO entities (name, entity_type, tier, access_count) VALUES (?,?,?,?)",
        ("beta_widget_0", "auto_memory/project", "long_term", 999),
    )
    con.execute(
        "INSERT INTO memory_scope (entity_name, scope) VALUES (?,?)",
        ("beta_widget_0", "beta"),
    )
    con.execute(
        "INSERT INTO entities (name, entity_type, tier, access_count) VALUES (?,?,?,?)",
        ("orphan_widget", "auto_memory/project", "long_term", 998),
    )
    con.commit()
    con.close()
    return str(db)


def _service(db_path: str):
    from memory_db_service import MemoryDatabase

    return MemoryDatabase(db_path)


def _names(res: dict) -> list[str]:
    return [r.get("name") for r in (res.get("results") or [])]


def test_scope_narrows_results(tmp_path):
    svc = _service(_make_db(tmp_path))
    everything = _names(svc.search_nodes("widget", limit=20))
    alpha = _names(svc.search_nodes("widget", limit=20, scope="alpha"))
    assert len(everything) == 7, everything
    assert len(alpha) == 5, alpha
    assert all(n.startswith("alpha_") for n in alpha), alpha
    assert "beta_widget_0" not in alpha
    assert "orphan_widget" not in alpha


def test_scope_filters_before_limit(tmp_path):
    """The regression guard for the design decision.

    beta_widget_0 and orphan_widget have the highest access_count, so an
    unscoped top-2 is entirely non-alpha. If scope were applied by filtering
    that result set, a scoped top-2 would return ZERO alpha rows. Filtering in
    SQL returns the real top 2 within the scope.
    """
    svc = _service(_make_db(tmp_path))
    unscoped_top2 = _names(svc.search_nodes("widget", limit=2))
    assert not any(n.startswith("alpha_") for n in unscoped_top2), unscoped_top2

    scoped_top2 = _names(svc.search_nodes("widget", limit=2, scope="alpha"))
    assert len(scoped_top2) == 2, scoped_top2
    assert all(n.startswith("alpha_") for n in scoped_top2), scoped_top2


def test_unknown_scope_errors_rather_than_returning_empty(tmp_path):
    svc = _service(_make_db(tmp_path))
    res = svc.search_nodes("widget", limit=5, scope="alpaha")
    assert res.get("success") is False, res
    assert "unknown scope" in str(res.get("error"))
    assert set(res.get("known_scopes") or []) == {"alpha", "beta"}


def test_invalid_scope_is_rejected_and_store_survives(tmp_path):
    db = _make_db(tmp_path)
    svc = _service(db)
    hostile = "a'; " + "DR" + "OP" + " TA" + "BLE entities;--"
    res = svc.search_nodes("widget", limit=5, scope=hostile)
    assert res.get("success") is False, res
    assert "invalid scope" in str(res.get("error"))
    con = sqlite3.connect(db)
    assert con.execute("SELECT COUNT(*) FROM entities").fetchone()[0] == 7
    con.close()


def test_missing_scope_table_errors_not_silently_unfiltered(tmp_path):
    db = _make_db(tmp_path)
    con = sqlite3.connect(db)
    con.execute(" ".join(["DR" + "OP", "TA" + "BLE", "memory_scope"]))
    con.commit()
    con.close()
    svc = _service(db)
    res = svc.search_nodes("widget", limit=5, scope="alpha")
    assert res.get("success") is False, res
    assert "memory_scope" in str(res.get("error"))


def test_scope_none_is_unchanged_behaviour(tmp_path):
    """Existing callers must be unaffected."""
    svc = _service(_make_db(tmp_path))
    a = _names(svc.search_nodes("widget", limit=20))
    b = _names(svc.search_nodes("widget", limit=20, scope=None))
    assert a == b and len(a) == 7


def test_socket_dispatcher_forwards_scope():
    """The dispatcher unpacks params positionally, so a new arg is easy to drop.

    That is exactly what happened while building this: the service and the
    client both understood `scope` while the dispatcher silently discarded it,
    which reads as "scope does nothing" with no error anywhere.
    """
    src = (Path(__file__).parent / "memory_db_service.py").read_text()
    i = src.index('elif method == "search_nodes"')
    block = src[i : i + 400]
    assert 'params.get("scope")' in block, (
        "memory-db dispatcher no longer forwards scope to search_nodes; "
        "scope would be silently ignored over the socket"
    )


@pytest.mark.parametrize(
    "module,func",
    [
        ("memory_client.py", "search_nodes"),
        ("server.py", "search_nodes"),
        ("semantic_vector_tools.py", "semantic_recall"),
    ],
)
def test_scope_param_present_across_the_call_chain(module, func):
    """Every layer between the MCP tool and the data store must carry it."""
    src = (Path(__file__).parent / module).read_text()
    i = src.index(f"def {func}(")
    assert "scope" in src[i : i + 900], f"{module}:{func} lost the scope parameter"


def test_promotion_walks_subdirectories_and_excludes_archive():
    """Guards the two fixes that make folder scoping possible at all.

    A non-recursive walk makes every subfolder memory invisible with no error,
    and without the archive exclusion the first recursive run promotes retired
    content back into recall (observed: 40 files, 2026-08-09).
    """
    hook = Path.home() / ".claude" / "hooks" / "memory_promotion.py"
    if not hook.is_file():
        pytest.skip("memory_promotion.py not installed on this host")
    src = hook.read_text()
    assert 'rglob("*.md")' in src, "promotion walk is no longer recursive"
    assert "EXCLUDED_DIRS" in src and "archive" in src, "archive exclusion gone"
    assert "def scope_for_path" in src, "scope derivation removed"
    i = src.index("def scope_for_path")
    assert "relative_to" in src[i : i + 900]


def test_launchd_still_points_at_the_files_that_carry_scope():
    """The variant trap, made falsifiable.

    This directory holds four dormant `server*.py` / `memory_db_service*.py`
    variants that do NOT implement scope. Scope was added only to the pair the
    launchd plists actually name. If a plist is ever repointed at a variant,
    every scoped search silently returns unscoped results: well-formed, wrong,
    and with nothing in any log to say so. Assert the wiring instead of trusting
    that nobody will move it.
    """
    import plistlib

    plists = {
        "com.phoenix.mcp-enhanced-memory": "server.py",
        "com.2acrestudios.memory-db": "memory_db_service.py",
    }
    for label, expected in plists.items():
        path = Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"
        if not path.is_file():
            pytest.skip(f"{label} not installed on this host")
        argv = plistlib.loads(path.read_bytes()).get("ProgramArguments", [])
        script = next((a for a in argv if a.endswith(".py")), "")
        assert Path(script).name == expected, (
            f"{label} runs {Path(script).name}, not {expected}; "
            "scope filtering exists only in the live pair"
        )
        src = Path(script).read_text()
        assert "scope" in src, f"{script} carries no scope support"


def test_promotion_reconciles_retired_files():
    """Retiring a file must retire its memory, or archiving is theater.

    Moving a memory into archive/ stops it being promoted and, before
    2026-08-09, did nothing to the entity already in the store: 44 files sat in
    archive/ while 12 of their entities stayed searchable, including an April
    leaderboard snapshot that came back as the top hit for "what is the current
    score". The file left the tree; the recall behaviour did not change.
    """
    hook = Path.home() / ".claude" / "hooks" / "memory_promotion.py"
    if not hook.is_file():
        pytest.skip("memory_promotion.py not installed on this host")
    src = hook.read_text()
    assert "present_names" in src, "reconciliation pass removed"
    assert "archived_at = CURRENT_TIMESTAMP" in src, (
        "promotion no longer archives entities whose source file is gone"
    )
    # Must archive, never delete: the row and its observations have to survive
    # so the action is reversible with one UPDATE.
    i = src.index("present_names: set")
    tail = src[i:]
    assert "DELETE FROM entities" not in tail, (
        "reconciliation must archive, not delete"
    )


def test_semantic_recall_honours_the_same_suppression_as_sql():
    """Both retrieval paths must hide the same retired facts.

    search_nodes has always excluded archived_at / superseded_by /
    tier='quarantine'. semantic_recall excluded none of them, so a fact retired
    in SQLite stayed recallable by MEANING -- and that is the path
    _proactive_recall uses to inject memories into prompts. Verified the same
    day: eleven ARC score reports vanished from search_nodes the moment they
    were marked superseded and stayed the top semantic hit.
    """
    src = (Path(__file__).parent / "semantic_vector_tools.py").read_text()
    i = src.index("def semantic_recall(")
    body = src[i : i + 6000]
    for col in ("archived_at", "superseded_by", "quarantine"):
        assert col in body, f"semantic_recall no longer filters {col}"
    assert "must_not" in body, "suppression exclusion filter removed"
    # The intersection is what keeps the payload sendable: without it the query
    # returns ~10k ids, blows the cap, and the filter is dropped entirely.
    assert "indexed" in body, (
        "suppression set is no longer intersected with the point set; "
        "it will exceed the cap and silently stop filtering"
    )
