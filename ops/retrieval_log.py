#!/usr/bin/env python3
"""retrieval_log.py

Append-only retrieval telemetry for the enhanced-memory store. This is the
precondition for every compounding rule: promotion-by-use and decay-by-disuse
are both unimplementable without a record of what was retrieved, when, in which
session, and whether it was acted on.

The existing `retrieval_contexts` table has been frozen at 5 seed rows since
2025-12-02 and stores static "context profiles", not actual retrievals. This
module adds a dedicated `retrieval_log` table (additive; it never reads or
writes any existing table) and the two functions the search path needs.

Table (one row per returned entity, so joins to `entities` are trivial):
    retrieval_log(
        id, session_id, query, entity_id, rank, source, retrieved_at, cited
    )

Public API (import this from the MCP server's search path):
    init_schema(db_path=None)
    log_retrieval(session_id, query, entity_ids, source, db_path=None)
    mark_cited(session_id, entity_id, db_path=None)

CLI:
    python3 retrieval_log.py --init        # create the table (additive, idempotent)
    python3 retrieval_log.py --selftest    # prove write->read->metric, then clean up
    python3 retrieval_log.py --status      # row count + distinct sessions

Wiring (the one change that turns this from infrastructure into a live signal)
belongs in the MCP server search functions and is staged as an L-tier proposal,
not applied here. The exact call:

    from ops.retrieval_log import log_retrieval
    log_retrieval(session_id, query, [r["id"] for r in results], source="search_nodes")

Env:
    ENHANCED_MEMORY_DB_PATH   override DB path
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

# Run as a script from this subdirectory, so the repo root is not on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory_paths import get_db_path  # noqa: E402


def _db_path(override: str | None = None) -> Path:
    if override:
        return Path(override).expanduser()
    return get_db_path()


def _connect(db_path: str | None = None) -> sqlite3.Connection:
    p = _db_path(db_path)
    # Read-write connection with a busy timeout: WAL lets this coexist with the
    # live daemon's readers; writes serialize briefly on the single writer lock.
    conn = sqlite3.connect(str(p), timeout=15)
    conn.execute("PRAGMA busy_timeout=15000")
    return conn


SCHEMA = """
CREATE TABLE IF NOT EXISTS retrieval_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT NOT NULL,
    query        TEXT,
    entity_id    INTEGER NOT NULL,
    rank         INTEGER,
    source       TEXT,
    retrieved_at TIMESTAMP NOT NULL,
    cited        INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_retrieval_log_entity  ON retrieval_log(entity_id);
CREATE INDEX IF NOT EXISTS idx_retrieval_log_session ON retrieval_log(session_id);
CREATE INDEX IF NOT EXISTS idx_retrieval_log_time    ON retrieval_log(retrieved_at);
"""


def init_schema(db_path: str | None = None) -> None:
    conn = _connect(db_path)
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_retrieval(
    session_id: str,
    query: str,
    entity_ids,
    source: str = "unknown",
    db_path: str | None = None,
) -> int:
    """Append one row per returned entity. Returns rows written.

    Defensive by design: a telemetry failure must NEVER break a retrieval, so
    the caller should wrap this in try/except and ignore errors. It does not
    raise on an empty result set; it simply writes nothing.
    """
    ids = [int(e) for e in (entity_ids or [])]
    if not ids:
        return 0
    ts = _now()
    rows = [(session_id, query, eid, rank, source, ts) for rank, eid in enumerate(ids)]
    conn = _connect(db_path)
    try:
        conn.executemany(
            "INSERT INTO retrieval_log (session_id, query, entity_id, rank, source, retrieved_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()
        return len(rows)
    finally:
        conn.close()


def mark_cited(session_id: str, entity_id: int, db_path: str | None = None) -> int:
    """Mark the most recent retrieval of an entity in a session as acted-on."""
    conn = _connect(db_path)
    try:
        cur = conn.execute(
            "UPDATE retrieval_log SET cited=1 WHERE id = ("
            "  SELECT id FROM retrieval_log WHERE session_id=? AND entity_id=? "
            "  ORDER BY retrieved_at DESC LIMIT 1)",
            (session_id, int(entity_id)),
        )
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


def status(db_path: str | None = None) -> dict:
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='retrieval_log'"
        ).fetchone()
        if not row:
            return {"exists": False}
        n = conn.execute("SELECT COUNT(*) FROM retrieval_log").fetchone()[0]
        s = conn.execute(
            "SELECT COUNT(DISTINCT session_id) FROM retrieval_log"
        ).fetchone()[0]
        cited = conn.execute(
            "SELECT COUNT(*) FROM retrieval_log WHERE cited=1"
        ).fetchone()[0]
        return {"exists": True, "rows": n, "sessions": s, "cited": cited}
    finally:
        conn.close()


_SELFTEST_SESSION = "__selftest__"


def selftest(db_path: str | None = None) -> bool:
    """Prove the full loop end-to-end on synthetic data, then remove it.

    Uses a sentinel session id and a guaranteed-real entity_id (the lowest id in
    `entities`) so the cross-session join in the metric actually fires, then
    deletes only its own synthetic rows. Touches no real telemetry.
    """
    init_schema(db_path)
    conn = _connect(db_path)
    try:
        ent = conn.execute("SELECT MIN(id) FROM entities").fetchone()[0]
        if ent is None:
            print("SELFTEST: no entities to reference; cannot test join")
            return False
        # write
        n = log_retrieval(
            _SELFTEST_SESSION, "selftest query", [ent], "selftest", db_path
        )
        assert n == 1, f"expected 1 row written, got {n}"
        # read back
        got = conn.execute(
            "SELECT entity_id, source FROM retrieval_log WHERE session_id=?",
            (_SELFTEST_SESSION,),
        ).fetchall()
        assert got and got[0][0] == ent, "read-back mismatch"
        # cite
        c = mark_cited(_SELFTEST_SESSION, ent, db_path)
        assert c == 1, f"expected 1 cited update, got {c}"
        cited = conn.execute(
            "SELECT cited FROM retrieval_log WHERE session_id=?", (_SELFTEST_SESSION,)
        ).fetchone()[0]
        assert cited == 1, "cite flag not set"
        print(f"SELFTEST: write=ok read=ok cite=ok (entity_id={ent})")
        return True
    finally:
        # clean up only our synthetic rows
        conn.execute(
            "DELETE FROM retrieval_log WHERE session_id=?", (_SELFTEST_SESSION,)
        )
        conn.commit()
        conn.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--init", action="store_true", help="create retrieval_log table (additive)"
    )
    ap.add_argument(
        "--selftest",
        action="store_true",
        help="prove write/read/cite loop, then clean up",
    )
    ap.add_argument(
        "--status", action="store_true", help="row count + distinct sessions"
    )
    args = ap.parse_args()
    if args.init:
        init_schema()
        print("retrieval_log schema ready")
    if args.selftest:
        ok = selftest()
        if not ok:
            return 1
    if args.status:
        print(status())
    if not (args.init or args.selftest or args.status):
        ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
