#!/usr/bin/env python3
"""memory_quarantine.py

Move never-retrieved bulk-imported entities out of the default search path so
recall precision over the memories that matter (the about-you layer) stops
being diluted by a corpus that has never been read.

Why this exists: 66% of the store is a single Jan-2026 `platonic_insight`
bulk import with 0 recorded retrievals in 5 months, plus a cold long tail of
generic types. Substring search over ~9.5k rows has poor precision; the 393
auto_memory/* entities that are actually about the user are the needle in that
haystack. Shrinking the haystack is the cheapest recall win available.

SAFETY MODEL
------------
- Default mode is DRY-RUN: read-only (URI mode=ro). It reports exactly what
  would be archived and the child-row blast radius. It mutates nothing.
- APPLY is reversible and additive: it adds a nullable `archived_at` column to
  `entities` (default NULL = unchanged behavior) and stamps matching rows. No
  row is deleted; no child row is touched; restore is a single UPDATE.
- APPLY is gated behind an explicit flag AND refuses to run while the memory
  daemon holds the DB open, because mutating 6k rows under a live 24/7 writer
  is exactly the class of action that should be done with the daemon quiesced.

QUARANTINE PREDICATE (conservative; protects everything that could compound)
    pinned = 0
    AND entity_type NOT LIKE 'auto_memory/%'   -- never archive about-you memory
    AND access_count = 0                        -- no recorded retrieval
    AND last_accessed = created_at              -- never re-accessed
    AND created_at < (now - AGE_DAYS)           -- old enough (default 90d)

The about-you layer is cold too, but archiving it would defeat the entire
purpose: we want those resurfaced, not hidden. They are explicitly excluded.

Usage:
    python3 memory_quarantine.py                       # dry-run (safe)
    python3 memory_quarantine.py --age-days 120        # tune the age cutoff
    python3 memory_quarantine.py --json                # machine-readable dry-run
    python3 memory_quarantine.py --apply --confirm     # gated; refuses if daemon is live
    python3 memory_quarantine.py --restore --confirm   # un-archive everything

Env:
    ENHANCED_MEMORY_DB_PATH   override DB path
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

PROTECT_TYPE_PREFIX = "auto_memory/"
DEFAULT_AGE_DAYS = 90

# Types that are safe to archive on the cold-proxy alone: known one-time bulk
# imports with zero plausible about-the-user value. Start minimal. Broadening
# this list on access_count=0 is a Goodhart trap, because access_count is the
# very signal we already proved unreliable. Anything not on this list must wait
# for the real retrieval_log signal before it is eligible for archive.
ARCHIVE_ALLOWLIST = ("platonic_insight",)

# Cold conditions shared by both modes. Protects the about-you layer and pins.
_COLD = (
    "pinned = 0 "
    "AND entity_type NOT LIKE 'auto_memory/%' "
    "AND access_count = 0 "
    "AND last_accessed = created_at "
    "AND created_at < datetime('now', ?)"
)


def _predicate(broad: bool) -> str:
    """Return the WHERE body. Default (safe) restricts to the allowlist; --broad
    trusts the cold proxy across all non-protected types and must only be used
    once retrieval_log corroborates disuse."""
    if broad:
        return _COLD
    placeholders = ",".join("?" for _ in ARCHIVE_ALLOWLIST)
    return f"{_COLD} AND entity_type IN ({placeholders})"


def _params(age_days: int, broad: bool) -> tuple:
    age = _age_arg(age_days)
    if broad:
        return (age,)
    return (age, *ARCHIVE_ALLOWLIST)


def _db_path() -> Path:
    env = os.environ.get("ENHANCED_MEMORY_DB_PATH")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".claude" / "enhanced_memories" / "memory.db"


def _connect_ro(p: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def _connect_rw(p: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(p), timeout=15)
    conn.execute("PRAGMA busy_timeout=15000")
    return conn


def _age_arg(age_days: int) -> str:
    return f"-{int(age_days)} days"


def _daemon_holds_db(p: Path) -> bool:
    """True if another process holds the DB open (the live MCP daemon)."""
    try:
        out = subprocess.run(
            ["lsof", str(p)], capture_output=True, text=True, timeout=10
        ).stdout
        return bool(out.strip())
    except Exception:
        # If we cannot tell, assume it IS held (fail-closed for safety).
        return True


def dry_run(age_days: int, broad: bool = False) -> dict:
    p = _db_path()
    pred = _predicate(broad)
    params = _params(age_days, broad)
    conn = _connect_ro(p)
    try:
        total_match = conn.execute(
            f"SELECT COUNT(*) FROM entities WHERE {pred}", params
        ).fetchone()[0]
        bytes_match = (
            conn.execute(
                f"SELECT COALESCE(SUM(compressed_size),0) FROM entities WHERE {pred}",
                params,
            ).fetchone()[0]
            or 0
        )
        by_type = [
            dict(r)
            for r in conn.execute(
                f"SELECT entity_type, COUNT(*) AS n FROM entities WHERE {pred} "
                "GROUP BY entity_type ORDER BY n DESC LIMIT 20",
                params,
            )
        ]
        store_total = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        # blast radius: child rows referencing the matched entities (informational)
        child = {}
        for tbl in ("observations", "episodic_memory", "forgetting_curves"):
            try:
                child[tbl] = conn.execute(
                    f"SELECT COUNT(*) FROM {tbl} WHERE entity_id IN "
                    f"(SELECT id FROM entities WHERE {pred})",
                    params,
                ).fetchone()[0]
            except sqlite3.OperationalError:
                child[tbl] = None
        # what's protected by the predicate (sanity: about-you layer untouched)
        protected_about_you = conn.execute(
            "SELECT COUNT(*) FROM entities WHERE entity_type LIKE 'auto_memory/%' "
            "AND access_count=0"
        ).fetchone()[0]
        return {
            "db_path": str(p),
            "mode": "broad (cold proxy, all non-protected types)"
            if broad
            else f"safe (allowlist: {', '.join(ARCHIVE_ALLOWLIST)})",
            "age_days": age_days,
            "store_total": store_total,
            "match_count": total_match,
            "match_pct_of_store": round(100 * total_match / store_total, 1)
            if store_total
            else 0,
            "reclaimable_compressed_bytes": bytes_match,
            "by_type": by_type,
            "child_rows_note": child,
            "protected_about_you_cold": protected_about_you,
            "remaining_after": store_total - total_match,
        }
    finally:
        conn.close()


def _ensure_archived_column(conn: sqlite3.Connection) -> None:
    cols = [r[1] for r in conn.execute("PRAGMA table_info(entities)")]
    if "archived_at" not in cols:
        conn.execute("ALTER TABLE entities ADD COLUMN archived_at TIMESTAMP")


def apply(age_days: int, confirm: bool, broad: bool = False) -> dict:
    if not confirm:
        raise SystemExit("apply requires --confirm")
    p = _db_path()
    if _daemon_holds_db(p):
        raise SystemExit(
            "REFUSING: the memory daemon holds the DB open. Quiesce it first "
            "(stop the enhanced-memory MCP daemon), then re-run with --apply --confirm. "
            "Mutating rows under a live 24/7 writer is not safe to do unilaterally."
        )
    pred = _predicate(broad)
    params = _params(age_days, broad)
    conn = _connect_rw(p)
    try:
        _ensure_archived_column(conn)
        cur = conn.execute(
            f"UPDATE entities SET archived_at=datetime('now') "
            f"WHERE archived_at IS NULL AND {pred}",
            params,
        )
        conn.commit()
        return {"archived": cur.rowcount}
    finally:
        conn.close()


def restore(confirm: bool) -> dict:
    if not confirm:
        raise SystemExit("restore requires --confirm")
    p = _db_path()
    conn = _connect_rw(p)
    try:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(entities)")]
        if "archived_at" not in cols:
            return {"restored": 0, "note": "no archived_at column; nothing to restore"}
        cur = conn.execute(
            "UPDATE entities SET archived_at=NULL WHERE archived_at IS NOT NULL"
        )
        conn.commit()
        return {"restored": cur.rowcount}
    finally:
        conn.close()


def render(d: dict) -> str:
    out = ["MEMORY QUARANTINE -- DRY RUN (nothing changed)", f"DB: {d['db_path']}", ""]
    out.append(f"mode: {d['mode']}")
    out.append(
        f"store={d['store_total']}  would-archive={d['match_count']} "
        f"({d['match_pct_of_store']}% of store)  remaining-after={d['remaining_after']}"
    )
    out.append(f"reclaimable (compressed bytes): {d['reclaimable_compressed_bytes']:,}")
    out.append(
        f"protected about-you cold memories (NOT touched): {d['protected_about_you_cold']}"
    )
    out.append("")
    out.append("would archive, by type:")
    for r in d["by_type"]:
        out.append(f"  {r['entity_type']:<26} {r['n']}")
    out.append("")
    out.append("child rows referencing matched entities (left intact; informational):")
    for k, v in d["child_rows_note"].items():
        out.append(f"  {k:<20} {v}")
    out.append("")
    out.append(
        "APPLY is gated: --apply --confirm, and only when the daemon is quiesced."
    )
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--age-days", type=int, default=DEFAULT_AGE_DAYS)
    ap.add_argument("--apply", action="store_true", help="stamp archived_at (gated)")
    ap.add_argument(
        "--restore", action="store_true", help="clear archived_at on all rows"
    )
    ap.add_argument("--confirm", action="store_true", help="required for apply/restore")
    ap.add_argument(
        "--broad",
        action="store_true",
        help="archive ALL non-protected cold types on the access_count proxy "
        "(Goodhart risk; only use once retrieval_log corroborates disuse)",
    )
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.restore:
        print(json.dumps(restore(args.confirm)))
        return 0
    if args.apply:
        print(json.dumps(apply(args.age_days, args.confirm, args.broad)))
        return 0
    d = dry_run(args.age_days, args.broad)
    print(json.dumps(d, indent=2, default=str) if args.json else render(d))
    return 0


if __name__ == "__main__":
    sys.exit(main())
