#!/usr/bin/env python3
"""Phase 0 spine repair migration (2026-07-02).

Per docs/plans/memory-world-class-audit-and-roadmap-2026-07.md items 0.2/0.3/0.5:
  1. Index observations(entity_id)            (hot join full-scanned a 64K table)
  2. Quarantine orphaned observations          (1,380 rows w/ no parent entity)
  3. FTS5 external-content index over observations + sync triggers
  4. Archive the platonic_insight legacy mass  (tier='archive', archived_at set)
  5. Quarantine zero-observation entities that also have no graph edges
  6. One-time working-tier TTL demotion        (recurring TTL: vector_write_indexer sweeper)
  7. entities.vector_indexed_at column         (write-path indexing bookkeeping)

Idempotent: every step guards on current state. Safe against the live WAL DB
(30s busy_timeout). Run with the mcp venv python. Backup taken first:
/Volumes/FILES/tmp/memdb-backup/memory-pre-phase0-20260702.db
"""

import sqlite3
import sys
from pathlib import Path

# Run as a script from this subdirectory, so the repo root is not on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory_paths import get_db_path  # noqa: E402

DB = str(get_db_path())


def main() -> int:
    conn = sqlite3.connect(DB, timeout=30)
    conn.execute("PRAGMA busy_timeout=30000")
    cur = conn.cursor()
    report = {}

    # 1. hot-join index
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_observations_entity ON observations(entity_id)"
    )
    report["observations_entity_index"] = "present"

    # 2. quarantine orphaned observations (keep rows, move them out)
    cur.execute(
        """CREATE TABLE IF NOT EXISTS quarantine_orphaned_observations
           AS SELECT o.* FROM observations o
              LEFT JOIN entities e ON o.entity_id = e.id
              WHERE e.id IS NULL AND 0"""
    )
    cur.execute(
        """INSERT INTO quarantine_orphaned_observations
           SELECT o.* FROM observations o
           LEFT JOIN entities e ON o.entity_id = e.id
           WHERE e.id IS NULL
           AND o.id NOT IN (SELECT id FROM quarantine_orphaned_observations)"""
    )
    cur.execute(
        """DELETE FROM observations WHERE id IN
           (SELECT id FROM quarantine_orphaned_observations)"""
    )
    report["orphaned_observations_quarantined"] = cur.execute(
        "SELECT COUNT(*) FROM quarantine_orphaned_observations"
    ).fetchone()[0]

    # 3. FTS5 over observations (external content) + sync triggers + rebuild
    fts_exists = cur.execute(
        "SELECT 1 FROM sqlite_master WHERE name='observations_fts'"
    ).fetchone()
    if not fts_exists:
        cur.execute(
            """CREATE VIRTUAL TABLE observations_fts USING fts5(
                 content, content='observations', content_rowid='id')"""
        )
        cur.execute(
            """CREATE TRIGGER obs_fts_ai AFTER INSERT ON observations BEGIN
                 INSERT INTO observations_fts(rowid, content) VALUES (new.id, new.content);
               END"""
        )
        cur.execute(
            """CREATE TRIGGER obs_fts_ad AFTER DELETE ON observations BEGIN
                 INSERT INTO observations_fts(observations_fts, rowid, content)
                 VALUES('delete', old.id, old.content);
               END"""
        )
        cur.execute(
            """CREATE TRIGGER obs_fts_au AFTER UPDATE ON observations BEGIN
                 INSERT INTO observations_fts(observations_fts, rowid, content)
                 VALUES('delete', old.id, old.content);
                 INSERT INTO observations_fts(rowid, content) VALUES (new.id, new.content);
               END"""
        )
        cur.execute("INSERT INTO observations_fts(observations_fts) VALUES('rebuild')")
    report["fts_rows"] = cur.execute(
        "SELECT COUNT(*) FROM observations_fts"
    ).fetchone()[0]

    # 4. archive the legacy platonic_insight mass
    cur.execute(
        """UPDATE entities SET tier='archive', archived_at=CURRENT_TIMESTAMP
           WHERE entity_type='platonic_insight' AND tier != 'archive'"""
    )
    report["platonic_archived_now"] = cur.rowcount
    report["archive_tier_total"] = cur.execute(
        "SELECT COUNT(*) FROM entities WHERE tier='archive'"
    ).fetchone()[0]

    # 5. quarantine empty entities with no graph edges (tier flag, no deletion)
    cur.execute(
        """UPDATE entities SET tier='quarantine', archived_at=CURRENT_TIMESTAMP
           WHERE id IN (
             SELECT e.id FROM entities e
             LEFT JOIN observations o ON o.entity_id = e.id
             WHERE o.id IS NULL
               AND e.tier NOT IN ('archive','quarantine')
               AND e.id NOT IN (SELECT from_entity_id FROM relations)
               AND e.id NOT IN (SELECT to_entity_id FROM relations)
               AND e.id NOT IN (SELECT cause_entity_id FROM causal_links)
               AND e.id NOT IN (SELECT effect_entity_id FROM causal_links)
           )"""
    )
    report["empty_entities_quarantined_now"] = cur.rowcount

    # 6. one-time TTL demotion of stale working-tier rows (config: 60 min TTL)
    cur.execute(
        """UPDATE entities SET tier='reference'
           WHERE tier='working'
             AND datetime(COALESCE(last_accessed, created_at)) < datetime('now','-60 minutes')"""
    )
    report["working_ttl_demoted_now"] = cur.rowcount

    # 7. vector indexing bookkeeping column
    cols = [r[1] for r in cur.execute("PRAGMA table_info(entities)").fetchall()]
    if "vector_indexed_at" not in cols:
        cur.execute("ALTER TABLE entities ADD COLUMN vector_indexed_at TIMESTAMP")
    report["vector_indexed_at_column"] = "present"

    conn.commit()

    # verification block
    report["remaining_orphans"] = cur.execute(
        """SELECT COUNT(*) FROM observations o
           LEFT JOIN entities e ON o.entity_id=e.id WHERE e.id IS NULL"""
    ).fetchone()[0]
    report["fts_smoke"] = cur.execute(
        "SELECT COUNT(*) FROM observations_fts WHERE observations_fts MATCH 'kumquat'"
    ).fetchone()[0]
    plan = cur.execute(
        "EXPLAIN QUERY PLAN SELECT * FROM observations WHERE entity_id=42"
    ).fetchall()
    report["obs_query_plan_uses_index"] = any(
        "idx_observations_entity" in str(r) for r in plan
    )
    report["tier_distribution"] = dict(
        cur.execute("SELECT tier, COUNT(*) FROM entities GROUP BY tier").fetchall()
    )
    conn.close()

    for k, v in report.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
