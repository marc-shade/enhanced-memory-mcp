#!/usr/bin/env python3
"""Write-path vector indexing for enhanced-memory (Phase 0 spine repair, 2026-07-02).

Root cause fixed here: create_entities never indexed into Qdrant, and the only
backfill ever run was --auto-only, leaving semantic_recall blind to 96% of the
store (audit: 364 points vs 9,841 entities).

Design: one self-healing choke point instead of per-writer hooks. Writers are
many (MCP create_entities, memory_promotion.py direct sqlite, consolidation
jobs), so correctness comes from the SWEEP, not from catching every write site:

  - `index_entities(ids)`   — immediate best-effort indexing (called by
                              create_entities for freshness).
  - `sweep(limit)`          — indexes every entity whose vector_indexed_at is
                              NULL or older than its newest observation. Catches
                              ALL writers, including direct-sqlite ones.
  - `start_sweeper(...)`    — daemon background thread: sweep every 5 min plus
                              recurring working-tier TTL demotion (config
                              ttl_minutes=60; the dedicated working_memory
                              table was never used, tier lives on entities).

Embedding: local ollama embeddinggemma (768d) via local_semantic_recall.embed,
byte-identical to the query path used by semantic_recall — no doc/query model
drift. Collection: collection_for(DEFAULT_MODEL). Archive/quarantine tiers are
excluded from indexing and evicted from the collection if present.

Fail-soft everywhere: an unreachable embedder leaves vector_indexed_at NULL and
the next sweep retries; nothing raises into the caller.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from typing import Iterable, Optional

from local_semantic_recall import (
    DB,
    DEFAULT_MODEL,
    QDRANT,
    collection_for,
    embed,
    ensure_collection,
)

logger = logging.getLogger(__name__)

EXCLUDED_TIERS = ("archive", "quarantine")
SWEEP_INTERVAL_SECONDS = 300
SWEEP_BATCH = 200
WORKING_TTL_MINUTES = 60
_TEXT_CAP = 4000
_OBS_PER_ENTITY = 8

_sweeper_thread: Optional[threading.Thread] = None
_stop = threading.Event()


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB, timeout=30)
    conn.execute("PRAGMA busy_timeout=30000")
    conn.row_factory = sqlite3.Row
    return conn


def _entity_text(conn: sqlite3.Connection, entity_id: int) -> Optional[tuple]:
    row = conn.execute(
        "SELECT id, name, entity_type, tier, superseded_by FROM entities WHERE id=?",
        (entity_id,),
    ).fetchone()
    if (
        row is None
        or (row["tier"] or "") in EXCLUDED_TIERS
        or row["superseded_by"] is not None
    ):
        return None
    obs = conn.execute(
        "SELECT content FROM observations WHERE entity_id=? ORDER BY id LIMIT ?",
        (entity_id, _OBS_PER_ENTITY),
    ).fetchall()
    text = (row["name"] + ": " + " ".join(o["content"] for o in obs)).strip()
    if len(text) <= 8:
        return None
    return (row["id"], row["name"], row["entity_type"], row["tier"], text[:_TEXT_CAP])


def index_entities(entity_ids: Iterable[int]) -> dict:
    """Embed + upsert the given entities. Best-effort; returns counts."""
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import PointStruct
    except ImportError as e:
        return {"indexed": 0, "skipped": 0, "error": f"qdrant_client missing: {e}"}

    conn = _connect()
    rows = []
    try:
        for eid in entity_ids:
            r = _entity_text(conn, int(eid))
            if r:
                rows.append(r)
        if not rows:
            return {"indexed": 0, "skipped": 0}
        try:
            vecs = embed([r[4] for r in rows], DEFAULT_MODEL)
        except Exception as e:
            logger.warning(f"vector indexing skipped (embedder unavailable): {e}")
            return {"indexed": 0, "skipped": len(rows), "error": str(e)}

        client = QdrantClient(url=QDRANT)
        coll = collection_for(DEFAULT_MODEL)
        ensure_collection(client, coll, len(vecs[0]))

        # Salience revival (Phase 2, 2026-07-02): every live write carried the
        # 0.5 schema default because nothing computed salience. Novelty is the
        # cheapest honest proxy and the embedding is already in hand: salience
        # = 1 - similarity(nearest OTHER point), queried BEFORE upserting this
        # batch so an entity never matches itself. Only rows still at the 0.5
        # default are updated (explicit salience is never clobbered).
        saliences: dict[int, float] = {}
        for r, v in zip(rows, vecs):
            try:
                hits = client.query_points(coll, query=v, limit=2).points
                others = [h for h in hits if h.id != r[0]]
                if others:
                    saliences[r[0]] = round(
                        max(0.0, min(1.0, 1.0 - others[0].score)), 3
                    )
            except Exception:
                pass  # salience stays at default; indexing must not fail on this

        client.upsert(
            coll,
            points=[
                PointStruct(
                    id=r[0],
                    vector=v,
                    payload={"name": r[1], "entity_type": r[2], "tier": r[3]},
                )
                for r, v in zip(rows, vecs)
            ],
        )
        conn.executemany(
            "UPDATE entities SET vector_indexed_at=CURRENT_TIMESTAMP WHERE id=?",
            [(r[0],) for r in rows],
        )
        if saliences:
            conn.executemany(
                "UPDATE entities SET salience_score=? WHERE id=? AND salience_score=0.5",
                [(s, eid) for eid, s in saliences.items()],
            )
        conn.commit()
        return {"indexed": len(rows), "skipped": 0, "salience_scored": len(saliences)}
    except Exception as e:
        logger.warning(f"vector indexing error: {e}")
        return {"indexed": 0, "skipped": 0, "error": str(e)}
    finally:
        conn.close()


def pending_ids(limit: int = SWEEP_BATCH) -> list[int]:
    """Entities needing (re-)indexing: never indexed, or observations newer
    than the last indexing. Excludes archive/quarantine tiers."""
    conn = _connect()
    try:
        rows = conn.execute(
            f"""SELECT e.id FROM entities e
                WHERE COALESCE(e.tier,'') NOT IN {EXCLUDED_TIERS!r}
                  AND e.superseded_by IS NULL
                  AND (e.vector_indexed_at IS NULL
                       OR e.vector_indexed_at <
                          (SELECT MAX(o.created_at) FROM observations o
                           WHERE o.entity_id = e.id))
                ORDER BY e.id LIMIT ?""",
            (limit,),
        ).fetchall()
        return [r["id"] for r in rows]
    finally:
        conn.close()


def evict_excluded() -> int:
    """Remove archive/quarantine-tier points from the collection (hygiene)."""
    try:
        from qdrant_client import QdrantClient

        conn = _connect()
        ids = [
            r["id"]
            for r in conn.execute(
                f"""SELECT id FROM entities
                    WHERE COALESCE(tier,'') IN {EXCLUDED_TIERS!r}
                      AND vector_indexed_at IS NOT NULL"""
            ).fetchall()
        ]
        if ids:
            QdrantClient(url=QDRANT).delete(
                collection_for(DEFAULT_MODEL), points_selector=ids
            )
            conn.executemany(
                "UPDATE entities SET vector_indexed_at=NULL WHERE id=?",
                [(i,) for i in ids],
            )
            conn.commit()
        conn.close()
        return len(ids)
    except Exception as e:
        logger.warning(f"evict_excluded error: {e}")
        return 0


def demote_stale_working() -> int:
    """Recurring TTL enforcement for the working tier (config: 60 min)."""
    conn = _connect()
    try:
        cur = conn.execute(
            """UPDATE entities SET tier='reference'
               WHERE tier='working'
                 AND datetime(COALESCE(last_accessed, created_at))
                     < datetime('now', ?)""",
            (f"-{WORKING_TTL_MINUTES} minutes",),
        )
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


def sweep(limit: int = SWEEP_BATCH) -> dict:
    """One sweep pass: index pending entities, enforce working TTL."""
    ids = pending_ids(limit)
    result = index_entities(ids) if ids else {"indexed": 0, "skipped": 0}
    result["pending_before"] = len(ids)
    result["ttl_demoted"] = demote_stale_working()
    return result


def coverage() -> dict:
    """Honest coverage stats for health reporting."""
    conn = _connect()
    try:
        eligible = conn.execute(
            f"SELECT COUNT(*) FROM entities WHERE COALESCE(tier,'') NOT IN {EXCLUDED_TIERS!r}"
        ).fetchone()[0]
        indexed = conn.execute(
            f"""SELECT COUNT(*) FROM entities
                WHERE COALESCE(tier,'') NOT IN {EXCLUDED_TIERS!r}
                  AND vector_indexed_at IS NOT NULL"""
        ).fetchone()[0]
    finally:
        conn.close()
    points = None
    try:
        from qdrant_client import QdrantClient

        points = QdrantClient(url=QDRANT).count(collection_for(DEFAULT_MODEL)).count
    except Exception:
        pass
    return {
        "eligible_entities": eligible,
        "indexed_entities": indexed,
        "coverage_pct": round(100.0 * indexed / eligible, 1) if eligible else 0.0,
        "qdrant_points": points,
        "collection": collection_for(DEFAULT_MODEL),
        "model": DEFAULT_MODEL,
    }


def backfill_salience(batch: int = 500) -> dict:
    """One-shot novelty-salience backfill for already-indexed entities still at
    the 0.5 schema default (Phase 2, 2026-07-02). Uses vectors already in
    Qdrant (retrieve by id, query nearest other) — no re-embedding."""
    try:
        from qdrant_client import QdrantClient
    except ImportError as e:
        return {"scored": 0, "error": str(e)}
    client = QdrantClient(url=QDRANT)
    coll = collection_for(DEFAULT_MODEL)
    conn = _connect()
    scored = 0
    try:
        ids = [
            r["id"]
            for r in conn.execute(
                f"""SELECT id FROM entities
                    WHERE COALESCE(tier,'') NOT IN {EXCLUDED_TIERS!r}
                      AND superseded_by IS NULL
                      AND vector_indexed_at IS NOT NULL
                      AND salience_score = 0.5"""
            ).fetchall()
        ]
        for i in range(0, len(ids), batch):
            chunk = ids[i : i + batch]
            recs = client.retrieve(coll, ids=chunk, with_vectors=True)
            updates = []
            for rec in recs:
                if rec.vector is None:
                    continue
                hits = client.query_points(coll, query=rec.vector, limit=2).points
                others = [h for h in hits if h.id != rec.id]
                if others:
                    sal = round(max(0.0, min(1.0, 1.0 - others[0].score)), 3)
                    updates.append((sal, rec.id))
            if updates:
                conn.executemany(
                    "UPDATE entities SET salience_score=? WHERE id=? AND salience_score=0.5",
                    updates,
                )
                conn.commit()
                scored += len(updates)
        return {"candidates": len(ids), "scored": scored}
    finally:
        conn.close()


def _sweeper_loop(interval: float) -> None:
    while not _stop.wait(interval):
        try:
            r = sweep()
            if r.get("indexed") or r.get("ttl_demoted"):
                logger.info(f"vector sweep: {r}")
        except Exception as e:
            logger.warning(f"vector sweeper iteration failed: {e}")


def sweeper_alive() -> bool:
    """Whether the background sweeper thread is running in THIS process.

    The sweeper is an in-process thread, not a launchd job, so it is invisible
    to `launchctl list` and (because this module's logger output does not reach
    the daemon log files) leaves no startup line either. Health checks that
    report only `pending_index` cannot distinguish a healthy sweeper behind a
    burst of writes from a dead one, so expose the mechanism directly.
    """
    return _sweeper_thread is not None and _sweeper_thread.is_alive()


def start_sweeper(interval: float = SWEEP_INTERVAL_SECONDS) -> bool:
    """Start the background sweeper thread (idempotent)."""
    global _sweeper_thread
    if _sweeper_thread is not None and _sweeper_thread.is_alive():
        return False
    _stop.clear()
    _sweeper_thread = threading.Thread(
        target=_sweeper_loop, args=(interval,), name="vector-sweeper", daemon=True
    )
    _sweeper_thread.start()
    logger.info(f"vector sweeper started (interval={interval}s, model={DEFAULT_MODEL})")
    return True


if __name__ == "__main__":
    import argparse
    import json

    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sweep", help="index pending entities")
    s.add_argument("--all", action="store_true", help="loop until no pending remain")
    sub.add_parser("coverage")
    sub.add_parser("evict", help="remove archive/quarantine points from collection")
    sub.add_parser("salience-backfill", help="novelty salience for 0.5-default rows")
    args = ap.parse_args()
    if args.cmd == "coverage":
        print(json.dumps(coverage(), indent=2))
    elif args.cmd == "evict":
        print(json.dumps({"evicted": evict_excluded()}))
    elif args.cmd == "salience-backfill":
        print(json.dumps(backfill_salience(), indent=2))
    else:
        total = 0
        t0 = time.time()
        while True:
            r = sweep()
            total += r.get("indexed", 0)
            print(f"  batch: {r}  (total {total}, {time.time() - t0:.0f}s)")
            if r.get("error") or r["pending_before"] < SWEEP_BATCH or not args.all:
                break
        print(json.dumps(coverage(), indent=2))
