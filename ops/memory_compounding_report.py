#!/usr/bin/env python3
"""memory_compounding_report.py

Read-only diagnostic for the enhanced-memory store. Answers one question:
is the memory getting smarter about the user across sessions, or just
accumulating?

It opens the live SQLite DB in read-only mode (URI mode=ro), so it is safe to
run while the MCP daemon holds the DB open in WAL mode. It mutates nothing.

Two classes of metric are reported and clearly separated:

  1. CURRENT STATE (proxy) -- derived from access_count / last_accessed /
     created_at, which is the only retrieval signal that exists today. This is
     a PROXY: access_count is only incremented by some code paths, so a 0 means
     "no recorded retrieval", not provably "never read".

  2. TRUE COMPOUNDING METRIC -- cross-session resurfacing rate. Requires the
     retrieval_log table (see retrieval_log.py). Until that table has data this
     section reports N/A and says so, rather than fabricating a number.

Usage:
    python3 memory_compounding_report.py              # human-readable report
    python3 memory_compounding_report.py --json       # machine-readable
    ENHANCED_MEMORY_DB_PATH=/path/to.db python3 memory_compounding_report.py

Env:
    ENHANCED_MEMORY_DB_PATH   override DB path (default ~/.claude/enhanced_memories/memory.db)
    MEMORY_MD_DIR             override MD-corpus dir (default ~/.claude/projects/-Users-marc/memory)
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

# Run as a script from this subdirectory, so the repo root is not on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory_paths import get_db_path  # noqa: E402


def db_path() -> Path:
    return get_db_path()


def md_dir() -> Path:
    env = os.environ.get("MEMORY_MD_DIR")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".claude" / "projects" / "-Users-marc" / "memory"


def connect_ro(p: Path) -> sqlite3.Connection:
    if not p.exists():
        raise SystemExit(f"DB not found: {p}")
    # Read-only URI connection: cannot write, safe alongside the live daemon.
    conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


def _scalar(conn: sqlite3.Connection, sql: str, params=()):
    row = conn.execute(sql, params).fetchone()
    return row[0] if row else None


def entity_metrics(conn: sqlite3.Connection) -> dict:
    total = _scalar(conn, "SELECT COUNT(*) FROM entities") or 0
    never = _scalar(conn, "SELECT COUNT(*) FROM entities WHERE access_count=0") or 0
    never_reaccessed = (
        _scalar(conn, "SELECT COUNT(*) FROM entities WHERE last_accessed=created_at")
        or 0
    )
    reaccessed = (
        _scalar(conn, "SELECT COUNT(*) FROM entities WHERE last_accessed>created_at")
        or 0
    )
    pinned = _scalar(conn, "SELECT COUNT(*) FROM entities WHERE pinned=1") or 0
    by_type = [
        dict(r)
        for r in conn.execute(
            """SELECT entity_type,
                      COUNT(*) AS total,
                      SUM(CASE WHEN access_count=0 THEN 1 ELSE 0 END) AS cold,
                      MAX(access_count) AS max_access
                 FROM entities GROUP BY entity_type
                 ORDER BY total DESC LIMIT 25"""
        )
    ]
    about_user = [
        dict(r)
        for r in conn.execute(
            """SELECT entity_type,
                      COUNT(*) AS total,
                      SUM(CASE WHEN access_count>0 THEN 1 ELSE 0 END) AS ever_read,
                      SUM(CASE WHEN last_accessed>created_at THEN 1 ELSE 0 END) AS reaccessed
                 FROM entities WHERE entity_type LIKE 'auto_memory/%'
                 GROUP BY entity_type ORDER BY total DESC"""
        )
    ]
    growth = [
        dict(r)
        for r in conn.execute(
            "SELECT substr(created_at,1,7) AS month, COUNT(*) AS n "
            "FROM entities GROUP BY month ORDER BY month"
        )
    ]
    return {
        "total": total,
        "never_accessed": never,
        "never_accessed_pct": round(100 * never / total, 1) if total else 0,
        "never_reaccessed": never_reaccessed,
        "reaccessed": reaccessed,
        "pinned": pinned,
        "by_type_top25": by_type,
        "about_user_layer": about_user,
        "growth_by_month": growth,
    }


def decay_metrics(conn: sqlite3.Connection) -> dict:
    if not _has_table(conn, "forgetting_curves"):
        return {"present": False}
    total = _scalar(conn, "SELECT COUNT(*) FROM forgetting_curves") or 0
    decayed = (
        _scalar(
            conn,
            "SELECT COUNT(*) FROM forgetting_curves WHERE current_strength<initial_strength",
        )
        or 0
    )
    # Eviction = decay that actually removes from the working set. Archive table
    # is the actuation surface; if absent, decay computes a number nobody acts on.
    evicted = 0
    if _has_table(conn, "memory_archive"):
        evicted = _scalar(conn, "SELECT COUNT(*) FROM memory_archive") or 0
    transitions = 0
    if _has_table(conn, "memory_transitions"):
        transitions = _scalar(conn, "SELECT COUNT(*) FROM memory_transitions") or 0
    return {
        "present": True,
        "curves": total,
        "decayed": decayed,
        "evicted_archived": evicted,
        "transitions_logged": transitions,
        "decay_actuated": evicted > 0,
    }


def consolidation_metrics(conn: sqlite3.Connection) -> dict:
    if not _has_table(conn, "consolidation_jobs"):
        return {"present": False}
    jobs = _scalar(conn, "SELECT COUNT(*) FROM consolidation_jobs") or 0
    promoted = (
        _scalar(
            conn, "SELECT COALESCE(SUM(memories_promoted),0) FROM consolidation_jobs"
        )
        or 0
    )
    last = _scalar(conn, "SELECT MAX(completed_at) FROM consolidation_jobs")
    by_type = [
        dict(r)
        for r in conn.execute(
            "SELECT job_type, COUNT(*) AS n, MAX(completed_at) AS last "
            "FROM consolidation_jobs GROUP BY job_type ORDER BY n DESC"
        )
    ]
    epi_total = epi_consolidated = 0
    if _has_table(conn, "episodic_memory"):
        epi_total = _scalar(conn, "SELECT COUNT(*) FROM episodic_memory") or 0
        epi_consolidated = (
            _scalar(
                conn,
                "SELECT COUNT(*) FROM episodic_memory WHERE consolidated_to_semantic=1",
            )
            or 0
        )
    return {
        "present": True,
        "jobs": jobs,
        "promotions_total": promoted,
        "promotions_per_job": round(promoted / jobs, 3) if jobs else 0,
        "last_run": last,
        "by_type": by_type,
        "episodic_total": epi_total,
        "episodic_consolidated": epi_consolidated,
    }


def md_corpus_metrics() -> dict:
    d = md_dir()
    index = d / "MEMORY.md"
    if not d.exists() or not index.exists():
        return {"present": False, "dir": str(d)}
    index_text = index.read_text(errors="ignore")
    files = [p for p in d.glob("*.md") if p.name != "MEMORY.md"]
    linked = [p for p in files if p.name in index_text]
    orphaned = [p for p in files if p.name not in index_text]
    return {
        "present": True,
        "dir": str(d),
        "md_files": len(files),
        "index_lines": len(index_text.splitlines()),
        "index_bytes": len(index_text.encode()),
        "linked": len(linked),
        "orphaned": len(orphaned),
        "orphaned_pct": round(100 * len(orphaned) / len(files), 1) if files else 0,
        "orphaned_sample": sorted(p.name for p in orphaned)[:15],
    }


# A rate is only trustworthy with enough independent observations. Below these
# floors the point estimate is noise (1.0 on 2 rows reads identically to 1.0 on
# 10k rows), so the report must say "insufficient_data" instead of a bare rate.
_MIN_DAYS = 7
_MIN_ENTITIES = 30


def _wilson(successes: int, n: int, z: float = 1.96) -> list:
    """Wilson score interval for a binomial rate. Renders 1/1 as [0.0, 1.0],
    not a deceptive 1.0. Returns [low, high] rounded to 3dp."""
    if n <= 0:
        return [0.0, 1.0]
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)) / denom
    return [round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3)]


def true_compounding_metric(conn: sqlite3.Connection) -> dict:
    """Cross-session resurfacing rate. The real 'getting smarter' number.

    Requires the retrieval_log table. Until it has rows we refuse to fabricate
    a value: we report status='unavailable' and the reason. Even with rows, a
    rate is only reported as trustworthy once it clears a minimum-n floor;
    below that it is flagged 'insufficient_data' with a Wilson interval so a
    1/1 = 1.0 cannot masquerade as a real signal.
    """
    if not _has_table(conn, "retrieval_log"):
        return {
            "status": "unavailable",
            "reason": "retrieval_log table does not exist; wire retrieval_log.py into the search path",
        }
    rows = _scalar(conn, "SELECT COUNT(*) FROM retrieval_log") or 0
    if rows == 0:
        return {
            "status": "unavailable",
            "reason": "retrieval_log table is empty; no retrievals have been logged yet",
        }
    sessions = (
        _scalar(conn, "SELECT COUNT(DISTINCT session_id) FROM retrieval_log") or 0
    )
    # A session "resurfaces" memory if it retrieves >=1 entity created in a prior session.
    resurfacing = (
        _scalar(
            conn,
            """SELECT COUNT(DISTINCT rl.session_id)
             FROM retrieval_log rl
             JOIN entities e ON e.id = rl.entity_id
            WHERE e.created_at < rl.retrieved_at""",
        )
        or 0
    )
    cited = _scalar(conn, "SELECT COUNT(*) FROM retrieval_log WHERE cited=1") or 0

    # session_id is "unknown" whenever the caller's session is not carried on
    # the MCP request envelope (the common case — MCP does not expose the Claude
    # session to the server). When all rows are "unknown", the session-based rate
    # is degenerate (1 pseudo-session), so we fall back to a TIME-BUCKETED proxy:
    # a day "resurfaces" if it retrieves >=1 entity created on a PRIOR day. This
    # needs only timestamps and is the honest signal until session_id propagation
    # (or a session proxy) exists.
    distinct_unknown = (
        _scalar(conn, "SELECT COUNT(*) FROM retrieval_log WHERE session_id='unknown'")
        or 0
    )
    session_degenerate = distinct_unknown == rows
    days = (
        _scalar(conn, "SELECT COUNT(DISTINCT date(retrieved_at)) FROM retrieval_log")
        or 0
    )
    days_resurfacing = (
        _scalar(
            conn,
            """SELECT COUNT(DISTINCT date(rl.retrieved_at))
                 FROM retrieval_log rl
                 JOIN entities e ON e.id = rl.entity_id
                WHERE date(e.created_at) < date(rl.retrieved_at)""",
        )
        or 0
    )
    distinct_entities = (
        _scalar(conn, "SELECT COUNT(DISTINCT entity_id) FROM retrieval_log") or 0
    )
    distinct_queries = (
        _scalar(conn, "SELECT COUNT(DISTINCT query) FROM retrieval_log") or 0
    )
    # Minimum-n gate: the rate is only trusted once it clears the floor on
    # BOTH independent days and distinct entities. Below that, the number is
    # noise and must be labelled so, not rendered as a confident rate.
    sufficient = days >= _MIN_DAYS and distinct_entities >= _MIN_ENTITIES
    return {
        "status": "available" if sufficient else "insufficient_data",
        "logged_retrievals": rows,
        "distinct_entities": distinct_entities,
        "distinct_queries": distinct_queries,
        "distinct_sessions": sessions,
        "session_id_degenerate": session_degenerate,
        "sessions_resurfacing_prior_memory": resurfacing,
        "resurfacing_rate": round(resurfacing / sessions, 3) if sessions else 0,
        "distinct_days": days,
        "days_resurfacing_prior_memory": days_resurfacing,
        "day_resurfacing_rate": round(days_resurfacing / days, 3) if days else 0,
        "day_resurfacing_ci95": _wilson(days_resurfacing, days),
        "min_n": {"days": _MIN_DAYS, "entities": _MIN_ENTITIES},
        "hit_utility_cited": cited,
        "hit_utility_rate": round(cited / rows, 3) if rows else 0,
    }


def build_report() -> dict:
    p = db_path()
    conn = connect_ro(p)
    try:
        return {
            "db_path": str(p),
            "entities": entity_metrics(conn),
            "decay": decay_metrics(conn),
            "consolidation": consolidation_metrics(conn),
            "md_corpus": md_corpus_metrics(),
            "compounding_metric": true_compounding_metric(conn),
        }
    finally:
        conn.close()


def render(rep: dict) -> str:
    e = rep["entities"]
    d = rep["decay"]
    c = rep["consolidation"]
    m = rep["md_corpus"]
    cm = rep["compounding_metric"]
    out = []
    out.append("MEMORY COMPOUNDING REPORT")
    out.append(f"DB: {rep['db_path']}")
    out.append("")
    out.append("== STORE (proxy: access_count, not provably retrieval) ==")
    out.append(
        f"  entities={e['total']}  never-recorded-read={e['never_accessed']} "
        f"({e['never_accessed_pct']}%)  re-accessed={e['reaccessed']}  pinned={e['pinned']}"
    )
    out.append("  about-you layer (auto_memory/*):")
    for r in e["about_user_layer"]:
        out.append(
            f"    {r['entity_type']:<26} total={r['total']:<4} ever-read={r['ever_read']:<4} re-accessed={r['reaccessed']}"
        )
    out.append("  largest types (with cold count):")
    for r in e["by_type_top25"][:6]:
        out.append(
            f"    {r['entity_type']:<26} total={r['total']:<5} cold={r['cold']:<5} max_access={r['max_access']}"
        )
    out.append("")
    out.append("== DECAY ==")
    if d.get("present"):
        out.append(
            f"  curves={d['curves']}  decayed={d['decayed']}  archived(evicted)={d['evicted_archived']}  "
            f"transitions_logged={d['transitions_logged']}"
        )
        out.append(
            f"  decay actuated (eviction wired)? {'YES' if d['decay_actuated'] else 'NO -- decay computes a score nothing acts on'}"
        )
    else:
        out.append("  forgetting_curves table absent")
    out.append("")
    out.append("== CONSOLIDATION ==")
    if c.get("present"):
        out.append(
            f"  jobs={c['jobs']}  promotions_total={c['promotions_total']}  "
            f"per_job={c['promotions_per_job']}  last_run={c['last_run']}"
        )
        out.append(
            f"  episodic={c['episodic_total']} consolidated_to_semantic={c['episodic_consolidated']} "
            f"({'DEAD pipeline' if c['episodic_total'] and not c['episodic_consolidated'] else 'ok'})"
        )
    else:
        out.append("  consolidation_jobs table absent")
    out.append("")
    out.append("== LIVE LAYER (MEMORY.md, the only force-loaded channel) ==")
    if m.get("present"):
        out.append(
            f"  md_files={m['md_files']}  index_lines={m['index_lines']}  index_bytes={m['index_bytes']}"
        )
        out.append(
            f"  linked-in-index={m['linked']}  ORPHANED={m['orphaned']} ({m['orphaned_pct']}%)"
        )
    else:
        out.append(f"  MD corpus not found at {m.get('dir')}")
    out.append("")
    out.append("== TRUE COMPOUNDING METRIC (cross-session resurfacing) ==")
    if cm["status"] == "insufficient_data":
        out.append(
            f"  INSUFFICIENT DATA: {cm['logged_retrievals']} retrievals, "
            f"{cm.get('distinct_entities', 0)} distinct entities, {cm['distinct_days']} days "
            f"(need >={cm['min_n']['days']} days AND >={cm['min_n']['entities']} entities)."
        )
        out.append(
            f"  day_resurfacing point={cm['day_resurfacing_rate']} but 95% CI={cm['day_resurfacing_ci95']} "
            "-> the rate is noise; do NOT trust it yet."
        )
        out.append(
            f"  hit_utility_rate={cm['hit_utility_rate']} (cited never wired; structurally 0)"
        )
    elif cm["status"] == "available":
        if cm.get("session_id_degenerate"):
            out.append(
                "  session-based rate DEGENERATE: all rows session_id='unknown' "
                "(MCP does not carry the caller session). Using time-bucket proxy:"
            )
            out.append(
                f"  day_resurfacing_rate={cm['day_resurfacing_rate']} "
                f"({cm['days_resurfacing_prior_memory']}/{cm['distinct_days']} days retrieve prior-day memory)"
            )
        else:
            out.append(
                f"  resurfacing_rate={cm['resurfacing_rate']} "
                f"({cm['sessions_resurfacing_prior_memory']}/{cm['distinct_sessions']} sessions)"
            )
            out.append(
                f"  day_resurfacing_rate={cm['day_resurfacing_rate']} "
                f"({cm['days_resurfacing_prior_memory']}/{cm['distinct_days']} days)"
            )
        out.append(
            f"  logged_retrievals={cm['logged_retrievals']}  "
            f"hit_utility_rate={cm['hit_utility_rate']} ({cm['hit_utility_cited']} cited)"
        )
    else:
        out.append(f"  UNAVAILABLE: {cm['reason']}")
        out.append("  -> This is THE number that answers 'smarter or accumulating'.")
        out.append("     It cannot be computed until retrievals are logged.")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="emit JSON")
    args = ap.parse_args()
    rep = build_report()
    if args.json:
        print(json.dumps(rep, indent=2, default=str))
    else:
        print(render(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
