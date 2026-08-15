#!/usr/bin/env python3
"""
Session Evidence Retention Tools

The "never lose the road back to evidence" mechanism: full tool outputs are
written to disk, a compact navigable sketch (node ids + one-line summaries)
is what stays in context, and the raw text is pulled back on demand by node
id. This is the compression-with-a-path design the harness compaction lacks:
compaction throws the session away and hands you a note; evidence retention
keeps the road back.

Storage:
  - `session_evidence` sqlite table: the index (session_id, node_id, parent,
    tool, summary, token_count, evidence_path).
  - `session_evidence_fts`: FTS5 over summaries (external-content, synced by
    triggers), mirroring the `observations_fts` pattern.
  - Raw outputs: `~/.claude/enhanced_memories/session_evidence/<session_id>/
    <node_id>.txt` (chunked as `<node_id>_NNNN.txt` if over the chunk cap).

Lifecycle:
  - Agent logs each expensive tool output with `evidence_log`.
  - Agent keeps `evidence_sketch` (a compact DAG) in context.
  - Agent pulls raw text with `evidence_get` when it needs detail.
  - `evidence_prune` GCs old raw files + rows; a SessionEnd bridge (in the
    session distiller) archives the sketch as an episode so the road back
    survives even after pruning.

Security: session_id and node_id are sanitized to [A-Za-z0-9._-] to prevent
path traversal, since both are used to build filesystem paths.
"""

import logging
import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger("session_evidence")

_CHUNK_CAP = 65536  # 64 KB per raw chunk file
_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Create the session_evidence tables + FTS + sync triggers (idempotent)."""
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS session_evidence (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            node_id TEXT NOT NULL,
            parent_node TEXT,
            tool TEXT NOT NULL,
            summary TEXT NOT NULL,
            token_count INTEGER DEFAULT 0,
            evidence_path TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(session_id, node_id)
        )
        """
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_session_evidence_session "
        "ON session_evidence(session_id)"
    )
    # FTS5 over summaries (external content, synced by triggers) — mirrors the
    # observations_fts pattern from the Phase 0.2 spine repair.
    if not cur.execute(
        "SELECT 1 FROM sqlite_master WHERE name='session_evidence_fts'"
    ).fetchone():
        cur.execute(
            """CREATE VIRTUAL TABLE session_evidence_fts USING fts5(
                 summary, content='session_evidence', content_rowid='id')"""
        )
        cur.execute(
            """CREATE TRIGGER se_fts_ai AFTER INSERT ON session_evidence BEGIN
                 INSERT INTO session_evidence_fts(rowid, summary)
                 VALUES (new.id, new.summary);
               END"""
        )
        cur.execute(
            """CREATE TRIGGER se_fts_ad AFTER DELETE ON session_evidence BEGIN
                 INSERT INTO session_evidence_fts(session_evidence_fts, rowid, summary)
                 VALUES('delete', old.id, old.summary);
               END"""
        )
        cur.execute(
            """CREATE TRIGGER se_fts_au AFTER UPDATE ON session_evidence BEGIN
                 INSERT INTO session_evidence_fts(session_evidence_fts, rowid, summary)
                 VALUES('delete', old.id, old.summary);
                 INSERT INTO session_evidence_fts(rowid, summary)
                 VALUES (new.id, new.summary);
               END"""
        )
    conn.commit()


def _validate_id(value: str, what: str) -> str:
    if not value or not _ID_PATTERN.match(value):
        raise ValueError(f"{what} must match {_ID_PATTERN.pattern!r}, got {value!r}")
    return value


def _write_raw(evidence_root: Path, session_id: str, node_id: str, raw: str) -> str:
    """Write raw text to disk (chunked over the cap). Returns the dir path."""
    node_dir = evidence_root / session_id / node_id
    node_dir.mkdir(parents=True, exist_ok=True)
    # Clear any prior chunks for this node so re-logs don't append stale data.
    for old in node_dir.glob("*.txt"):
        old.unlink()
    if len(raw) <= _CHUNK_CAP:
        (node_dir / "raw.txt").write_text(raw, encoding="utf-8")
    else:
        for i in range(0, len(raw), _CHUNK_CAP):
            chunk = raw[i : i + _CHUNK_CAP]
            (node_dir / f"raw_{i // _CHUNK_CAP:04d}.txt").write_text(
                chunk, encoding="utf-8"
            )
    return str(node_dir)


def _read_raw(evidence_path: str) -> str:
    node_dir = Path(evidence_path)
    if not node_dir.is_dir():
        raise FileNotFoundError(f"evidence files missing: {evidence_path}")
    parts = sorted(p for p in node_dir.glob("raw*.txt"))
    if not parts:
        raise FileNotFoundError(f"no raw files under {evidence_path}")
    return "".join(p.read_text(encoding="utf-8") for p in parts)


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def register_session_evidence_tools(app, db_path):
    """Register session evidence MCP tools."""

    evidence_root = Path(db_path).parent / "session_evidence"
    evidence_root.mkdir(parents=True, exist_ok=True)

    def _connect() -> sqlite3.Connection:
        conn = sqlite3.connect(db_path, timeout=30)
        conn.execute("PRAGMA busy_timeout = 30000")
        _ensure_schema(conn)
        return conn

    @app.tool()
    async def evidence_log(
        session_id: str,
        node_id: str,
        tool: str,
        summary: str,
        raw: str,
        parent_node: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Log a tool output as session evidence: raw text goes to disk, a compact
        index row (with a one-line summary) is kept for the in-context sketch.

        Call this for every expensive tool output (big logs, file dumps, search
        results). The raw text leaves context; retrieve it later with
        evidence_get using the returned node_id.

        Args:
            session_id: Identifier for the session (letters, digits, ._-)
            node_id: Unique id for this node within the session (letters, digits, ._-)
            tool: Tool or step name that produced the output
            summary: One-line summary shown in the sketch
            raw: The full output text (written to disk)
            parent_node: Optional parent node id for the evidence DAG

        Returns:
            Dict with node_id, evidence_path, token_count
        """
        _validate_id(session_id, "session_id")
        _validate_id(node_id, "node_id")
        if parent_node:
            _validate_id(parent_node, "parent_node")

        path = _write_raw(evidence_root, session_id, node_id, raw)
        tokens = _approx_tokens(raw)

        conn = _connect()
        try:
            conn.execute(
                """
                INSERT INTO session_evidence
                    (session_id, node_id, parent_node, tool, summary,
                     token_count, evidence_path)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id, node_id) DO UPDATE SET
                    parent_node = excluded.parent_node,
                    tool = excluded.tool,
                    summary = excluded.summary,
                    token_count = excluded.token_count,
                    evidence_path = excluded.evidence_path
                """,
                (session_id, node_id, parent_node, tool, summary, tokens, path),
            )
            conn.commit()
        finally:
            conn.close()

        return {
            "success": True,
            "session_id": session_id,
            "node_id": node_id,
            "evidence_path": path,
            "token_count": tokens,
            "bytes": len(raw),
        }

    @app.tool()
    async def evidence_get(
        session_id: str,
        node_id: str,
    ) -> Dict[str, Any]:
        """
        Pull the raw text for a single evidence node (the road back to detail).

        Args:
            session_id: Session identifier
            node_id: Node id returned by evidence_log

        Returns:
            Dict with the node's tool, summary, token_count and full raw text
        """
        _validate_id(session_id, "session_id")
        _validate_id(node_id, "node_id")

        conn = _connect()
        try:
            row = conn.execute(
                """
                SELECT node_id, parent_node, tool, summary, token_count, evidence_path
                FROM session_evidence
                WHERE session_id = ? AND node_id = ?
                """,
                (session_id, node_id),
            ).fetchone()
        finally:
            conn.close()

        if row is None:
            return {
                "success": False,
                "error": f"node {node_id} not found in session {session_id}",
            }

        try:
            raw = _read_raw(row[5])
        except (FileNotFoundError, OSError) as e:
            return {"success": False, "error": f"evidence on disk unreadable: {e}"}

        return {
            "success": True,
            "node_id": row[0],
            "parent_node": row[1],
            "tool": row[2],
            "summary": row[3],
            "token_count": row[4],
            "raw": raw,
        }

    @app.tool()
    async def evidence_sketch(
        session_id: str,
        max_chars: int = 2000,
    ) -> Dict[str, Any]:
        """
        Return a compact DAG of the session's evidence for keeping in context.

        The sketch is an indented tree of `node_id [tool]: summary` lines,
        capped at max_chars so it stays a tiny fraction of the context budget.
        Detail lives on disk; expand any node with evidence_get.

        Args:
            session_id: Session identifier
            max_chars: Cap for the sketch text (default 2000)

        Returns:
            Dict with sketch text, node_count, total_tokens, truncated
        """
        _validate_id(session_id, "session_id")

        conn = _connect()
        try:
            rows = conn.execute(
                """
                SELECT node_id, parent_node, tool, summary, token_count
                FROM session_evidence
                WHERE session_id = ?
                ORDER BY id
                """,
                (session_id,),
            ).fetchall()
        finally:
            conn.close()

        if not rows:
            return {"success": True, "sketch": "", "node_count": 0, "total_tokens": 0}

        children: Dict[str, List[Dict[str, Any]]] = {}
        by_id: Dict[str, Dict[str, Any]] = {}
        roots: List[Dict[str, Any]] = []
        for node_id, parent, tool, summary, tokens in rows:
            item = {
                "node_id": node_id,
                "tool": tool,
                "summary": summary,
                "tokens": tokens,
            }
            by_id[node_id] = item
            children.setdefault(parent, []).append(item)
            if parent is None:
                roots.append(item)

        lines: List[str] = []

        def walk(item: Dict[str, Any], depth: int) -> None:
            indent = "  " * depth
            summary = item["summary"] or ""
            if len(summary) > 120:
                summary = summary[:117] + "..."
            lines.append(
                f"{indent}{item['node_id']} [{item['tool']}] ({item['tokens']}t): {summary}"
            )
            for child in children.get(item["node_id"], []):
                walk(child, depth + 1)

        # Orphan items (parent not in session) are treated as roots.
        for node_id, item in by_id.items():
            parent = next((r[1] for r in rows if r[0] == node_id), None)
            if parent not in by_id:
                if item not in roots:
                    roots.append(item)

        for root in roots:
            walk(root, 0)

        total_tokens = sum(r[4] or 0 for r in rows)
        sketch = "\n".join(lines)
        truncated = False
        if len(sketch) > max_chars:
            sketch = sketch[:max_chars]
            truncated = True

        return {
            "success": True,
            "sketch": sketch,
            "node_count": len(rows),
            "total_tokens": total_tokens,
            "truncated": truncated,
            "hint": "expand any node with evidence_get(session_id, node_id)",
        }

    @app.tool()
    async def evidence_search(
        session_id: str,
        query: str,
        limit: int = 20,
    ) -> Dict[str, Any]:
        """
        Find evidence nodes whose summaries match a query (FTS5, per-token AND).

        Args:
            session_id: Session identifier
            query: Search terms
            limit: Max results (default 20)

        Returns:
            Dict with matching nodes (node_id, tool, summary, token_count)
        """
        _validate_id(session_id, "session_id")

        tokens = re.findall(r"[\w']+", query)
        if not tokens:
            return {"success": True, "results": [], "query": query}

        fts_query = " ".join('"' + t.replace('"', '""') + '"' for t in tokens)

        conn = _connect()
        try:
            try:
                rows = conn.execute(
                    """
                    SELECT se.node_id, se.tool, se.summary, se.token_count,
                           MIN(se_fts.rank) AS rank
                    FROM session_evidence_fts se_fts
                    JOIN session_evidence se ON se.id = se_fts.rowid
                    WHERE session_evidence_fts MATCH ? AND se.session_id = ?
                    GROUP BY se.id
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (fts_query, session_id, limit),
                ).fetchall()
            except sqlite3.OperationalError as e:
                # FTS syntax failure (e.g. a bare operator) — fall back to LIKE.
                logger.warning("FTS search failed (%s); LIKE fallback", e)
                like = "%" + query.replace("%", "%%") + "%"
                rows = conn.execute(
                    """
                    SELECT node_id, tool, summary, token_count, 0.0 AS rank
                    FROM session_evidence
                    WHERE session_id = ? AND summary LIKE ?
                    LIMIT ?
                    """,
                    (session_id, like, limit),
                ).fetchall()
        finally:
            conn.close()

        return {
            "success": True,
            "query": query,
            "results": [
                {
                    "node_id": r[0],
                    "tool": r[1],
                    "summary": r[2],
                    "token_count": r[3],
                }
                for r in rows
            ],
        }

    @app.tool()
    async def evidence_prune(
        session_id: str,
        older_than_hours: int = 24,
    ) -> Dict[str, Any]:
        """
        Garbage-collect evidence older than a threshold: delete raw files and rows.

        Use after a session ends (or on demand). The archived sketch in episodic
        memory is the durable record; this frees the raw bytes.

        Args:
            session_id: Session identifier
            older_than_hours: Delete evidence older than this many hours (default 24)

        Returns:
            Dict with deleted row/file counts
        """
        _validate_id(session_id, "session_id")

        conn = _connect()
        try:
            rows = conn.execute(
                """
                SELECT id, node_id, evidence_path
                FROM session_evidence
                WHERE session_id = ? AND created_at <= datetime('now', ?)
                """,
                (session_id, f"-{older_than_hours} hours"),
            ).fetchall()

            deleted_files = 0
            deleted_rows = 0
            for row_id, node_id, evidence_path in rows:
                node_dir = Path(evidence_path)
                if node_dir.is_dir():
                    for f in node_dir.glob("*.txt"):
                        try:
                            f.unlink()
                            deleted_files += 1
                        except OSError as e:
                            logger.warning("could not unlink %s: %s", f, e)
                    try:
                        node_dir.rmdir()
                    except OSError:
                        pass  # not empty or busy; fine
                conn.execute("DELETE FROM session_evidence WHERE id = ?", (row_id,))
                deleted_rows += 1

            conn.commit()
        finally:
            conn.close()

        return {
            "success": True,
            "session_id": session_id,
            "deleted_rows": deleted_rows,
            "deleted_files": deleted_files,
        }

    logger.info("Registered 5 session evidence MCP tools")
    return True
