#!/usr/bin/env python3
"""
Knowledge CodeGraph MCP tools: index and query the code symbol + call-graph asset.

Registration follows the feature-module pattern used elsewhere in this
directory: export `register_knowledge_codegraph_tools(app, db_path)` where
`app` is a FastMCP-like object with an `@app.tool()` decorator, and storage is
direct sqlite3 against `db_path` (schema created idempotently inside
codegraph_indexer).

Tools:
  codegraph_index(path)    - full clear + reparse via Universal Ctags
  codegraph_callers(name)  - who calls a symbol (all candidates if ambiguous)
  codegraph_callees(name)  - what a symbol calls
  codegraph_impact(name)   - transitive closure over INCOMING 'call' edges
                             (everything that would be affected by changing
                             this symbol), depth-limited BFS, deduped
  codegraph_search(prefix) - symbols whose name starts with prefix

Ambiguity rule: when a symbol name resolves to multiple indexed symbols, every
query returns ALL of them with their file:line and marks `ambiguous: true`.
We never guess which one the caller meant.

Env: CTAGS_BIN overrides the ctags binary path (default /opt/homebrew/bin/ctags).
"""

import logging
import sqlite3
from collections import deque
from typing import Any, Dict, List, Set, Tuple

from codegraph_indexer import _ensure_schema, index_codegraph

logger = logging.getLogger("knowledge_codegraph")


def _resolve_symbols(
    conn: sqlite3.Connection, name: str
) -> List[Tuple[int, str, str, int, str, str]]:
    """All indexed symbols with this exact name."""
    return conn.execute(
        "SELECT id, name, file, line, kind, language FROM code_symbols "
        "WHERE name = ? ORDER BY file, line",
        (name,),
    ).fetchall()


def _neighbors(
    conn: sqlite3.Connection,
    symbol_ids: List[int],
    direction: str,
    limit: int,
) -> List[Dict[str, Any]]:
    """direction='out' -> callees of the symbol; 'in' -> callers."""
    if not symbol_ids:
        return []
    placeholders = ",".join("?" for _ in symbol_ids)
    if direction == "out":
        join_col, sel_col = "e.to_symbol_id", "cs.id"
        where = f"e.from_symbol_id IN ({placeholders})"
    else:
        join_col, sel_col = "e.from_symbol_id", "cs.id"
        where = f"e.to_symbol_id IN ({placeholders})"
    rows = conn.execute(
        f"""
        SELECT cs.id, cs.name, cs.file, cs.line, cs.kind, e.ambiguous
        FROM code_edges e
        JOIN code_symbols cs ON cs.id = {join_col}
        WHERE {where} AND e.edge_kind IN ('call')
        ORDER BY cs.file, cs.line
        LIMIT ?
        """,
        tuple(symbol_ids) + (limit,),
    ).fetchall()
    return [
        {
            "id": r[0],
            "name": r[1],
            "file": r[2],
            "line": r[3],
            "kind": r[4],
            "edge_ambiguous": bool(r[5]),
        }
        for r in rows
    ]


def _impact_closure(
    conn: sqlite3.Connection,
    symbol_ids: List[int],
    max_depth: int,
) -> List[Dict[str, Any]]:
    """BFS over INCOMING 'call' edges (upstream callers), depth-limited, deduped.

    Changing a symbol affects its transitive callers, so impact walks from the
    symbol up the call graph. Returns nodes with their depth.
    """
    if not symbol_ids:
        return []
    seen: Set[int] = set(symbol_ids)
    results: List[Dict[str, Any]] = []
    queue: deque[Tuple[int, int]] = deque((sid, 0) for sid in symbol_ids)
    while queue:
        sid, depth = queue.popleft()
        if depth > 0:
            row = conn.execute(
                "SELECT name, file, line, kind FROM code_symbols WHERE id = ?",
                (sid,),
            ).fetchone()
            if row is not None:
                results.append(
                    {
                        "id": sid,
                        "name": row[0],
                        "file": row[1],
                        "line": row[2],
                        "kind": row[3],
                        "depth": depth,
                    }
                )
        if depth >= max_depth:
            continue
        rows = conn.execute(
            """
            SELECT e.from_symbol_id
            FROM code_edges e
            WHERE e.to_symbol_id = ? AND e.edge_kind IN ('call')
            """,
            (sid,),
        ).fetchall()
        for (caller_id,) in rows:
            if caller_id in seen:
                continue
            seen.add(caller_id)
            queue.append((caller_id, depth + 1))
    results.sort(key=lambda r: (r["depth"], r["file"], r["line"]))
    return results


def register_knowledge_codegraph_tools(app, db_path: str):
    """Register the knowledge-codegraph MCP tools."""

    def _connect() -> sqlite3.Connection:
        conn = sqlite3.connect(db_path, timeout=30)
        conn.execute("PRAGMA busy_timeout = 30000")
        _ensure_schema(conn)
        return conn

    @app.tool()
    async def codegraph_index(path: str) -> Dict[str, Any]:
        """
        Index a code file or directory into the code-symbol/call-graph asset.

        Runs Universal Ctags (`--fields=+nKzl --output-format=json -R`) as a
        subprocess and parses the JSON output. This is a FULL clear + reparse:
        the previous codegraph is replaced. The ctags binary path is taken from
        the CTAGS_BIN env var, defaulting to /opt/homebrew/bin/ctags.

        Args:
            path: File or directory to index

        Returns:
            Dict with success, symbols, files, call_edges, ambiguous_edges,
            include_edges (or success False + error when ctags is unavailable
            or the path does not exist)
        """
        try:
            stats = index_codegraph(db_path, path)
        except Exception as e:  # surface, never crash the MCP call
            logger.exception("codegraph_index failed")
            return {"success": False, "error": f"{type(e).__name__}: {e}"}
        return stats

    @app.tool()
    async def codegraph_callers(symbol: str, limit: int = 50) -> Dict[str, Any]:
        """
        Return which indexed symbols call the named symbol.

        If the name resolves to multiple symbols (ambiguity), ALL of them are
        returned with their file:line and `ambiguous: true`; the caller's
        intent is never guessed. Each match lists its direct callers.

        Args:
            symbol: Symbol name
            limit: Max callers per match (default 50)

        Returns:
            Dict with matches: [{symbol, file, line, kind, ambiguous, callers}]
        """
        conn = _connect()
        try:
            rows = _resolve_symbols(conn, symbol)
            if not rows:
                return {
                    "success": True,
                    "symbol": symbol,
                    "matches": [],
                    "match_count": 0,
                }
            ambiguous = len(rows) > 1
            matches = []
            for sid, name, file, line, kind, language in rows:
                callers = _neighbors(conn, [sid], "in", limit)
                matches.append(
                    {
                        "symbol": name,
                        "file": file,
                        "line": line,
                        "kind": kind,
                        "language": language,
                        "ambiguous": ambiguous,
                        "caller_count": len(callers),
                        "callers": callers,
                    }
                )
        finally:
            conn.close()
        return {
            "success": True,
            "symbol": symbol,
            "matches": matches,
            "match_count": len(matches),
        }

    @app.tool()
    async def codegraph_callees(symbol: str, limit: int = 50) -> Dict[str, Any]:
        """
        Return which indexed symbols the named symbol likely calls.

        Same ambiguity handling as codegraph_callers: all candidate symbols
        are returned with `ambiguous: true`, never a guessed one.

        Args:
            symbol: Symbol name
            limit: Max callees per match (default 50)

        Returns:
            Dict with matches: [{symbol, file, line, kind, ambiguous, callees}]
        """
        conn = _connect()
        try:
            rows = _resolve_symbols(conn, symbol)
            if not rows:
                return {
                    "success": True,
                    "symbol": symbol,
                    "matches": [],
                    "match_count": 0,
                }
            ambiguous = len(rows) > 1
            matches = []
            for sid, name, file, line, kind, language in rows:
                callees = _neighbors(conn, [sid], "out", limit)
                matches.append(
                    {
                        "symbol": name,
                        "file": file,
                        "line": line,
                        "kind": kind,
                        "language": language,
                        "ambiguous": ambiguous,
                        "callee_count": len(callees),
                        "callees": callees,
                    }
                )
        finally:
            conn.close()
        return {
            "success": True,
            "symbol": symbol,
            "matches": matches,
            "match_count": len(matches),
        }

    @app.tool()
    async def codegraph_impact(symbol: str, depth: int = 3) -> Dict[str, Any]:
        """
        Return the transitive impact set of a symbol: everything that would be
        affected by changing it, i.e. the depth-limited BFS closure over
        INCOMING 'call' edges (its callers, their callers, ...).

        Args:
            symbol: Symbol name
            depth: Max BFS depth (default 3)

        Returns:
            Dict with matches (one per candidate symbol) each carrying the
            reachable set with per-node depth
        """
        conn = _connect()
        try:
            rows = _resolve_symbols(conn, symbol)
            if not rows:
                return {
                    "success": True,
                    "symbol": symbol,
                    "matches": [],
                    "match_count": 0,
                }
            ambiguous = len(rows) > 1
            matches = []
            for sid, name, file, line, kind, language in rows:
                closure = _impact_closure(conn, [sid], max(0, depth))
                matches.append(
                    {
                        "symbol": name,
                        "file": file,
                        "line": line,
                        "kind": kind,
                        "language": language,
                        "ambiguous": ambiguous,
                        "impact_count": len(closure),
                        "impact": closure,
                    }
                )
        finally:
            conn.close()
        return {
            "success": True,
            "symbol": symbol,
            "matches": matches,
            "match_count": len(matches),
        }

    @app.tool()
    async def codegraph_search(prefix: str, limit: int = 20) -> Dict[str, Any]:
        """
        Search indexed symbols by name prefix (case-insensitive).

        Returns every matching symbol, including duplicates of the same name in
        different files, each with its file:line.

        Args:
            prefix: Symbol name prefix
            limit: Max results (default 20)

        Returns:
            Dict with results: [{id, name, file, line, kind, language}]
        """
        conn = _connect()
        try:
            like = prefix.replace("%", "%%") + "%"
            rows = conn.execute(
                "SELECT id, name, file, line, kind, language FROM code_symbols "
                "WHERE name LIKE ? ESCAPE '\\' ORDER BY file, line LIMIT ?",
                (like, limit),
            ).fetchall()
        finally:
            conn.close()
        return {
            "success": True,
            "prefix": prefix,
            "results": [
                {
                    "id": r[0],
                    "name": r[1],
                    "file": r[2],
                    "line": r[3],
                    "kind": r[4],
                    "language": r[5],
                }
                for r in rows
            ],
        }

    logger.info("Registered 5 knowledge-codegraph MCP tools")
    return True
