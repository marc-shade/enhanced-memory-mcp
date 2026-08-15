#!/usr/bin/env python3
"""
Knowledge Wiki MCP tools: ingest and query the markdown wiki asset.

Registration follows the feature-module pattern used elsewhere in this
directory: export `register_knowledge_wiki_tools(app, db_path)` where `app` is
a FastMCP-like object with an `@app.tool()` decorator, and storage is direct
sqlite3 against `db_path` (schema created idempotently inside wiki_indexer).

Tools:
  wiki_ingest(dir)      - ingest (or re-ingest) a markdown tree
  wiki_search(query)    - FTS5 over title + sections (LIKE fallback)
  wiki_get_page(id)     - one page with its parsed sections
  wiki_link_graph(id)   - outbound/inbound neighbors + edge labels

Default ingest target when `dir` is omitted: `$STORAGE_BASE/docs`, else the
`sibling-of-repo/docs` directory (the agentic-system docs dir).
"""

import json
import logging
import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional

from wiki_indexer import _ensure_schema, ingest_wiki

logger = logging.getLogger("knowledge_wiki")


def _default_wiki_dir() -> str:
    base = os.environ.get("STORAGE_BASE")
    if base:
        return os.path.join(base, "docs")
    # Sibling docs dir of the agentic-system repo containing this module.
    return str(Path(__file__).resolve().parents[2] / "docs")


def register_knowledge_wiki_tools(app, db_path: str):
    """Register the knowledge-wiki MCP tools."""

    def _connect() -> sqlite3.Connection:
        conn = sqlite3.connect(db_path, timeout=30)
        conn.execute("PRAGMA busy_timeout = 30000")
        _ensure_schema(conn)
        return conn

    @app.tool()
    async def wiki_ingest(dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Ingest a tree of markdown files into the wiki asset (idempotent).

        Unchanged files are skipped (content_hash), changed files updated,
        stale files deleted, and the page-to-page link graph rebuilt.

        Args:
            dir: Directory to scan for .md files. Omitted -> $STORAGE_BASE/docs,
                 else the agentic-system docs/ dir.

        Returns:
            Dict with success, files_scanned, new, updated, unchanged,
            deleted, links, dir
        """
        target = dir or _default_wiki_dir()
        try:
            stats = ingest_wiki(db_path, target)
        except FileNotFoundError as e:
            return {"success": False, "error": str(e)}
        except Exception as e:  # surface, never crash the MCP call
            logger.exception("wiki_ingest failed")
            return {"success": False, "error": f"{type(e).__name__}: {e}"}
        stats["success"] = True
        return stats

    @app.tool()
    async def wiki_search(query: str, limit: int = 20) -> Dict[str, Any]:
        """
        Search wiki pages by title + section text (FTS5, per-token AND).

        Falls back to a LIKE scan if the query breaks FTS syntax.

        Args:
            query: Search terms
            limit: Max results (default 20)

        Returns:
            Dict with results: [{id, title, path, score}]
        """
        tokens = re.findall(r"[\w']+", query)
        if not tokens:
            return {"success": True, "query": query, "results": []}
        fts_query = " ".join('"' + t.replace('"', '""') + '"' for t in tokens)

        conn = _connect()
        try:
            try:
                rows = conn.execute(
                    """
                    SELECT p.id, p.title, p.path, MIN(f.rank) AS rank
                    FROM wiki_pages_fts f
                    JOIN wiki_pages p ON p.id = f.rowid
                    WHERE wiki_pages_fts MATCH ?
                    GROUP BY p.id
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (fts_query, limit),
                ).fetchall()
            except sqlite3.OperationalError as e:
                logger.warning("wiki FTS search failed (%s); LIKE fallback", e)
                like = "%" + query.replace("%", "%%") + "%"
                rows = conn.execute(
                    """
                    SELECT id, title, path, 0.0 AS rank
                    FROM wiki_pages
                    WHERE title LIKE ? OR sections_json LIKE ?
                    ORDER BY updated_at DESC
                    LIMIT ?
                    """,
                    (like, like, limit),
                ).fetchall()
        finally:
            conn.close()

        return {
            "success": True,
            "query": query,
            "results": [
                {"id": r[0], "title": r[1], "path": r[2], "score": round(r[3], 4)}
                for r in rows
            ],
        }

    @app.tool()
    async def wiki_get_page(page_id: int) -> Dict[str, Any]:
        """
        Get one wiki page with its parsed sections.

        Args:
            page_id: Page id from wiki_search / wiki_link_graph

        Returns:
            Dict with id, title, path, sections (list of {heading, content}),
            updated_at
        """
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT id, title, path, sections_json, updated_at "
                "FROM wiki_pages WHERE id = ?",
                (page_id,),
            ).fetchone()
        finally:
            conn.close()

        if row is None:
            return {"success": False, "error": f"page {page_id} not found"}

        try:
            sections = json.loads(row[3])
        except (ValueError, TypeError):
            sections = []

        return {
            "success": True,
            "id": row[0],
            "title": row[1],
            "path": row[2],
            "sections": sections,
            "updated_at": row[4],
        }

    @app.tool()
    async def wiki_link_graph(page_id: int) -> Dict[str, Any]:
        """
        Return a page's neighbors in the wiki link graph with edge labels.

        Args:
            page_id: Page id

        Returns:
            Dict with the page, outbound links, and inbound links
        """
        conn = _connect()
        try:
            page = conn.execute(
                "SELECT id, title, path FROM wiki_pages WHERE id = ?",
                (page_id,),
            ).fetchone()
            if page is None:
                return {"success": False, "error": f"page {page_id} not found"}
            outbound = conn.execute(
                """
                SELECT l.to_page_id, p.title, p.path, l.label
                FROM wiki_links l
                JOIN wiki_pages p ON p.id = l.to_page_id
                WHERE l.from_page_id = ?
                ORDER BY p.title
                """,
                (page_id,),
            ).fetchall()
            inbound = conn.execute(
                """
                SELECT l.from_page_id, p.title, p.path, l.label
                FROM wiki_links l
                JOIN wiki_pages p ON p.id = l.from_page_id
                WHERE l.to_page_id = ?
                ORDER BY p.title
                """,
                (page_id,),
            ).fetchall()
        finally:
            conn.close()

        return {
            "success": True,
            "page": {"id": page[0], "title": page[1], "path": page[2]},
            "outbound": [
                {
                    "page_id": r[0],
                    "title": r[1],
                    "path": r[2],
                    "label": r[3],
                }
                for r in outbound
            ],
            "inbound": [
                {
                    "page_id": r[0],
                    "title": r[1],
                    "path": r[2],
                    "label": r[3],
                }
                for r in inbound
            ],
        }

    logger.info("Registered 4 knowledge-wiki MCP tools")
    return True
