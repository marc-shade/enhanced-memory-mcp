#!/usr/bin/env python3
"""
Cold-start backfill MCP Tools.

Point the memory system at an existing codebase, docs tree, and past session
transcripts and it ingests all three into the knowledge assets:
  - code_dir    -> CodeGraph (code symbols + call graph via universal-ctags)
  - docs_dir    -> Wiki (markdown pages + link graph)
  - sessions_dir -> episodic memory (one episode per transcript, distilled)

Idempotent: wiki/codegraph use content hashes / full-reparse; transcripts are
tracked in a `backfill_log` table keyed on the transcript path hash, so a
re-run only ingests new transcripts.
"""

import hashlib
import json
import logging
import sqlite3
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger("backfill")

_MAX_PROMPTS = 20
_MAX_CHARS_PER_PROMPT = 2000
_MAX_FINAL_ASSISTANT = 4000


def _ensure_backfill_log(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS backfill_log (
            transcript_path TEXT PRIMARY KEY,
            path_hash TEXT NOT NULL,
            ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()


def _extract_transcript(transcript_path: str) -> dict:
    """User prompts + final assistant text from a Claude Code transcript JSONL.

    Mirrors workflows/inngest/session_distill._extract_transcript (which cannot
    be imported here: the server venv has no `inngest` package).
    """
    p = Path(transcript_path)
    if not p.exists():
        return {"ok": False, "reason": f"transcript not found: {transcript_path}"}
    prompts: list[str] = []
    final_assistant = ""
    with open(p, errors="replace") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            typ = rec.get("type")
            msg = rec.get("message") or {}
            if typ == "user" and not rec.get("isMeta"):
                content = msg.get("content")
                if isinstance(content, str) and content.strip():
                    text = content.strip()
                elif isinstance(content, list):
                    text = " ".join(
                        c.get("text", "")
                        for c in content
                        if isinstance(c, dict) and c.get("type") == "text"
                    ).strip()
                else:
                    continue
                if text and not text.startswith(("<", "Caveat:")):
                    prompts.append(text[:_MAX_CHARS_PER_PROMPT])
            elif typ == "assistant":
                content = msg.get("content")
                if isinstance(content, list):
                    text = " ".join(
                        c.get("text", "")
                        for c in content
                        if isinstance(c, dict) and c.get("type") == "text"
                    ).strip()
                    if text:
                        final_assistant = text
    if not prompts:
        return {"ok": False, "reason": "no user prompts found"}
    return {
        "ok": True,
        "prompts": prompts[-_MAX_PROMPTS:],
        "final_assistant": final_assistant[:_MAX_FINAL_ASSISTANT],
        "prompt_count": len(prompts),
    }


def _ingest_transcripts(db_path: str, sessions_dir: str, limit: int) -> Dict[str, Any]:
    conn = sqlite3.connect(db_path, timeout=30)
    try:
        _ensure_backfill_log(conn)
        cur = conn.cursor()
        files = sorted(Path(sessions_dir).glob("*.jsonl"))[:limit]
        stats = {"scanned": len(files), "ingested": 0, "skipped": 0, "errors": 0}
        for f in files:
            path_hash = hashlib.sha256(str(f).encode()).hexdigest()[:16]
            if cur.execute(
                "SELECT 1 FROM backfill_log WHERE path_hash = ?", (path_hash,)
            ).fetchone():
                stats["skipped"] += 1
                continue
            extracted = _extract_transcript(str(f))
            if not extracted.get("ok"):
                stats["errors"] += 1
                continue
            episode_data = {
                "source": "backfill",
                "transcript_path": str(f),
                "prompt_count": extracted["prompt_count"],
                "first_prompt": (
                    extracted["prompts"][0] if extracted["prompts"] else ""
                )[:500],
                "final_assistant_preview": extracted["final_assistant"][:500],
            }
            cur.execute(
                """
                INSERT INTO episodic_memory
                    (event_type, episode_data, significance_score, tags)
                VALUES (?, ?, ?, ?)
                """,
                (
                    "session_backfill",
                    json.dumps(episode_data),
                    0.5,
                    json.dumps(["backfill", "session"]),
                ),
            )
            cur.execute(
                "INSERT INTO backfill_log (transcript_path, path_hash) VALUES (?, ?)",
                (str(f), path_hash),
            )
            stats["ingested"] += 1
        conn.commit()
        return stats
    finally:
        conn.close()


def register_backfill_tools(app, db_path):
    @app.tool()
    async def backfill(
        code_dir: Optional[str] = None,
        docs_dir: Optional[str] = None,
        sessions_dir: Optional[str] = None,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """
        Cold-start backfill: index a codebase, docs tree, and past session
        transcripts into the memory knowledge assets.

        Chains CodeGraph (code symbols + call graph), Wiki (markdown pages +
        link graph), and episodic memory (one distilled episode per transcript).
        Idempotent: re-running only ingests new/changed content.

        Args:
            code_dir: Directory to index into CodeGraph (optional)
            docs_dir: Directory of markdown docs to ingest into Wiki (optional)
            sessions_dir: Directory of Claude transcript .jsonl files (optional)
            limit: Max transcripts to ingest (default 50)

        Returns:
            Dict with per-asset results
        """
        results: Dict[str, Any] = {}
        if code_dir:
            try:
                from codegraph_indexer import index_codegraph

                results["codegraph"] = index_codegraph(db_path, code_dir)
            except Exception as e:
                results["codegraph"] = {
                    "success": False,
                    "error": f"{type(e).__name__}: {e}",
                }
        if docs_dir:
            try:
                from wiki_indexer import ingest_wiki

                results["wiki"] = ingest_wiki(db_path, docs_dir)
            except Exception as e:
                results["wiki"] = {
                    "success": False,
                    "error": f"{type(e).__name__}: {e}",
                }
        if sessions_dir:
            try:
                results["sessions"] = _ingest_transcripts(db_path, sessions_dir, limit)
            except Exception as e:
                results["sessions"] = {
                    "success": False,
                    "error": f"{type(e).__name__}: {e}",
                }
        if not results:
            return {
                "success": False,
                "error": "provide at least one of code_dir, docs_dir, sessions_dir",
            }
        return {"success": True, "results": results}

    logger.info("Registered backfill MCP tool")
    return True
