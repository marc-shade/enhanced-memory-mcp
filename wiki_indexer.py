#!/usr/bin/env python3
"""
Wiki indexer: parse a tree of markdown files into a structured wiki asset.

Storage (all sqlite, idempotent CREATE TABLE IF NOT EXISTS):

  wiki_pages(id, title, path, content_hash, sections_json, updated_at)
      UNIQUE(path)
  wiki_links(from_page_id, to_page_id, label)
      UNIQUE(from_page_id, to_page_id)   -- "the unique pair" of the spec;
      multiple labels between the same two pages are joined with " | "
  wiki_pages_fts  -- FTS5 external-content table over (title, sections_json),
      synced by AFTER INSERT/DELETE/UPDATE triggers on wiki_pages (mirrors the
      observations_fts pattern). A 'rebuild' is issued at the end of every
      ingest so the index is guaranteed consistent regardless of trigger edge
      cases.

Parsing (parse_markdown):
  - title   : first `# H1`, else the file basename (stem)
  - sections: content split at heading lines (`#{1,6}`); content before the
    first heading is a section with heading ""
  - links   : `[label](target)` over the whole text. External schemes
    (http/https/mailto) are skipped. `[label](#anchor)` becomes a self-edge
    (the page references its own anchor). `[label](path.md)` is resolved
    relative to the current file and, when it points at another ingested page,
    becomes a real page->page edge. Unresolved path links produce no edge.

Idempotency (ingest_wiki):
  - content_hash is sha256 of the file text. Files whose hash is unchanged are
    not re-parsed into sections and not rewritten; only a light link pass runs
    over their text so the link graph can be rebuilt against the current page
    set. Changed files are re-parsed + updated; new files inserted. Files that
    were previously ingested from THIS tree but no longer exist under it are
    deleted (stale), with their links in both directions removed first. Pages
    ingested from other roots (separate wiki_ingest calls on different dirs)
    are left untouched: the wiki is a union of every ingested tree.
"""

import hashlib
import json
import logging
import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("wiki_indexer")

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_LINK_RE = re.compile(r"\[([^\]]*)\]\(([^)]*)\)")
_EXTERNAL_SCHEME_RE = re.compile(r"^(?:[a-zA-Z][a-zA-Z0-9+.-]*):")


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Create wiki tables + FTS + sync triggers (idempotent)."""
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS wiki_pages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            path TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            sections_json TEXT NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(path)
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_wiki_pages_title ON wiki_pages(title)")
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS wiki_links (
            from_page_id INTEGER NOT NULL,
            to_page_id INTEGER NOT NULL,
            label TEXT,
            UNIQUE(from_page_id, to_page_id)
        )
        """
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_wiki_links_from ON wiki_links(from_page_id)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_wiki_links_to ON wiki_links(to_page_id)"
    )
    # FTS5 over title + sections (external content, synced by triggers).
    if not cur.execute(
        "SELECT 1 FROM sqlite_master WHERE name='wiki_pages_fts'"
    ).fetchone():
        cur.execute(
            """CREATE VIRTUAL TABLE wiki_pages_fts USING fts5(
                 title, sections_json, content='wiki_pages', content_rowid='id')"""
        )
        cur.execute(
            """CREATE TRIGGER wiki_fts_ai AFTER INSERT ON wiki_pages BEGIN
                 INSERT INTO wiki_pages_fts(rowid, title, sections_json)
                 VALUES (new.id, new.title, new.sections_json);
               END"""
        )
        cur.execute(
            """CREATE TRIGGER wiki_fts_ad AFTER DELETE ON wiki_pages BEGIN
                 INSERT INTO wiki_pages_fts(wiki_pages_fts, rowid, title, sections_json)
                 VALUES('delete', old.id, old.title, old.sections_json);
               END"""
        )
        cur.execute(
            """CREATE TRIGGER wiki_fts_au AFTER UPDATE ON wiki_pages BEGIN
                 INSERT INTO wiki_pages_fts(wiki_pages_fts, rowid, title, sections_json)
                 VALUES('delete', old.id, old.title, old.sections_json);
                 INSERT INTO wiki_pages_fts(rowid, title, sections_json)
                 VALUES (new.id, new.title, new.sections_json);
               END"""
        )
    conn.commit()


def _norm_path(path: os.PathLike) -> str:
    """Stable absolute, normalized string for a filesystem path."""
    return os.path.abspath(os.path.normpath(os.fspath(path)))


def parse_markdown(text: str, file_path: str) -> Dict[str, Any]:
    """Parse markdown text into {title, sections, links}.

    sections: list of {"heading": str, "content": str} in document order.
    links:    list of {"label": str, "target": str} for non-external links.
    """
    lines = text.splitlines()
    title: Optional[str] = None
    sections: List[Dict[str, str]] = []
    links: List[Dict[str, str]] = []
    current_heading = ""
    current_content: List[str] = []

    def flush() -> None:
        # Append a section only when it has content; content before the first
        # heading becomes an implicit section with heading "".
        if not current_content:
            current_content.clear()
            return
        content = "\n".join(current_content).strip()
        sections.append({"heading": current_heading, "content": content})
        current_content.clear()

    for line in lines:
        m = _HEADING_RE.match(line)
        if m:
            flush()
            level = len(m.group(1))
            heading = m.group(2).strip()
            if title is None and level == 1:
                title = heading
            current_heading = heading
        else:
            current_content.append(line)
    flush()

    if not title:
        title = Path(file_path).stem

    for m in _LINK_RE.finditer(text):
        label = m.group(1).strip()
        target = m.group(2).strip()
        if not target:
            continue
        if _EXTERNAL_SCHEME_RE.match(target):
            continue
        links.append({"label": label, "target": target})

    return {"title": title, "sections": sections, "links": links}


def walk_markdown_files(root: os.PathLike) -> List[Path]:
    """Recursively list *.md files under root, skipping hidden paths and .git."""
    results: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            d for d in dirnames if not d.startswith(".") and d != ".git"
        )
        for fn in sorted(filenames):
            if fn.endswith(".md") and not fn.startswith("."):
                results.append(Path(dirpath) / fn)
    return results


def _resolve_link_target(
    base_dir: str, target: str, id_by_path: Dict[str, int]
) -> Optional[str]:
    """Resolve a relative markdown link target to an ingested page path."""
    t = target.split("#", 1)[0].strip()
    if not t:
        return None
    cand = _norm_path(os.path.join(base_dir, t))
    if cand in id_by_path:
        return cand
    if not Path(cand).suffix:
        cand2 = cand + ".md"
        if cand2 in id_by_path:
            return cand2
    return None


def ingest_wiki(db_path: str, root: os.PathLike) -> Dict[str, Any]:
    """Idempotently ingest a markdown tree into the wiki asset.

    Returns a stats dict: files_scanned, new, updated, unchanged, deleted,
    links, and dir.
    """
    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(f"wiki ingest dir not found: {root_path}")

    files = walk_markdown_files(root_path)
    file_paths = {_norm_path(p): p for p in files}

    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA busy_timeout = 30000")
    _ensure_schema(conn)

    stats: Dict[str, Any] = {
        "files_scanned": len(files),
        "new": 0,
        "updated": 0,
        "unchanged": 0,
        "deleted": 0,
        "links": 0,
    }

    try:
        existing = dict(
            conn.execute("SELECT path, content_hash FROM wiki_pages").fetchall()
        )
        id_by_path = {
            p: i for i, p in conn.execute("SELECT id, path FROM wiki_pages").fetchall()
        }

        # Pass 1: insert/update pages (skip unchanged content).
        parsed: Dict[str, Dict[str, Any]] = {}
        for norm_p, p in file_paths.items():
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError as e:
                logger.warning("wiki ingest: cannot read %s: %s", p, e)
                continue
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            data = parse_markdown(text, str(p))
            # Light link pass is always run so the graph can be rebuilt against
            # the current page set; section re-parse is skipped when unchanged.
            parsed[norm_p] = data
            if existing.get(norm_p) == digest:
                stats["unchanged"] += 1
                continue
            sections_json = json.dumps(data["sections"], ensure_ascii=False)
            if norm_p in existing:
                conn.execute(
                    """UPDATE wiki_pages
                       SET title=?, content_hash=?, sections_json=?,
                           updated_at=CURRENT_TIMESTAMP
                       WHERE path=?""",
                    (data["title"], digest, sections_json, norm_p),
                )
                stats["updated"] += 1
            else:
                conn.execute(
                    """INSERT INTO wiki_pages
                           (title, path, content_hash, sections_json, updated_at)
                       VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)""",
                    (data["title"], norm_p, digest, sections_json),
                )
                stats["new"] += 1
        conn.commit()

        # Pass 2: remove stale pages (files gone from THIS tree). Pages that
        # came from a different ingest root (e.g. a separately-ingested docs
        # subtree) are left alone so the wiki stays a union of ingested trees.
        root_norm = _norm_path(root_path)

        def _under_root(p: str) -> bool:
            return p == root_norm or p.startswith(root_norm + os.sep)

        stale = [p for p in existing if _under_root(p) and p not in file_paths]
        for norm_p in stale:
            pid = id_by_path.get(norm_p)
            if pid is not None:
                conn.execute(
                    "DELETE FROM wiki_links WHERE from_page_id=? OR to_page_id=?",
                    (pid, pid),
                )
            conn.execute("DELETE FROM wiki_pages WHERE path=?", (norm_p,))
            stats["deleted"] += 1
        conn.commit()

        # Pass 3: rebuild the link graph from scratch.
        conn.execute("DELETE FROM wiki_links")
        id_by_path = {
            p: i for i, p in conn.execute("SELECT id, path FROM wiki_pages").fetchall()
        }
        for norm_p, data in parsed.items():
            from_id = id_by_path.get(norm_p)
            if from_id is None:
                continue
            base_dir = os.path.dirname(norm_p)
            seen = set()
            for link in data["links"]:
                target = link["target"]
                if target.startswith("#"):
                    # Anchor link: self-edge (the page references its own anchor).
                    to_id = from_id
                else:
                    resolved = _resolve_link_target(base_dir, target, id_by_path)
                    if resolved is None:
                        continue
                    to_id = id_by_path[resolved]
                label = link["label"] or target
                key = (from_id, to_id)
                if key in seen:
                    continue
                seen.add(key)
                conn.execute(
                    """INSERT INTO wiki_links (from_page_id, to_page_id, label)
                       VALUES (?, ?, ?)
                       ON CONFLICT(from_page_id, to_page_id) DO UPDATE SET
                           label = wiki_links.label || ' | ' || excluded.label""",
                    (from_id, to_id, label),
                )
                stats["links"] += 1
        conn.commit()

        # Guarantee FTS consistency (trigger edge cases / historical rows).
        conn.execute("INSERT INTO wiki_pages_fts(wiki_pages_fts) VALUES('rebuild')")
        conn.commit()
    finally:
        conn.close()

    stats["dir"] = _norm_path(root_path)
    return stats
