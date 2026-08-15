#!/usr/bin/env python3
"""
CodeGraph indexer: build a code symbol + call-graph asset via Universal Ctags.

Backend: `ctags --fields=+nKzl --output-format=json -R --languages=...` run as a
subprocess (shell=False). ctags emits one JSON object per line; we parse those
into `code_symbols`. The spec command is `--fields=+nKz`; the `l` (language)
field is added because the schema requires a `language` column. ctags does not
emit a `signature` in JSON mode even with `+s` (verified against 6.2.1), so the
signature is recovered from the `pattern` field when present.

Storage (all sqlite, idempotent CREATE TABLE IF NOT EXISTS):

  code_symbols(id, language, kind, name, file, line, signature)
      UNIQUE(file, line, name)
  code_edges(from_symbol_id, to_symbol_id, edge_kind, ambiguous)
      UNIQUE(from_symbol_id, to_symbol_id, edge_kind)
      edge_kind: 'call' (symbol->symbol) or 'includes' (file->file).
      ambiguous: 1 when the same referenced name maps to >1 indexed symbols,
                 so the edge targets every candidate and the query layer can
                 report the ambiguity honestly (never guessing).

Re-index is a full clear + reparse (DELETE then INSERT).

Callee detection (deliberately coarse, honest v1 heuristic): for each symbol,
the lines between its definition line and the next indexed symbol in the same
file approximate its body. Those lines are scanned for other indexed symbol
names (word-boundary); every distinct reference becomes a 'call' edge to all
candidate symbols. Same-name ambiguity -> one edge per candidate, all marked
ambiguous. This over-approximates (a nested def truncates its parent's span)
and under-approximates (unindexed call targets are invisible); it is NOT
precise intra-procedural analysis.

File->file 'includes' edges: each indexed file is scanned for
#include / import / require lines; when the target resolves to another indexed
file (relative path or unambiguous basename match) a synthetic 'file' symbol
(line=0) is created for both files and an 'includes' edge links them.

Env: CTAGS_BIN overrides the ctags binary path (default /opt/homebrew/bin/ctags).
"""

import json
import logging
import os
import re
import sqlite3
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("codegraph_indexer")

DEFAULT_CTAGS_BIN = "/opt/homebrew/bin/ctags"
CTAGS_LANGUAGES = "Python,JavaScript,TypeScript,Go,Rust,C,C++"
# A referenced name whose definitions span more than this many files is
# library-level common (main/get/run/init...), not a resolvable call target.
# Ambiguous edges for such names are skipped to keep the edge table usable.
MAX_AMBIGUOUS_FILES = 10
CTAGS_EXCLUDES = [
    "node_modules",
    ".git",
    "build",
    "dist",
    "__pycache__",
    # wildcards: cover .venv, .venv_py313_backup, venv3, ... (a stray venv
    # backup indexed 473k torch/scipy symbols on the enhanced-memory-mcp tree)
    ".venv*",
    "venv*",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    # junk / backup / tool dirs that appear in real repos and would otherwise
    # drag ctags across thousands of irrelevant files (e.g. a stray "~" dir
    # resolving to $HOME indexed 473k symbols on the enhanced-memory-mcp tree)
    ".archive",
    ".benchmarks",
    ".swarm",
    ".claude",
    ".claude-flow",
    ".coverage",
    "~",
    "$HOME",
    ".hg",
    ".svn",
]

# include/import/require line patterns (leading whitespace tolerated).
# Order matters: more specific forms (ESM `import x from 'y'`, `from x import`,
# paren-require anywhere in the line) run before the bare `import x` form so
# the captured module path is the right token.
_IMPORT_PATTERNS = [
    re.compile(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]'),
    re.compile(r"^\s*from\s+([\w.]+)\s+import\b"),
    re.compile(r"\b(?:import|require)\s*\(\s*['\"]([^'\"]+)['\"]\s*\)"),
    re.compile(r"\brequire\s+['\"]([^'\"]+)['\"]"),
    re.compile(r"^\s*import\s+[^'\"\n]+?\s+from\s+['\"]([^'\"]+)['\"]"),
    re.compile(r"^\s*import\s+['\"]([^'\"]+)['\"]"),
    re.compile(r"^\s*import\s+([\w.]+)\b"),
]

# suffix candidates appended to bare include/import targets when resolving.
_IMPORT_SUFFIXES = [
    ".py",
    ".h",
    ".c",
    ".cc",
    ".cpp",
    ".hpp",
    ".go",
    ".rs",
    ".js",
    ".ts",
    ".tsx",
]

_SIGNATURE_RE_CACHE: Dict[str, re.Pattern] = {}


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Create codegraph tables + indexes (idempotent)."""
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS code_symbols (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            language TEXT,
            kind TEXT,
            name TEXT NOT NULL,
            file TEXT NOT NULL,
            line INTEGER,
            signature TEXT,
            UNIQUE(file, line, name)
        )
        """
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_code_symbols_name ON code_symbols(name)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_code_symbols_file ON code_symbols(file)"
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS code_edges (
            from_symbol_id INTEGER NOT NULL,
            to_symbol_id INTEGER NOT NULL,
            edge_kind TEXT NOT NULL,
            ambiguous INTEGER NOT NULL DEFAULT 0,
            UNIQUE(from_symbol_id, to_symbol_id, edge_kind)
        )
        """
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_code_edges_from ON code_edges(from_symbol_id)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_code_edges_to ON code_edges(to_symbol_id)"
    )
    conn.commit()


def _ctags_bin() -> str:
    return os.environ.get("CTAGS_BIN", DEFAULT_CTAGS_BIN)


def _extract_signature(entry: Dict[str, Any]) -> str:
    """Recover a signature: the `signature` field if present, else the pattern."""
    sig = entry.get("signature")
    if sig:
        return str(sig)
    pattern = entry.get("pattern", "")
    if pattern.startswith("/^") and pattern.endswith("$/"):
        text = pattern[2:-2]
    else:
        text = pattern
    name = entry.get("name", "")
    if not name:
        return ""
    cache_key = re.escape(name)
    rx = _SIGNATURE_RE_CACHE.get(cache_key)
    if rx is None:
        rx = re.compile(r"\b" + re.escape(name) + r"\s*\(([^)]*)\)")
        _SIGNATURE_RE_CACHE[cache_key] = rx
    m = rx.search(text)
    if m:
        return f"{name}({m.group(1)})"
    return ""


def run_ctags(path: str, ctags_bin: Optional[str] = None) -> Dict[str, Any]:
    """Run ctags over a path; return parsed JSON entries or a clean error."""
    bin_ = ctags_bin or _ctags_bin()
    if not os.path.isfile(bin_):
        return {
            "success": False,
            "error": f"ctags binary not found at {bin_}; set CTAGS_BIN",
        }
    if not (os.path.isdir(path) or os.path.isfile(path)):
        return {"success": False, "error": f"path not found: {path}"}

    cmd = [
        bin_,
        "--fields=+nKzl",
        "--output-format=json",
        "-R",
        f"--languages={CTAGS_LANGUAGES}",
    ]
    for ex in CTAGS_EXCLUDES:
        cmd.append(f"--exclude={ex}")
    cmd.append(path)

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600, shell=False
        )
    except FileNotFoundError:
        return {
            "success": False,
            "error": f"ctags binary not found at {bin_}; set CTAGS_BIN",
        }
    except (subprocess.SubprocessError, OSError) as e:
        return {"success": False, "error": f"ctags failed to run: {e}"}

    # ctags returns 0 for a missing input path (verified), so stdout emptiness
    # is the real signal on top of a non-zero exit.
    if proc.returncode != 0 and not proc.stdout.strip():
        return {
            "success": False,
            "error": f"ctags exited {proc.returncode}: {proc.stderr[:500]}",
        }

    entries: List[Dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if entry.get("_type") == "tag":
            entries.append(entry)

    return {
        "success": True,
        "entries": entries,
        "stdout_bytes": len(proc.stdout),
        "stderr": proc.stderr[:500],
    }


def _norm(path: str) -> str:
    return os.path.abspath(os.path.normpath(path))


def _resolve_include(
    target: str,
    from_file: str,
    file_basenames: Dict[str, List[str]],
    indexed_files: Set[str],
) -> Optional[str]:
    """Resolve an include/import target to an indexed file (or None)."""
    t = target.strip()
    if not t:
        return None
    # Python relative imports: ".executor" -> "executor" in this dir,
    # "..pkg.executor" -> "../pkg/executor" relative to this dir.
    # "./x" / "../x" are plain paths and are left as-is.
    m = re.match(r"^(\.+)([a-zA-Z_][\w.]*)$", t)
    if m:
        ndots = len(m.group(1))
        t = m.group(2)
        base_dir = os.path.dirname(from_file)
        for _ in range(ndots - 1):
            base_dir = os.path.dirname(base_dir)
    else:
        base_dir = os.path.dirname(from_file)
    if not t:
        return None
    # Python module paths: foo.bar -> foo/bar.py candidates handled below.
    candidates: List[str] = [t]
    if not Path(t).suffix:
        candidates.extend(t + s for s in _IMPORT_SUFFIXES)
        if "." in t:
            dotted_path = t.replace(".", "/")
            candidates.append(dotted_path)
            candidates.extend(dotted_path + s for s in _IMPORT_SUFFIXES)

    for cand in candidates:
        # 1) relative-path resolution against the including file's dir
        abs_cand = _norm(os.path.join(base_dir, cand))
        if abs_cand in indexed_files:
            return abs_cand
        # 2) unambiguous basename match anywhere in the index
        base = os.path.basename(cand)
        if not base:
            continue
        matches = file_basenames.get(base)
        if matches is not None:
            if len(matches) == 1:
                return matches[0]
            # ambiguous basename: do not guess
            return None
    return None


def index_codegraph(
    db_path: str, path: str, ctags_bin: Optional[str] = None
) -> Dict[str, Any]:
    """Full clear + reparse of the codegraph asset for `path`.

    Returns a stats dict with success, symbols, files, call_edges,
    ambiguous_edges, include_edges (or success False + error).
    """
    result = run_ctags(path, ctags_bin)
    if not result["success"]:
        return result  # {"success": False, "error": ...}
    entries = result["entries"]

    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA busy_timeout = 30000")
    _ensure_schema(conn)

    stats: Dict[str, Any] = {
        "success": True,
        "symbols": 0,
        "files": 0,
        "call_edges": 0,
        "ambiguous_edges": 0,
        "skipped_ambiguous": 0,
        "include_edges": 0,
    }

    try:
        # Full clear + reparse.
        conn.execute("DELETE FROM code_edges")
        conn.execute("DELETE FROM code_symbols")
        conn.commit()

        cwd = os.getcwd()
        inserted: List[Tuple[int, str, str, int, str, str]] = []
        file_set: Set[str] = set()
        for e in entries:
            name = e.get("name", "")
            if not name:
                continue
            file_abs = _norm(os.path.join(cwd, e.get("path", "")))
            if not file_abs:
                continue
            line = e.get("line")
            if not isinstance(line, int):
                line = None
            cur = conn.execute(
                """INSERT OR REPLACE INTO code_symbols
                       (language, kind, name, file, line, signature)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    e.get("language", ""),
                    e.get("kind", ""),
                    name,
                    file_abs,
                    line,
                    _extract_signature(e),
                ),
            )
            inserted.append(
                (
                    cur.lastrowid,
                    name,
                    file_abs,
                    line,
                    e.get("kind", ""),
                    e.get("language", ""),
                )
            )
            file_set.add(file_abs)
        conn.commit()
        stats["symbols"] = len(inserted)
        stats["files"] = len(file_set)

        # name -> symbol ids
        name_index: Dict[str, List[int]] = {}
        for sid, name, *_rest in inserted:
            name_index.setdefault(name, []).append(sid)

        # Per-file line cache (read once per file).
        lines_cache: Dict[str, Optional[List[str]]] = {}

        def file_lines(file_abs: str) -> Optional[List[str]]:
            if file_abs not in lines_cache:
                try:
                    with open(file_abs, "r", encoding="utf-8", errors="replace") as f:
                        lines_cache[file_abs] = f.readlines()
                except OSError:
                    lines_cache[file_abs] = None
            return lines_cache[file_abs]

        # Per-file symbols sorted by line (for body-span approximation).
        by_file: Dict[str, List[Tuple[int, int]]] = {}
        for sid, _name, file_abs, line, _kind, _lang in inserted:
            if line is None or line < 1:
                continue
            by_file.setdefault(file_abs, []).append((line, sid))
        for lst in by_file.values():
            lst.sort()

        # Body-span: lines [line, next_symbol_line) approximate the body.
        # A symbol at line L covers up to the next indexed symbol in the file.
        span_end: Dict[int, int] = {}
        for file_abs, lst in by_file.items():
            for i, (line, sid) in enumerate(lst):
                next_line = lst[i + 1][0] if i + 1 < len(lst) else None
                span_end[sid] = next_line if next_line is not None else None

        names = list(name_index.keys())
        name_re = (
            re.compile(r"\b(?:" + "|".join(re.escape(n) for n in names) + r")\b")
            if names
            else None
        )

        # Names whose definitions span too many files are library-level common
        # words (main/get/run/init...), not a specific call target. Creating
        # ambiguous edges to ALL of them is pure noise (measured: 15 names in
        # >10 files generated most of 213k ambiguous edges on the full index).
        # Skip ambiguous edges for those; keep genuine 2..N-file ambiguity.
        skip_ambiguous: Set[str] = set()
        if names:
            row = conn.execute(
                """SELECT name FROM code_symbols
                   WHERE kind != 'file'
                   GROUP BY name HAVING COUNT(DISTINCT file) > ?""",
                (MAX_AMBIGUOUS_FILES,),
            ).fetchall()
            skip_ambiguous = {r[0] for r in row}

        # Call edges: scan each symbol's body span for referenced names.
        for sid, name, file_abs, line, kind, language in inserted:
            if line is None or line < 1 or name_re is None:
                continue
            lines = file_lines(file_abs)
            if not lines:
                continue
            end = span_end.get(sid)
            chunk = lines[line - 1 : end if end is not None else None]
            if not chunk:
                continue
            text = "".join(chunk)
            for ref in name_re.findall(text):
                if ref in skip_ambiguous:
                    stats["skipped_ambiguous"] += 1
                    continue
                target_ids = name_index.get(ref)
                if not target_ids:
                    continue
                amb = len(target_ids) > 1
                for tid in target_ids:
                    if tid == sid:
                        continue
                    conn.execute(
                        """INSERT OR IGNORE INTO code_edges
                               (from_symbol_id, to_symbol_id, edge_kind, ambiguous)
                           VALUES (?, ?, 'call', ?)""",
                        (sid, tid, 1 if amb else 0),
                    )
                    if amb:
                        stats["ambiguous_edges"] += 1
                    else:
                        stats["call_edges"] += 1
        conn.commit()

        # File->file includes edges via synthetic 'file' symbols (line=0).
        file_basenames: Dict[str, List[str]] = {}
        for f in file_set:
            file_basenames.setdefault(os.path.basename(f), []).append(f)

        def ensure_file_node(file_abs: str) -> Optional[int]:
            name = os.path.basename(file_abs)
            conn.execute(
                """INSERT OR IGNORE INTO code_symbols
                       (language, kind, name, file, line, signature)
                   VALUES ('', 'file', ?, ?, 0, '')""",
                (name, file_abs),
            )
            row = conn.execute(
                "SELECT id FROM code_symbols WHERE file=? AND line=0 AND kind='file'",
                (file_abs,),
            ).fetchone()
            return row[0] if row else None

        for file_abs in sorted(file_set):
            lines = file_lines(file_abs)
            if not lines:
                continue
            from_id = ensure_file_node(file_abs)
            if from_id is None:
                continue
            seen_to: Set[int] = set()
            for line in lines:
                for rx in _IMPORT_PATTERNS:
                    # search() (not match()) so the unanchored require(...)
                    # patterns fire mid-line, e.g. `const x = require('y')`.
                    m = rx.search(line)
                    if not m:
                        continue
                    target = next((g for g in m.groups() if g), None)
                    if target:
                        resolved = _resolve_include(
                            target, file_abs, file_basenames, file_set
                        )
                        if resolved and resolved != file_abs:
                            to_id = ensure_file_node(resolved)
                            if to_id is not None and to_id not in seen_to:
                                seen_to.add(to_id)
                                conn.execute(
                                    """INSERT OR IGNORE INTO code_edges
                                           (from_symbol_id, to_symbol_id,
                                            edge_kind, ambiguous)
                                       VALUES (?, ?, 'includes', 0)""",
                                    (from_id, to_id),
                                )
                                stats["include_edges"] += 1
                    break  # first matching import pattern per line
        conn.commit()
    finally:
        conn.close()

    return stats
