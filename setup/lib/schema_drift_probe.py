#!/usr/bin/env python3
"""Catch the bug class where code INSERTs a column the schema never created.

Written after a real one: memory_db_service.py created the entities table
without `modality`, then inserted into `modality`. Every write on a fresh
database failed, and because the failure was caught per entity and the response
still said success, the caller saw an empty database instead of an error. It
only worked on machines where the MCP server had already added the column.

That was found by writing a probe row and reading it back, which proves the one
table the probe touches. This proves the rest statically and cheaply: every
literal INSERT in the two files that own this database is checked against the
live schema.

Deliberate limits, so the output is not read as more than it is:

* Only memory_db_service.py and server.py are scanned by default. They own
  memory.db. Other modules keep their own databases, and comparing their
  statements against this schema would produce confident nonsense.
* Only INSERT statements with an explicit column list are checked.
  `INSERT INTO t VALUES (...)` names no columns, and UPDATE ... SET takes
  expressions that a regex has no business parsing.

  IF YOU ARE ABOUT TO BUILD AN INSERT DYNAMICALLY, READ THIS. This probe reads
  source text, not runtime behaviour. A statement assembled from f-strings,
  joined column lists or a query builder is invisible to it, and its coverage
  drops silently: the check still passes, still prints a count, and no longer
  covers your statement. Nothing warns you. If you add dynamic SQL here, either
  keep a literal form for the column list or extend this probe in the same
  change.
* A table absent from the database is skipped, not failed. Several are created
  lazily by the component that uses them, so absence here means "not created
  yet", not "wrong".

So a PASS means: no literal INSERT names a column missing from a table that
exists. It does not mean the schema is complete or correct.

Output: RESULT <PASS|FAIL|WARN> <check-id> <message> on stdout. Exit 1 on FAIL.
"""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

# INSERT [OR REPLACE|OR IGNORE|...] INTO <table> ( <columns> )
INSERT_RE = re.compile(
    r"INSERT\s+(?:OR\s+\w+\s+)?INTO\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)",
    re.IGNORECASE | re.DOTALL,
)

DEFAULT_SOURCES = ("memory_db_service.py", "server.py")

_failed = False


def result(status: str, check: str, message: str) -> None:
    global _failed
    if status == "FAIL":
        _failed = True
    print(f"RESULT {status} {check} {message}", flush=True)


def statements(source: Path) -> List[Tuple[str, Set[str], int]]:
    """(table, columns, line number) for every literal INSERT with a column list."""
    text = source.read_text(errors="replace")
    found = []
    for match in INSERT_RE.finditer(text):
        table = match.group(1)
        columns = set()
        for raw in match.group(2).split(","):
            name = raw.strip().strip('"').strip("`").strip("[]").strip()
            # Skip anything that is not a bare identifier: an expression in a
            # column list means this is not the simple form we can verify.
            if name and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                columns.add(name)
        if columns:
            line = text.count("\n", 0, match.start()) + 1
            found.append((table, columns, line))
    return found


def live_schema(db_path: Path) -> Dict[str, Set[str]]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10)
    try:
        tables = [
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        ]
        return {
            table: {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
            for table in tables
        }
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, help="database to check against")
    parser.add_argument(
        "--repo",
        default=str(Path(__file__).resolve().parents[2]),
        help="checkout root holding the sources to scan",
    )
    parser.add_argument("--source", action="append", default=None)
    args = parser.parse_args()

    db_path = Path(os.path.expandvars(os.path.expanduser(args.db)))
    if not db_path.exists():
        result(
            "WARN",
            "schema",
            f"{db_path} does not exist yet; start the daemon once, then re-run",
        )
        return 0

    repo = Path(args.repo)
    sources = [repo / name for name in (args.source or DEFAULT_SOURCES)]
    present = [path for path in sources if path.exists()]
    if not present:
        result("WARN", "schema", f"none of {sources} exist; nothing to check")
        return 0

    try:
        schema = live_schema(db_path)
    except sqlite3.Error as exc:
        result("FAIL", "schema", f"cannot read schema from {db_path}: {exc}")
        return 1

    checked = 0
    skipped_tables: Set[str] = set()
    problems: List[str] = []

    for source in present:
        for table, columns, line in statements(source):
            if table not in schema:
                skipped_tables.add(table)
                continue
            checked += 1
            missing = sorted(columns - schema[table])
            if missing:
                problems.append(
                    f"{source.name}:{line} inserts into {table}."
                    f"{{{','.join(missing)}}} which the schema does not have"
                )

    if problems:
        for problem in problems:
            result("FAIL", "schema", problem)
        result(
            "FAIL",
            "schema",
            "every write through those statements fails at runtime, and the "
            "daemon reports the failure per row rather than raising",
        )
        return 1

    detail = f"{checked} literal INSERT column lists match the live schema"
    if skipped_tables:
        detail += (
            f"; {len(skipped_tables)} table(s) not in this database were skipped "
            f"({', '.join(sorted(skipped_tables)[:4])}"
            f"{', ...' if len(skipped_tables) > 4 else ''})"
        )
    result("PASS", "schema", detail)
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
