"""Regression tests for GitHub issue #9 (filed 2026-08-17).

`setup/lib/schema_drift_probe.py` built each table's allowed column set from
PRAGMA table_info alone. For an FTS5 virtual table that model is wrong: SQLite
also accepts `rowid` (the external-content rowid mapping) and a control column
named after the table itself, as in

    INSERT INTO observations_fts(observations_fts) VALUES('rebuild')

Since e9ca30c added exactly those forms in the FTS sync triggers, the probe
emitted five FAIL lines against a healthy database, healthcheck.sh counted them
in N_FAIL, and a brand-new clone could no longer finish "Required checks
passed". The writes it warned about work fine at runtime. Reported from a
Windows/Python 3.12 fresh install, reproduced on macOS/Python 3.14.

The last test here is the one that matters most. Widening an allowed set is the
kind of fix that quietly turns a gate into a rubber stamp, so `test_probe_still_
fails_on_a_genuinely_missing_fts_column` pins that a wrong column name on the
very same FTS5 table is still a FAIL. Without it, this file would prove only
that the probe stopped complaining.

Gaps / not covered: fts3/fts4 are not exercised (this repository uses fts5
only, and they additionally accept `docid`); the probe's regex-level parsing of
INSERT statements is unchanged and still cannot see dynamically-built SQL, which
is the documented limit it carries in its own module docstring.
"""

import importlib.util
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = REPO_ROOT / "setup" / "lib" / "schema_drift_probe.py"


def _load_probe():
    """Import the probe by path: setup/lib is not an importable package."""
    spec = importlib.util.spec_from_file_location("schema_drift_probe", PROBE_PATH)
    assert spec and spec.loader, f"cannot load probe from {PROBE_PATH}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def probe():
    return _load_probe()


@pytest.fixture()
def fts_db(tmp_path):
    """A database shaped like the real one: an ordinary table and an FTS5 index."""
    path = tmp_path / "fts.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_id INTEGER,
            content TEXT
        );
        CREATE VIRTUAL TABLE observations_fts USING fts5(
            content, content='observations', content_rowid='id');
        INSERT INTO observations (id, entity_id, content)
            VALUES (1, 1, 'the quick brown fox');
        """
    )
    conn.commit()
    conn.close()
    return path


def test_table_info_alone_does_not_describe_an_fts5_table(fts_db):
    """The premise of the bug, pinned so the fix is not mistaken for a workaround."""
    conn = sqlite3.connect(fts_db)
    try:
        declared = {
            row[1] for row in conn.execute("PRAGMA table_info(observations_fts)")
        }
        assert declared == {"content"}

        # ...yet all three flagged forms are accepted by SQLite. Order matters:
        # this is an external-content index, so the rows must exist in
        # `observations` first and 'delete' must name the exact indexed content.
        # Driving it in any other order corrupts the index instead of raising a
        # column error, which is its own reason not to model it from table_info.
        conn.execute("INSERT INTO observations_fts(observations_fts) VALUES('rebuild')")
        conn.execute(
            "INSERT INTO observations_fts(observations_fts, rowid, content) "
            "VALUES('delete', 1, 'the quick brown fox')"
        )
        conn.execute(
            "INSERT INTO observations_fts(rowid, content) "
            "VALUES (1, 'the quick brown fox')"
        )
        conn.commit()

        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        matched = conn.execute(
            "SELECT count(*) FROM observations_fts WHERE observations_fts MATCH 'brown'"
        ).fetchone()[0]
        assert matched == 1
    finally:
        conn.close()


def test_live_schema_grants_fts5_rowid_and_control_column(probe, fts_db):
    schema = probe.live_schema(fts_db)
    assert {"content", "rowid", "observations_fts"} <= schema["observations_fts"]


def test_live_schema_leaves_ordinary_tables_strict(probe, fts_db):
    """The widening must not leak to non-virtual tables."""
    schema = probe.live_schema(fts_db)
    assert schema["observations"] == {"id", "entity_id", "content"}
    # Neither special name is granted to a non-virtual table, including the
    # control-column name it would receive if the check keyed off the name
    # rather than the DDL.
    assert "rowid" not in schema["observations"]
    assert "observations" not in schema["observations"]


def test_probe_passes_against_a_fresh_service_database(tmp_path):
    """End-to-end: the fresh-install path a new user actually runs."""
    sys.path.insert(0, str(REPO_ROOT))
    from memory_db_service import MemoryDatabase

    db_path = tmp_path / "fresh.db"
    MemoryDatabase(db_path).init_database()

    completed = subprocess.run(
        [sys.executable, str(PROBE_PATH), "--db", str(db_path)],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "RESULT FAIL" not in completed.stdout
    assert "RESULT PASS" in completed.stdout


def test_probe_still_fails_on_a_genuinely_missing_fts_column(probe, fts_db, tmp_path):
    """Prove the gate can still fail: a wrong column on the FTS5 table is a FAIL.

    This is the check that keeps the issue-#9 fix from becoming a rubber stamp.
    """
    source = tmp_path / "fake_source.py"
    source.write_text(
        "INSERT INTO observations_fts(bogus_col, content) VALUES (1, 2)\n"
    )

    schema = probe.live_schema(fts_db)
    statements = probe.statements(source)
    assert statements, "probe did not parse the planted INSERT"

    table, columns = statements[0][0], statements[0][1]
    assert sorted(columns - schema[table]) == ["bogus_col"]
