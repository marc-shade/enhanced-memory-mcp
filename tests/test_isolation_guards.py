#!/usr/bin/env python3
"""Regression tests for the test suite's own isolation guards.

These exist because of a real incident, not as a theoretical safeguard. On
2026-08-14, tests/code_exec/test_integration.py::test_api_access connected to a
running production memory-db daemon and asserted against its 11,952 entities
while reporting itself as a passing unit test. That result was reported as
green before anyone noticed what it had been measuring.

Two guards in conftest.py now prevent it, and these tests prevent the guards
from being removed or quietly neutered. A guard nobody verifies is the same
class of problem as the bug it was written for: something that looks like
protection and is not.

Gaps / not covered: these assert that the guards are installed and that they
fire. They cannot prove no reachable code path exists that evades them -- see
test_no_module_bypasses_the_sqlite_guard for the one evasion that is checkable
statically.
"""

import os
import sqlite3
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_MEMORY_DB = Path.home() / ".claude" / "enhanced_memories" / "memory.db"


class TestProductionDatabaseGuard:
    """The suite must not be able to open the operator's real database."""

    def test_direct_connect_is_blocked(self):
        with pytest.raises(RuntimeError, match="real memory database"):
            sqlite3.connect(REAL_MEMORY_DB)

    def test_string_path_is_blocked(self):
        with pytest.raises(RuntimeError, match="real memory database"):
            sqlite3.connect(str(REAL_MEMORY_DB))

    def test_readonly_uri_is_blocked(self):
        """The URI form is a separate code path and was checked separately."""
        with pytest.raises(RuntimeError, match="real memory database"):
            sqlite3.connect(f"file:{REAL_MEMORY_DB}?mode=ro", uri=True)

    def test_sibling_files_in_the_real_directory_are_blocked(self):
        """The whole directory is off limits, not just memory.db.

        neural_memory_fabric resolves nmf.db in the same directory, and it is
        imported transitively by server.py.
        """
        with pytest.raises(RuntimeError, match="real memory database"):
            sqlite3.connect(REAL_MEMORY_DB.parent / "nmf.db")

    def test_unrelated_databases_still_open(self, tmp_path):
        """The guard must not be so broad that it breaks legitimate tests."""
        scratch = tmp_path / "scratch.db"
        conn = sqlite3.connect(scratch)
        try:
            conn.execute("CREATE TABLE t (x INTEGER)")
        finally:
            conn.close()
        assert scratch.exists()

    def test_guard_resolves_symlinks(self):
        """Path resolution is load-bearing, not incidental.

        On the machine this was written on, ~/.claude/enhanced_memories is a
        symlink into a different volume. A guard comparing path *strings* would
        match nothing and report a clean run with full confidence. This asserts
        the guard catches the resolved location too, whenever the two differ.
        """
        resolved = REAL_MEMORY_DB.parent.resolve()
        if resolved == REAL_MEMORY_DB.parent:
            pytest.skip("home memory directory is not a symlink on this machine")
        with pytest.raises(RuntimeError, match="real memory database"):
            sqlite3.connect(resolved / "memory.db")


class TestProductionMkdirGuard:
    """Creating storage directories is a separate escape route from connecting.

    `import server` runs `MEMORY_DIR.mkdir(parents=True, exist_ok=True)` at
    module scope. With no override that resolved to ~/.claude/enhanced_memories,
    so importing server from a test touched the operator's real memory directory
    -- and on a machine where it did not exist yet, created it. The connect
    guard could never have seen this: no connection is opened.
    """

    def test_mkdir_in_the_real_directory_is_blocked(self):
        with pytest.raises(RuntimeError, match="real memory store"):
            os.mkdir(REAL_MEMORY_DB.parent / "should_never_appear")

    def test_makedirs_in_the_real_directory_is_blocked(self):
        with pytest.raises(RuntimeError, match="real memory store"):
            os.makedirs(REAL_MEMORY_DB.parent / "a" / "b", exist_ok=True)

    def test_mkdir_elsewhere_still_works(self, tmp_path):
        target = tmp_path / "fine"
        os.mkdir(target)
        assert target.is_dir()

    def test_memory_path_env_is_redirected_away_from_the_real_store(self):
        """The guard is the backstop; the redirect is the actual fix.

        Without these overrides, server.py resolves the real directory at import
        time and the guard would fire on a plain `import server`.
        """
        for var in ("ENHANCED_MEMORY_DIR", "ENHANCED_MEMORY_DB_PATH"):
            value = os.environ.get(var)
            assert value, f"{var} is not set; test modules would resolve the real store"
            resolved = Path(value).resolve()
            assert REAL_MEMORY_DB.parent.resolve() not in [
                resolved,
                *resolved.parents,
            ], f"{var}={value} points into the real memory store"

    def test_importing_server_does_not_touch_the_real_store(self):
        """The end-to-end version of the above: the import that caused it."""
        import server

        assert Path(server.MEMORY_DIR).resolve() != REAL_MEMORY_DB.parent.resolve()


class TestSocketIsolation:
    """No test may inherit a socket pointing at a live daemon."""

    def test_socket_env_points_somewhere_that_does_not_exist(self):
        socket_path = os.environ.get("MEMORY_DB_SOCKET_PATH")
        assert socket_path, "the isolation fixture did not set MEMORY_DB_SOCKET_PATH"
        assert not Path(socket_path).exists(), (
            f"{socket_path} exists: this test could be talking to a real daemon"
        )

    def test_socket_is_not_the_shared_default(self):
        """Unset is as dangerous as the default, and must fail the same way.

        As first written this compared the variable to the shared path and
        nothing else, so it passed when the variable was absent -- which is
        precisely the case where MemoryClient falls back to
        /tmp/memory-db.sock and reaches the operator's daemon. Caught by
        disabling the isolation fixture and noticing this test stayed green
        while its neighbours went red.
        """
        socket_path = os.environ.get("MEMORY_DB_SOCKET_PATH")
        assert socket_path is not None, (
            "MEMORY_DB_SOCKET_PATH is unset: MemoryClient would fall back to "
            "the shared socket"
        )
        assert socket_path != "/tmp/memory-db.sock"

    def test_memory_client_cannot_connect_by_default(self):
        """The failure has to be reaching nothing, not reaching something else."""
        from memory_client import MemoryClient

        with pytest.raises(FileNotFoundError):
            MemoryClient().get_memory_status_sync()


def test_no_module_bypasses_the_sqlite_guard():
    """No module may bind sqlite3.connect directly.

    The guard replaces the attribute on the sqlite3 module, so
    `import sqlite3; sqlite3.connect(...)` is intercepted at call time. A module
    written as `from sqlite3 import connect` holds its own reference and would
    never see the guard -- it would bypass the isolation silently, which is
    exactly the failure mode the guard exists to prevent.

    Measured 2026-08-14: no module in the tree uses that form. This test keeps
    it that way, and gives a specific instruction if one appears.
    """
    offenders = []
    for path in REPO_ROOT.rglob("*.py"):
        if ".venv" in path.parts or path.name == __file__.rsplit("/", 1)[-1]:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if "from sqlite3 import" in text:
            offenders.append(str(path.relative_to(REPO_ROOT)))

    assert not offenders, (
        "These modules bind sqlite3.connect directly and would bypass the test "
        f"suite's production-database guard: {offenders}. Use `import sqlite3` "
        "and call `sqlite3.connect(...)` so the guard can intercept it."
    )
