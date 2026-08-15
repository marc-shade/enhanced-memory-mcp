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

        neural_memory_fabric used to resolve nmf.db in the real directory no
        matter what ENHANCED_MEMORY_DIR said; that is fixed at source now and
        TestNeuralMemoryFabricStoragePaths asserts it. This stays as the
        backstop for the next module that reinvents the same hardcoded path.
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


class TestAgiModulesHonourTheConfiguredDatabase:
    """Isolation by environment variable only isolates modules that read it.

    On 2026-08-14 test_agi_phase1.py ran under a harness that overrode every
    ENHANCED_MEMORY_* variable and still wrote 11 rows into the operator's real
    database: the agi/ modules built ~/.claude/enhanced_memories/memory.db
    inline and opened SQLite on it directly, so nothing the harness set was ever
    consulted. These tests fail against that code.
    """

    def test_no_agi_module_resolves_the_memory_directory_inline(self):
        """Static sweep: the whole package, not just the two modules that bit.

        Twenty-one of them carried the same two lines. A test naming only the
        ones in the reproduction would go green the moment a twenty-second file
        copies the pattern.
        """
        offenders = []
        for module in sorted((REPO_ROOT / "agi").glob("*.py")):
            text = module.read_text(encoding="utf-8", errors="ignore")
            if "enhanced_memories" in text and "Path.home()" in text:
                offenders.append(module.name)

        assert not offenders, (
            "These agi modules resolve the memory directory themselves and will "
            f"ignore ENHANCED_MEMORY_DB_PATH: {offenders}. Import "
            "get_memory_paths from memory_paths instead."
        )

    def test_agi_modules_resolve_the_configured_database(self):
        """The runtime half: what the modules in the incident actually opened."""
        import agi.action_tracker
        import agi.agent_identity

        configured = Path(os.environ["ENHANCED_MEMORY_DB_PATH"]).resolve()
        for module in (agi.action_tracker, agi.agent_identity):
            assert Path(module.DB_PATH).resolve() == configured, (
                f"{module.__name__}.DB_PATH is {module.DB_PATH}, not the "
                f"configured {configured}"
            )


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


class TestNeuralMemoryFabricStoragePaths:
    """NMF must put nmf.db and nmf_files where the environment says.

    Until 2026-08-15 the fallbacks were the literal string
    ~/.claude/enhanced_memories/nmf.db, and the shipped nmf_config.yaml names
    that same directory -- so an install pointed at its own ENHANCED_MEMORY_DIR
    still wrote both backends into the operator's real store. The sqlite guard
    above catches that inside this suite; these tests catch it at the source,
    where it also protects real second installs.
    """

    @staticmethod
    def _resolve(config_path=None):
        from neural_memory_fabric import NeuralMemoryFabric

        config = NeuralMemoryFabric._load_config(None, config_path)
        return (
            Path(config["storage"]["sqlite"]["path"]).expanduser(),
            Path(config["storage"]["files"]["root"]).expanduser(),
        )

    @staticmethod
    def _write_config(tmp_path, sqlite_path, files_root):
        import yaml

        config = tmp_path / "nmf_config.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "storage": {
                        "sqlite": {"path": str(sqlite_path)},
                        "files": {"root": str(files_root)},
                    }
                }
            )
        )
        return config

    def test_shipped_config_does_not_reach_the_real_store(self):
        """The config file as shipped, with the suite's ENHANCED_MEMORY_DIR."""
        sqlite_path, files_root = self._resolve()
        real_dir = REAL_MEMORY_DB.parent.expanduser().resolve()
        override = Path(os.environ["ENHANCED_MEMORY_DIR"]).resolve()
        for resolved in (sqlite_path.resolve(), files_root.resolve()):
            assert real_dir not in resolved.parents and resolved != real_dir, (
                f"NMF resolved {resolved} inside the operator's real store"
            )
            assert override in resolved.parents, (
                f"NMF resolved {resolved}, which is not under ENHANCED_MEMORY_DIR"
            )

    def test_a_config_naming_the_default_directory_is_relocated(self, tmp_path):
        """The default is what this repo ships, not a decision anyone made."""
        config = self._write_config(
            tmp_path,
            REAL_MEMORY_DB.parent / "nmf.db",
            REAL_MEMORY_DB.parent / "nmf_files",
        )
        sqlite_path, files_root = self._resolve(config)
        override = Path(os.environ["ENHANCED_MEMORY_DIR"]).resolve()
        # Resolve both sides: on macOS /tmp is a symlink to /private/tmp, so a
        # raw comparison fails on paths that are in fact the same file.
        assert sqlite_path.resolve() == (override / "nmf.db").resolve()
        assert files_root.resolve() == (override / "nmf_files").resolve()

    def test_a_deliberate_config_path_still_wins(self, tmp_path):
        """An operator who named an existing directory keeps it."""
        chosen = tmp_path / "chosen"
        chosen.mkdir()
        config = self._write_config(tmp_path, chosen / "nmf.db", chosen / "nmf_files")
        sqlite_path, files_root = self._resolve(config)
        assert sqlite_path == chosen / "nmf.db"
        assert files_root == chosen / "nmf_files"

    def test_the_env_overrides_beat_everything(self, tmp_path, monkeypatch):
        chosen = tmp_path / "chosen"
        chosen.mkdir()
        config = self._write_config(tmp_path, chosen / "nmf.db", chosen / "nmf_files")
        monkeypatch.setenv("NMF_SQLITE_PATH", str(tmp_path / "custom.db"))
        monkeypatch.setenv("NMF_FILES_ROOT", str(tmp_path / "custom_files"))
        sqlite_path, files_root = self._resolve(config)
        assert sqlite_path == tmp_path / "custom.db"
        assert files_root == tmp_path / "custom_files"


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
