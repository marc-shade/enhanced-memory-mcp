#!/usr/bin/env python3
"""A second daemon must not take a live socket away from the first.

The incident these tests come from, measured on a host install 2026-08-15: a
second checkout was started with the default socket path while the operator's
daemon was serving 11,952 entities on it. The new daemon unlinked the socket,
bound its own, and every existing client -- MCP server, hooks, CLI -- began
reading a brand new empty database. Nothing errored. The store simply reported
zero, which is indistinguishable from a store that is genuinely empty.

The three cases below are the whole contract: refuse when somebody is
answering, take over when the file is stale, do nothing special when the path
is free. The stale path is the one the launcher already handled correctly, so
these tests also pin that it did not regress.

Gaps / not covered: the probe/bind race (two daemons starting in the same few
milliseconds both pass the probe; the loser fails on bind instead of taking
over), and non-daemon programs bound to the same path, which the guard
deliberately treats as "occupied" without identifying them.
"""

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from socket_guard import (  # noqa: E402
    SocketInUseError,
    claim_socket_path,
    probe_socket,
)

DAEMON = REPO_ROOT / "memory_db_service.py"


def call(sock_path, method, params=None):
    """One JSON request/response round trip against a daemon socket."""
    payload = json.dumps({"method": method, "params": params or {}}).encode()
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(10)
    try:
        sock.connect(str(sock_path))
        sock.sendall(payload)
        chunks = []
        while True:
            chunk = sock.recv(1 << 16)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        sock.close()
    return json.loads(b"".join(chunks).decode())


class Daemon:
    """A memory-db daemon with its own database, socket and log."""

    def __init__(self, work: Path, socket_path: Path, name: str):
        self.dir = work / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.dir / "memory.db"
        self.socket_path = socket_path
        self.log_path = self.dir / "daemon.log"
        self._log = open(self.log_path, "w")
        env = dict(os.environ)
        env["ENHANCED_MEMORY_DIR"] = str(self.dir)
        env["ENHANCED_MEMORY_DB_PATH"] = str(self.db_path)
        env["MEMORY_DIR"] = str(self.dir)
        env["MEMORY_DB_PATH"] = str(self.db_path)
        env["MEMORY_DB_SOCKET_PATH"] = str(socket_path)
        self.process = subprocess.Popen(
            [sys.executable, str(DAEMON)],
            cwd=str(REPO_ROOT),
            env=env,
            stdout=self._log,
            stderr=subprocess.STDOUT,
        )

    def wait_serving(self, timeout=20):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.socket_path.exists():
                try:
                    call(self.socket_path, "get_memory_status")
                    return True
                except OSError:
                    # Bound but not accepting yet. Keep waiting until timeout;
                    # the deadline below is what turns this into a failure.
                    time.sleep(0.05)
            if self.process.poll() is not None:
                return False
            time.sleep(0.1)
        return False

    def wait_exit(self, timeout=20):
        try:
            return self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None

    @property
    def log(self):
        if not self._log.closed:
            self._log.flush()
        return self.log_path.read_text()

    def stop(self, sig=signal.SIGTERM):
        if self.process.poll() is None:
            self.process.send_signal(sig)
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self._log.close()


@pytest.fixture
def work():
    """A short /tmp directory: AF_UNIX truncates paths past ~104 bytes."""
    path = Path(tempfile.mkdtemp(prefix="em-guard-", dir="/tmp"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


class TestProbe:
    def test_free_path_probes_as_free(self, work):
        assert probe_socket(work / "nothing.sock") is None

    def test_stale_socket_file_probes_as_free(self, work):
        """A bound-then-abandoned socket file: connect gets ECONNREFUSED."""
        stale = work / "stale.sock"
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(str(stale))
        listener.listen(1)
        listener.close()  # file survives, nothing is listening
        assert stale.exists()
        assert probe_socket(stale) is None

    def test_live_listener_probes_as_occupied(self, work):
        live = work / "live.sock"
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(str(live))
        listener.listen(1)
        try:
            occupant = probe_socket(live, timeout=1)
        finally:
            listener.close()
        assert occupant is not None
        # This listener never replies, so the probe cannot have "answered" --
        # and an accepted connection alone must still count as occupied.
        assert occupant["answered"] is False

    def test_claim_refuses_a_live_listener_and_leaves_the_file(self, work):
        live = work / "live.sock"
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(str(live))
        listener.listen(1)
        try:
            with pytest.raises(SocketInUseError) as caught:
                claim_socket_path(live, timeout=1)
            assert str(live) in str(caught.value)
            assert live.exists(), "a refused claim must not remove the socket"
        finally:
            listener.close()

    def test_claim_removes_a_stale_file(self, work):
        stale = work / "stale.sock"
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(str(stale))
        listener.listen(1)
        listener.close()
        assert claim_socket_path(stale) is True
        assert not stale.exists()

    def test_claim_on_a_free_path_is_a_no_op(self, work):
        assert claim_socket_path(work / "nothing.sock") is False


class TestDaemonStartup:
    def test_second_daemon_refuses_a_live_socket(self, work):
        """The incident case: the first daemon keeps its socket and its data."""
        sock = work / "db.sock"
        first = Daemon(work, sock, "first")
        try:
            assert first.wait_serving(), f"first daemon did not start: {first.log}"
            written = call(
                sock,
                "create_entities",
                {
                    "entities": [
                        {
                            "name": "socket-guard-canary",
                            "entityType": "test",
                            "observations": ["written to the first daemon"],
                        }
                    ]
                },
            )
            assert written.get("created") == 1, written

            second = Daemon(work, sock, "second")
            rc = second.wait_exit()
            second.stop()

            assert rc is not None, "second daemon did not exit; it took the socket"
            assert rc != 0, f"second daemon exited 0 after refusing: log {second.log}"
            assert "REFUSING TO START" in second.log
            assert str(sock) in second.log
            assert str(first.db_path) in second.log, (
                "the refusal must name the database of the daemon that answered"
            )
            assert not second.db_path.exists() or _entity_count(second.db_path) == 0

            # The point of the whole exercise: clients of the first daemon are
            # still talking to the first daemon.
            status = call(sock, "get_memory_status")
            assert status["database_path"] == str(first.db_path)
            assert status["entities"]["total"] == 1
            found = call(sock, "search_nodes", {"query": "socket-guard-canary"})
            assert found["count"] == 1, found
        finally:
            first.stop()

    def test_stale_socket_is_taken_over(self, work):
        """SIGKILL leaves the file behind; the next daemon must still start."""
        sock = work / "db.sock"
        first = Daemon(work, sock, "first")
        assert first.wait_serving(), f"first daemon did not start: {first.log}"
        first.stop(sig=signal.SIGKILL)
        assert sock.exists(), "SIGKILL should leave the socket file behind"

        second = Daemon(work, sock, "second")
        try:
            assert second.wait_serving(), (
                f"stale socket blocked a legitimate start: {second.log}"
            )
            assert "removed stale socket" in second.log
            status = call(sock, "get_memory_status")
            assert status["database_path"] == str(second.db_path)
        finally:
            second.stop()

    def test_fresh_path_starts_without_touching_anything(self, work):
        sock = work / "db.sock"
        daemon = Daemon(work, sock, "only")
        try:
            assert daemon.wait_serving(), f"daemon did not start: {daemon.log}"
            assert "removed stale socket" not in daemon.log
            assert "REFUSING TO START" not in daemon.log
            status = call(sock, "get_memory_status")
            assert status["database_path"] == str(daemon.db_path)
        finally:
            daemon.stop()


class TestNewDatabasePermissions:
    """A database the daemon creates must not be world readable.

    Lives here because this file is where a real daemon already runs against a
    throwaway directory. It matters because setup.sh stopped chmod'ing a
    database it did not create: without this, a fresh install's memory.db would
    inherit the umask (0644 on a default account) until somebody happened to
    re-run the installer.
    """

    def test_a_fresh_database_is_created_600(self, work):
        sock = work / "db.sock"
        daemon = Daemon(work, sock, "only")
        try:
            assert daemon.wait_serving(), f"daemon did not start: {daemon.log}"
            mode = daemon.db_path.stat().st_mode & 0o777
            assert mode == 0o600, f"new database is mode {oct(mode)}"
        finally:
            daemon.stop()

    def test_an_existing_database_keeps_its_mode(self, work):
        """Same principle as the installer: do not re-permission what exists."""
        socket_dir = work / "existing"
        socket_dir.mkdir()
        db_path = socket_dir / "memory.db"
        db_path.touch()
        db_path.chmod(0o644)
        sock = work / "db.sock"
        daemon = Daemon(work, sock, "existing")
        try:
            assert daemon.wait_serving(), f"daemon did not start: {daemon.log}"
            assert db_path.stat().st_mode & 0o777 == 0o644
        finally:
            daemon.stop()


def _entity_count(db_path: Path) -> int:
    import sqlite3

    conn = sqlite3.connect(db_path)
    try:
        return conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    finally:
        conn.close()
