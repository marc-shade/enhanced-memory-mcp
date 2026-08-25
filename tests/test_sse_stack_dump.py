"""An HTTP-transport server arms SIGUSR1 stack dumps; the stdio one does not.

This is the only test that starts server.py as a real process under
MCP_TRANSPORT=sse. It exists because the arming code sits in the `__main__`
transport branch, which no import-level test reaches, and a daemon that
claims "ask me for my stacks" but never registered the handler would die of
SIGUSR1 instead (the signal's default action is termination).

The check is behavioural: wait for the port to accept a connection, send
SIGUSR1, and read the dump file. The process must still be alive afterwards.

Gaps / not covered: the daemon socket is a dead path here (the server boots
without one; tools would fail, the transport does not), and only the SSE
transport is exercised, not streamable-http.
"""

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_listening(port: int, proc: subprocess.Popen, timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise AssertionError(
                f"server exited rc={proc.returncode} before listening:\n"
                + proc.stderr.read().decode(errors="replace")[-3000:]
            )
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.25)
    raise AssertionError(f"port {port} never accepted a connection in {timeout}s")


@pytest.mark.skipif(not hasattr(signal, "SIGUSR1"), reason="platform has no SIGUSR1")
def test_sse_server_arms_sigusr1_stack_dump(tmp_path):
    port = _free_port()
    dump_dir = tmp_path / "dumps"
    env = dict(
        os.environ,
        MCP_TRANSPORT="sse",
        MCP_HOST="127.0.0.1",
        MCP_PORT=str(port),
        STACK_DUMP_DIR=str(dump_dir),
        MEMORY_PROFILE="minimal",
        MEMORY_DB_SOCKET_PATH=str(tmp_path / "no-daemon.sock"),
        PYTHONUNBUFFERED="1",
    )
    proc = subprocess.Popen(
        [sys.executable, str(REPO_ROOT / "server.py")],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        _wait_listening(port, proc, timeout=90)
        stacks = dump_dir / f"port-{port}.stacks.log"
        assert stacks.exists(), (
            "install() should have written the armed marker before listening"
        )
        armed = stacks.read_text()
        assert f"armed pid {proc.pid}" in armed

        os.kill(proc.pid, signal.SIGUSR1)
        deadline = time.time() + 10
        dumped = ""
        while time.time() < deadline:
            dumped = stacks.read_text()
            if 'File "' in dumped and dumped.count("\n") > armed.count("\n") + 2:
                break
            time.sleep(0.2)
        assert 'File "' in dumped, f"no stack frames after SIGUSR1:\n{dumped[-1500:]}"
        assert "Thread" in dumped, "faulthandler dump should name threads"
        assert proc.poll() is None, "SIGUSR1 must dump, not terminate, an armed daemon"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
