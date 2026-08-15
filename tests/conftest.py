"""Shared pytest configuration for the enhanced-memory-mcp test suite.

Three jobs:

1. Put the repository root on sys.path, so `import server` and
   `import memory_client` work no matter which directory pytest is invoked
   from.

2. Run `async def` tests. The suite marks them `@pytest.mark.asyncio`, which is
   pytest-asyncio's marker, but pytest-asyncio is not a dependency of this
   package -- and without it pytest does not fail those tests, it *errors* them
   with "async def functions are not natively supported", which reads like
   broken code rather than a missing plugin. Rather than add a plugin to the
   install just to run the tests, the hook below executes coroutine test
   functions directly. If pytest-asyncio is installed it takes precedence and
   this hook never fires.

3. Keep the suite off any real memory database. `MemoryClient()` built with no
   argument falls back to the shared socket at /tmp/memory-db.sock, so a test
   that calls the memory API on a developer machine silently reads and writes
   the operator's live store. That was not hypothetical: before this fixture,
   tests/code_exec/test_integration.py::test_api_access connected to a running
   production daemon and asserted against its 11,952 entities while reporting
   itself as a passing unit test.

   `_isolate_memory_socket` is autouse, so the default for every test is a
   socket path that does not exist -- an accidental memory call fails loudly
   instead of reaching production. A test that needs a real round-trip asks for
   the `memory_daemon` fixture, which starts a daemon of its own against a
   throwaway database.

Gaps / not covered: the async hook handles test *functions* only; async
fixtures and generators are unsupported because the suite has none. The
`memory_daemon` fixture starts one daemon per module and does not exercise
concurrent clients.
"""

import asyncio
import inspect
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SOCKET_ENV = "MEMORY_DB_SOCKET_PATH"

# Redirect the memory paths BEFORE any test module is imported. This has to
# happen at conftest import time, not in a fixture: `import server` runs
# `MEMORY_DIR.mkdir(parents=True, exist_ok=True)` at module scope, and with no
# override that resolves to ~/.claude/enhanced_memories. Measured 2026-08-14 --
# importing server from a test called mkdir on the operator's real memory
# directory, and on a machine where that directory does not exist yet, a test
# run would have created it.
_SESSION_MEMORY_DIR = Path(tempfile.mkdtemp(prefix="em-session-", dir="/tmp"))
os.environ["ENHANCED_MEMORY_DIR"] = str(_SESSION_MEMORY_DIR)
os.environ["ENHANCED_MEMORY_DB_PATH"] = str(_SESSION_MEMORY_DIR / "memory.db")
os.environ["MEMORY_DIR"] = str(_SESSION_MEMORY_DIR)
os.environ["MEMORY_DB_PATH"] = str(_SESSION_MEMORY_DIR / "memory.db")


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "asyncio: run this coroutine test function in an event loop"
    )


def pytest_unconfigure(config):
    shutil.rmtree(_SESSION_MEMORY_DIR, ignore_errors=True)


@pytest.fixture(autouse=True, scope="session")
def _forbid_production_mkdir():
    """Fail any test that creates a directory inside the real memory directory.

    The sqlite3.connect guard cannot see this: `mkdir` is a different syscall,
    and a module that only ever creates its storage directory would slip past a
    connection-level check entirely. neural_memory_fabric does exactly that for
    its files root, and server.py does it for MEMORY_DIR at import time.

    An earlier attempt to catch this class by watching the real directory's
    mtime across a test run was abandoned: on a machine running a daemon the
    directory changes constantly on its own (measured: 3 of 4 idle windows), so
    the observation carries no signal. Interception does.
    """
    real_dir = (Path.home() / ".claude" / "enhanced_memories").expanduser().resolve()
    original_mkdir = os.mkdir
    original_makedirs = os.makedirs

    def _is_inside(path) -> bool:
        try:
            resolved = Path(path).expanduser().resolve()
        except (OSError, ValueError, RuntimeError):
            return False
        return resolved == real_dir or real_dir in resolved.parents

    def guarded_mkdir(path, *args, **kwargs):
        if _is_inside(path):
            raise RuntimeError(
                f"Test created a directory in the real memory store: {path}\n"
                "The module under test resolves its own storage path instead of "
                "honouring ENHANCED_MEMORY_DIR."
            )
        return original_mkdir(path, *args, **kwargs)

    def guarded_makedirs(path, *args, **kwargs):
        if _is_inside(path):
            raise RuntimeError(
                f"Test created a directory tree in the real memory store: {path}\n"
                "The module under test resolves its own storage path instead of "
                "honouring ENHANCED_MEMORY_DIR."
            )
        return original_makedirs(path, *args, **kwargs)

    os.mkdir = guarded_mkdir
    os.makedirs = guarded_makedirs
    try:
        yield
    finally:
        os.mkdir = original_mkdir
        os.makedirs = original_makedirs


@pytest.fixture(autouse=True, scope="session")
def _forbid_production_database():
    """Fail any test that opens the real memory database directly.

    The socket isolation below only redirects clients that go through
    MemoryClient. Roughly a dozen modules resolve
    ~/.claude/enhanced_memories/memory.db themselves and open SQLite on it --
    api.memory.update_entity did until it was fixed, and it is a WRITE path.
    Those bypass the socket entirely, so a test that imports one of them can
    still reach the operator's real store.

    This wraps sqlite3.connect for the whole session and raises on any path
    under the real memory directory. It is deliberately loud: there is no
    legitimate reason for a unit test to touch that database, so a failure here
    is a finding, not an inconvenience to be silenced.
    """
    real_dir = (Path.home() / ".claude" / "enhanced_memories").expanduser().resolve()
    original_connect = sqlite3.connect

    def guarded_connect(database, *args, **kwargs):
        target = str(database)
        if target.startswith("file:"):
            target = target[len("file:") :].split("?", 1)[0]
        try:
            resolved = Path(target).expanduser().resolve()
        except (OSError, ValueError, RuntimeError):
            resolved = None
        if resolved is not None and (
            resolved == real_dir or real_dir in resolved.parents
        ):
            raise RuntimeError(
                f"Test opened the real memory database: {resolved}\n"
                "Nothing in this suite may read or write the operator's memory store. "
                "The module under test resolves its own path instead of honouring "
                "ENHANCED_MEMORY_DB_PATH, which is a product bug worth reporting, not "
                "a test to be exempted."
            )
        return original_connect(database, *args, **kwargs)

    sqlite3.connect = guarded_connect
    try:
        yield
    finally:
        sqlite3.connect = original_connect


@pytest.fixture(autouse=True)
def _isolate_memory_socket(monkeypatch):
    """Point every test at a socket that does not exist, unless it opts out.

    A test that genuinely wants the socket overrides this itself (the
    MemoryClient tests do, to pin the documented default), or takes the
    `memory_daemon` fixture.

    The path is short and under /tmp on purpose. Using pytest's tmp_path also
    blocks the connection, but with "AF_UNIX path too long" rather than "no
    such file or directory" -- a developer chasing that message would look at
    socket lengths instead of at this fixture.
    """
    monkeypatch.setenv(SOCKET_ENV, f"/tmp/em-no-daemon-{os.getpid()}.sock")


@pytest.fixture(scope="module")
def memory_daemon():
    """Run a memory-db daemon against a throwaway database.

    Yields the database path. The socket lives in a short /tmp directory
    because AF_UNIX truncates paths past ~104 bytes, and pytest's own tmp_path
    is comfortably longer than that.
    """
    work = Path(tempfile.mkdtemp(prefix="em-test-", dir="/tmp"))
    socket_path = work / "db.sock"
    db_path = work / "memory.db"

    env = dict(os.environ)
    env["ENHANCED_MEMORY_DB_PATH"] = str(db_path)
    env["ENHANCED_MEMORY_DIR"] = str(work)
    env[SOCKET_ENV] = str(socket_path)

    log = open(work / "daemon.log", "w")
    process = subprocess.Popen(
        [sys.executable, str(REPO_ROOT / "memory_db_service.py")],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )

    deadline = time.time() + 20
    while time.time() < deadline and not socket_path.exists():
        if process.poll() is not None:
            log.close()
            pytest.fail(f"memory-db daemon exited: {(work / 'daemon.log').read_text()}")
        time.sleep(0.1)
    if not socket_path.exists():
        process.kill()
        log.close()
        pytest.fail("memory-db daemon did not create its socket within 20s")

    previous = os.environ.get(SOCKET_ENV)
    os.environ[SOCKET_ENV] = str(socket_path)
    try:
        yield db_path
    finally:
        if previous is None:
            os.environ.pop(SOCKET_ENV, None)
        else:
            os.environ[SOCKET_ENV] = previous
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        log.close()
        shutil.rmtree(work, ignore_errors=True)


@pytest.fixture
def live_memory(memory_daemon, monkeypatch):
    """Re-point the socket at the module's daemon, past the autouse isolation.

    The autouse fixture runs per test and would otherwise put the dead socket
    back after `memory_daemon` set the real one.
    """
    monkeypatch.setenv(SOCKET_ENV, str(memory_daemon.parent / "db.sock"))
    return memory_daemon


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    """Execute a coroutine test function, or defer to a real async plugin."""
    test_func = pyfuncitem.obj
    if not inspect.iscoroutinefunction(test_func):
        return None

    if pyfuncitem.config.pluginmanager.hasplugin("asyncio"):
        return None  # pytest-asyncio is installed; let it do the work.

    kwargs = {
        name: pyfuncitem.funcargs[name] for name in pyfuncitem._fixtureinfo.argnames
    }
    asyncio.run(test_func(**kwargs))
    return True
