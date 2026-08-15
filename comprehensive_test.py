#!/usr/bin/env python3
"""Post-install gate for enhanced-memory-mcp.

Run this after installing to find out whether the server actually works on this
machine. It starts a real memory-db daemon and a real MCP server over stdio,
drives them through the real protocol, and asserts against the contracts the
current code returns -- not against a remembered older API.

    python comprehensive_test.py        # exit 0 = every assertion held

ISOLATION
    By default the gate builds its own sandbox: a fresh SQLite database and a
    fresh Unix socket in a temporary directory. It never reads or writes
    ~/.claude/enhanced_memories/memory.db and never binds /tmp/memory-db.sock,
    so a green run says something about the code rather than about whatever
    happens to be in the operator's memory store.

    If ENHANCED_MEMORY_DB_PATH / MEMORY_DB_PATH / ENHANCED_MEMORY_DIR /
    MEMORY_DIR / MEMORY_DB_SOCKET_PATH are set in the environment, those win --
    the gate exercises the deployment you configured. Because the gate WRITES
    entities, it refuses to run against a database that already holds any, and
    refuses the default production path outright. Set ENHANCED_MEMORY_TEST_FORCE=1
    to override that refusal.

WHAT A FAILURE MEANS
    Assertions here are written against contracts read out of server.py and
    memory_db_service.py at the time of writing (2026-08-14, server 2.14.1).
    A failure is one of two things and the message tries to say which: the
    install is broken, or the contract moved and this file is now stale. Both
    are worth knowing; neither should be silenced by deleting the assertion.

Gaps / not covered:
  * Qdrant / vector indexing. create_entities reports a `vector_indexing`
    block; the gate prints it but asserts nothing, because the vector store is
    an optional dependency and is expected to be absent on a fresh install.
  * The LLM-backed paths. Contextual enrichment runs with `using_llm: false`
    here; the enrichment quality path is unexercised.
  * Concurrency. One client at a time. The socket daemon exists to serve
    concurrent readers/writers and that property is not tested.
  * Tools beyond the front door. 200+ tools register; this gate calls five.
    A tool being *listed* is not evidence that it works.
  * Cross-platform. Written and measured on macOS (AF_UNIX path limit 104).
    Linux allows 108; the gate uses the stricter bound.
"""

import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent
SERVER_PATH = REPO_DIR / "server.py"
DAEMON_PATH = REPO_DIR / "memory_db_service.py"

# AF_UNIX sun_path is 104 bytes on macOS / 108 on Linux, NUL included. Use the
# stricter bound: a socket path over this fails at bind() with a truncated,
# misleading error.
SOCKET_PATH_LIMIT = 104

DEFAULT_PRODUCTION_DB = Path.home() / ".claude" / "enhanced_memories" / "memory.db"
DEFAULT_SOCKET = "/tmp/memory-db.sock"

ENV_KEYS = (
    "ENHANCED_MEMORY_DB_PATH",
    "MEMORY_DB_PATH",
    "ENHANCED_MEMORY_DIR",
    "MEMORY_DIR",
    "MEMORY_DB_SOCKET_PATH",
)


class Colors:
    """ANSI codes, blanked when stdout is not a terminal."""

    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    END = "\033[0m"

    @classmethod
    def disable(cls):
        for name in ("GREEN", "RED", "YELLOW", "BLUE", "CYAN", "BOLD", "END"):
            setattr(cls, name, "")


if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
    Colors.disable()


class TestResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
        self.details = []

    def assert_test(self, condition, test_name, error_msg=""):
        if condition:
            self.passed += 1
            print(f"  {Colors.GREEN}PASS{Colors.END} {test_name}")
            return True
        self.failed += 1
        self.errors.append(f"{test_name}: {error_msg}")
        print(
            f"  {Colors.RED}FAIL{Colors.END} {test_name}: {Colors.RED}{error_msg}{Colors.END}"
        )
        return False

    def add_detail(self, detail):
        self.details.append(detail)
        print(f"    {Colors.CYAN}->{Colors.END} {detail}")

    def print_summary(self, mode: str = ""):
        total = self.passed + self.failed
        print(f"\n{Colors.BOLD}=== SUMMARY ==={Colors.END}")
        if total == 0:
            print(
                f"{Colors.RED}No assertions ran. Treat this as a failure, not a pass.{Colors.END}"
            )
            return
        rate = (self.passed / total) * 100
        color = Colors.GREEN if self.failed == 0 else Colors.RED
        if mode:
            print(f"Mode: {mode}")
        print(f"Assertions: {total}")
        print(f"{Colors.GREEN}Passed: {self.passed}{Colors.END}")
        print(f"{Colors.RED}Failed: {self.failed}{Colors.END}")
        print(f"Success rate: {color}{rate:.1f}%{Colors.END}")
        if self.errors:
            print(f"\n{Colors.BOLD}Failures:{Colors.END}")
            for err in self.errors:
                print(f"  {Colors.RED}x{Colors.END} {err}")
        # The assertion total is mode-dependent and environment-dependent. Two
        # green runs can legitimately report different counts, so the count is
        # not the result -- the exit code is.
        print(
            "\nThe pass COUNT is not a comparable figure between machines or modes; "
            "the EXIT CODE is the result. Cite that."
        )


# --------------------------------------------------------------------------
# Environment resolution
# --------------------------------------------------------------------------


class Sandbox:
    """Where this run's database and socket live, and who chose them."""

    def __init__(self, work_dir: Path, db_path: Path, socket_path: str, from_env: bool):
        self.work_dir = work_dir
        self.db_path = db_path
        self.socket_path = socket_path
        self.from_env = from_env

    def child_env(self) -> dict:
        env = dict(os.environ)
        env["ENHANCED_MEMORY_DB_PATH"] = str(self.db_path)
        env["ENHANCED_MEMORY_DIR"] = str(self.db_path.parent)
        env["MEMORY_DB_SOCKET_PATH"] = self.socket_path
        # The daemon is authoritative; keep the legacy aliases pointing at the
        # same place so nothing resolves a second database behind our back.
        env["MEMORY_DB_PATH"] = str(self.db_path)
        env["MEMORY_DIR"] = str(self.db_path.parent)
        return env


def _same_path(a: Path, b: Path) -> bool:
    """Compare two paths, resolving symlinks only where the target exists.

    Path.resolve() on a non-existent path still normalizes it, but a symlinked
    home directory would make an unresolved comparison miss.
    """
    return a.expanduser().resolve() == b.expanduser().resolve()


def _entity_count(db_path: Path) -> int:
    """Entities in an existing database, or 0 if there is no table yet."""
    if not db_path.exists():
        return 0
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)
        try:
            row = conn.execute("SELECT COUNT(*) FROM entities").fetchone()
            return int(row[0]) if row else 0
        finally:
            conn.close()
    except sqlite3.Error:
        return 0


def resolve_sandbox() -> Sandbox:
    """Pick the database and socket for this run, or exit with a reason."""
    env_set = [k for k in ENV_KEYS if os.environ.get(k)]

    if not env_set:
        work = Path(tempfile.mkdtemp(prefix="em-gate-", dir="/tmp"))
        return Sandbox(work, work / "memory.db", str(work / "db.sock"), from_env=False)

    db_override = os.environ.get("ENHANCED_MEMORY_DB_PATH") or os.environ.get(
        "MEMORY_DB_PATH"
    )
    if db_override:
        db_path = Path(os.path.expandvars(os.path.expanduser(db_override)))
    else:
        dir_override = os.environ.get("ENHANCED_MEMORY_DIR") or os.environ.get(
            "MEMORY_DIR"
        )
        db_path = (
            Path(os.path.expandvars(os.path.expanduser(dir_override))) / "memory.db"
        )

    socket_path = os.environ.get("MEMORY_DB_SOCKET_PATH", DEFAULT_SOCKET)
    forced = os.environ.get("ENHANCED_MEMORY_TEST_FORCE") == "1"

    if not forced:
        if _same_path(db_path, DEFAULT_PRODUCTION_DB):
            sys.exit(
                f"{Colors.RED}Refusing to run: the environment points this gate at the default\n"
                f"production database ({DEFAULT_PRODUCTION_DB}).\n"
                f"This gate WRITES entities. Unset {', '.join(env_set)} to run isolated,\n"
                f"or set ENHANCED_MEMORY_TEST_FORCE=1 if you truly mean it.{Colors.END}"
            )
        existing = _entity_count(db_path)
        if existing:
            sys.exit(
                f"{Colors.RED}Refusing to run: {db_path} already holds {existing} entities.\n"
                f"This gate WRITES entities and would mix test data into a live store.\n"
                f"Point {env_set[0]} at a scratch path, or set ENHANCED_MEMORY_TEST_FORCE=1.{Colors.END}"
            )

    db_path.parent.mkdir(parents=True, exist_ok=True)
    return Sandbox(db_path.parent, db_path, socket_path, from_env=True)


def resolve_python() -> str:
    """The interpreter used for the daemon and server subprocesses."""
    venv = REPO_DIR / ".venv" / "bin" / "python"
    if venv.exists():
        return str(venv)
    return sys.executable


# --------------------------------------------------------------------------
# Process helpers
# --------------------------------------------------------------------------


class MemoryDaemon:
    """The memory-db Unix socket service. Every front-door tool goes through it."""

    def __init__(self, python_path, sandbox: Sandbox, log_path: Path, env=None):
        self.python_path = python_path
        self.sandbox = sandbox
        self.log_path = log_path
        self.env = env if env is not None else sandbox.child_env()
        self.process = None

    def start(self, timeout=20) -> bool:
        self.log = open(self.log_path, "w")
        self.process = subprocess.Popen(
            [self.python_path, str(DAEMON_PATH)],
            cwd=str(REPO_DIR),
            env=self.env,
            stdout=self.log,
            stderr=subprocess.STDOUT,
        )
        deadline = time.time() + timeout
        while time.time() < deadline:
            if os.path.exists(self.sandbox.socket_path):
                return True
            if self.process.poll() is not None:
                return False
            time.sleep(0.1)
        return False

    def log_text(self) -> str:
        try:
            return Path(self.log_path).read_text()
        except OSError:
            return ""

    def stop(self):
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
        if getattr(self, "log", None):
            self.log.close()
            self.log = None
        try:
            os.unlink(self.sandbox.socket_path)
        except FileNotFoundError:
            # The daemon removes its own socket on a clean shutdown; losing the
            # race with it is the expected case, not an error.
            pass


class MCPServer:
    """server.py over stdio, spoken to in JSON-RPC as a client would."""

    def __init__(self, python_path, env, log_path: Path):
        self.python_path = python_path
        self.env = env
        self.log_path = log_path
        self.process = None
        self.request_id = 0
        self.log = None

    def start(self) -> bool:
        try:
            self.log = open(self.log_path, "a")
            self.process = subprocess.Popen(
                [self.python_path, str(SERVER_PATH)],
                cwd=str(REPO_DIR),
                env=self.env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.log,
                text=True,
                bufsize=1,
            )
            return True
        except OSError as e:
            print(f"{Colors.RED}Failed to start server: {e}{Colors.END}")
            return False

    def _write(self, payload: dict) -> bool:
        try:
            self.process.stdin.write(json.dumps(payload) + "\n")
            self.process.stdin.flush()
            return True
        except (BrokenPipeError, ValueError, AttributeError):
            return False

    def request(self, method, params=None, timeout=60):
        """Send a JSON-RPC request, return the parsed response or an error dict."""
        if not self.process:
            return {"error": "no_process"}
        self.request_id += 1
        payload = {"jsonrpc": "2.0", "id": self.request_id, "method": method}
        if params is not None:
            payload["params"] = params
        if not self._write(payload):
            return {"error": "write_failed"}

        line = [None]

        def read():
            try:
                line[0] = self.process.stdout.readline()
            except (ValueError, AttributeError):
                line[0] = None

        thread = threading.Thread(target=read, daemon=True)
        thread.start()
        thread.join(timeout)
        if thread.is_alive():
            return {"error": f"timeout after {timeout}s"}
        if not line[0]:
            return {"error": "no_response (server exited?)"}
        try:
            return json.loads(line[0].strip())
        except json.JSONDecodeError as e:
            return {"error": f"unparseable response: {e}: {line[0][:200]!r}"}

    def notify(self, method, params=None):
        payload = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        self._write(payload)

    def handshake(self, timeout=60):
        response = self.request(
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "comprehensive-test", "version": "1.0.0"},
            },
            timeout=timeout,
        )
        if response and "result" in response:
            self.notify("notifications/initialized")
        return response

    def call_tool(self, name, arguments=None, timeout=60):
        """Call a tool and return its decoded payload, or a diagnostic dict.

        The payload is the tool's own return value. A transport-level problem
        surfaces as {"__transport_error__": ...} so it can never be mistaken
        for a tool that returned nothing.
        """
        response = self.request(
            "tools/call", {"name": name, "arguments": arguments or {}}, timeout=timeout
        )
        if not response or "result" not in response:
            return {"__transport_error__": response}
        result = response["result"]
        structured = result.get("structuredContent")
        if isinstance(structured, dict):
            return structured
        content = result.get("content") or []
        if content and content[0].get("type") == "text":
            try:
                return json.loads(content[0]["text"])
            except json.JSONDecodeError as e:
                return {"__transport_error__": f"tool text not JSON: {e}"}
        return {"__transport_error__": f"unexpected result shape: {list(result)}"}

    def stop(self):
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
        if self.log:
            self.log.close()
            self.log = None


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


def test_isolation(sandbox: Sandbox, result: TestResult):
    """TEST 1: the run is pointed where we think it is."""
    print(f"\n{Colors.BOLD}=== TEST 1: Isolation and path resolution ==={Colors.END}")

    result.add_detail(f"database: {sandbox.db_path}")
    result.add_detail(
        f"socket:   {sandbox.socket_path} ({len(sandbox.socket_path)} chars)"
    )
    result.add_detail(
        "source:   environment overrides"
        if sandbox.from_env
        else "source:   generated sandbox"
    )

    result.assert_test(
        len(sandbox.socket_path) < SOCKET_PATH_LIMIT,
        f"Socket path fits AF_UNIX limit ({SOCKET_PATH_LIMIT})",
        f"{len(sandbox.socket_path)} chars: bind() would fail or silently truncate",
    )

    if not sandbox.from_env:
        result.assert_test(
            not _same_path(sandbox.db_path, DEFAULT_PRODUCTION_DB),
            "Default run does not touch the production database",
            f"resolved to {sandbox.db_path}",
        )
        result.assert_test(
            sandbox.socket_path != DEFAULT_SOCKET,
            "Default run does not bind the shared socket",
            f"would collide with a running daemon at {DEFAULT_SOCKET}",
        )
        result.assert_test(
            ".claude" not in sandbox.db_path.parts,
            "Sandbox database lives outside ~/.claude",
            str(sandbox.db_path),
        )
        result.assert_test(
            _entity_count(sandbox.db_path) == 0,
            "Sandbox database starts empty",
            f"found {_entity_count(sandbox.db_path)} entities before any test wrote one",
        )
    else:
        # The three checks above describe a sandbox this run did not generate,
        # so they cannot be asserted here. Say so out loud: a run that quietly
        # performs fewer checks than another run of the same file, and reports
        # only a total, invites the reader to compare two numbers that are not
        # measuring the same thing.
        result.add_detail(
            "operator-directed mode: 3 sandbox-only checks are not applicable "
            "(production-path, shared-socket and ~/.claude location all assume a "
            "generated sandbox). This run therefore reports 4 fewer assertions than "
            "a default run: those 3 plus the sandbox entity-count check in TEST 8."
        )
        if os.environ.get("ENHANCED_MEMORY_TEST_FORCE") == "1":
            result.add_detail(
                "ENHANCED_MEMORY_TEST_FORCE=1: the empty-database precondition is "
                "waived, so this run may be writing into a populated store."
            )
        else:
            # Still assertable, and the property that actually matters: whatever
            # the operator pointed at, this run did not start on top of data.
            result.assert_test(
                _entity_count(sandbox.db_path) == 0,
                "Configured database started empty",
                f"found {_entity_count(sandbox.db_path)} entities before any test wrote one",
            )
    return True


def test_fresh_db_daemon_first(python_path, result: TestResult):
    """TEST 2: REGRESSION (2026-08-14) -- fresh database, daemon started alone.

    The daemon's CREATE TABLE for `entities` omitted the modality and
    raw_data_pointer columns that its own INSERT writes. server.py runs a
    migration at startup that adds them, so the bug was invisible whenever the
    server had ever run against the database. Daemon-first on a fresh install
    is the ordering that exposes it, and it is exactly the ordering a new user
    hits: start the service, then connect a client.

    The pre-fix failure was silent. The daemon returned:
        {"success": true, "created": 0, "updated": 0, "failed": 1, "count": 0}
    and server.py, seeing success, reported created=0 / failed=0 to the caller.
    Both assertions below are needed: the count, and the refusal to call that
    shape a success.
    """
    print(
        f"\n{Colors.BOLD}=== TEST 2: Fresh DB, daemon-first (regression) ==={Colors.END}"
    )

    work = Path(tempfile.mkdtemp(prefix="em-regress-", dir="/tmp"))
    fresh = Sandbox(work, work / "memory.db", str(work / "db.sock"), from_env=False)
    daemon = MemoryDaemon(python_path, fresh, work / "daemon.log")

    try:
        started = daemon.start()
        result.assert_test(
            started, "Daemon starts on a fresh database", daemon.log_text()[-400:]
        )
        if not started:
            return False

        # No server.py is started against `fresh` anywhere in this function.
        # That is the point of the test: server.py's startup migration adds the
        # missing columns, so any run that touches it can no longer see the bug.
        sys.path.insert(0, str(REPO_DIR))
        import asyncio

        from memory_client import MemoryClient

        client = MemoryClient(fresh.socket_path)
        response = asyncio.run(
            client.create_entities(
                [
                    {
                        "name": "regression_fresh_db_entity",
                        "entityType": "test",
                        "observations": [
                            "written by the daemon before any server started"
                        ],
                    }
                ]
            )
        )
        result.add_detail(f"daemon response: {json.dumps(response, sort_keys=True)}")

        created = response.get("created", 0)
        failed = response.get("failed", 0)

        result.assert_test(
            created == 1,
            "Entity written on a schema the daemon created itself",
            f"created={created} failed={failed}; daemon log: {daemon.log_text()[-300:]}",
        )
        result.assert_test(failed == 0, "No entity failed", f"failed={failed}")
        result.assert_test(
            not (response.get("success") and failed > 0),
            "Daemon does not report success while entities failed",
            f"success={response.get('success')} with failed={failed}: a caller cannot tell "
            "a working write from a dropped one",
        )

        # The column the INSERT needs must exist in the schema the daemon built.
        conn = sqlite3.connect(fresh.db_path)
        try:
            columns = {row[1] for row in conn.execute("PRAGMA table_info(entities)")}
        finally:
            conn.close()
        for column in ("modality", "raw_data_pointer"):
            result.assert_test(
                column in columns,
                f"Daemon-created schema has '{column}'",
                f"columns: {sorted(columns)}",
            )

        row_count = _entity_count(fresh.db_path)
        result.assert_test(
            row_count == 1,
            "Row is actually in the database, not just in the response",
            f"SELECT COUNT(*) returned {row_count}",
        )
    finally:
        daemon.stop()
        shutil.rmtree(work, ignore_errors=True)
    return True


def test_mcp_protocol(server: MCPServer, result: TestResult):
    """TEST 3: MCP handshake and the tool surface that actually ships."""
    print(f"\n{Colors.BOLD}=== TEST 3: MCP protocol and tool surface ==={Colors.END}")

    init = server.handshake()
    ok = result.assert_test(
        "result" in init, "initialize succeeds", f"response: {json.dumps(init)[:300]}"
    )
    if not ok:
        return False

    info = init["result"]
    result.assert_test("protocolVersion" in info, "Response carries a protocol version")
    result.assert_test("serverInfo" in info, "Response carries serverInfo")
    result.assert_test(
        info.get("serverInfo", {}).get("name") == "enhanced-memory",
        "Server identifies as 'enhanced-memory'",
        f"got {info.get('serverInfo')}",
    )
    result.add_detail(
        f"server {info.get('serverInfo', {}).get('version')} "
        f"speaking MCP {info.get('protocolVersion')}"
    )

    listing = server.request("tools/list")
    ok = result.assert_test(
        "result" in listing,
        "tools/list succeeds",
        f"response: {json.dumps(listing)[:300]}",
    )
    if not ok:
        return False

    tools = [t["name"] for t in listing["result"].get("tools", [])]
    result.add_detail(f"{len(tools)} tools registered")

    # The front door: eagerly loaded, and what this gate exercises below.
    for name in ("create_entities", "search_nodes", "get_memory_status"):
        result.assert_test(name in tools, f"Front-door tool '{name}' registered")

    # Removed surface. These are knowledge-graph tools from the upstream memory
    # server that this server does not implement; relations are carried on
    # entities. Asserting their absence keeps a future re-add from going unnoticed.
    for name in ("read_graph", "create_relations"):
        result.assert_test(
            name not in tools,
            f"Removed tool '{name}' is absent",
            "tool reappeared: this file's assumptions about the surface are stale",
        )
    return True


def test_create_entities(server: MCPServer, result: TestResult):
    """TEST 4: create_entities returns what the current server returns."""
    print(f"\n{Colors.BOLD}=== TEST 4: create_entities contract ==={Colors.END}")

    entities = [
        {
            "name": "gate_orchestrator_core",
            "entityType": "system_role",
            "observations": ["Core orchestrator functionality", "Always active"],
        },
        {
            "name": "gate_project_alpha",
            "entityType": "project",
            "observations": [
                "Machine learning project",
                "Uses Python and TensorFlow",
                "Data analysis focus",
            ],
        },
        {
            "name": "gate_project_beta",
            "entityType": "project",
            "observations": ["Web development project", "JavaScript and React"],
        },
        {
            "name": "gate_person_alice",
            "entityType": "person",
            "observations": ["Senior developer", "Python expert"],
        },
    ]

    payload = server.call_tool("create_entities", {"entities": entities})
    ok = result.assert_test(
        "__transport_error__" not in payload,
        "create_entities call completes",
        str(payload.get("__transport_error__"))[:300],
    )
    if not ok:
        return False

    result.assert_test(
        payload.get("created") == len(entities),
        f"All {len(entities)} entities created",
        f"created={payload.get('created')} failed={payload.get('failed')} "
        f"error={payload.get('error')}",
    )
    result.assert_test(
        payload.get("failed") == 0,
        "No entity failed",
        f"failed={payload.get('failed')}",
    )
    result.assert_test(
        "daemon" not in payload,
        "Success envelope carries no daemon-failure marker",
        f"daemon={payload.get('daemon')}",
    )

    results = payload.get("results", [])
    result.assert_test(
        len(results) == len(entities),
        "One result row per entity",
        f"got {len(results)}",
    )
    if results:
        row = results[0]
        for field in ("name", "id", "compression_ratio"):
            result.assert_test(field in row, f"Result row has '{field}'", f"row: {row}")

    for section in ("contextual_enrichment", "tpu_scoring"):
        result.assert_test(section in payload, f"Response includes '{section}'")

    enrichment = payload.get("contextual_enrichment", {})
    result.assert_test(
        enrichment.get("enriched", 0) + enrichment.get("failed", 0) == len(entities),
        "Enrichment accounts for every entity",
        f"enriched={enrichment.get('enriched')} failed={enrichment.get('failed')}",
    )
    result.add_detail(f"enrichment: {json.dumps(enrichment, sort_keys=True)}")
    result.add_detail(
        f"tpu scoring: {json.dumps(payload.get('tpu_scoring', {}), sort_keys=True)}"
    )
    if "vector_indexing" in payload:
        # Asserted on purpose: Qdrant is optional and absent on a fresh install.
        result.add_detail(
            f"vector indexing (not asserted): {payload['vector_indexing']}"
        )
    return True


def test_partial_failure(server: MCPServer, result: TestResult):
    """TEST 5: a write where some entities fail is not reported as a success.

    Forced with an entity that has no name, which the NOT NULL constraint on
    entities.name rejects. The daemon previously set success=True before the
    loop ran and never revised it, so a call that stored nothing returned
    created=0/failed=1/success=true and server.py relayed created=0/failed=0.
    The caller had no field that distinguished a dropped write from an empty
    request.
    """
    print(f"\n{Colors.BOLD}=== TEST 5: Partial failure ==={Colors.END}")

    payload = server.call_tool(
        "create_entities",
        {
            "entities": [
                {
                    "name": "gate_partial_survivor",
                    "entityType": "test",
                    "observations": ["this one is well formed"],
                },
                {"entityType": "test", "observations": ["no name: violates NOT NULL"]},
            ]
        },
    )
    result.add_detail(
        f"partial-write response: {json.dumps(payload, sort_keys=True)[:400]}"
    )

    result.assert_test(
        payload.get("status") == "failed",
        "Partial write reports status='failed'",
        f"status={payload.get('status')!r}",
    )
    result.assert_test(
        "error" in payload,
        "Partial write carries an error message",
        f"keys: {sorted(payload)}",
    )
    result.assert_test(
        "created" not in payload,
        "Partial write does not report a 'created' count",
        "a caller reading `created` on a failed call cannot tell it failed",
    )
    result.assert_test(
        payload.get("failed") == 1,
        "Partial write reports how many entities failed",
        f"failed={payload.get('failed')}",
    )
    result.assert_test(
        payload.get("stored") == 1,
        "Partial write reports how many entities did land",
        f"stored={payload.get('stored')}: the survivor is a measured fact and must be visible",
    )
    result.assert_test(
        payload.get("requested") == 2,
        "Partial write echoes how many were requested",
        f"requested={payload.get('requested')}",
    )
    result.assert_test(
        str(payload.get("daemon", "")).startswith("ERROR ("),
        "Partial write is marked ERROR, not UNREACHABLE",
        f"daemon={payload.get('daemon')!r}: the daemon answered, so this is not a transport failure",
    )

    # A caller retrying a partial batch has to know which names NOT to rewrite,
    # so the survivors are identified, not just counted.
    survivors = payload.get("results", [])
    result.assert_test(
        [row.get("name") for row in survivors] == ["gate_partial_survivor"],
        "Partial write names the entities that survived, not just how many",
        f"results={survivors!r}: a retry cannot tell which of the batch already landed",
    )
    result.assert_test(
        all("id" in row for row in survivors),
        "Each surviving entity carries its id",
        f"results={survivors!r}",
    )

    # The claim that one entity survived has to be true on disk, not just in
    # the response.
    found = server.call_tool("search_nodes", {"query": "gate_partial_survivor"})
    result.assert_test(
        found.get("count", 0) == 1,
        "The entity reported as stored is actually retrievable",
        f"search returned {found.get('count')}",
    )
    return True


def test_compression(server: MCPServer, result: TestResult):
    """TEST 6: compression is real, measured on a large entity."""
    print(f"\n{Colors.BOLD}=== TEST 6: Compression ==={Colors.END}")

    big = {
        "name": "gate_compression_entity",
        "entityType": "test",
        "observations": [
            "This is a very long observation with lots of repeated text. " * 50,
            "Another long observation with different content but still lengthy. " * 30,
        ],
    }
    payload = server.call_tool("create_entities", {"entities": [big]})
    ok = result.assert_test(
        payload.get("created") == 1,
        "Large entity created",
        f"payload: {json.dumps(payload)[:300]}",
    )
    if not ok:
        return False

    ratio_text = payload["results"][0].get("compression_ratio", "")
    result.assert_test(
        ratio_text.endswith("%"),
        "Per-entity compression ratio reported as a percentage",
        f"got {ratio_text!r}",
    )
    try:
        ratio = float(ratio_text.rstrip("%"))
    except ValueError:
        result.assert_test(
            False, "Compression ratio parses as a number", repr(ratio_text)
        )
        return False

    result.assert_test(
        0 < ratio < 100,
        "Compressed output is smaller than the input",
        f"ratio {ratio}% -- 100% or more means compression did nothing",
    )
    result.add_detail(f"repetitive text compressed to {ratio:.2f}% of original")

    status = server.call_tool("get_memory_status")
    compression = status.get("compression", {})
    result.assert_test(
        compression.get("total_original_kb", 0)
        > compression.get("total_compressed_kb", -1),
        "Store-wide totals show a net saving",
        f"original={compression.get('total_original_kb')}KB "
        f"compressed={compression.get('total_compressed_kb')}KB",
    )
    result.add_detail(f"store-wide savings: {compression.get('ratio')}")
    return True


def test_search(server: MCPServer, result: TestResult):
    """TEST 7: search_nodes contract and field names."""
    print(f"\n{Colors.BOLD}=== TEST 7: Search ==={Colors.END}")

    payload = server.call_tool("search_nodes", {"query": "gate_project"})
    ok = result.assert_test(
        "__transport_error__" not in payload,
        "search_nodes call completes",
        str(payload.get("__transport_error__"))[:300],
    )
    if not ok:
        return False

    for field in ("query", "count", "confidence", "low_confidence", "results"):
        result.assert_test(field in payload, f"Search response has '{field}'")

    result.assert_test(
        payload.get("query") == "gate_project", "Response echoes the query back"
    )
    hits = payload.get("results", [])
    result.assert_test(
        payload.get("count") == len(hits),
        "count matches the number of rows returned",
        f"count={payload.get('count')} rows={len(hits)}",
    )
    result.assert_test(
        len(hits) >= 2,
        "Finds both entities created with the 'gate_project' prefix",
        f"found {len(hits)}: {[h.get('name') for h in hits]}",
    )

    if hits:
        row = hits[0]
        # entityType is camelCase here and snake_case nowhere: asserting
        # entity_type is the mistake the previous version of this file made.
        for field in (
            "id",
            "name",
            "entityType",
            "observations",
            "tier",
            "access_count",
        ):
            result.assert_test(
                field in row, f"Search row has '{field}'", f"keys: {sorted(row)}"
            )
        result.assert_test(
            isinstance(row.get("observations"), list),
            "Observations come back decompressed as a list",
            f"got {type(row.get('observations')).__name__}",
        )
        result.add_detail(f"hits: {[h.get('name') for h in hits]}")

    # A query that matches nothing must be distinguishable from a broken search.
    empty = server.call_tool("search_nodes", {"query": "zzz_no_such_entity_zzz"})
    result.assert_test(
        empty.get("count") == 0 and "error" not in empty,
        "A no-match search returns count 0 without an error",
        f"payload: {json.dumps(empty)[:300]}",
    )
    result.assert_test(
        empty.get("low_confidence") is True,
        "A no-match search is flagged low confidence",
        f"low_confidence={empty.get('low_confidence')}",
    )
    return True


def test_status(server: MCPServer, sandbox: Sandbox, result: TestResult):
    """TEST 8: get_memory_status, and the proof this ran against the sandbox."""
    print(f"\n{Colors.BOLD}=== TEST 8: Memory status ==={Colors.END}")

    payload = server.call_tool("get_memory_status")
    ok = result.assert_test(
        "__transport_error__" not in payload,
        "get_memory_status call completes",
        str(payload.get("__transport_error__"))[:300],
    )
    if not ok:
        return False

    for section in ("entities", "compression", "tiers", "database_path"):
        result.assert_test(section in payload, f"Status has '{section}'")

    total = payload.get("entities", {}).get("total")
    result.assert_test(
        isinstance(total, int) and total > 0,
        "Entity total is a positive integer",
        f"total={total!r}",
    )
    result.add_detail(f"entities in this run's store: {total}")
    result.add_detail(f"tiers: {json.dumps(payload.get('tiers', {}), sort_keys=True)}")

    reported = payload.get("database_path", "")
    result.assert_test(
        Path(reported).resolve() == sandbox.db_path.resolve(),
        "Daemon served THIS run's database",
        f"daemon says {reported}, gate expected {sandbox.db_path}",
    )
    if not sandbox.from_env:
        # The isolation claim, stated as a number rather than an intention.
        result.assert_test(
            total < 100,
            "Entity count is this run's writes only, not a populated store",
            f"total={total}: the gate is reading somebody else's database",
        )
    return True


def test_persistence(
    server: MCPServer, python_path, sandbox: Sandbox, result: TestResult
):
    """TEST 9: data survives a server restart."""
    print(f"\n{Colors.BOLD}=== TEST 9: Persistence across restart ==={Colors.END}")

    before = server.call_tool("get_memory_status").get("entities", {}).get("total", 0)
    result.add_detail(f"entities before restart: {before}")

    server.stop()
    time.sleep(1)
    restarted = server.start()
    result.assert_test(restarted, "Server restarts")
    if not restarted:
        return False
    init = server.handshake()
    result.assert_test(
        "result" in init, "Re-initialize after restart", json.dumps(init)[:200]
    )

    after = server.call_tool("get_memory_status").get("entities", {}).get("total", 0)
    result.assert_test(
        after == before,
        "Entity count preserved across restart",
        f"before={before} after={after}",
    )

    found = server.call_tool("search_nodes", {"query": "gate_orchestrator_core"})
    result.assert_test(
        found.get("count", 0) > 0,
        "A specific entity is still retrievable after restart",
        f"payload: {json.dumps(found)[:300]}",
    )
    return True


def test_daemon_unreachable(python_path, sandbox: Sandbox, result: TestResult):
    """TEST 10: a dead daemon does not look like an empty memory system.

    This is the honesty property the failure envelope exists for. A response
    carrying `entities: {total: 0}` alongside an error renders as a healthy,
    empty store in any caller that reads the data keys first.
    """
    print(f"\n{Colors.BOLD}=== TEST 10: Daemon-unreachable envelope ==={Colors.END}")

    dead_socket = str(sandbox.work_dir / "not-a-socket.sock")
    env = sandbox.child_env()
    env["MEMORY_DB_SOCKET_PATH"] = dead_socket
    result.assert_test(
        not os.path.exists(dead_socket),
        "Socket under test genuinely does not exist",
        dead_socket,
    )

    server = MCPServer(python_path, env, sandbox.work_dir / "server-dead-socket.log")
    if not result.assert_test(server.start(), "Server starts with no daemon behind it"):
        return False
    try:
        init = server.handshake()
        if not result.assert_test(
            "result" in init,
            "Server initializes despite the dead socket",
            json.dumps(init)[:200],
        ):
            return False

        status = server.call_tool("get_memory_status")
        result.add_detail(
            f"status with dead socket: {json.dumps(status, sort_keys=True)[:300]}"
        )

        result.assert_test(
            "error" in status, "Failure response carries 'error'", str(status)[:200]
        )
        daemon_field = status.get("daemon", "")
        result.assert_test(
            daemon_field.startswith("UNREACHABLE ("),
            "Failure response names the state as UNREACHABLE",
            f"daemon={daemon_field!r}",
        )
        result.assert_test(
            dead_socket in daemon_field,
            "Failure response names the socket it could not reach",
            f"daemon={daemon_field!r}",
        )
        result.assert_test(
            status.get("status") == "failed",
            "Failure response sets status='failed'",
            f"status={status.get('status')!r}",
        )
        for key in ("entities", "compression", "tiers", "count", "results", "created"):
            result.assert_test(
                key not in status,
                f"Failure response carries no data-shaped key '{key}'",
                f"{key}={status.get(key)!r}: an empty store and a dead daemon would render alike",
            )

        search = server.call_tool("search_nodes", {"query": "anything"})
        result.assert_test(
            "error" in search and "results" not in search and "count" not in search,
            "Failed search returns an error with no empty result list",
            f"payload: {json.dumps(search, sort_keys=True)[:300]}",
        )

        create = server.call_tool(
            "create_entities",
            {
                "entities": [
                    {
                        "name": "unreachable_probe",
                        "entityType": "test",
                        "observations": [],
                    }
                ]
            },
        )
        result.assert_test(
            "error" in create and create.get("created") is None,
            "Failed create returns an error and never claims created=0 as a result",
            f"payload: {json.dumps(create, sort_keys=True)[:300]}",
        )
        # A total failure omits the partial-write keys rather than zero-filling
        # them. `stored: 0` and `results: []` would be indistinguishable from a
        # write that legitimately stored nothing, which is the whole point of
        # the envelope. Absent is the contract; empty is not.
        for key in ("stored", "results"):
            result.assert_test(
                key not in create,
                f"Total failure omits '{key}' entirely rather than zero-filling it",
                f"{key}={create.get(key)!r}: an empty value and a broken write must not "
                "serialize alike",
            )
    finally:
        server.stop()
    return True


def test_database_schema(sandbox: Sandbox, result: TestResult):
    """TEST 11: on-disk structure and integrity of the database this run used."""
    print(f"\n{Colors.BOLD}=== TEST 11: Database structure ==={Colors.END}")

    db_path = sandbox.db_path
    if not result.assert_test(db_path.exists(), "Database file exists", str(db_path)):
        return False

    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        tables = {
            row[0]
            for row in cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        for table in ("entities", "observations"):
            result.assert_test(
                table in tables, f"Table '{table}' exists", f"tables: {sorted(tables)}"
            )

        columns = {row[1] for row in cursor.execute("PRAGMA table_info(entities)")}
        expected = {
            "id",
            "name",
            "entity_type",
            "tier",
            "compressed_data",
            "original_size",
            "compressed_size",
            "compression_ratio",
            "checksum",
            "access_count",
            "created_at",
            "last_accessed",
            "modality",
            "raw_data_pointer",
        }
        missing = expected - columns
        result.assert_test(
            not missing,
            "entities table has every column the writers use",
            f"missing: {sorted(missing)}",
        )

        indexes = {
            row[0]
            for row in cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            )
        }
        result.assert_test(
            any("name" in i for i in indexes) and any("type" in i for i in indexes),
            "Name and type indexes exist on entities",
            f"indexes: {sorted(indexes)}",
        )

        entities = cursor.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        observations = cursor.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
        result.add_detail(f"{entities} entities, {observations} observations on disk")
        result.assert_test(
            observations > 0,
            "Observations were persisted, not just entity rows",
            f"count={observations}",
        )

        # Compressed blobs must round-trip. A checksum column that nothing
        # verifies is decoration.
        row = cursor.execute(
            "SELECT compressed_data, checksum FROM entities WHERE name = ?",
            ("gate_project_alpha",),
        ).fetchone()
        if result.assert_test(
            row is not None, "Known entity present for a round-trip check"
        ):
            import hashlib
            import zlib

            blob, checksum = row
            result.assert_test(
                hashlib.sha256(blob).hexdigest() == checksum,
                "Stored checksum matches the stored blob",
                "checksum mismatch: silent corruption or a checksum that is never computed",
            )
            decoded = json.loads(zlib.decompress(blob).decode())
            result.assert_test(
                decoded.get("name") == "gate_project_alpha",
                "Compressed blob decompresses to the original entity",
                f"decoded: {str(decoded)[:200]}",
            )

        integrity = cursor.execute("PRAGMA integrity_check").fetchone()[0]
        result.assert_test(
            integrity == "ok", "PRAGMA integrity_check passes", f"result: {integrity}"
        )
    finally:
        conn.close()
    return True


# --------------------------------------------------------------------------


def main():
    print(
        f"{Colors.BOLD}{Colors.BLUE}=== enhanced-memory-mcp post-install gate ==={Colors.END}"
    )
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if not SERVER_PATH.exists() or not DAEMON_PATH.exists():
        sys.exit(
            f"{Colors.RED}Missing server.py or memory_db_service.py in {REPO_DIR}{Colors.END}"
        )

    sandbox = resolve_sandbox()
    python_path = resolve_python()
    result = TestResult()
    result.add_detail(f"interpreter: {python_path}")
    mode = (
        "operator-directed (environment overrides)"
        if sandbox.from_env
        else "isolated sandbox (default)"
    )

    probe = subprocess.run(
        [python_path, "-c", "import fastmcp, mcp"], capture_output=True, text=True
    )
    if probe.returncode != 0:
        sys.exit(
            f"{Colors.RED}{python_path} cannot import fastmcp/mcp -- dependencies are not "
            f"installed.\n{probe.stderr.strip()[:400]}{Colors.END}"
        )

    test_isolation(sandbox, result)

    # Runs before anything starts a server: the bug it guards is only visible
    # when no server.py has ever migrated the database.
    try:
        test_fresh_db_daemon_first(python_path, result)
    except Exception as e:
        result.assert_test(
            False, "test_fresh_db_daemon_first", f"{type(e).__name__}: {e}"
        )

    daemon = MemoryDaemon(python_path, sandbox, sandbox.work_dir / "daemon.log")
    server = MCPServer(
        python_path, sandbox.child_env(), sandbox.work_dir / "server.log"
    )

    try:
        if not result.assert_test(
            daemon.start(), "memory-db daemon starts", daemon.log_text()[-400:]
        ):
            result.print_summary(mode)
            return 1
        if not result.assert_test(server.start(), "MCP server process starts"):
            result.print_summary(mode)
            return 1

        for name, test in (
            ("test_mcp_protocol", lambda: test_mcp_protocol(server, result)),
            ("test_create_entities", lambda: test_create_entities(server, result)),
            ("test_partial_failure", lambda: test_partial_failure(server, result)),
            ("test_compression", lambda: test_compression(server, result)),
            ("test_search", lambda: test_search(server, result)),
            ("test_status", lambda: test_status(server, sandbox, result)),
            (
                "test_persistence",
                lambda: test_persistence(server, python_path, sandbox, result),
            ),
        ):
            try:
                test()
            except Exception as e:
                # An exception is a failed assertion, never a skipped test: it
                # counts against the run and names itself in the summary.
                result.assert_test(False, name, f"{type(e).__name__}: {e}")

        server.stop()

        try:
            test_daemon_unreachable(python_path, sandbox, result)
        except Exception as e:
            result.assert_test(
                False, "test_daemon_unreachable", f"{type(e).__name__}: {e}"
            )

        try:
            test_database_schema(sandbox, result)
        except Exception as e:
            result.assert_test(
                False, "test_database_schema", f"{type(e).__name__}: {e}"
            )
    finally:
        server.stop()
        daemon.stop()

    result.print_summary(mode)
    print(f"\nDatabase used: {sandbox.db_path}")
    if not sandbox.from_env:
        if result.failed == 0:
            shutil.rmtree(sandbox.work_dir, ignore_errors=True)
            print("Sandbox removed (clean run).")
        else:
            print(f"Sandbox kept for inspection: {sandbox.work_dir}")
            print(f"  daemon log: {sandbox.work_dir / 'daemon.log'}")
            print(f"  server log: {sandbox.work_dir / 'server.log'}")

    return 0 if result.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
