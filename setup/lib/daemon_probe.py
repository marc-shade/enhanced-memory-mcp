#!/usr/bin/env python3
"""Round-trip probe for the memory-db daemon.

Answers one question honestly: does a write sent to the daemon socket land in
the configured database and come back out of a search?

Why this is not a socket ping. When the daemon is down, the MCP server does not
raise; it returns a well-formed object with a zero in it, for example::

    {"error": "Memory-DB service error: [Errno 2] No such file or directory",
     "entities": {"total": 0}, "compression": {"ratio": "N/A"}}
    {"query": "...", "count": 0, "results": [], "error": "..."}

Both shapes parse, both have the keys a caller expects, and both mean the store
is deaf. A check that reads ``entities.total`` and sees 0 reports "empty
database, fine". So this probe treats an ``error`` or ``daemon`` key, a falsy
``success``, or a nonzero ``failed`` count as failure regardless of the rest of
the payload, and it is not satisfied until it reads its own write back.

The daemon protocol is JSON over AF_UNIX: {"method": ..., "params": {...}} in,
one JSON object out, then the daemon closes the connection. Its methods are
create_entities, search_nodes and get_memory_status. Note that ``ping`` is NOT
among them: memory_client.py offers a ping() helper but the daemon answers it
with {"error": "Unknown method: ping"}, so a ping-based liveness check would
report failure against a perfectly healthy daemon.

Output: RESULT <PASS|FAIL|WARN> <check-id> <message> lines on stdout for
healthcheck.sh to tally, diagnostics on stderr. Exit 1 if any check FAILed.

Gaps / not covered: proves the daemon's own read/write path only. It does not
exercise the MCP tool layer (healthcheck.sh check 4 does that), Qdrant, ollama,
or concurrent access from several clients.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sqlite3
import sys
import uuid
from pathlib import Path
from typing import Any, Dict

TIMEOUT_S = 15.0
PROBE_PREFIX = "healthcheck-probe"

_failed = False


def result(status: str, check: str, message: str) -> None:
    global _failed
    if status == "FAIL":
        _failed = True
    print(f"RESULT {status} {check} {message}", flush=True)


def note(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def call(sock_path: str, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """One request/response round trip. Raises on transport failure."""
    payload = json.dumps({"method": method, "params": params}).encode()
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(TIMEOUT_S)
    try:
        sock.connect(sock_path)
        sock.sendall(payload)
        chunks = []
        while True:
            chunk = sock.recv(1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        sock.close()
    raw = b"".join(chunks)
    if not raw:
        raise RuntimeError(f"daemon closed the connection without answering {method}")
    return json.loads(raw.decode())


def reject_zero_shape(resp: Dict[str, Any], method: str) -> str:
    """Return a failure reason, or "" when the response is genuinely successful."""
    if not isinstance(resp, dict):
        return f"{method} returned {type(resp).__name__}, expected an object"
    for key in ("error", "daemon"):
        if key in resp and resp[key]:
            return f"{method} carries {key}={resp[key]!r}"
    if "success" in resp and not resp["success"]:
        return f"{method} returned success={resp['success']!r}"
    return ""


def resolve_expected_db() -> Path:
    """Resolve the database path exactly as both python processes do."""
    db_override = os.environ.get("ENHANCED_MEMORY_DB_PATH") or os.environ.get(
        "MEMORY_DB_PATH"
    )
    dir_override = os.environ.get("ENHANCED_MEMORY_DIR") or os.environ.get("MEMORY_DIR")
    if db_override:
        return Path(os.path.expandvars(os.path.expanduser(db_override)))
    base = (
        Path(os.path.expandvars(os.path.expanduser(dir_override)))
        if dir_override
        else Path.home() / ".claude" / "enhanced_memories"
    )
    return base / "memory.db"


def cleanup(db_path: str, name: str) -> None:
    """Delete the probe row. The daemon exposes no delete method, so this goes
    straight to SQLite. The daemon opens and closes a connection per request, so
    there is no long-lived lock to fight; busy_timeout covers the overlap."""
    conn = sqlite3.connect(db_path, timeout=10)
    try:
        conn.execute("PRAGMA busy_timeout = 10000")
        row = conn.execute("SELECT id FROM entities WHERE name = ?", (name,)).fetchone()
        if row is None:
            result("WARN", "cleanup", f"probe row {name} was already gone")
            return
        conn.execute("DELETE FROM observations WHERE entity_id = ?", (row[0],))
        conn.execute("DELETE FROM entities WHERE id = ?", (row[0],))
        conn.commit()
        result("PASS", "cleanup", f"probe row {name} removed")
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--socket",
        default=os.environ.get("MEMORY_DB_SOCKET_PATH", "/tmp/memory-db.sock"),
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="leave the probe entity in the database (for debugging)",
    )
    parser.add_argument(
        "--read-only",
        action="store_true",
        help="stop after the status call: liveness without writing a probe row. "
        "For repeating container healthchecks. Weaker evidence than the full "
        "round trip, which is why it is not the default.",
    )
    args = parser.parse_args()

    sock_path = args.socket
    expected_db = resolve_expected_db()

    # --- 1. transport ------------------------------------------------------
    if not os.path.exists(sock_path):
        result(
            "FAIL", "socket", f"{sock_path} does not exist: the daemon is not running"
        )
        note("start it with: setup/bin/memory-db-daemon.sh")
        return 1
    try:
        status = call(sock_path, "get_memory_status", {})
    except ConnectionRefusedError:
        result(
            "FAIL",
            "socket",
            f"{sock_path} exists but refuses connections: stale socket from a "
            "killed daemon. Remove it and restart the daemon.",
        )
        return 1
    except socket.timeout:
        result(
            "FAIL", "socket", f"{sock_path} accepted the connection but never answered"
        )
        return 1
    except OSError as exc:
        result("FAIL", "socket", f"{sock_path}: {exc}")
        return 1
    except Exception as exc:  # noqa: BLE001 - report anything, hide nothing
        result("FAIL", "socket", f"{sock_path}: {type(exc).__name__}: {exc}")
        return 1

    reason = reject_zero_shape(status, "get_memory_status")
    if reason:
        result("FAIL", "socket", reason)
        return 1
    result("PASS", "socket", f"daemon answered on {sock_path}")

    # --- 2. which database is it actually holding? -------------------------
    reported = status.get("database_path")
    total = status.get("entities", {}).get("total")
    if not reported:
        result(
            "WARN",
            "db-agreement",
            "daemon did not report database_path; cannot verify both processes "
            "resolved the same file",
        )
    elif Path(reported).resolve() != expected_db.resolve():
        result(
            "FAIL",
            "db-agreement",
            f"SPLIT BRAIN: daemon holds {reported}, this environment resolves "
            f"{expected_db}. Writes and statistics would come from different "
            "files. Restart the daemon via setup/bin/memory-db-daemon.sh so it "
            "reads the same .env.",
        )
    else:
        result("PASS", "db-agreement", f"both resolve {expected_db} ({total} entities)")

    if args.read_only:
        return 1 if _failed else 0

    # --- 3. write, read back, delete ---------------------------------------
    token = f"{PROBE_PREFIX}-{uuid.uuid4().hex[:12]}"
    entity = {
        "name": token,
        "entityType": "healthcheck",
        "observations": [
            f"Probe written by healthcheck.sh. Safe to delete. token={token}"
        ],
    }
    try:
        created = call(sock_path, "create_entities", {"entities": [entity]})
    except Exception as exc:  # noqa: BLE001
        result("FAIL", "write", f"create_entities raised {type(exc).__name__}: {exc}")
        return 1

    reason = reject_zero_shape(created, "create_entities")
    if reason:
        result("FAIL", "write", reason)
        return 1
    # success stays True even when individual entities fail, so count the rows.
    n_written = int(created.get("created", 0)) + int(created.get("updated", 0))
    n_failed = int(created.get("failed", 0))
    if n_failed or n_written < 1:
        result(
            "FAIL",
            "write",
            f"create_entities reported success but wrote {n_written} rows with "
            f"{n_failed} failures: {json.dumps(created)[:300]}",
        )
        return 1
    result("PASS", "write", f"created probe entity {token}")

    try:
        found = call(sock_path, "search_nodes", {"query": token, "limit": 10})
    except Exception as exc:  # noqa: BLE001
        result("FAIL", "read", f"search_nodes raised {type(exc).__name__}: {exc}")
        found = {}

    reason = reject_zero_shape(found, "search_nodes") if found else "no response"
    if reason:
        result("FAIL", "read", reason)
    else:
        names = [r.get("name") for r in found.get("results", []) if isinstance(r, dict)]
        if token in names:
            result(
                "PASS", "read", f"search_nodes returned the probe ({len(names)} hits)"
            )
        else:
            result(
                "FAIL",
                "read",
                f"search_nodes returned {len(names)} results and none was the "
                f"probe just written. The write and read paths disagree.",
            )

    # --- 4. leave no trace -------------------------------------------------
    if args.keep:
        result("WARN", "cleanup", f"--keep: probe entity {token} left in place")
    else:
        db_for_cleanup = reported or str(expected_db)
        try:
            cleanup(db_for_cleanup, token)
        except Exception as exc:  # noqa: BLE001
            result(
                "WARN",
                "cleanup",
                f"could not remove probe {token} from {db_for_cleanup}: "
                f"{type(exc).__name__}: {exc}",
            )

    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
