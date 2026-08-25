"""On-signal thread-stack dumps + timestamped logging for the MCP daemons.

Built 2026-08-23 after the third unexplained agent-runtime stall (a local
SSE daemon accepts TCP connections and then answers nothing; sightings
2026-08-19, 2026-08-22, and at a session start 2026-08-23, cause never
established). Two obstacles made those stalls undiagnosable after the
fact: the daemons' stderr is bare ``INFO:...`` lines with no timestamps,
so nothing correlates with an incident window; and py-spy requires root
on macOS, so nobody can ask a wedged daemon where it is stuck.

``install(name)`` gives a daemon both halves:

- ``SIGUSR1`` dumps every thread's stack to
  ``$STACK_DUMP_DIR/<name>.stacks.log`` (default
  ``~/.claude/logs/mcp-daemons/``). The handler is ``faulthandler``'s —
  registered C code that writes frames without touching the interpreter
  loop, so it works even while Python code is wedged, which is the whole
  point. To ask a wedged daemon where it is: ``kill -USR1 <pid>`` and
  read the file. The file records ``armed pid N`` when the handler is
  registered; never signal a pid that has not written that line, because
  SIGUSR1's default action is process termination.
- The root logger gains an ISO-timestamp format (``force=True`` replaces
  the bare ``logging.basicConfig(level=INFO)`` most servers ran first).

Fail-soft everywhere: a diagnosis helper must never take a daemon down,
so an unwritable directory or a platform without SIGUSR1 degrades to a
stderr note and the daemon runs on.

In this repository ``server.py`` calls ``install()`` when it starts an HTTP
transport (``MCP_TRANSPORT=sse`` or ``streamable-http``). The stdio
transport does not arm it: a stdio server is owned by one client process
and dies with it, so there is no long-lived daemon to interrogate.

Gaps / not covered: the dump file's faulthandler output carries no
timestamps of its own (write a marker line to the file before signalling
if you need a clock); dumps append forever with no rotation (a dump is
~2 KB and only written on demand).
"""

from __future__ import annotations

import faulthandler
import logging
import os
import signal
import sys
from pathlib import Path

LOG_FORMAT = "%(asctime)s %(levelname)s:%(name)s:%(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_installed: dict = {}


def dump_dir() -> Path:
    return Path(
        os.environ.get("STACK_DUMP_DIR")
        or Path.home() / ".claude" / "logs" / "mcp-daemons"
    )


def stacks_path(name: str) -> Path:
    return dump_dir() / f"{name}.stacks.log"


def install(name: str) -> bool:
    """Register the SIGUSR1 stack dump and timestamped logging. Idempotent.

    Returns True when the dump handler is armed, False when it degraded
    (unwritable dir, unsupported platform); the logging half is applied
    either way.
    """
    logging.basicConfig(
        level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT, force=True
    )

    if name in _installed:
        return True
    try:
        path = stacks_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        # The file object must outlive this call: faulthandler holds the fd.
        fh = open(path, "a", encoding="utf-8")
        faulthandler.register(signal.SIGUSR1, file=fh, all_threads=True)
        # Declare the handler ARMED for this pid. The probe refuses to signal
        # any pid that has not declared itself: SIGUSR1's default action is
        # process termination, so signalling an unarmed daemon would let the
        # diagnostic kill the patient (the test suite died of exactly this,
        # rc 158 = 128+SIGUSR1, before the declaration existed).
        from datetime import datetime, timezone

        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        fh.write(f"=== {stamp} armed pid {os.getpid()} ===\n")
        fh.flush()
        _installed[name] = fh
        return True
    except (OSError, AttributeError, ValueError) as exc:
        # AttributeError: no SIGUSR1 (never on macOS/Linux, but stay soft).
        print(
            f"stack_dump_on_signal: degraded, no dump handler: {exc}", file=sys.stderr
        )
        return False
