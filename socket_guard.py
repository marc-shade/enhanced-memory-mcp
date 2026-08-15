#!/usr/bin/env python3
"""Refuse to take a Unix socket away from a daemon that is already serving it.

The daemon used to unlink whatever it found at its socket path and bind a fresh
one. That is right for a socket file left behind by a killed process and
catastrophic for a live one: the second daemon inherits every client of the
first, and because those clients then get well-formed replies out of an empty
database, the failure reads as "the memory store is empty" rather than as "you
are talking to a different daemon". Measured on a host install 2026-08-15 -- a
store holding 11,952 entities started answering zeros the moment a second
install was started with the default socket path.

The launcher (setup/bin/memory-db-daemon.sh) carries a stale-socket guard, but
it only runs when nothing is answering, which is the case that was already
safe, and it is bypassable: a real ~/.claude.json execs the module directly.
The check therefore lives here, on the path every start goes through.

An accepted connection is by itself decisive -- something is bound to that
address, so this process must not unlink it. The status request that follows
only enriches the refusal message; if it fails, the refusal still stands.
A connect that is refused (ECONNREFUSED) or finds nothing (ENOENT) means the
file outlived its daemon and is safe to remove.

Gaps / not covered: this cannot tell one daemon apart from an unrelated program
that happens to be bound to the same path, and it deliberately does not try.
It also does not close the race between the probe and the bind -- two daemons
starting within the same few milliseconds can still both pass the probe, and
the loser then fails on bind with EADDRINUSE rather than taking over.
"""

from __future__ import annotations

import errno
import json
import os
import socket
from typing import Any, Dict, Optional

# Long enough for a busy daemon to accept, short enough that a real stale-socket
# start is not visibly delayed. Only the connect and one status round trip use it.
PROBE_TIMEOUT_S = float(os.environ.get("MEMORY_DB_PROBE_TIMEOUT", "2"))

STATUS_REQUEST = json.dumps({"method": "get_memory_status", "params": {}}).encode()


class SocketInUseError(RuntimeError):
    """Another daemon is answering on the requested socket path."""


def probe_socket(
    socket_path: str, timeout: Optional[float] = None
) -> Optional[Dict[str, Any]]:
    """Ask whoever owns ``socket_path`` to identify itself.

    Returns ``None`` when the path is free or holds a stale socket file, which
    is the only case in which a caller may unlink it. Returns a dict when the
    address is occupied: ``answered`` says whether a status reply came back,
    and ``status`` carries it when it did.
    """
    timeout = PROBE_TIMEOUT_S if timeout is None else timeout
    # socket.connect() rejects a PathLike, so normalise before anything else:
    # the daemon passes a str, but every other caller here holds a Path.
    socket_path = os.fspath(socket_path)
    if not os.path.lexists(socket_path):
        return None

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        try:
            sock.connect(socket_path)
        except OSError as exc:
            if exc.errno in (errno.ECONNREFUSED, errno.ENOENT):
                return None
            # ENOTSOCK, EACCES, a connect timeout on a full backlog: something
            # is there, or we cannot tell. Neither is a licence to unlink.
            return {"answered": False, "reason": f"{type(exc).__name__}: {exc}"}

        occupant: Dict[str, Any] = {
            "answered": False,
            "reason": "connection accepted",
        }
        try:
            sock.sendall(STATUS_REQUEST)
            chunks = []
            while True:
                chunk = sock.recv(1 << 16)
                if not chunk:
                    break
                chunks.append(chunk)
            reply = json.loads(b"".join(chunks).decode())
        except (OSError, ValueError):
            return occupant
        if isinstance(reply, dict):
            occupant = {"answered": True, "status": reply}
        return occupant
    finally:
        sock.close()


def format_refusal(socket_path: str, occupant: Dict[str, Any]) -> str:
    """Build the message an operator sees when a start is refused."""
    status = occupant.get("status") or {}
    lines = [
        f"REFUSING TO START: another daemon is already serving {socket_path}",
        "Taking that socket over would point every client of the running daemon "
        "at this process's database. They would read zeros, not an error.",
    ]
    database_path = status.get("database_path")
    if database_path:
        lines.append(f"  the daemon holding it uses database: {database_path}")
    entities = (status.get("entities") or {}).get("total")
    if entities is not None:
        lines.append(f"  it reports {entities} entities")
    if not occupant.get("answered"):
        lines.append(
            f"  it did not answer a status request ({occupant.get('reason')}), "
            "but the address is bound"
        )
    lines += [
        "Either run this instance alongside the other one:",
        "  MEMORY_DB_SOCKET_PATH=/tmp/memory-db-<name>.sock \\",
        "  ENHANCED_MEMORY_DIR=<a directory of your own> ...   (or set both in .env)",
        "or stop the daemon that owns the socket first:",
        f"  lsof {socket_path}   # or: ss -xlp | grep {os.path.basename(socket_path)}",
    ]
    return "\n".join(lines)


def claim_socket_path(socket_path: str, timeout: Optional[float] = None) -> bool:
    """Make ``socket_path`` safe to bind, or refuse.

    Returns True when a stale socket file was removed, False when the path was
    already free. Raises :class:`SocketInUseError` when a daemon answers there.
    """
    socket_path = os.fspath(socket_path)
    occupant = probe_socket(socket_path, timeout)
    if occupant is not None:
        raise SocketInUseError(format_refusal(socket_path, occupant))
    if os.path.lexists(socket_path):
        os.unlink(socket_path)
        return True
    return False
