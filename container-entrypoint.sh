#!/usr/bin/env bash
# Container entrypoint: supervise both processes, or fail loudly.
#
# The image runs the two processes this system needs:
#
#   1. memory_db_service.py   owns the SQLite file, listens on a Unix socket
#   2. server.py              the MCP server, on the SSE transport here because a
#                             container has no stdio peer to talk to
#
# Order matters and so does supervision. The MCP server starts happily without
# the daemon and answers every tool call with {"count": 0, "results": [],
# "error": ...}: a container that looked healthy while serving nothing is exactly
# the outcome this file exists to prevent. So:
#
#   * the daemon starts first and the socket must appear before the server starts
#   * if either process exits, both are stopped and the container exits nonzero,
#     letting the restart policy do its job instead of hiding a dead half
set -euo pipefail

REPO="${EMM_REPO:-/app}"
SOCKET="${MEMORY_DB_SOCKET_PATH:-/tmp/memory-db.sock}"
WAIT_SECONDS="${EMM_SOCKET_TIMEOUT:-30}"
PY="${EMM_PYTHON:-python}"

say() { printf '[entrypoint] %s\n' "$*" >&2; }

# A caller can still ask for a one-off command, e.g.
#   podman run --rm <image> ./healthcheck.sh --skip-mcp
if [ "$#" -gt 0 ]; then
    say "running: $*"
    exec "$@"
fi

cd "$REPO"

# Stale socket from an unclean stop of a previous container generation.
if [ -e "$SOCKET" ] && ! "$PY" - "$SOCKET" <<'PY' >/dev/null 2>&1
import socket, sys
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.settimeout(3)
try:
    s.connect(sys.argv[1])
finally:
    s.close()
PY
then
    say "removing stale socket $SOCKET"
    rm -f "$SOCKET"
fi

say "starting memory-db daemon (socket $SOCKET)"
"$PY" "${REPO}/memory_db_service.py" &
DAEMON_PID=$!

shutdown() {
    local code="${1:-0}"
    say "shutting down (exit $code)"
    kill "$DAEMON_PID" "${SERVER_PID:-}" 2>/dev/null || true
    wait "$DAEMON_PID" 2>/dev/null || true
    [ -n "${SERVER_PID:-}" ] && wait "$SERVER_PID" 2>/dev/null || true
    exit "$code"
}
trap 'shutdown 0' INT TERM

waited=0
until "$PY" "${REPO}/setup/lib/daemon_probe.py" --socket "$SOCKET" --read-only >/dev/null 2>&1; do
    if ! kill -0 "$DAEMON_PID" 2>/dev/null; then
        say "FATAL: the daemon exited before opening $SOCKET"
        wait "$DAEMON_PID" || true
        exit 1
    fi
    waited=$((waited + 1))
    if [ "$waited" -ge "$WAIT_SECONDS" ]; then
        say "FATAL: $SOCKET did not answer within ${WAIT_SECONDS}s"
        shutdown 1
    fi
    sleep 1
done
say "daemon is answering on $SOCKET"

# 0.0.0.0 is correct inside a container: the address is scoped to the container's
# own network namespace, and what is actually exposed is decided by the published
# port. compose.yaml publishes it on the host loopback only. Do NOT copy this
# value into a workstation .env.
export MCP_TRANSPORT="${MCP_TRANSPORT:-sse}"
export MCP_HOST="${MCP_HOST:-0.0.0.0}"
export MCP_PORT="${MCP_PORT:-9106}"

say "starting MCP server (${MCP_TRANSPORT} on ${MCP_HOST}:${MCP_PORT})"
"$PY" "${REPO}/server.py" &
SERVER_PID=$!

# Whichever process exits first takes the container with it. Without this, a dead
# daemon leaves a live server answering every query with a well-formed zero.
#
# errexit is suspended across the wait on purpose. `wait -n` returns the dead
# child's status, so under `set -e` a nonzero one kills this script on the spot:
# the container still exits, which looks correct from outside, but the shutdown
# below never runs and the log never says which half died. Verified 2026-08-14 by
# killing the daemon in a running container: exit code 1, and not one line of
# the diagnosis that follows.
set +e
wait -n "$DAEMON_PID" "$SERVER_PID"
FIRST_EXIT=$?
set -e

if ! kill -0 "$DAEMON_PID" 2>/dev/null; then
    say "the memory-db daemon exited (status ${FIRST_EXIT}); stopping the MCP server too"
else
    say "the MCP server exited (status ${FIRST_EXIT}); stopping the daemon too"
fi
shutdown "${FIRST_EXIT:-1}"
