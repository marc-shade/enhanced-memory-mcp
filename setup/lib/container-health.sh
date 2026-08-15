#!/usr/bin/env bash
# Container HEALTHCHECK: is each half of the pair actually up?
#
# Two independent facts, because either one alone lies:
#   1. the daemon answers on its Unix socket (a live MCP server next to a dead
#      daemon returns well-formed zeros, and the port would still be open)
#   2. the MCP port accepts a TCP connection
#
# Read-only on purpose: this runs every 30 seconds. The write/read/delete round
# trip lives in ./healthcheck.sh and is meant to be run once, by a human.
set -euo pipefail

REPO="${EMM_REPO:-/app}"
PY="${EMM_PYTHON:-python}"

"$PY" "${REPO}/setup/lib/daemon_probe.py" \
    --socket "${MEMORY_DB_SOCKET_PATH:-/tmp/memory-db.sock}" \
    --read-only >/dev/null

exec "$PY" - "${MCP_PORT:-9106}" <<'PY'
import socket, sys

port = int(sys.argv[1])
connection = socket.create_connection(("127.0.0.1", port), timeout=5)
connection.close()
PY
