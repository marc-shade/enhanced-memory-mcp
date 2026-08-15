#!/usr/bin/env bash
# Launch the MCP server with the checkout's .env applied.
#
# Point your MCP client at THIS script rather than at python server.py. Doing so
# is what guarantees the server and the daemon resolve the same database: both
# launchers source the same .env. A client that execs python directly inherits
# only whatever environment that client happened to have, which is how a split
# brain starts.
#
# stdout belongs to the MCP protocol. Every diagnostic here goes to stderr; a
# single stray line on stdout breaks the client handshake with a JSON parse error.
set -euo pipefail

_self="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
. "${_self}/../lib/common.sh"

load_env || true

PY="$(python_for_run || true)"
[ -n "$PY" ] || {
    printf 'FAIL no venv at %s and no python >= %s on PATH. Run setup/setup.sh first.\n' \
        "$VENV_DIR" "$EMM_PYTHON_MIN" >&2
    exit 1
}

if ! [ -S "$(socket_path)" ]; then
    printf 'WARN memory-db daemon socket %s is absent.\n' "$(socket_path)" >&2
    printf 'WARN Tools will return {"error": ..., "count": 0} instead of data.\n' >&2
    printf 'WARN Start it with: setup/bin/memory-db-daemon.sh\n' >&2
fi

cd "$REPO_ROOT"
exec "$PY" "${REPO_ROOT}/server.py" "$@"
