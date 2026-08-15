#!/usr/bin/env bash
# Launch the memory-db daemon with the checkout's .env applied.
#
# This is one of the two launchers that share a single .env. Start the daemon any
# other way and you are responsible for giving it byte-identical
# ENHANCED_MEMORY_DIR / ENHANCED_MEMORY_DB_PATH / MEMORY_DB_SOCKET_PATH values to
# whatever the MCP server sees. Disagreement does not raise: the server writes to
# one database and reports statistics from another.
set -euo pipefail

_self="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
. "${_self}/../lib/common.sh"

load_env || true

PY="$(python_for_run || true)"
[ -n "$PY" ] || die "no venv at $VENV_DIR and no python >= ${EMM_PYTHON_MIN} on PATH. Run setup/setup.sh first."

check_socket_path_length || exit 1

sock="$(socket_path)"
mkdir -p "$(dirname "$sock")"

# A socket file left behind by a killed daemon makes the next start fail with
# "Address already in use". The daemon unlinks a stale path itself at startup;
# this is here so a failure to do so is visible in the log rather than silent.
if [ -S "$sock" ] && ! daemon_running "$PY"; then
    printf 'removing stale socket %s\n' "$sock" >&2
    rm -f "$sock"
fi

cd "$REPO_ROOT"
exec "$PY" "${REPO_ROOT}/memory_db_service.py" "$@"
