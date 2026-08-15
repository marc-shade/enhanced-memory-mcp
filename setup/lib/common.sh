#!/usr/bin/env bash
# Shared helpers for setup.sh, healthcheck.sh, the launchers and the service
# installers. Sourced, never executed.
#
# Written for bash 3.2 because that is what macOS ships. No associative arrays,
# no mapfile, no ${var,,}.

# --- repository layout -----------------------------------------------------

# Absolute path of the checkout, derived from this file's own location so every
# caller agrees regardless of the working directory it was invoked from.
_emm_common_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${_emm_common_dir}/../.." && pwd)"
export REPO_ROOT

VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"
VENV_PY="${VENV_DIR}/bin/python"
ENV_FILE="${ENV_FILE:-${REPO_ROOT}/.env}"
# shellcheck disable=SC2034  # read by setup.sh, which sources this file
ENV_EXAMPLE="${REPO_ROOT}/.env.example"

# --- output ----------------------------------------------------------------

# shellcheck disable=SC2034  # C_BOLD is used by the scripts that source this file
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
    C_RED=$'\033[31m'; C_GREEN=$'\033[32m'; C_YELLOW=$'\033[33m'
    C_BLUE=$'\033[34m'; C_BOLD=$'\033[1m'; C_OFF=$'\033[0m'
else
    C_RED=''; C_GREEN=''; C_YELLOW=''; C_BLUE=''; C_BOLD=''; C_OFF=''
fi

log()  { printf '%s\n' "$*"; }
info() { printf '%s==>%s %s\n' "$C_BLUE" "$C_OFF" "$*"; }
ok()   { printf '%sPASS%s %s\n' "$C_GREEN" "$C_OFF" "$*"; }
warn() { printf '%sWARN%s %s\n' "$C_YELLOW" "$C_OFF" "$*" >&2; }
fail() { printf '%sFAIL%s %s\n' "$C_RED" "$C_OFF" "$*" >&2; }
die()  { fail "$*"; exit 1; }

# Octal permission bits of a path, for reporting what was left alone.
# BSD stat and GNU stat disagree on the flag, so try both before giving up.
file_mode() {
    # GNU first, BSD as the fallback - NOT the other way around. On GNU
    # coreutils `stat -f` means FILESYSTEM status, so it SUCCEEDS with the
    # wrong output and a BSD-first chain never falls through: the mode line
    # rendered as btrfs filesystem info on the exact platform CFGI runs
    # (measured on fedora, 2026-08-15). GNU `stat -c` fails cleanly on
    # macOS, so this order works on both.
    stat -c '%a' "$1" 2>/dev/null || stat -f '%Lp' "$1" 2>/dev/null || printf '?'
}

# --- .env ------------------------------------------------------------------

# Load .env, exporting each assignment so both python processes inherit an
# identical view.
#
# A variable already present in the environment WINS over the file. That
# precedence is deliberate: it is what lets `MEMORY_DB_SOCKET_PATH=/tmp/other.sock
# ./healthcheck.sh` mean what it says, and what lets a container runtime's
# environment take effect without an image rebuild. Plain `source` would do the
# opposite and silently overwrite the caller.
#
# Values are eval'd, so the usual shell expansions ($HOME, ${VAR:-default}) work
# inside .env. That also means .env is executable code; it is your own file, kept
# out of git, and readable only by you (setup.sh chmods it 600).
load_env() {
    [ -f "$ENV_FILE" ] || return 1
    local line name value
    while IFS= read -r line || [ -n "$line" ]; do
        case "$line" in
            ''|'#'*|' '*'#'*) continue ;;
        esac
        line="${line#export }"
        case "$line" in
            *=*) ;;
            *) continue ;;
        esac
        name="${line%%=*}"
        value="${line#*=}"
        case "$name" in
            ''|*[!A-Za-z0-9_]*) continue ;;
        esac
        # Already in the environment, including deliberately empty: leave it.
        if [ -n "${!name+x}" ]; then
            continue
        fi
        eval "export ${name}=${value}"
    done < "$ENV_FILE"
    return 0
}

# --- python discovery ------------------------------------------------------

# Minimum interpreter version. Override with EMM_PYTHON_MIN=3.12 etc.
EMM_PYTHON_MIN="${EMM_PYTHON_MIN:-3.11}"

_py_version_ok() {
    # $1 = interpreter path. Compares against EMM_PYTHON_MIN numerically so that
    # 3.10 does not sort above 3.9 as a string would.
    "$1" - "$EMM_PYTHON_MIN" <<'PY' >/dev/null 2>&1
import sys
want = tuple(int(p) for p in sys.argv[1].split(".")[:2])
sys.exit(0 if sys.version_info[:2] >= want else 1)
PY
}

# Echoes the first interpreter at or above EMM_PYTHON_MIN, or returns 1.
# EMM_PYTHON=/path/to/python forces a specific one.
find_python() {
    local candidate
    if [ -n "${EMM_PYTHON:-}" ]; then
        if [ -x "$EMM_PYTHON" ] && _py_version_ok "$EMM_PYTHON"; then
            printf '%s\n' "$EMM_PYTHON"; return 0
        fi
        fail "EMM_PYTHON=$EMM_PYTHON is not an executable python >= $EMM_PYTHON_MIN"
        return 1
    fi
    # Newest first. Bare python3 is last on purpose: on several macOS machines it
    # still resolves to the 3.9 that ships with the Command Line Tools.
    for candidate in python3.14 python3.13 python3.12 python3.11 python3 python; do
        local resolved
        resolved="$(command -v "$candidate" 2>/dev/null)" || continue
        if _py_version_ok "$resolved"; then
            printf '%s\n' "$resolved"; return 0
        fi
    done
    return 1
}

# The interpreter to actually run the two processes with: this checkout's venv
# when it exists, otherwise any interpreter new enough. The fallback is what lets
# the same launchers work inside the container image, which installs into the
# system python and has no venv.
python_for_run() {
    if [ -x "$VENV_PY" ]; then
        printf '%s\n' "$VENV_PY"; return 0
    fi
    find_python
}

# --- paths -----------------------------------------------------------------

# Resolve the database path the way memory_db_service.py and server.py do, using
# the same precedence, so callers can compare intent against what the running
# daemon reports. Requires an interpreter as $1.
resolve_db_path() {
    "$1" - <<'PY'
import os
from pathlib import Path

db_override = os.environ.get("ENHANCED_MEMORY_DB_PATH") or os.environ.get("MEMORY_DB_PATH")
dir_override = os.environ.get("ENHANCED_MEMORY_DIR") or os.environ.get("MEMORY_DIR")
if db_override:
    db = Path(os.path.expandvars(os.path.expanduser(db_override)))
else:
    base = (
        Path(os.path.expandvars(os.path.expanduser(dir_override)))
        if dir_override
        else Path.home() / ".claude" / "enhanced_memories"
    )
    db = base / "memory.db"
print(db)
PY
}

socket_path() {
    printf '%s\n' "${MEMORY_DB_SOCKET_PATH:-/tmp/memory-db.sock}"
}

# AF_UNIX caps sun_path at 104 bytes on macOS and 108 on Linux. Use the smaller
# of the two so a repository that works on a Mac also works on Linux, and leave a
# byte for the NUL terminator.
AF_UNIX_MAX=103

check_socket_path_length() {
    local path len
    path="$(socket_path)"
    len=${#path}
    if [ "$len" -gt "$AF_UNIX_MAX" ]; then
        fail "MEMORY_DB_SOCKET_PATH is ${len} bytes, over the AF_UNIX limit of ${AF_UNIX_MAX}:"
        fail "  $path"
        fail "bind() would fail with an OSError that does not mention the limit."
        fail "Use a short path such as /tmp/em-$(basename "$REPO_ROOT").sock"
        return 1
    fi
    case "$path" in
        "$REPO_ROOT"/*)
            warn "socket lives inside the checkout ($path)."
            warn "It is ${len}/${AF_UNIX_MAX} bytes today and grows with the checkout path."
            ;;
    esac
    return 0
}

# --- platform --------------------------------------------------------------

platform() {
    case "$(uname -s)" in
        Darwin) printf 'macos\n' ;;
        Linux)  printf 'linux\n' ;;
        *)      printf 'unsupported\n' ;;
    esac
}

# First available container engine, preferring podman per the delivery standard.
container_engine() {
    local engine
    for engine in podman docker container; do
        if command -v "$engine" >/dev/null 2>&1; then
            printf '%s\n' "$engine"; return 0
        fi
    done
    return 1
}

# --- process helpers -------------------------------------------------------

daemon_running() {
    local sock
    sock="$(socket_path)"
    [ -S "$sock" ] || return 1
    # A socket file surviving an unclean shutdown is indistinguishable from a
    # live one by stat(2), so connect for real.
    "${1:-$VENV_PY}" - "$sock" <<'PY' >/dev/null 2>&1
import socket, sys
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.settimeout(5)
try:
    s.connect(sys.argv[1])
finally:
    s.close()
PY
}

wait_for_socket() {
    # $1 socket path, $2 timeout seconds, $3 interpreter
    local sock="$1" timeout="${2:-30}" py="${3:-$VENV_PY}" waited=0
    while [ "$waited" -lt "$timeout" ]; do
        if [ -S "$sock" ] && MEMORY_DB_SOCKET_PATH="$sock" daemon_running "$py"; then
            return 0
        fi
        sleep 1
        waited=$((waited + 1))
    done
    return 1
}
