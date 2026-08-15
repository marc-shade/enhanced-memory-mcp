#!/usr/bin/env bash
# Enhanced Memory MCP: post-install gate.
#
# Run this after setup/setup.sh and after any configuration change. It is the
# only claim in this repository that "the install works", so it is built to be
# able to FAIL: every check reads back something it wrote, or compares two
# independently resolved values, rather than asking a component whether it feels
# well.
#
# The specific trap it exists to catch: when the memory-db daemon is down, the
# MCP server does not error. Its tools return well-formed objects containing
# zeros and an "error" field, for example {"count": 0, "results": [], "error":
# "Memory-DB service error: ..."}. A check that reads the count sees 0 and calls
# it an empty database. So check 2 writes a probe entity through the socket,
# searches it back, and treats an "error" or "daemon" key as failure no matter
# what else the payload contains.
#
# Exit codes: 0 all required checks passed, 1 at least one required check failed.
# Optional services (Qdrant, ollama) are reported and never fail the gate unless
# you pass --require-optional.
#
# Gaps / not covered: this gate does not test the SSE/HTTP transport (the
# container's compose healthcheck does), does not call individual MCP tools (it
# lists them), does not test concurrent multi-client access, and does not
# measure recall quality with or without the vector stack.
set -uo pipefail

_self="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=setup/lib/common.sh
. "${_self}/setup/lib/common.sh"

# --- arguments -------------------------------------------------------------

EXPECT_TOOLS="${EXPECTED_TOOL_COUNT:-}"
MIN_TOOLS="${MIN_TOOL_COUNT:-1}"
SKIP_MCP=0
REQUIRE_OPTIONAL=0

usage() {
    cat <<'USAGE'
usage: ./healthcheck.sh [options]

  --expect-tools N     require exactly N tools from tools/list
                       (default: $EXPECTED_TOOL_COUNT from .env, else unpinned)
  --min-tools N        minimum tool count when the exact count is unpinned (default 1)
  --skip-mcp           skip the MCP stdio handshake (fast daemon-only check)
  --require-optional   treat a missing Qdrant or ollama as a failure
  -h, --help           this text
USAGE
}

while [ $# -gt 0 ]; do
    case "$1" in
        --expect-tools) EXPECT_TOOLS="${2:-}"; shift 2 ;;
        --min-tools)    MIN_TOOLS="${2:-1}"; shift 2 ;;
        --skip-mcp)     SKIP_MCP=1; shift ;;
        --require-optional) REQUIRE_OPTIONAL=1; shift ;;
        -h|--help)      usage; exit 0 ;;
        *) fail "unknown argument: $1"; usage; exit 2 ;;
    esac
done

# --- tally -----------------------------------------------------------------

N_PASS=0; N_FAIL=0; N_WARN=0; N_SKIP=0

record() {
    # record STATUS CHECK-ID MESSAGE...
    local status="$1" check="$2"; shift 2
    case "$status" in
        PASS) N_PASS=$((N_PASS + 1)); printf '  %sPASS%s %-16s %s\n' "$C_GREEN" "$C_OFF" "$check" "$*" ;;
        FAIL) N_FAIL=$((N_FAIL + 1)); printf '  %sFAIL%s %-16s %s\n' "$C_RED" "$C_OFF" "$check" "$*" ;;
        WARN) N_WARN=$((N_WARN + 1)); printf '  %sWARN%s %-16s %s\n' "$C_YELLOW" "$C_OFF" "$check" "$*" ;;
        SKIP) N_SKIP=$((N_SKIP + 1)); printf '  %sSKIP%s %-16s %s\n' "$C_BLUE" "$C_OFF" "$check" "$*" ;;
    esac
}

# Run a probe that speaks the RESULT protocol and fold its lines into the tally.
run_probe() {
    local out rc line status check message
    # Not `mktemp -t emm-probe`: BSD mktemp treats -t's argument as a prefix,
    # GNU mktemp requires a template ending in at least three X's, and the GNU
    # form fails outright inside the container image. An explicit template is
    # the only spelling both accept.
    out="$(mktemp "${TMPDIR:-/tmp}/emm-probe.XXXXXX")"
    "$@" >"$out" 2>"${out}.err"
    rc=$?
    while IFS= read -r line; do
        case "$line" in
            "RESULT "*)
                status="$(printf '%s' "$line" | cut -d' ' -f2)"
                check="$(printf '%s' "$line" | cut -d' ' -f3)"
                message="$(printf '%s' "$line" | cut -d' ' -f4-)"
                record "$status" "$check" "$message"
                ;;
            *) [ -n "$line" ] && printf '       %s\n' "$line" ;;
        esac
    done < "$out"
    # A probe that dies before emitting anything must not pass silently.
    if [ "$rc" -ne 0 ] && ! grep -q '^RESULT FAIL' "$out"; then
        record FAIL "probe" "$(basename "$1") exited $rc without reporting a failure"
        sed 's/^/       /' "${out}.err" | tail -15
    elif [ -s "${out}.err" ] && [ "${VERBOSE:-0}" = "1" ]; then
        sed 's/^/       /' "${out}.err" | tail -30
    fi
    rm -f "$out" "${out}.err"
    return $rc
}

# GET a URL, echo the body, return nonzero if unreachable. Uses the venv python
# so the gate does not depend on curl being installed.
http_get() {
    "$PY" - "$1" <<'PY' 2>/dev/null
import sys, urllib.request
try:
    with urllib.request.urlopen(sys.argv[1], timeout=5) as response:
        sys.stdout.write(response.read().decode("utf-8", "replace")[:4000])
except Exception:
    sys.exit(1)
PY
}

# --- header ----------------------------------------------------------------

printf '%sEnhanced Memory MCP :: health check%s\n' "$C_BOLD" "$C_OFF"

if load_env; then
    ENV_STATE="$ENV_FILE"
else
    ENV_STATE="absent (using built-in defaults)"
fi

PY="$VENV_PY"
[ -x "$PY" ] || PY="$(find_python || true)"

printf '  repo    %s\n' "$REPO_ROOT"
printf '  env     %s\n' "$ENV_STATE"
if [ -n "$PY" ] && [ -x "$PY" ]; then
    printf '  python  %s (%s)\n' "$PY" "$("$PY" -c 'import platform;print(platform.python_version())' 2>/dev/null)"
    printf '  db      %s\n' "$(resolve_db_path "$PY")"
fi
printf '  socket  %s\n' "$(socket_path)"
printf '\n'

# --- 1. environment --------------------------------------------------------

printf '%s[1/4] environment%s\n' "$C_BOLD" "$C_OFF"

if [ -x "$VENV_PY" ]; then
    record PASS "venv" "$VENV_PY"
elif [ -n "$PY" ] && [ -x "$PY" ]; then
    # The container image installs into the system interpreter and has no venv.
    # What this gate actually requires is a usable interpreter with the
    # dependencies importable, not a directory named .venv, so demanding one
    # would fail a perfectly good install.
    record WARN "venv" "no venv at $VENV_DIR; using $PY instead (normal inside a container)"
else
    record FAIL "venv" "no interpreter at $VENV_PY and none on PATH. Run setup/setup.sh."
fi

if [ -n "$PY" ] && [ -x "$PY" ]; then
    if _py_version_ok "$PY"; then
        record PASS "python" "$("$PY" -c 'import sys;print(".".join(map(str,sys.version_info[:3])))') >= $EMM_PYTHON_MIN"
    else
        record FAIL "python" "$PY is older than $EMM_PYTHON_MIN"
    fi
else
    record FAIL "python" "no python >= $EMM_PYTHON_MIN found"
fi

if [ -f "$ENV_FILE" ]; then
    record PASS "env-file" "$ENV_FILE"
else
    record WARN "env-file" "no .env; defaults are in use. cp .env.example .env to pin them."
fi

_sock="$(socket_path)"
if check_socket_path_length >/dev/null 2>&1; then
    record PASS "socket-path" "${#_sock}/${AF_UNIX_MAX} bytes, within the AF_UNIX limit"
else
    record FAIL "socket-path" "$(socket_path) exceeds the AF_UNIX limit of $AF_UNIX_MAX bytes; bind() would fail with an unhelpful OSError"
fi

if [ -f "${REPO_ROOT}/memory_db_service.py" ] && [ -f "${REPO_ROOT}/server.py" ]; then
    record PASS "sources" "memory_db_service.py and server.py present"
else
    record FAIL "sources" "memory_db_service.py or server.py missing from $REPO_ROOT"
fi

# --- 2. daemon round trip --------------------------------------------------

printf '\n%s[2/4] memory-db daemon and database schema%s\n' "$C_BOLD" "$C_OFF"

if [ -n "$PY" ] && [ -x "$PY" ]; then
    run_probe "$PY" "${REPO_ROOT}/setup/lib/daemon_probe.py" --socket "$(socket_path)" || true
    # The round trip proves one table. This proves the rest: every literal INSERT
    # in the two files that own this database is checked against the live schema,
    # because a column the schema lacks fails every write at runtime while the
    # daemon reports it per row rather than raising.
    run_probe "$PY" "${REPO_ROOT}/setup/lib/schema_drift_probe.py" \
        --db "$(resolve_db_path "$PY")" --repo "$REPO_ROOT" || true
else
    record FAIL "roundtrip" "skipped: no usable interpreter"
fi

# --- 3. MCP stdio ----------------------------------------------------------

printf '\n%s[3/4] MCP server over stdio%s\n' "$C_BOLD" "$C_OFF"

if [ "$SKIP_MCP" = "1" ]; then
    record SKIP "mcp" "--skip-mcp"
elif [ -z "$PY" ] || [ ! -x "$PY" ]; then
    record FAIL "mcp" "skipped: no usable interpreter"
else
    if [ -n "$EXPECT_TOOLS" ]; then
        run_probe "$PY" "${REPO_ROOT}/setup/lib/mcp_stdio_probe.py" \
            --command "${REPO_ROOT}/setup/bin/mcp-server.sh" \
            --expect "$EXPECT_TOOLS" || true
    else
        run_probe "$PY" "${REPO_ROOT}/setup/lib/mcp_stdio_probe.py" \
            --command "${REPO_ROOT}/setup/bin/mcp-server.sh" \
            --min "$MIN_TOOLS" || true
    fi
fi

# --- 4. optional services --------------------------------------------------

printf '\n%s[4/4] optional services%s\n' "$C_BOLD" "$C_OFF"

optional_status() {
    # optional_status CHECK MESSAGE  -> WARN normally, FAIL under --require-optional
    if [ "$REQUIRE_OPTIONAL" = "1" ]; then
        record FAIL "$1" "$2"
    else
        record SKIP "$1" "$2"
    fi
}

QDRANT_URL="${MEMORY_QDRANT_URL:-http://localhost:6333}"
if [ -z "$PY" ] || [ ! -x "$PY" ]; then
    record SKIP "qdrant" "no interpreter to probe with"
elif http_get "${QDRANT_URL%/}/readyz" >/dev/null 2>&1 || http_get "${QDRANT_URL%/}/" >/dev/null 2>&1; then
    record PASS "qdrant" "reachable at $QDRANT_URL (OPTIONAL, enables semantic recall)"
else
    optional_status "qdrant" "unreachable at $QDRANT_URL (OPTIONAL). Recall degrades from semantic to lexical: keyword matches still work, meaning-based ones do not. Provision: setup/setup.sh --with-qdrant"
fi

OLLAMA_URL="${MEMORY_OLLAMA_URL:-http://127.0.0.1:11434}"
EMBED_MODEL="${MEMORY_EMBED_MODEL:-embeddinggemma}"
if [ -z "$PY" ] || [ ! -x "$PY" ]; then
    record SKIP "ollama" "no interpreter to probe with"
else
    tags="$(http_get "${OLLAMA_URL%/}/api/tags" 2>/dev/null)"
    if [ -n "$tags" ]; then
        case "$tags" in
            *"\"$EMBED_MODEL"*) record PASS "ollama" "reachable at $OLLAMA_URL with $EMBED_MODEL (OPTIONAL)" ;;
            *) record WARN "ollama" "reachable at $OLLAMA_URL but $EMBED_MODEL is not pulled (OPTIONAL). ollama pull $EMBED_MODEL" ;;
        esac
    else
        optional_status "ollama" "unreachable at $OLLAMA_URL (OPTIONAL). Without embeddings nothing gets indexed for vector search, so recall stays lexical even with Qdrant up. Provision: setup/setup.sh --with-ollama"
    fi
fi

# --- summary ---------------------------------------------------------------

printf '\n%sSummary%s  %d passed, %d failed, %d warnings, %d skipped\n' \
    "$C_BOLD" "$C_OFF" "$N_PASS" "$N_FAIL" "$N_WARN" "$N_SKIP"

if [ "$N_FAIL" -gt 0 ]; then
    printf '%sInstall is NOT healthy.%s See README.md "Troubleshooting".\n' "$C_RED" "$C_OFF"
    exit 1
fi

if [ -z "$EXPECT_TOOLS" ] && [ "$SKIP_MCP" != "1" ]; then
    printf 'Required checks passed. Note: the tool count was not pinned, so it was\n'
    printf 'checked as ">= %s" rather than exactly. Set EXPECTED_TOOL_COUNT in .env.\n' "$MIN_TOOLS"
else
    printf '%sRequired checks passed.%s\n' "$C_GREEN" "$C_OFF"
fi
exit 0
