#!/usr/bin/env bash
# Enhanced Memory MCP: from-scratch installer for macOS and Linux.
#
# Idempotent by design. Run it as often as you like: an existing venv is reused,
# an existing .env is never overwritten, and an already-provisioned Qdrant
# container is started rather than recreated.
#
# It never uses sudo. Everything lands in the checkout (.venv), your home
# directory (the database), or a user-scoped container.
#
# What it does NOT do: it does not start the daemon as a background service (see
# setup/service/install-services.sh), does not modify ~/.claude.json (the README
# gives you the snippet to paste), and does not verify the install (that is
# ./healthcheck.sh, which you should run next).
set -euo pipefail

_self="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
. "${_self}/lib/common.sh"

WITH_QDRANT=0
WITH_OLLAMA=0
WITH_OPTIONAL=0
FOREGROUND=0

usage() {
    cat <<'USAGE'
usage: setup/setup.sh [options]

  --with-optional   also install requirements-optional.txt (heavier extras:
                    re-ranking models, provider SDKs)
  --with-qdrant     provision a local Qdrant container (podman or docker,
                    whichever is installed) on 127.0.0.1:6333 with a named
                    volume. Enables semantic recall.
  --with-ollama     verify ollama and pull the embedding model. Required for
                    semantic recall to actually index anything.
  --foreground      after installing, run the daemon and an SSE server in the
                    foreground so you can try it immediately. Ctrl-C stops both.
  --python PATH     use this interpreter instead of searching
  -h, --help        this text

Environment:
  EMM_PYTHON_MIN    minimum python version (default 3.11)
USAGE
}

while [ $# -gt 0 ]; do
    case "$1" in
        --with-optional) WITH_OPTIONAL=1; shift ;;
        --with-qdrant)   WITH_QDRANT=1; shift ;;
        --with-ollama)   WITH_OLLAMA=1; shift ;;
        --foreground)    FOREGROUND=1; shift ;;
        --python)        EMM_PYTHON="${2:?--python needs a path}"; export EMM_PYTHON; shift 2 ;;
        -h|--help)       usage; exit 0 ;;
        *) die "unknown argument: $1 (try --help)" ;;
    esac
done

PLATFORM="$(platform)"
[ "$PLATFORM" = "unsupported" ] && die "unsupported platform: $(uname -s). macOS and Linux only."

printf '%sEnhanced Memory MCP :: setup%s\n' "$C_BOLD" "$C_OFF"
printf '  checkout %s\n' "$REPO_ROOT"
printf '  platform %s\n\n' "$PLATFORM"

# --- 1. interpreter --------------------------------------------------------

info "Locating python >= ${EMM_PYTHON_MIN}"
PYTHON="$(find_python || true)"
if [ -z "$PYTHON" ]; then
    fail "no python >= ${EMM_PYTHON_MIN} on PATH."
    case "$PLATFORM" in
        macos) fail "install one with: brew install python@3.11" ;;
        linux) fail "install one with your package manager, e.g. dnf install python3.11" ;;
    esac
    fail "Note that a bare 'python3' is 3.9 on some macOS machines, which is why"
    fail "this script looks for versioned names first."
    exit 1
fi
log "  $PYTHON ($("$PYTHON" -c 'import platform; print(platform.python_version())'))"

# --- 2. virtualenv ---------------------------------------------------------

if [ -x "$VENV_PY" ]; then
    info "Reusing existing venv at ${VENV_DIR}"
else
    info "Creating venv at ${VENV_DIR}"
    "$PYTHON" -m venv "$VENV_DIR"
fi
[ -x "$VENV_PY" ] || die "venv creation did not produce ${VENV_PY}"

# --- 3. dependencies -------------------------------------------------------

pip_install() {
    # uv when available (much faster, same wheels), pip otherwise.
    if command -v uv >/dev/null 2>&1; then
        uv pip install --python "$VENV_PY" "$@"
    else
        "$VENV_PY" -m pip install "$@"
    fi
}

info "Upgrading pip tooling"
"$VENV_PY" -m pip install --quiet --upgrade pip setuptools wheel

REQ="${REPO_ROOT}/requirements.txt"
[ -f "$REQ" ] || die "requirements.txt not found at $REQ"
info "Installing core dependencies from requirements.txt"
pip_install -r "$REQ"

REQ_OPT="${REPO_ROOT}/requirements-optional.txt"
if [ "$WITH_OPTIONAL" = "1" ]; then
    # Hard failure, not a warning. A flag whose entire purpose is "also install
    # the optional set" that quietly installs nothing when the file is missing is
    # a switch that does not switch: the run exits 0, the extras are absent, and
    # the first symptom is a tool silently not registering later.
    [ -f "$REQ_OPT" ] || die "--with-optional needs $REQ_OPT, which does not exist"
    info "Installing optional dependencies from requirements-optional.txt"
    pip_install -r "$REQ_OPT"
elif [ -f "$REQ_OPT" ]; then
    log "  (optional extras available: re-run with --with-optional)"
fi

# --- 4. configuration ------------------------------------------------------

if [ -f "$ENV_FILE" ]; then
    info "Keeping existing .env (not overwritten)"
else
    [ -f "$ENV_EXAMPLE" ] || die ".env.example missing; cannot generate .env"
    info "Generating .env from .env.example"
    cp "$ENV_EXAMPLE" "$ENV_FILE"
    chmod 600 "$ENV_FILE"
fi

load_env || true

# --- 5. storage ------------------------------------------------------------

DB_PATH="$(resolve_db_path "$VENV_PY")"
DB_DIR="$(dirname "$DB_PATH")"
info "Preparing database directory ${DB_DIR}"
mkdir -p "$DB_DIR"
chmod 700 "$DB_DIR" 2>/dev/null || warn "could not chmod 700 $DB_DIR"
if [ -f "$DB_PATH" ]; then
    log "  existing database kept: $DB_PATH"
    chmod 600 "$DB_PATH" 2>/dev/null || true
else
    log "  new database will be created on first daemon start: $DB_PATH"
fi

info "Checking the daemon socket path"
check_socket_path_length || die "fix MEMORY_DB_SOCKET_PATH in .env, then re-run"
log "  $(socket_path) (${#DB_PATH} byte db path, socket within the AF_UNIX limit)"

# --- 6. optional: Qdrant ---------------------------------------------------

if [ "$WITH_QDRANT" = "1" ]; then
    info "Provisioning Qdrant"
    ENGINE="$(container_engine || true)"
    if [ -z "$ENGINE" ]; then
        warn "no podman or docker found; skipping Qdrant."
        warn "Install podman (recommended) or docker, then re-run with --with-qdrant."
    else
        QDRANT_NAME="${QDRANT_CONTAINER_NAME:-enhanced-memory-qdrant}"
        QDRANT_VOLUME="${QDRANT_VOLUME_NAME:-enhanced-memory-qdrant-data}"
        if "$ENGINE" ps --format '{{.Names}}' 2>/dev/null | grep -qx "$QDRANT_NAME"; then
            log "  $QDRANT_NAME already running"
        elif "$ENGINE" ps -a --format '{{.Names}}' 2>/dev/null | grep -qx "$QDRANT_NAME"; then
            log "  starting existing container $QDRANT_NAME"
            "$ENGINE" start "$QDRANT_NAME" >/dev/null
        else
            log "  creating $QDRANT_NAME with volume $QDRANT_VOLUME on 127.0.0.1:6333"
            "$ENGINE" volume create "$QDRANT_VOLUME" >/dev/null 2>&1 || true
            # Published on the loopback interface only: an unauthenticated vector
            # store must not be reachable from the network.
            "$ENGINE" run -d \
                --name "$QDRANT_NAME" \
                -p 127.0.0.1:6333:6333 \
                -p 127.0.0.1:6334:6334 \
                -v "${QDRANT_VOLUME}:/qdrant/storage" \
                docker.io/qdrant/qdrant:latest >/dev/null
        fi
    fi
fi

# --- 7. optional: ollama ---------------------------------------------------

if [ "$WITH_OLLAMA" = "1" ]; then
    info "Checking ollama"
    if ! command -v ollama >/dev/null 2>&1; then
        warn "ollama is not installed. Install it from https://ollama.com/download"
        warn "(macOS: brew install ollama). Semantic recall stays lexical without it."
    elif ! ollama list >/dev/null 2>&1; then
        warn "ollama is installed but its server is not answering."
        warn "Start it with: ollama serve"
    else
        MODEL="${MEMORY_EMBED_MODEL:-embeddinggemma}"
        if ollama list 2>/dev/null | awk 'NR>1 {print $1}' | grep -q "^${MODEL}"; then
            log "  $MODEL already pulled"
        else
            log "  pulling $MODEL (this downloads a few hundred MB)"
            ollama pull "$MODEL"
        fi
    fi
fi

# --- 8. what to do next ----------------------------------------------------

printf '\n%sSetup complete.%s Nothing has been verified yet. Next:\n\n' "$C_GREEN" "$C_OFF"
cat <<NEXT
  1. Start the memory-db daemon (REQUIRED: without it every tool returns
     zeros with an "error" field instead of data):

       ${REPO_ROOT}/setup/bin/memory-db-daemon.sh

     ... or install it as a background service that starts at login:

       ${REPO_ROOT}/setup/service/install-services.sh

  2. Verify the install:

       ${REPO_ROOT}/healthcheck.sh

  3. Register the server with your MCP client by adding this to ~/.claude.json:

       "enhanced-memory": {
         "command": "${REPO_ROOT}/setup/bin/mcp-server.sh"
       }

     Use the launcher rather than python server.py directly: it applies the same
     .env the daemon uses, which is what keeps both processes on one database.

NEXT

# --- 9. optional: foreground trial ----------------------------------------

if [ "$FOREGROUND" = "1" ]; then
    info "Starting daemon and SSE server in the foreground (Ctrl-C stops both)"
    "${REPO_ROOT}/setup/bin/memory-db-daemon.sh" &
    DAEMON_PID=$!
    # shellcheck disable=SC2064  # expand PID now, on purpose
    trap "kill $DAEMON_PID 2>/dev/null || true" EXIT INT TERM

    if ! wait_for_socket "$(socket_path)" 30 "$VENV_PY"; then
        kill "$DAEMON_PID" 2>/dev/null || true
        die "daemon did not open $(socket_path) within 30s. Run it directly to see why."
    fi
    ok "daemon listening on $(socket_path)"

    log ""
    log "MCP server on http://${MCP_HOST:-127.0.0.1}:${MCP_PORT:-9106} (SSE transport)"
    log "In another terminal, verify with: ${REPO_ROOT}/healthcheck.sh"
    log ""
    MCP_TRANSPORT=sse "${REPO_ROOT}/setup/bin/mcp-server.sh"
fi
