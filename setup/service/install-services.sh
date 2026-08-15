#!/usr/bin/env bash
# Install the memory-db daemon (and optionally the SSE server) as a background
# service that starts at login.
#
#   macOS  launchd user agents in ~/Library/LaunchAgents
#   Linux  systemd user units in ~/.config/systemd/user
#
# No sudo, no system-wide units. Every path is rendered from this checkout's
# location and its .env, so two checkouts can each have their own service as long
# as they use different labels (--label-prefix) and different sockets.
#
# Installing the SSE service is optional and usually unnecessary: stdio clients
# such as Claude Code spawn their own server process per session. Install it only
# if you want one shared HTTP server.
set -euo pipefail

_self="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
. "${_self}/../lib/common.sh"

TEMPLATES="${_self}/templates"
WITH_SSE=0
LABEL_PREFIX="${LABEL_PREFIX:-com.enhanced-memory}"

usage() {
    cat <<'USAGE'
usage: setup/service/install-services.sh [options]

  --with-sse             also install the SSE/HTTP MCP server service
  --label-prefix NAME    service name prefix (default com.enhanced-memory).
                         Change it when running two checkouts side by side.
  -h, --help             this text

Uninstall with setup/service/uninstall-services.sh (same flags).
USAGE
}

while [ $# -gt 0 ]; do
    case "$1" in
        --with-sse)      WITH_SSE=1; shift ;;
        --label-prefix)  LABEL_PREFIX="${2:?--label-prefix needs a value}"; shift 2 ;;
        -h|--help)       usage; exit 0 ;;
        *) die "unknown argument: $1 (try --help)" ;;
    esac
done

PLATFORM="$(platform)"
load_env || warn "no .env found; the services will run on built-in defaults"

# Same rule as the launchers: a venv is the normal case, but any interpreter new
# enough will do. Demanding a directory named .venv would refuse installs that
# work fine, such as one against a system python with the dependencies present.
PY="$(python_for_run || true)"
[ -n "$PY" ] || die "no venv at $VENV_DIR and no python >= ${EMM_PYTHON_MIN} on PATH. Run setup/setup.sh first."

case "$PLATFORM" in
    macos) LOGDIR="${HOME}/Library/Logs/enhanced-memory" ;;
    linux) LOGDIR="${XDG_STATE_HOME:-${HOME}/.local/state}/enhanced-memory/log" ;;
    *)     die "unsupported platform: $(uname -s)" ;;
esac
mkdir -p "$LOGDIR"

render() {
    # render TEMPLATE LABEL EXEC DESC > OUTPUT
    sed -e "s|@LABEL@|$2|g" \
        -e "s|@EXEC@|$3|g" \
        -e "s|@DESC@|$4|g" \
        -e "s|@REPO@|${REPO_ROOT}|g" \
        -e "s|@LOGDIR@|${LOGDIR}|g" \
        "$1"
}

install_macos() {
    # install_macos LABEL EXEC DESC
    local label="$1" exec_path="$2" desc="$3"
    local plist="${HOME}/Library/LaunchAgents/${label}.plist"
    mkdir -p "${HOME}/Library/LaunchAgents"
    render "${TEMPLATES}/launchd.plist.template" "$label" "$exec_path" "$desc" > "$plist"
    info "wrote $plist"

    # bootout first so a re-run replaces the running job rather than failing with
    # "service already loaded". Absent job is not an error here.
    launchctl bootout "gui/$(id -u)/${label}" >/dev/null 2>&1 || true
    if ! launchctl bootstrap "gui/$(id -u)" "$plist" 2>/tmp/emm-launchctl.$$; then
        fail "launchctl bootstrap failed:"
        sed 's/^/  /' "/tmp/emm-launchctl.$$" >&2 || true
        rm -f "/tmp/emm-launchctl.$$"
        return 1
    fi
    rm -f "/tmp/emm-launchctl.$$"
    launchctl kickstart -k "gui/$(id -u)/${label}" >/dev/null 2>&1 || true
    ok "loaded $label"
}

install_linux() {
    # install_linux NAME EXEC DESC
    local name="$1" exec_path="$2" desc="$3"
    local unit_dir="${HOME}/.config/systemd/user"
    local unit="${unit_dir}/${name}.service"
    mkdir -p "$unit_dir"
    render "${TEMPLATES}/systemd.service.template" "$name" "$exec_path" "$desc" > "$unit"
    info "wrote $unit"
    systemctl --user daemon-reload
    systemctl --user enable --now "${name}.service"
    ok "enabled ${name}.service"
}

DB_LABEL="${LABEL_PREFIX}.db"
SSE_LABEL="${LABEL_PREFIX}.sse"

case "$PLATFORM" in
    macos)
        install_macos "$DB_LABEL" "${REPO_ROOT}/setup/bin/memory-db-daemon.sh" \
            "Enhanced Memory database daemon"
        if [ "$WITH_SSE" = "1" ]; then
            install_macos "$SSE_LABEL" \
                "${REPO_ROOT}/setup/bin/mcp-server-sse.sh" \
                "Enhanced Memory MCP server (SSE)"
        fi
        ;;
    linux)
        # systemd unit names cannot contain the dots a launchd label uses.
        DB_LABEL="$(printf '%s' "$DB_LABEL" | tr '.' '-')"
        SSE_LABEL="$(printf '%s' "$SSE_LABEL" | tr '.' '-')"
        install_linux "$DB_LABEL" "${REPO_ROOT}/setup/bin/memory-db-daemon.sh" \
            "Enhanced Memory database daemon"
        if [ "$WITH_SSE" = "1" ]; then
            install_linux "$SSE_LABEL" \
                "${REPO_ROOT}/setup/bin/mcp-server-sse.sh" \
                "Enhanced Memory MCP server (SSE)"
        fi
        if ! loginctl show-user "$USER" 2>/dev/null | grep -q 'Linger=yes'; then
            warn "lingering is off, so these units stop when you log out."
            warn "Enable it with: loginctl enable-linger $USER"
        fi
        ;;
esac

# --- did it actually start? ------------------------------------------------

printf '\n'
info "Waiting for the daemon socket at $(socket_path)"
if wait_for_socket "$(socket_path)" 30 "$PY"; then
    ok "daemon is listening"
    printf '\nLogs: %s\n' "$LOGDIR"
    printf 'Verify the whole install with: %s/healthcheck.sh\n' "$REPO_ROOT"
else
    fail "the socket did not appear within 30s. The service is installed but not working."
    fail "Look at ${LOGDIR}/${DB_LABEL}.err.log"
    if [ -f "${LOGDIR}/${DB_LABEL}.err.log" ]; then
        printf '\n--- last 20 lines ---\n'
        tail -20 "${LOGDIR}/${DB_LABEL}.err.log"
    fi
    exit 1
fi
