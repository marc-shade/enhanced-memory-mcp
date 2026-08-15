#!/usr/bin/env bash
# Remove the services installed by setup/service/install-services.sh.
#
# Leaves your database, your .env and the venv alone. Removing the services stops
# the daemon, which means MCP tools will start returning zeros with an "error"
# field until you start it again by hand, so do this on purpose.
set -euo pipefail

_self="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
. "${_self}/../lib/common.sh"

LABEL_PREFIX="${LABEL_PREFIX:-com.enhanced-memory}"
KEEP_LOGS=1

usage() {
    cat <<'USAGE'
usage: setup/service/uninstall-services.sh [options]

  --label-prefix NAME   service name prefix used at install time
                        (default com.enhanced-memory)
  --remove-logs         also delete the service log directory
  -h, --help            this text
USAGE
}

while [ $# -gt 0 ]; do
    case "$1" in
        --label-prefix) LABEL_PREFIX="${2:?--label-prefix needs a value}"; shift 2 ;;
        --remove-logs)  KEEP_LOGS=0; shift ;;
        -h|--help)      usage; exit 0 ;;
        *) die "unknown argument: $1 (try --help)" ;;
    esac
done

# Needed for the closing socket note. Without it this script falls back to the
# built-in default and reports on /tmp/memory-db.sock no matter which socket the
# install being removed actually used, which on a machine running two checkouts
# points the reader straight at the wrong one.
load_env || true

PLATFORM="$(platform)"
removed=0

case "$PLATFORM" in
    macos)
        LOGDIR="${HOME}/Library/Logs/enhanced-memory"
        for label in "${LABEL_PREFIX}.db" "${LABEL_PREFIX}.sse"; do
            plist="${HOME}/Library/LaunchAgents/${label}.plist"
            if launchctl print "gui/$(id -u)/${label}" >/dev/null 2>&1; then
                launchctl bootout "gui/$(id -u)/${label}" >/dev/null 2>&1 || true
                ok "unloaded $label"
                removed=$((removed + 1))
            fi
            if [ -f "$plist" ]; then
                rm -f "$plist"
                ok "removed $plist"
                removed=$((removed + 1))
            fi
        done
        ;;
    linux)
        LOGDIR="${XDG_STATE_HOME:-${HOME}/.local/state}/enhanced-memory/log"
        prefix="$(printf '%s' "$LABEL_PREFIX" | tr '.' '-')"
        for name in "${prefix}-db" "${prefix}-sse"; do
            unit="${HOME}/.config/systemd/user/${name}.service"
            if systemctl --user list-unit-files "${name}.service" >/dev/null 2>&1 &&
               [ -f "$unit" ]; then
                systemctl --user disable --now "${name}.service" >/dev/null 2>&1 || true
                rm -f "$unit"
                ok "removed ${name}.service"
                removed=$((removed + 1))
            fi
        done
        systemctl --user daemon-reload
        ;;
    *) die "unsupported platform: $(uname -s)" ;;
esac

if [ "$removed" -eq 0 ]; then
    warn "nothing to remove for prefix '${LABEL_PREFIX}'."
    warn "If you installed with a different --label-prefix, pass the same one here."
fi

if [ "$KEEP_LOGS" = "0" ] && [ -d "$LOGDIR" ]; then
    rm -rf "$LOGDIR"
    ok "removed $LOGDIR"
elif [ -d "$LOGDIR" ]; then
    log "logs kept at $LOGDIR (--remove-logs to delete)"
fi

sock="$(socket_path)"
if [ -S "$sock" ]; then
    warn "$sock still exists. If no daemon is running it is a stale file; the next"
    warn "daemon start removes it."
fi
