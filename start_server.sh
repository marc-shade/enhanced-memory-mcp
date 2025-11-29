#!/bin/bash
# Enhanced Memory MCP Server Startup Script
# Runs migrations and starts the server

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${AGENTIC_SYSTEM_PATH:-${AGENTIC_SYSTEM_PATH:-/opt/agentic}}/.venv"

# Activate venv
source "$VENV_PATH/bin/activate"

cd "$SCRIPT_DIR"

# Run migrations first (will handle locked database gracefully)
echo "Running database migrations..." >&2
python3 migrate_database.py 2>&1 || {
    echo "Migration failed or database locked - will retry on startup" >&2
}

# Start the server
echo "Starting enhanced-memory MCP server..." >&2
exec python3 server.py
