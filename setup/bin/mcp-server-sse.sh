#!/usr/bin/env bash
# Launch the MCP server on the SSE/HTTP transport instead of stdio.
#
# For the container image and for the optional background service. Desktop MCP
# clients (Claude Code, Claude Desktop) want stdio and should use mcp-server.sh.
#
# MCP_HOST stays whatever .env says, which defaults to 127.0.0.1. This server has
# no authentication: binding it to 0.0.0.0 on a workstation publishes your entire
# memory store to the local network. The container image overrides the host
# because a container's 0.0.0.0 is scoped to its own namespace, and compose.yaml
# republishes it on the host loopback only.
set -euo pipefail

_self="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MCP_TRANSPORT="${MCP_TRANSPORT_OVERRIDE:-sse}"
exec "${_self}/mcp-server.sh" "$@"
