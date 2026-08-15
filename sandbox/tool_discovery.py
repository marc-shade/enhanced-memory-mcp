"""
Progressive Tool Discovery System

Implements Anthropic's file tree pattern for on-demand tool loading.
Instead of loading all tool definitions upfront, agents explore a filesystem.

Token Savings: ~90% reduction (2000 -> 200 tokens for tool discovery)

Architecture:
    servers/
    ├── enhanced-memory/
    │   ├── search_nodes.json      (schema)
    │   ├── create_entities.json
    │   └── index.json             (list of tools)
    ├── voice-mode/
    │   ├── converse.json
    │   └── index.json
    └── agent-runtime/
        ├── create_goal.json
        └── index.json
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Tool registry directory (created on first use)
TOOL_REGISTRY_DIR = Path(__file__).parent.parent / "tool_registry"


@dataclass
class ToolSchema:
    """Lightweight tool schema for progressive disclosure"""

    name: str
    server: str
    description: str
    parameters: Dict[str, Any]
    returns: Optional[str] = None
    examples: Optional[List[str]] = None
    cost_estimate: Optional[str] = None  # e.g., "low", "medium", "high"


# Declared MCP servers and their tools.
#
# EMPTY BY DEFAULT, ON PURPOSE. This used to ship a fixed in-source dict naming
# six servers -- voice-mode, arduino-surface, agent-runtime, safla-enhanced,
# sequential-thinking and enhanced-memory -- which init_tool_registry() wrote to
# disk and list_servers() then reported as available. On a standalone install
# five of those do not exist, so agent code inside execute_code was told it
# could call tools that were not there. This module cannot know what a given
# installation runs, so its honest default is to claim nothing.
#
# To declare the servers you actually run, point MEMORY_TOOL_REGISTRY_FILE at a
# JSON file of the same shape:
#
#   {"my-server": [{"name": "do_thing",
#                   "description": "...",
#                   "parameters": {"x": "str"},
#                   "returns": "...",
#                   "cost_estimate": "low"}]}
#
# Anything already present as a directory under tool_registry/ is also reported
# by list_servers(), so an installation can declare servers by writing that tree
# directly instead.
MCP_TOOL_REGISTRY: Dict[str, List[Dict[str, Any]]] = {}


def _load_declared_registry() -> Dict[str, List[Dict[str, Any]]]:
    """Read the operator-declared registry, if one is configured.

    A malformed or unreadable file is reported and treated as "nothing
    declared". It is never silently swallowed: a registry that quietly empties
    itself is the failure this module was changed to stop.
    """
    import os

    path = os.environ.get("MEMORY_TOOL_REGISTRY_FILE")
    if not path:
        return {}
    try:
        declared = json.loads(Path(path).expanduser().read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "MEMORY_TOOL_REGISTRY_FILE=%s could not be read (%s: %s); "
            "no servers declared",
            path,
            type(exc).__name__,
            exc,
        )
        return {}
    if not isinstance(declared, dict):
        logger.warning(
            "MEMORY_TOOL_REGISTRY_FILE=%s must contain a JSON object mapping "
            "server name -> list of tools; got %s",
            path,
            type(declared).__name__,
        )
        return {}
    return declared


def init_tool_registry():
    """Write the declared registry to disk.

    With nothing declared this creates an empty directory and reports no
    servers, which is the correct answer when the installation has not said
    what it runs. It does NOT invent entries.
    """
    TOOL_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)

    declared = {**MCP_TOOL_REGISTRY, **_load_declared_registry()}
    if not declared:
        logger.info(
            "Tool registry at %s: no servers declared "
            "(set MEMORY_TOOL_REGISTRY_FILE to declare them)",
            TOOL_REGISTRY_DIR,
        )
        return

    for server_name, tools in declared.items():
        server_dir = TOOL_REGISTRY_DIR / server_name
        server_dir.mkdir(exist_ok=True)

        # Create index.json with tool list
        index = {
            "server": server_name,
            "tool_count": len(tools),
            "tools": [t["name"] for t in tools],
        }
        (server_dir / "index.json").write_text(json.dumps(index, indent=2))

        # Create individual tool schema files
        for tool in tools:
            tool_file = server_dir / f"{tool['name']}.json"
            tool_file.write_text(json.dumps(tool, indent=2))

    logger.info(f"Tool registry initialized at {TOOL_REGISTRY_DIR}")


def list_servers() -> List[str]:
    """List available MCP servers (directory names)"""
    if not TOOL_REGISTRY_DIR.exists():
        init_tool_registry()
    return [d.name for d in TOOL_REGISTRY_DIR.iterdir() if d.is_dir()]


def list_tools(server: str) -> List[str]:
    """List tools for a server (names only - minimal tokens)"""
    server_dir = TOOL_REGISTRY_DIR / server
    if not server_dir.exists():
        return []

    index_file = server_dir / "index.json"
    if index_file.exists():
        index = json.loads(index_file.read_text())
        return index.get("tools", [])
    return []


def get_tool_schema(server: str, tool: str) -> Optional[Dict[str, Any]]:
    """Get full schema for a specific tool (on-demand loading)"""
    tool_file = TOOL_REGISTRY_DIR / server / f"{tool}.json"
    if tool_file.exists():
        return json.loads(tool_file.read_text())
    return None


def search_tools(query: str, detail_level: str = "names") -> Dict[str, Any]:
    """
    Search for tools matching query across all servers.

    Progressive disclosure levels:
    - "names": Just tool names (minimal tokens)
    - "brief": Names + one-line descriptions
    - "full": Complete schemas (use sparingly)

    Args:
        query: Search term (matches name or description)
        detail_level: How much detail to return

    Returns:
        Matching tools with requested detail level
    """
    if not TOOL_REGISTRY_DIR.exists():
        init_tool_registry()

    query_lower = query.lower()
    matches = []

    for server_name, tools in MCP_TOOL_REGISTRY.items():
        for tool in tools:
            name_match = query_lower in tool["name"].lower()
            desc_match = query_lower in tool["description"].lower()

            if name_match or desc_match:
                if detail_level == "names":
                    matches.append(f"{server_name}/{tool['name']}")
                elif detail_level == "brief":
                    matches.append(
                        {
                            "path": f"{server_name}/{tool['name']}",
                            "description": tool["description"][:80],
                        }
                    )
                else:  # full
                    matches.append({"server": server_name, **tool})

    return {
        "query": query,
        "detail_level": detail_level,
        "count": len(matches),
        "matches": matches,
    }


def get_tool_cost_estimate(server: str, tool: str) -> str:
    """Get estimated cost/complexity of a tool call"""
    schema = get_tool_schema(server, tool)
    if schema:
        return schema.get("cost_estimate", "unknown")
    return "unknown"


# API functions for code execution context
def search_tools_api(query: str, detail: str = "brief") -> Dict[str, Any]:
    """Search tools API for code execution context"""
    return search_tools(query, detail)


def list_servers_api() -> List[str]:
    """List servers API for code execution context"""
    return list_servers()


def list_tools_api(server: str) -> List[str]:
    """List tools API for code execution context"""
    return list_tools(server)


def get_schema_api(server: str, tool: str) -> Optional[Dict]:
    """Get schema API for code execution context"""
    return get_tool_schema(server, tool)


# Initialize on import
if not TOOL_REGISTRY_DIR.exists():
    try:
        init_tool_registry()
    except Exception as e:
        logger.warning(f"Failed to initialize tool registry: {e}")
