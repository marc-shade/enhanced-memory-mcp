"""
MCP Tools package for Enhanced Memory Server.

Provides tool registration functions for FastMCP.
"""

from .core import register_core_tools
from .git_ops import register_git_tools
from .planning import register_planning_tools
from .execution import register_execution_tools
from .compaction_tools import register_compaction_tools
from .integrity_tools import register_integrity_tools
from .prompt_injection_tools import register_prompt_injection_tools
from .feed_tools import register_feed_tools


def register_all_tools(app, memory_client):
    """
    Register all server tools with FastMCP app.

    Args:
        app: FastMCP application instance
        memory_client: MemoryClient instance for database operations
    """
    register_core_tools(app, memory_client)
    register_git_tools(app)
    register_planning_tools(app, memory_client)
    register_execution_tools(app)
    register_compaction_tools(app)
    register_integrity_tools(app)
    register_prompt_injection_tools(app)
    register_feed_tools(app)


__all__ = [
    'register_all_tools',
    'register_core_tools',
    'register_git_tools',
    'register_planning_tools',
    'register_execution_tools',
    'register_compaction_tools',
    'register_integrity_tools',
    'register_prompt_injection_tools',
    'register_feed_tools',
]
