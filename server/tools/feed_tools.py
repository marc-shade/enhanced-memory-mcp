"""
Phoenix Feed MCP Tools (B1)
=============================

5 tools for the internal agent social network:
- feed_post: Create a new post
- feed_list: List posts by category
- feed_reply: Reply to a post
- feed_upvote: Upvote a post
- feed_search: Search posts by content
"""

import os
import sys
import logging
from typing import Any, Dict

logger = logging.getLogger("feed_tools")

# Add security module to path
sys.path.insert(0, os.environ.get("AGENTIC_SYSTEM_PATH", "/Volumes/SSDRAID0/agentic-system"))


def _get_feed():
    """Get or create PhoenixFeed instance."""
    try:
        from security.phoenix_feed import PhoenixFeed

        node_id = os.environ.get("NODE_ID", "orchestrator")
        return PhoenixFeed(node_id=node_id)
    except ImportError:
        logger.warning("Phoenix Feed not available")
        return None


def register_feed_tools(app):
    """Register Phoenix Feed tools with FastMCP app."""

    @app.tool()
    async def feed_post(category: str, content: str) -> Dict[str, Any]:
        """
        Post to the Phoenix Feed — the cluster's internal social network.

        Share learnings, discoveries, anomalies, or start discussions.

        Categories:
        - m/learnings: Patterns, insights, knowledge gained
        - m/anomalies: Unusual behavior, security events
        - m/discoveries: New capabilities, tools, approaches
        - m/discussions: Open questions, proposals, debates

        Args:
            category: Feed category (e.g., "m/learnings")
            content: Post content text
        """
        feed = _get_feed()
        if not feed:
            return {"error": "Phoenix Feed not available", "success": False}
        return feed.create_post(category, content)

    @app.tool()
    async def feed_list(category: str = "", limit: int = 20) -> Dict[str, Any]:
        """
        List recent posts from the Phoenix Feed.

        Args:
            category: Filter by category (empty for all posts)
            limit: Maximum number of posts to return (default: 20)
        """
        feed = _get_feed()
        if not feed:
            return {"error": "Phoenix Feed not available", "success": False}
        return feed.list_posts(category=category or None, limit=limit)

    @app.tool()
    async def feed_reply(parent_id: str, content: str) -> Dict[str, Any]:
        """
        Reply to a post on the Phoenix Feed.

        Args:
            parent_id: ID of the post to reply to
            content: Reply content text
        """
        feed = _get_feed()
        if not feed:
            return {"error": "Phoenix Feed not available", "success": False}
        return feed.reply(parent_id, content)

    @app.tool()
    async def feed_upvote(post_id: str) -> Dict[str, Any]:
        """
        Upvote a post on the Phoenix Feed. Each node can upvote once per post.

        Args:
            post_id: ID of the post to upvote
        """
        feed = _get_feed()
        if not feed:
            return {"error": "Phoenix Feed not available", "success": False}
        return feed.upvote(post_id)

    @app.tool()
    async def feed_search(query: str, limit: int = 20) -> Dict[str, Any]:
        """
        Search posts on the Phoenix Feed by content.

        Args:
            query: Search query string
            limit: Maximum results to return (default: 20)
        """
        feed = _get_feed()
        if not feed:
            return {"error": "Phoenix Feed not available", "success": False}
        return feed.search(query, limit)

    return {
        "feed_post": feed_post,
        "feed_list": feed_list,
        "feed_reply": feed_reply,
        "feed_upvote": feed_upvote,
        "feed_search": feed_search,
    }
