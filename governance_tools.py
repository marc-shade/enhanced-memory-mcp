#!/usr/bin/env python3
"""
Governance MCP Tools: per-agent ACL on memory entities.

Wraps the visibility sidecar (entity_visibility + entity_acl) as MCP tools so
agents can set/get visibility, set an owner, and grant/revoke per-agent read
access. Retrieval scoping itself is enforced in the socket service's search path
(viewer_agent param): a scoped viewer sees only PUBLIC/CLUSTER entities plus
PRIVATE ones they own or are granted. Untagged entities default to PRIVATE
(opt-in sharing).

Storage lives in the same memory.db as the entities, via visibility.py in this
repository. That module used to be reached by walking out of the tree into a
separate checkout, which meant this whole group failed to register anywhere
else -- and said so only in a log file under /tmp.
"""

import logging
import sqlite3
from typing import Dict, Any, Optional

logger = logging.getLogger("governance_tools")

# visibility.py ships in this repository. It used to be reached by walking up
# out of the tree into a separate private checkout, which meant this whole tool
# group failed to register on any machine that did not have that checkout --
# silently, because the only record was a line in a log file under /tmp.
from visibility import (
    VisibilityTag,
    can_view,
    ensure_sidecar_tables,
    get_acl,
    get_visibility_row,
    grant_access,
    revoke_access,
    set_visibility,
)

VALID_TAGS = {"private", "cluster", "public"}


def _resolve_entity_id(
    db_path: str, entity_id: Optional[int], name: Optional[str]
) -> int:
    if entity_id is not None:
        return int(entity_id)
    if not name:
        raise ValueError("provide either entity_id or name")
    conn = sqlite3.connect(db_path, timeout=10)
    try:
        row = conn.execute("SELECT id FROM entities WHERE name = ?", (name,)).fetchone()
    finally:
        conn.close()
    if row is None:
        raise ValueError(f"no entity named {name!r}")
    return row[0]


def _sanitize_agent(agent_id: str) -> str:
    """Agent ids are used as identifiers, not paths; allow a safe charset."""
    import re

    if not re.match(r"^[A-Za-z0-9._-]{1,64}$", agent_id):
        raise ValueError(
            f"agent_id must match [A-Za-z0-9._-]{{1,64}}, got {agent_id!r}"
        )
    return agent_id


def register_governance_tools(app, db_path):
    """Register governance MCP tools."""

    @app.tool()
    async def set_entity_visibility(
        entity_id: Optional[int] = None,
        name: Optional[str] = None,
        visibility: str = "private",
        owner_agent: Optional[str] = None,
        set_by: str = "agent",
    ) -> Dict[str, Any]:
        """
        Set the visibility (private/cluster/public) and optional owner for an entity.

        Untagged entities default to private. 'cluster'/'public' entities are
        visible to any scoped viewer; 'private' entities are visible only to
        their owner or explicitly granted agents.

        Args:
            entity_id: Entity id (either this or name required)
            name: Entity name (either this or entity_id required)
            visibility: private, cluster, or public
            owner_agent: Agent that owns this entity (used for private scoping)
            set_by: Caller identifier for audit

        Returns:
            Dict with entity_id and visibility
        """
        tag = visibility.lower()
        if tag not in VALID_TAGS:
            raise ValueError(f"visibility must be one of {sorted(VALID_TAGS)}")
        if owner_agent:
            owner_agent = _sanitize_agent(owner_agent)
        eid = _resolve_entity_id(db_path, entity_id, name)
        ensure_sidecar_tables(db_path)
        set_visibility(
            db_path, eid, VisibilityTag(tag), set_by=set_by, owner_agent=owner_agent
        )
        return {
            "success": True,
            "entity_id": eid,
            "visibility": tag,
            "owner_agent": owner_agent,
        }

    @app.tool()
    async def get_entity_visibility(
        entity_id: Optional[int] = None,
        name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get an entity's visibility tag, owner, and granted agents.

        Args:
            entity_id: Entity id (either this or name required)
            name: Entity name (either this or entity_id required)

        Returns:
            Dict with visibility, owner_agent, acl (granted agents)
        """
        eid = _resolve_entity_id(db_path, entity_id, name)
        row = get_visibility_row(db_path, eid)
        acl = get_acl(db_path, eid)
        return {
            "success": True,
            "entity_id": eid,
            "visibility": (row or {}).get("visibility", "private"),
            "owner_agent": (row or {}).get("owner_agent"),
            "set_by": (row or {}).get("set_by"),
            "acl": acl,
            "defaulted_to_private": row is None,
        }

    @app.tool()
    async def grant_entity_access(
        agent_id: str,
        entity_id: Optional[int] = None,
        name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Grant an agent read access to an entity (e.g. a private one it does not own).

        Args:
            agent_id: The agent to grant access to
            entity_id: Entity id (either this or name required)
            name: Entity name (either this or entity_id required)

        Returns:
            Dict confirming the grant
        """
        agent_id = _sanitize_agent(agent_id)
        eid = _resolve_entity_id(db_path, entity_id, name)
        grant_access(db_path, eid, agent_id)
        return {"success": True, "entity_id": eid, "granted_to": agent_id}

    @app.tool()
    async def revoke_entity_access(
        agent_id: str,
        entity_id: Optional[int] = None,
        name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Revoke an agent's read access to an entity.

        Args:
            agent_id: The agent whose access to revoke
            entity_id: Entity id (either this or name required)
            name: Entity name (either this or entity_id required)

        Returns:
            Dict confirming the revocation
        """
        agent_id = _sanitize_agent(agent_id)
        eid = _resolve_entity_id(db_path, entity_id, name)
        revoke_access(db_path, eid, agent_id)
        return {"success": True, "entity_id": eid, "revoked_from": agent_id}

    @app.tool()
    async def can_agent_view(
        agent_id: str,
        entity_id: Optional[int] = None,
        name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Check whether a scoped agent can view an entity (fail-closed).

        Args:
            agent_id: The viewer agent
            entity_id: Entity id (either this or name required)
            name: Entity name (either this or entity_id required)

        Returns:
            Dict with can_view bool and the visibility context
        """
        agent_id = _sanitize_agent(agent_id)
        eid = _resolve_entity_id(db_path, entity_id, name)
        allowed = can_view(db_path, eid, agent_id)
        row = get_visibility_row(db_path, eid)
        return {
            "success": True,
            "entity_id": eid,
            "agent_id": agent_id,
            "can_view": allowed,
            "visibility": (row or {}).get("visibility", "private"),
            "owner_agent": (row or {}).get("owner_agent"),
        }

    logger.info("Registered 5 governance MCP tools")
    return True
