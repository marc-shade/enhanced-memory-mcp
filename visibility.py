"""
visibility.py — Read/write visibility tags on enhanced-memory entities.

Tags live in the sidecar table `entity_visibility`, which this module
creates if it doesn't exist.  The canonical entity table is never modified.

Default for any entity NOT in the sidecar: VisibilityTag.PRIVATE (opt-in to
sharing, not opt-out).
"""

from __future__ import annotations

import sqlite3
from enum import Enum
from typing import Optional


class VisibilityTag(str, Enum):
    PRIVATE = "private"
    CLUSTER = "cluster"
    PUBLIC = "public"


# DDL for the sidecar tables we own.
_CREATE_VISIBILITY = """
CREATE TABLE IF NOT EXISTS entity_visibility (
    entity_id  INTEGER PRIMARY KEY,
    visibility TEXT    NOT NULL CHECK (visibility IN ('private', 'cluster', 'public')),
    owner_agent TEXT,
    set_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    set_by     TEXT,
    FOREIGN KEY (entity_id) REFERENCES entities(id)
);
"""

_CREATE_ACL = """
CREATE TABLE IF NOT EXISTS entity_acl (
    entity_id  INTEGER NOT NULL,
    agent_id   TEXT    NOT NULL,
    access     TEXT    DEFAULT 'read',
    granted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (entity_id, agent_id),
    FOREIGN KEY (entity_id) REFERENCES entities(id)
);
"""

_CREATE_CONFLICTS = """
CREATE TABLE IF NOT EXISTS federation_conflicts (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id         INTEGER NOT NULL,
    conflicted_with_id INTEGER NOT NULL,
    resolution        TEXT    NOT NULL,
    resolved_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (entity_id)           REFERENCES entities(id),
    FOREIGN KEY (conflicted_with_id)  REFERENCES entities(id)
);
"""


def _connect(db_path: str) -> sqlite3.Connection:
    """Open a WAL-mode, busy-timeout-safe connection."""
    conn = sqlite3.connect(db_path, timeout=3.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=3000")
    conn.row_factory = sqlite3.Row
    return conn


def ensure_sidecar_tables(db_path: str) -> None:
    """Create entity_visibility, entity_acl and federation_conflicts if absent."""
    with _connect(db_path) as conn:
        conn.execute(_CREATE_VISIBILITY)
        conn.execute(_CREATE_ACL)
        conn.execute(_CREATE_CONFLICTS)
        # Idempotent migration for pre-existing DBs (owner_agent added 2026-08-05).
        cols = {r[1] for r in conn.execute("PRAGMA table_info(entity_visibility)")}
        if "owner_agent" not in cols:
            conn.execute("ALTER TABLE entity_visibility ADD COLUMN owner_agent TEXT")
        conn.commit()


def set_visibility(
    db_path: str,
    entity_id: int,
    tag: VisibilityTag,
    set_by: str = "system",
    owner_agent: Optional[str] = None,
) -> None:
    """Upsert a visibility tag (and optionally an owner agent) on an entity."""
    ensure_sidecar_tables(db_path)
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO entity_visibility (entity_id, visibility, set_by, owner_agent)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(entity_id) DO UPDATE SET
                visibility = excluded.visibility,
                set_at     = CURRENT_TIMESTAMP,
                set_by     = excluded.set_by,
                owner_agent = COALESCE(excluded.owner_agent, entity_visibility.owner_agent)
            """,
            (entity_id, tag.value, set_by, owner_agent),
        )
        conn.commit()


def set_visibility_conn(
    conn: sqlite3.Connection,
    entity_id: int,
    tag: VisibilityTag,
    set_by: str = "system",
    owner_agent: Optional[str] = None,
) -> None:
    """
    Upsert a visibility tag using an already-open connection.
    Does NOT commit — caller owns the transaction.
    Sidecar tables must already exist (call ensure_sidecar_tables first).
    """
    conn.execute(
        """
        INSERT INTO entity_visibility (entity_id, visibility, set_by, owner_agent)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(entity_id) DO UPDATE SET
            visibility = excluded.visibility,
            set_at     = CURRENT_TIMESTAMP,
            set_by     = excluded.set_by,
            owner_agent = COALESCE(excluded.owner_agent, entity_visibility.owner_agent)
        """,
        (entity_id, tag.value, set_by, owner_agent),
    )


def get_visibility(db_path: str, entity_id: int) -> VisibilityTag:
    """Return the tag for entity_id, defaulting to PRIVATE if absent."""
    ensure_sidecar_tables(db_path)
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT visibility FROM entity_visibility WHERE entity_id = ?",
            (entity_id,),
        ).fetchone()
    if row is None:
        return VisibilityTag.PRIVATE
    return VisibilityTag(row["visibility"])


def list_shareable_entity_ids(
    db_path: str,
    max_entities: int = 1000,
) -> list[int]:
    """
    Return entity IDs tagged cluster or public, ordered by l_score DESC.

    Joins the sidecar against the canonical entities table so that entities
    deleted from `entities` but orphaned in the sidecar are excluded.
    Capped at max_entities.
    """
    ensure_sidecar_tables(db_path)
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT ev.entity_id
            FROM   entity_visibility ev
            JOIN   entities          e  ON e.id = ev.entity_id
            WHERE  ev.visibility IN ('cluster', 'public')
            ORDER  BY COALESCE(e.l_score, 0.5) DESC
            LIMIT  ?
            """,
            (max_entities,),
        ).fetchall()
    return [r["entity_id"] for r in rows]


def get_visibility_row(db_path: str, entity_id: int) -> Optional[dict]:
    """Return the full visibility row (tag, owner, set_by) or None if untagged."""
    ensure_sidecar_tables(db_path)
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT entity_id, visibility, owner_agent, set_at, set_by "
            "FROM entity_visibility WHERE entity_id = ?",
            (entity_id,),
        ).fetchone()
    return dict(row) if row else None


def grant_access(db_path: str, entity_id: int, agent_id: str) -> None:
    """Grant an agent read access to a (typically private) entity."""
    ensure_sidecar_tables(db_path)
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO entity_acl (entity_id, agent_id, access)
            VALUES (?, ?, 'read')
            ON CONFLICT(entity_id, agent_id) DO UPDATE SET
                granted_at = CURRENT_TIMESTAMP
            """,
            (entity_id, agent_id),
        )
        conn.commit()


def revoke_access(db_path: str, entity_id: int, agent_id: str) -> None:
    """Revoke an agent's read access to an entity."""
    ensure_sidecar_tables(db_path)
    with _connect(db_path) as conn:
        conn.execute(
            "DELETE FROM entity_acl WHERE entity_id = ? AND agent_id = ?",
            (entity_id, agent_id),
        )
        conn.commit()


def get_acl(db_path: str, entity_id: int) -> list[str]:
    """Return the list of agents granted access to an entity."""
    ensure_sidecar_tables(db_path)
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT agent_id FROM entity_acl WHERE entity_id = ?",
            (entity_id,),
        ).fetchall()
    return [r["agent_id"] for r in rows]


def can_view(db_path: str, entity_id: int, viewer_agent: str) -> bool:
    """Fail-closed access check for a scoped viewer.

    - PUBLIC or CLUSTER tagged entities: visible to any viewer.
    - PRIVATE (or untagged, which defaults to private): visible only to the
      owner or to an explicitly granted agent.
    """
    row = get_visibility_row(db_path, entity_id)
    if row is None:
        # Untagged entities default to private.
        return viewer_agent in get_acl(db_path, entity_id)
    tag = row["visibility"]
    if tag in ("cluster", "public"):
        return True
    if row.get("owner_agent") == viewer_agent:
        return True
    return viewer_agent in get_acl(db_path, entity_id)
