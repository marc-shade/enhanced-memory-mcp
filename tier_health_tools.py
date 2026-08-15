#!/usr/bin/env python3
"""
Tier Health + Promotion MCP Tools.

The live store is dominated by archive (8396) / quarantine (1553) / reference
(1412); the semantic (28) and episodic (2) tiers are nearly empty, so the
4-tier claim is structural, not operational. These tools promote genuinely-used
archive/reference entities up into semantic (timeless knowledge) and episodic
(experience) tiers, using real usage (access_count) plus provenance quality
(l_score) as the signal — not a significance guess. Promotion un-archives
(clears archived_at) and is fully reversible (tier is just a column).
"""

import logging
import sqlite3
from typing import Dict, Any

logger = logging.getLogger("tier_health")


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


def _promote(
    db_path: str,
    target_tier: str,
    limit: int,
    extra_where: str,
    order_by: str,
) -> Dict[str, Any]:
    conn = _connect(db_path)
    try:
        cur = conn.cursor()
        rows = cur.execute(
            f"""
            SELECT id, name FROM entities
            WHERE tier IN ('archive', 'reference')
              AND (pinned IS NULL OR pinned = 0)
              AND COALESCE(access_count, 0) > 0
              -- Exclude test/audit artifacts: high access_count from being
              -- probed makes them look important, but they are not knowledge.
              AND entity_type NOT LIKE 'audit%'
              AND entity_type NOT LIKE '%test%'
              AND entity_type NOT IN ('test', 'diagnostic', 'verification')
              {extra_where}
            ORDER BY {order_by}
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        promoted = []
        for eid, name in rows:
            cur.execute(
                """
                UPDATE entities
                SET tier = ?, archived_at = NULL, last_accessed = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (target_tier, eid),
            )
            promoted.append({"id": eid, "name": name})
        conn.commit()
        return {
            "success": True,
            "target_tier": target_tier,
            "promoted": len(promoted),
            "entities": promoted[:50],
        }
    finally:
        conn.close()


def register_tier_health_tools(app, db_path):
    @app.tool()
    async def tier_health() -> Dict[str, Any]:
        """
        Report the live tier distribution and the promotion candidate pool.

        The 4-tier claim (working/episodic/semantic/procedural) is structural;
        this shows whether it is operational. Semantic and episodic have been
        near-empty; promote_to_* moves accessed archive/reference entities up.

        Returns:
            Dict with per-tier counts, total, and candidate pool sizes
        """
        conn = _connect(db_path)
        try:
            cur = conn.cursor()
            rows = cur.execute(
                "SELECT tier, COUNT(*) FROM entities GROUP BY tier ORDER BY 2 DESC"
            ).fetchall()
            tiers = {t: c for t, c in rows}
            total = sum(tiers.values())
            candidates = cur.execute(
                """
                SELECT
                  SUM(CASE WHEN tier IN ('archive','reference')
                            AND COALESCE(access_count,0) > 0 THEN 1 ELSE 0 END),
                  SUM(CASE WHEN tier IN ('archive','reference')
                            AND COALESCE(access_count,0) > 0
                            AND COALESCE(l_score,0) >= 0.5 THEN 1 ELSE 0 END)
                FROM entities
                """
            ).fetchone()
            return {
                "success": True,
                "total_entities": total,
                "tiers": tiers,
                "promotion_candidates": {
                    "accessed": candidates[0] or 0,
                    "accessed_and_high_lscore": candidates[1] or 0,
                },
                "note": (
                    "run promote_to_semantic / promote_to_episodic to move "
                    "accessed archive/reference entities up"
                ),
            }
        finally:
            conn.close()

    @app.tool()
    async def promote_to_semantic(limit: int = 50) -> Dict[str, Any]:
        """
        Promote accessed archive/reference entities to the semantic tier.

        Semantic = timeless knowledge. Candidates are entities with real usage
        (access_count > 0), not quarantined, not pinned, ordered by provenance
        quality (l_score) then usage. Reversible (tier is a column).

        Args:
            limit: Max entities to promote (default 50)

        Returns:
            Dict with count and the promoted entities
        """
        return _promote(
            db_path,
            "semantic",
            limit,
            "AND COALESCE(l_score, 0.5) >= 0.5",
            "COALESCE(l_score, 0.5) DESC, access_count DESC, last_accessed DESC",
        )

    @app.tool()
    async def promote_to_episodic(limit: int = 50) -> Dict[str, Any]:
        """
        Promote accessed archive/reference entities to the episodic tier.

        Episodic = time-bound experiences. Candidates are recent entities (from
        the last 90 days) with usage, not quarantined, not pinned, ordered by
        recency.

        Args:
            limit: Max entities to promote (default 50)

        Returns:
            Dict with count and the promoted entities
        """
        return _promote(
            db_path,
            "episodic",
            limit,
            "AND created_at > datetime('now', '-90 days')",
            "last_accessed DESC, access_count DESC",
        )

    logger.info("Registered 3 tier-health MCP tools")
    return True
