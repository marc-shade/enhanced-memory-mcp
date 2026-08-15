#!/usr/bin/env python3
"""
Retrieval Diagnostics MCP Tools.

Kills quiet retrieval misses: when a recall step returns weak or empty results,
there is no error and the agent blames the model. `retrieval_diagnostics` runs
the same query through both retrieval backends (keyword/FTS search_nodes and
vector semantic_recall), reports the confidence each produced and why, and gives
the caller an actionable escalation path (broaden scope, fall back to raw text,
ask the user).

This is a read-only diagnostic; it never mutates memory.
"""

import logging

logger = logging.getLogger("retrieval_diagnostics")


def register_retrieval_diagnostics_tools(app, memory_client):
    @app.tool()
    async def retrieval_diagnostics(
        query: str,
        viewer_agent: str | None = None,
    ) -> dict:
        """
        Run a query through both retrieval backends and explain what matched.

        Use this when a search returned weak or empty results and you need to
        know WHY before deciding the next step. It reports the keyword (FTS +
        name) path and the vector (semantic) path side by side, with confidence
        and a low-confidence flag for each.

        Args:
            query: The query that underperformed.
            viewer_agent: Optional scoped viewer (same semantics as search_nodes).

        Returns:
            Dict with per-backend results, confidence, and a recommendation.
        """
        out = {"query": query, "backends": {}}

        # Keyword / FTS path via the socket service.
        try:
            kw = await memory_client.search_nodes(query, 10, viewer_agent)
            out["backends"]["keyword"] = {
                "count": kw.get("count", 0),
                "confidence": kw.get("confidence", 0.0),
                "low_confidence": kw.get("low_confidence", True),
                "top_names": [r.get("name") for r in (kw.get("results") or [])][:5],
            }
        except Exception as e:
            out["backends"]["keyword"] = {"error": f"{type(e).__name__}: {e}"}

        # Vector path (same embed + Qdrant search semantic_recall uses).
        try:
            from local_semantic_recall import (
                embed,
                QDRANT,
                DEFAULT_MODEL,
                collection_for,
            )
            from qdrant_client import QdrantClient
            from retrieval_quality import vector_low_confidence

            qv = embed([query], DEFAULT_MODEL)[0]
            client = QdrantClient(url=QDRANT)
            hits = client.query_points(
                collection_for(DEFAULT_MODEL), query=qv, limit=5
            ).points
            top_score = round(float(hits[0].score), 4) if hits else 0.0
            out["backends"]["vector"] = {
                "count": len(hits),
                "confidence": top_score,
                "low_confidence": vector_low_confidence(top_score),
                "top_names": [(h.payload or {}).get("name") for h in hits],
            }
        except Exception as e:
            out["backends"]["vector"] = {"error": f"{type(e).__name__}: {e}"}

        # Recommendation from the per-backend verdicts.
        kw = out["backends"].get("keyword") or {}
        vec = out["backends"].get("vector") or {}
        kw_weak = kw.get("low_confidence", True)
        vec_weak = vec.get("low_confidence", True)
        if kw_weak and vec_weak:
            out["recommendation"] = (
                "Both backends returned weak/no matches. The memory is likely "
                "absent or phrased very differently: broaden the query, search a "
                "broader entity_type prefix, or ask the user directly."
            )
        elif kw_weak and not vec_weak:
            out["recommendation"] = (
                "Vector search found a plausible match the keyword path missed. "
                "The concept is stored but the wording differs: use the vector "
                "hit's name to pull the entity."
            )
        elif not kw_weak and vec_weak:
            out["recommendation"] = (
                "Keyword/FTS matched but the vector path did not. The memory is "
                "stored with exact wording; the semantic index may not cover it "
                "yet (check vector_semantic_health for indexing coverage)."
            )
        else:
            out["recommendation"] = (
                "Both backends matched with healthy confidence. If the answer "
                "still seems wrong, the miss is in synthesis, not retrieval."
            )
        return out

    logger.info("Registered retrieval_diagnostics MCP tool")
    return True
