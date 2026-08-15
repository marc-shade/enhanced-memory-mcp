"""Native vector (semantic) search MCP tool for enhanced-memory.

Adds `semantic_recall`: embeds the query via the fedora ollama node
(nomic-embed-text, 768d — the SAME model that backfilled the index) and searches
the Qdrant `enhanced_memory` collection (unnamed 768d). Returns top matches by
MEANING, complementing the substring-only `search_nodes`.

Reuses `local_semantic_recall.embed` so query embeddings are byte-for-byte the
same path as the backfill (no doc-vs-query model drift). Fail-soft: returns a
JSON error rather than raising, so a degraded embedding backend never breaks the
MCP server.
"""

import json
import logging
import sqlite3

logger = logging.getLogger("enhanced-memory")


def _failure(error, backend, **echo):
    """JSON failure envelope for semantic_recall.

    Carries no `count` and no `results`: an empty result list and a broken
    embedding/vector backend must not serialize to the same thing. Callers that
    index `results` fail loudly instead of rendering "no memories found".
    """
    return json.dumps({"status": "failed", "error": error, "backend": backend, **echo})


def _make_can_view(db_path):
    """Lazily import the visibility ACL checker (governance sidecar)."""
    import sys
    from pathlib import Path

    _mf = (
        Path(__file__).resolve().parent.parent.parent
        / "intelligent-agents"
        / "memory_federation"
    )
    if str(_mf) not in sys.path and _mf.exists():
        sys.path.insert(0, str(_mf))

    def can_view(entity_id, viewer):
        from visibility import can_view as _cv

        return _cv(db_path, entity_id, viewer)

    return can_view


def register_semantic_vector_tools(app, db_path):
    @app.tool()
    async def semantic_recall(
        query: str,
        limit: int = 5,
        viewer_agent: str | None = None,
        scope: str | None = None,
    ) -> str:
        """Semantic vector search over stored memories by MEANING (not substring).

        Embeds the query with the cluster embedding model (fedora nomic-embed-text,
        768d) and returns the top-k most similar memory entities from the Qdrant
        vector index. Use when a keyword/substring search (`search_nodes`) would
        miss conceptually-related memories.

        Args:
            query: Natural-language query.
            limit: Max results (default 5).
            viewer_agent: Optional scoped viewer. When set, results are filtered
                fail-closed: only PUBLIC/CLUSTER entities plus PRIVATE ones this
                agent owns or is granted.
            scope: Optional project filter ('cfgi', 'arc-agi3', 'harness',
                'hardware', 'research', 'ops', 'kre', 'business', 'global'),
                matching search_nodes. An unknown scope errors with the valid
                list rather than returning an empty result set.

        Returns:
            JSON string: {query, count, confidence, low_confidence,
                          results: [{name, entity_type, score}]}
        """
        try:
            from local_semantic_recall import (
                embed,
                QDRANT,
                DEFAULT_MODEL,
                collection_for,
            )
            from qdrant_client import QdrantClient
            from retrieval_quality import vector_low_confidence

            conditions = []
            must_not = []

            # Suppression parity with search_nodes (added 2026-08-09).
            #
            # memory_db_service._visibility() excludes archived_at, superseded_by
            # and tier='quarantine' from every SQL search. This path honoured
            # NONE of them, so a fact retired in SQLite stayed fully recallable
            # by meaning -- and this is the path _proactive_recall uses to inject
            # memories into prompts, so it is the one that matters most.
            #
            # Demonstrated the same day: eleven April-2026 ARC score reports were
            # marked superseded, disappeared from search_nodes immediately, and
            # "what is the current leaderboard score" still returned the April
            # snapshot as the TOP semantic hit. This is the identical shape as
            # the 2026-07-24 quarantine incident, where search_nodes and the
            # indexer honoured tier='quarantine' and the injection path did not.
            #
            # Excluded by id rather than payload, because suppression lives in
            # SQLite and is not mirrored into the Qdrant payload.
            try:
                _c = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
                try:
                    suppressed = [
                        r[0]
                        for r in _c.execute(
                            "SELECT id FROM entities WHERE archived_at IS NOT NULL "
                            "OR superseded_by IS NOT NULL OR tier = 'quarantine'"
                        )
                    ]
                finally:
                    _c.close()
            except Exception as exc:  # reported, never silently skipped
                logging.getLogger(__name__).warning(
                    "semantic_recall: could not load suppression set (%s); "
                    "results may include retired memories",
                    exc,
                )
                suppressed = []

            # Only the suppressed entities that are actually INDEXED matter.
            # Taking the whole SQLite suppression set produced 9,962 ids, blew
            # the cap, and disabled the filter entirely -- while just 11 of them
            # existed as points. Intersecting with the collection is what makes
            # the payload small enough to send.
            #
            # The scroll costs one id-only request against a ~2k-point
            # collection, which is nothing beside the embedding call this
            # function already makes over the network.
            SUPPRESS_CAP = 4000
            if suppressed:
                from qdrant_client.models import HasIdCondition

                try:
                    _client = QdrantClient(url=QDRANT)
                    _pts, _ = _client.scroll(
                        collection_name=collection_for(DEFAULT_MODEL),
                        limit=100000,
                        with_payload=False,
                        with_vectors=False,
                    )
                    indexed = {p.id for p in _pts}
                    suppressed = [i for i in suppressed if i in indexed]
                except Exception as exc:
                    logging.getLogger(__name__).warning(
                        "semantic_recall: could not read the point set (%s); "
                        "falling back to the unintersected suppression list",
                        exc,
                    )

                if len(suppressed) > SUPPRESS_CAP:
                    # Say so rather than quietly dropping the filter: an
                    # unfiltered result that looks filtered is the exact failure
                    # this block exists to prevent.
                    logging.getLogger(__name__).warning(
                        "semantic_recall: %d indexed suppressed entities exceeds "
                        "the %d cap; retired memories may be returned",
                        len(suppressed),
                        SUPPRESS_CAP,
                    )
                elif suppressed:
                    must_not.append(HasIdCondition(has_id=suppressed))

            qfilter = None
            if scope is not None:
                # Scope lives in SQLite, not in the Qdrant payload, so it cannot
                # be expressed as a payload condition. Qdrant point ids ARE the
                # entities-table ids (local_semantic_recall backfill), so the
                # scope resolves to an id set and filters SERVER-SIDE via
                # HasIdCondition.
                #
                # Deliberately not "fetch limit, then drop out-of-scope hits":
                # that is how the ACL filter below works and it is a known wart,
                # because asking for 5 and discarding 4 returns 1 while looking
                # like a top-5. Filtering before the limit gives the real top-k
                # within the project. Scope sets here are small (largest folder
                # is ~113 entities), so the id list is cheap.
                import re as _re

                if not _re.fullmatch(r"[A-Za-z0-9_-]{1,64}", scope):
                    return _failure(
                        f"invalid scope {scope!r}", "scope-filter", query=query
                    )
                con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
                try:
                    has_tbl = con.execute(
                        "SELECT 1 FROM sqlite_master WHERE type='table' "
                        "AND name='memory_scope'"
                    ).fetchone()
                    if not has_tbl:
                        return _failure(
                            "memory_scope table missing; run memory_promotion.py",
                            "scope-filter",
                            query=query,
                        )
                    known = {
                        r[0]
                        for r in con.execute("SELECT DISTINCT scope FROM memory_scope")
                    }
                    if scope not in known:
                        return _failure(
                            f"unknown scope {scope!r}",
                            "scope-filter",
                            query=query,
                            known_scopes=sorted(known),
                        )
                    ids = [
                        r[0]
                        for r in con.execute(
                            "SELECT e.id FROM memory_scope ms "
                            "JOIN entities e ON e.name = ms.entity_name "
                            "WHERE ms.scope = ?",
                            (scope,),
                        )
                    ]
                finally:
                    con.close()
                from qdrant_client.models import HasIdCondition

                conditions.append(HasIdCondition(has_id=ids))

            if conditions or must_not:
                from qdrant_client.models import Filter

                qfilter = Filter(must=conditions or None, must_not=must_not or None)

            qv = embed([query], DEFAULT_MODEL)[0]
            client = QdrantClient(url=QDRANT)
            hits = client.query_points(
                collection_for(DEFAULT_MODEL),
                query=qv,
                limit=limit,
                query_filter=qfilter,
            ).points
            # Fail-closed ACL scoping for scoped viewers (Phase D, 2026-08-05).
            can_view = _make_can_view(db_path) if viewer_agent else None
            results = [
                {
                    "name": (h.payload or {}).get("name"),
                    "entity_type": (h.payload or {}).get("entity_type"),
                    "score": round(float(h.score), 4),
                }
                for h in hits
                if not can_view or can_view(int(h.id), viewer_agent)
            ]
            # Retrieval-quality signal (Phase G, 2026-08-05): the top cosine score
            # is the confidence, and a weak top match (or no match) flags a quiet
            # miss so the caller knows to broaden scope or fall back to raw text.
            top_score = results[0]["score"] if results else 0.0
            low_confidence = vector_low_confidence(top_score)
            # Append-only retrieval telemetry for the vector path. Qdrant point
            # ids are the entities-table ids (local_semantic_recall backfill).
            # Swallowed: telemetry must never break a retrieval. session_id is
            # not carried here today, so it logs "unknown" (see proposal caveat).
            try:
                from ops.retrieval_log import log_retrieval

                log_retrieval(
                    "unknown", query, [h.id for h in hits], source="semantic_recall"
                )
            except Exception as _log_err:
                logger.warning(
                    "semantic_recall: retrieval telemetry not written (%s: %s)",
                    type(_log_err).__name__,
                    _log_err,
                )
            return json.dumps(
                {
                    "query": query,
                    "count": len(results),
                    "confidence": round(top_score, 4),
                    "low_confidence": low_confidence,
                    "results": results,
                }
            )
        except Exception as e:
            logger.warning(f"semantic_recall failed: {type(e).__name__}: {e}")
            return _failure(f"{type(e).__name__}: {e}", "embedding/vector", query=query)
