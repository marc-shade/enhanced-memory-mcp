"""
OmniMEM N-Hop Graph Retrieval Tools

Implements bounded N-hop graph traversal with distance-decaying relevance scoring,
inspired by OmniMEM (April 2026, UNC Chapel Hill et al.).

Key differences from existing spread_activation:
- spread_activation: Neural priming model (activation propagates and decays via strength * (1-decay))
- nhop_retrieval: Retrieval-focused (finds seed entities from query, expands N hops,
  scores with 1/(1+d) distance decay, merges with hybrid search via set union)

The OmniMEM approach:
1. Given a query, identify seed entities mentioned in the query
2. Perform bounded neighborhood expansion (N hops) through relations and associations
3. Score each reached entity with distance-decaying relevance: score = 1 / (1 + d)
4. Merge graph-reached entities with hybrid search results (vector + BM25)
5. Return unified ranked results

Reference: "OmniMEM: Auto-Research Guided Discovery of Lifelong Multimodal Agent Memory"
"""

import logging
import sqlite3
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from qdrant_client import QdrantClient  # type: ignore[import-untyped]

    _QDRANT_AVAILABLE = True
except ImportError:
    QdrantClient = None  # type: ignore[assignment,misc]
    _QDRANT_AVAILABLE = False

logger = logging.getLogger("omnimem-nhop")

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
# Phase 0 spine repair (2026-07-02): repointed at the live write-path-indexed
# collection + its embedder (was the stale 364-point "enhanced_memory" with a
# different model, whose sqlite_id filter matched nothing, so seeds always fell
# back to the broken text matcher — audit: ARC milestones at match_score 1.0).
# Single source of truth: local_semantic_recall (same as semantic_recall and
# vector_write_indexer).
from local_semantic_recall import DEFAULT_MODEL as OLLAMA_EMBED_MODEL
from local_semantic_recall import OLLAMA_URL as OLLAMA_HOST
from local_semantic_recall import collection_for as _collection_for

QDRANT_COLLECTION = _collection_for(OLLAMA_EMBED_MODEL)


def _ollama_embed(text: str) -> Optional[List[float]]:
    """Generate embedding via Ollama API. Returns None on failure."""
    import urllib.request
    import json as _json

    try:
        req = urllib.request.Request(
            f"{OLLAMA_HOST}/api/embeddings",
            data=_json.dumps({"model": OLLAMA_EMBED_MODEL, "prompt": text}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = _json.loads(resp.read())
            embedding = data.get("embedding", [])
            return embedding if embedding else None
    except Exception as e:
        logger.warning(f"Ollama embedding failed: {e}")
        return None


MEMORY_DIR = Path.home() / ".claude" / "enhanced_memories"
DB_PATH = MEMORY_DIR / "memory.db"


def _find_seed_entities_text(
    cursor: sqlite3.Cursor, query: str, limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Find seed entities by substring/term-overlap matching against entity names
    and observations.  This is the original text-based implementation, retained
    as the fallback when Qdrant is unavailable.

    Returns list of dicts with entity_id and match_score.
    """
    seeds = {}
    query_lower = query.lower()
    query_terms = [t.strip() for t in query_lower.split() if len(t.strip()) > 2]

    if not query_terms:
        return []

    # Match against entity names (exact substring match scores highest).
    # Suppression predicate added 2026-08-11: this read was unfiltered, so
    # archived/superseded/quarantined memories reached callers through the
    # n-hop path while search_nodes and semantic_recall suppressed them.
    cursor.execute(
        "SELECT id, name, entity_type FROM entities "
        "WHERE archived_at IS NULL AND superseded_by IS NULL "
        "AND COALESCE(tier,'') != 'quarantine'"
    )
    for row in cursor.fetchall():
        entity_id, name, entity_type = row
        name_lower = name.lower()
        # Exact name match
        if name_lower in query_lower or query_lower in name_lower:
            seeds[entity_id] = {
                "entity_id": entity_id,
                "name": name,
                "type": entity_type,
                "match_score": 1.0,
            }
            continue
        # Term overlap
        name_terms = (
            set(name_lower.split("_"))
            .union(name_lower.split("-"))
            .union(name_lower.split())
        )
        overlap = sum(
            1 for t in query_terms if any(t in nt or nt in t for nt in name_terms)
        )
        if overlap > 0:
            score = overlap / max(len(query_terms), 1)
            if score > 0.2:
                seeds[entity_id] = {
                    "entity_id": entity_id,
                    "name": name,
                    "type": entity_type,
                    "match_score": score,
                }

    # Match against observations (check if query terms appear)
    if len(seeds) < limit:
        like_clauses = " OR ".join(["content LIKE ?"] * len(query_terms))
        if like_clauses:
            params = [f"%{t}%" for t in query_terms]
            cursor.execute(
                f"""
                SELECT DISTINCT o.entity_id, e.name, e.entity_type,
                       COUNT(*) as hit_count
                FROM observations o
                JOIN entities e ON o.entity_id = e.id
                WHERE e.archived_at IS NULL AND e.superseded_by IS NULL
                  AND COALESCE(e.tier,'') != 'quarantine'
                  AND {like_clauses}
                GROUP BY o.entity_id
                ORDER BY hit_count DESC
                LIMIT ?
                """,
                params + [limit * 2],
            )
            for row in cursor.fetchall():
                eid, name, etype, hits = row
                if eid not in seeds:
                    score = min(1.0, hits / max(len(query_terms), 1) * 0.6)
                    seeds[eid] = {
                        "entity_id": eid,
                        "name": name,
                        "type": etype,
                        "match_score": score,
                    }

    # Sort by match score, return top N
    sorted_seeds = sorted(seeds.values(), key=lambda x: x["match_score"], reverse=True)
    return sorted_seeds[:limit]


def _find_seed_entities_vector(
    cursor: sqlite3.Cursor, query: str, limit: int = 10
) -> Optional[List[Dict[str, Any]]]:
    """
    Find seed entities using Qdrant vector search via Ollama embeddings.

    Encodes the query via Ollama (nomic-embed-text-v2-moe, 768-dim) and
    retrieves top-scoring points from Qdrant. Each point's payload contains
    ``sqlite_id`` (int) mapping to the entities table.

    Returns a list of dicts with entity_id, name, type, and match_score on
    success, or None if Qdrant/Ollama is unavailable (falls back to text matching).
    """
    if not _QDRANT_AVAILABLE:
        return None

    query_vector = _ollama_embed(query)
    if query_vector is None:
        return None

    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=3)

        # Verify the collection exists before querying
        collections = [c.name for c in client.get_collections().collections]
        if QDRANT_COLLECTION not in collections:
            logger.warning(
                "Qdrant collection '%s' not found — falling back to text matching",
                QDRANT_COLLECTION,
            )
            return None

        # Point id IS the sqlite entity id in the write-path-indexed collection
        # (vector_write_indexer sets PointStruct(id=entity_id)); no payload
        # filter needed. The old sqlite_id Range filter matched nothing in the
        # live collection and silently forced the broken text fallback.
        response = client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )
        hits = response.points
    except Exception as e:
        logger.warning(
            "Qdrant vector search unavailable (%s) — falling back to text matching", e
        )
        return None

    if not hits:
        return None

    # Map Qdrant results back to entity details via SQLite
    # Payload schema: {sqlite_id: int, name: str, entity_type: str, ...}
    # Some points may also use "entity_id" or "memory_id" — try all.
    seeds: List[Dict[str, Any]] = []
    seen_ids: set = set()
    for hit in hits:
        payload = hit.payload or {}
        # point id is the entity id in the live collection; legacy payload
        # keys kept as fallbacks for old points
        entity_id = hit.id or payload.get("sqlite_id") or payload.get("entity_id")
        if entity_id is None:
            continue
        entity_id = int(entity_id)
        if entity_id in seen_ids:
            continue
        seen_ids.add(entity_id)

        # Use payload name/type if available, otherwise look up in SQLite
        name = payload.get("name")
        entity_type = payload.get("entity_type")
        if not name:
            cursor.execute(
                "SELECT name, entity_type FROM entities WHERE id = ?",
                (entity_id,),
            )
            row = cursor.fetchone()
            if row is None:
                continue
            name, entity_type = row

        seeds.append(
            {
                "entity_id": entity_id,
                "name": name,
                "type": entity_type or "unknown",
                # Qdrant cosine scores are already in [0, 1] for cosine distance
                "match_score": round(max(0.0, min(1.0, hit.score)), 4),
            }
        )

    return seeds if seeds else None


def _find_seed_entities(
    cursor: sqlite3.Cursor, query: str, limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Find seed entities relevant to a query.

    Tries Qdrant vector search first (semantic similarity, avoids greedy
    substring matches).  Falls back to text-based name/observation matching
    when Qdrant is down or the collection doesn't exist.

    Returns list of dicts with entity_id, name, type, and match_score.
    """
    vector_results = _find_seed_entities_vector(cursor, query, limit=limit)
    if vector_results is not None:
        logger.debug(
            "Seed selection: vector search returned %d candidates", len(vector_results)
        )
        return vector_results

    logger.debug("Seed selection: using text fallback")
    return _find_seed_entities_text(cursor, query, limit=limit)


def _bounded_nhop_expansion(
    cursor: sqlite3.Cursor,
    seed_ids: List[int],
    max_hops: int = 3,
    min_relation_weight: float = 0.0,
    min_association_strength: float = 0.1,
) -> Dict[int, int]:
    """
    BFS expansion from seed entities through relations and associations.

    Returns dict mapping entity_id -> shortest_path_distance from any seed.
    Seeds themselves have distance 0.
    """
    distances: Dict[int, int] = {}
    queue = deque()

    # Initialize with seeds at distance 0
    for sid in seed_ids:
        distances[sid] = 0
        queue.append((sid, 0))

    while queue:
        current_id, current_dist = queue.popleft()

        if current_dist >= max_hops:
            continue

        next_dist = current_dist + 1
        neighbors = set()

        # Expand via relations table (directed graph)
        cursor.execute(
            """
            SELECT to_entity_id FROM relations
            WHERE from_entity_id = ?
            """,
            (current_id,),
        )
        for (neighbor_id,) in cursor.fetchall():
            neighbors.add(neighbor_id)

        cursor.execute(
            """
            SELECT from_entity_id FROM relations
            WHERE to_entity_id = ?
            """,
            (current_id,),
        )
        for (neighbor_id,) in cursor.fetchall():
            neighbors.add(neighbor_id)

        # Expand via memory_associations table (undirected, strength-gated)
        try:
            cursor.execute(
                """
                SELECT
                    CASE WHEN entity_a_id = ? THEN entity_b_id ELSE entity_a_id END as neighbor_id
                FROM memory_associations
                WHERE (entity_a_id = ? OR entity_b_id = ?)
                  AND association_strength >= ?
                """,
                (current_id, current_id, current_id, min_association_strength),
            )
            for (neighbor_id,) in cursor.fetchall():
                neighbors.add(neighbor_id)
        except sqlite3.OperationalError:
            pass  # memory_associations table may not exist yet

        # Expand via causal_links table (directed)
        try:
            cursor.execute(
                """
                SELECT effect_entity_id FROM causal_links WHERE cause_entity_id = ? AND strength >= ?
                UNION
                SELECT cause_entity_id FROM causal_links WHERE effect_entity_id = ? AND strength >= ?
                """,
                (current_id, min_relation_weight, current_id, min_relation_weight),
            )
            for (neighbor_id,) in cursor.fetchall():
                neighbors.add(neighbor_id)
        except sqlite3.OperationalError:
            pass  # causal_links table may not exist yet

        # BFS: only visit unvisited nodes (shortest path guarantee)
        for nid in neighbors:
            if nid not in distances:
                distances[nid] = next_dist
                queue.append((nid, next_dist))

    return distances


def _distance_decaying_relevance(distance: int) -> float:
    """OmniMEM distance-decaying relevance: score = 1 / (1 + d)"""
    return 1.0 / (1.0 + distance)


def nhop_graph_retrieve(
    query: str,
    max_hops: int = 3,
    max_seeds: int = 5,
    min_association_strength: float = 0.1,
    limit: int = 20,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    OmniMEM-style N-hop graph retrieval with distance-decaying relevance.

    1. Find seed entities from query
    2. BFS expand through relations/associations up to max_hops
    3. Score with 1/(1+d) distance decay
    4. Return ranked entities with their graph context
    """
    db = db_path or DB_PATH
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    try:
        # Step 1: Find seed entities
        seeds = _find_seed_entities(cursor, query, limit=max_seeds)
        if not seeds:
            return {
                "success": True,
                "query": query,
                "seeds_found": 0,
                "results": [],
                "message": "No seed entities found for query",
            }

        seed_ids = [s["entity_id"] for s in seeds]

        # Step 2: Bounded N-hop expansion
        distances = _bounded_nhop_expansion(
            cursor,
            seed_ids,
            max_hops=max_hops,
            min_association_strength=min_association_strength,
        )

        # Step 3: Score with distance-decaying relevance
        scored_entities = []
        for entity_id, distance in distances.items():
            relevance = _distance_decaying_relevance(distance)

            # Boost seeds by their match score
            seed_match = next((s for s in seeds if s["entity_id"] == entity_id), None)
            if seed_match:
                relevance *= 1.0 + seed_match["match_score"]

            # Fetch entity details
            cursor.execute(
                """
                SELECT id, name, entity_type, tier, modality, raw_data_pointer,
                       access_count, created_at, last_accessed
                FROM entities WHERE id = ?
                """,
                (entity_id,),
            )
            row = cursor.fetchone()
            if not row:
                continue

            # Fetch observations (summary)
            cursor.execute(
                "SELECT content FROM observations WHERE entity_id = ? LIMIT 5",
                (entity_id,),
            )
            observations = [r[0] for r in cursor.fetchall()]

            scored_entities.append(
                {
                    "entity_id": row["id"],
                    "name": row["name"],
                    "entity_type": row["entity_type"],
                    "tier": row["tier"],
                    "modality": row["modality"] if "modality" in row.keys() else "text",
                    "raw_data_pointer": row["raw_data_pointer"]
                    if "raw_data_pointer" in row.keys()
                    else None,
                    "observations": observations,
                    "graph_distance": distance,
                    "relevance_score": round(relevance, 4),
                    "is_seed": entity_id in seed_ids,
                }
            )

        # Step 4: Sort by relevance, return top N
        scored_entities.sort(key=lambda x: x["relevance_score"], reverse=True)
        results = scored_entities[:limit]

        return {
            "success": True,
            "query": query,
            "seeds_found": len(seeds),
            "seed_entities": [
                {"name": s["name"], "match_score": round(s["match_score"], 3)}
                for s in seeds
            ],
            "total_graph_entities": len(distances),
            "results_returned": len(results),
            "max_hops": max_hops,
            "results": results,
        }

    except Exception as e:
        logger.error(f"N-hop graph retrieval failed: {e}", exc_info=True)
        return {"success": False, "error": str(e), "query": query}
    finally:
        conn.close()


def register_omnimem_nhop_tools(app, db_path: Optional[Path] = None):
    """Register OmniMEM N-hop graph retrieval tools with FastMCP app."""

    _db_path: Path = db_path or DB_PATH

    @app.tool()
    async def omnimem_nhop_retrieve(
        query: str,
        max_hops: int = 3,
        max_seeds: int = 5,
        min_association_strength: float = 0.1,
        limit: int = 20,
    ) -> Dict[str, Any]:
        """
        OmniMEM-style N-hop graph retrieval with distance-decaying relevance.

        Finds seed entities matching the query, expands through the knowledge graph
        (relations, associations, causal links) up to N hops, and scores results
        using 1/(1+d) distance decay. Seed entities get a match-quality boost.

        Inspired by OmniMEM (April 2026): progressive pyramid retrieval with
        bounded neighborhood expansion and hybrid search fusion.

        Args:
            query: Natural language search query
            max_hops: Maximum graph traversal depth (1-5, default 3)
            max_seeds: Maximum seed entities to start from (default 5)
            min_association_strength: Minimum association strength to traverse (0.0-1.0)
            limit: Maximum results to return

        Returns:
            Ranked entities with graph distance, relevance scores, and observations
        """
        max_hops = max(1, min(5, max_hops))
        max_seeds = max(1, min(20, max_seeds))
        min_association_strength = max(0.0, min(1.0, min_association_strength))

        return nhop_graph_retrieve(
            query=query,
            max_hops=max_hops,
            max_seeds=max_seeds,
            min_association_strength=min_association_strength,
            limit=limit,
            db_path=_db_path,
        )

    @app.tool()
    async def omnimem_hybrid_graph_search(
        query: str,
        max_hops: int = 2,
        max_seeds: int = 5,
        graph_weight: float = 0.4,
        text_weight: float = 0.6,
        limit: int = 15,
    ) -> Dict[str, Any]:
        """
        Fused OmniMEM retrieval: combines N-hop graph traversal with text search.

        Implements OmniMEM's set-union fusion strategy:
        1. N-hop graph retrieval (distance-decaying relevance)
        2. Text-based entity search (name + observation matching)
        3. Set union of results, re-scored with configurable weights

        This mirrors OmniMEM's "parallel multiview retrieval" that simultaneously
        queries vector (dense), BM25 (sparse), and graph (structured) indexes,
        then fuses via set union with per-view scoring.

        Args:
            query: Natural language search query
            max_hops: Graph traversal depth (default 2)
            max_seeds: Maximum seed entities (default 5)
            graph_weight: Weight for graph-based relevance (0.0-1.0)
            text_weight: Weight for text-based relevance (0.0-1.0)
            limit: Maximum results to return

        Returns:
            Fused results with both graph and text relevance scores
        """
        max_hops = max(1, min(5, max_hops))
        graph_weight = max(0.0, min(1.0, graph_weight))
        text_weight = max(0.0, min(1.0, text_weight))

        # Normalize weights
        total = graph_weight + text_weight
        if total > 0:
            graph_weight /= total
            text_weight /= total

        # Get graph results
        graph_results = nhop_graph_retrieve(
            query=query,
            max_hops=max_hops,
            max_seeds=max_seeds,
            limit=limit * 2,
            db_path=_db_path,
        )

        # Build entity score map from graph
        entity_scores = {}
        if graph_results.get("success"):
            for r in graph_results.get("results", []):
                eid = r["entity_id"]
                entity_scores[eid] = {
                    "graph_score": r["relevance_score"],
                    "text_score": 0.0,
                    **r,
                }

        # Text search: find entities by observation content matching
        db = _db_path or DB_PATH
        conn = sqlite3.connect(db)
        cursor = conn.cursor()
        try:
            query_terms = [
                t.strip() for t in query.lower().split() if len(t.strip()) > 2
            ]
            if query_terms:
                like_clauses = " OR ".join(["content LIKE ?"] * len(query_terms))
                params = [f"%{t}%" for t in query_terms]
                cursor.execute(
                    f"""
                    SELECT o.entity_id, e.name, e.entity_type, e.tier,
                           COUNT(*) as hits, GROUP_CONCAT(SUBSTR(o.content, 1, 100), ' | ') as snippets
                    FROM observations o
                    JOIN entities e ON o.entity_id = e.id
                    WHERE {like_clauses}
                    GROUP BY o.entity_id
                    ORDER BY hits DESC
                    LIMIT ?
                    """,
                    params + [limit * 2],
                )
                for row in cursor.fetchall():
                    eid = row[0]
                    text_score = min(1.0, row[4] / max(len(query_terms), 1))
                    if eid in entity_scores:
                        entity_scores[eid]["text_score"] = text_score
                    else:
                        entity_scores[eid] = {
                            "entity_id": eid,
                            "name": row[1],
                            "entity_type": row[2],
                            "tier": row[3],
                            "graph_score": 0.0,
                            "text_score": text_score,
                            "graph_distance": -1,
                            "is_seed": False,
                            "observations": (row[5] or "").split(" | ")[:3],
                        }
        finally:
            conn.close()

        # Fuse scores
        fused = []
        for eid, data in entity_scores.items():
            combined = (graph_weight * data["graph_score"]) + (
                text_weight * data["text_score"]
            )
            fused.append(
                {
                    "entity_id": eid,
                    "name": data.get("name", ""),
                    "entity_type": data.get("entity_type", ""),
                    "tier": data.get("tier", ""),
                    "graph_score": round(data["graph_score"], 4),
                    "text_score": round(data["text_score"], 4),
                    "combined_score": round(combined, 4),
                    "graph_distance": data.get("graph_distance", -1),
                    "is_seed": data.get("is_seed", False),
                    "observations": data.get("observations", [])[:3],
                }
            )

        fused.sort(key=lambda x: x["combined_score"], reverse=True)
        results = fused[:limit]

        return {
            "success": True,
            "query": query,
            "fusion_method": "set_union_weighted",
            "weights": {"graph": round(graph_weight, 2), "text": round(text_weight, 2)},
            "total_candidates": len(entity_scores),
            "results_returned": len(results),
            "results": results,
        }

    logger.info("OmniMEM N-hop graph retrieval tools registered")
