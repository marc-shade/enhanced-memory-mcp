"""MCP tool registration for the AtomMem upgrades.

register_atommem_tools(app, db_path) adds these tools (all additive; no existing
tool is modified):

  extract_atomic_facts        (Δ1) raw text -> standalone, time-anchored facts + metadata
  atommem_graph_recall        (Δ2) IDF-weighted PPR recall over real entities
  atommem_keyword_neighbors   (Δ2) IDF-weighted keyword neighbors of a named entity
  verify_fact_before_store    (Δ3) dedup + conflict + residual decision for a new fact
  upsert_temporal_profile     (Δ4) merge a stable attribute into a subject's timeline
  query_temporal_profile      (Δ4) point-in-time profile retrieval
  atommem_status              diagnostics (LLM providers, embedder, profile count)

Tools read the live memory.db read-only (entities + observations.content). The
only writer is upsert_temporal_profile, which touches the isolated, newly-created
temporal_profiles table.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Dict, List, Optional

from .atomic_facts import AtomicFactExtractor
from .fact_verification import FactVerifier, VerifyConfig
from .idf_keyword_graph import IDFKeywordGraph, jaccard_similarity
from .keywords import extract_keywords
from .llm_cli import get_llm
from .temporal_profile import TemporalProfileStore

_DEFAULT_ENTITY_SCAN = 800  # mirror detect_memory_conflicts' bounded scan


def _load_entity_items(
    db_path: str, limit: int = _DEFAULT_ENTITY_SCAN
) -> List[Dict[str, Any]]:
    """Load the newest `limit` entities with concatenated observation text and
    derived keywords. Read-only."""
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT e.id AS id, e.name AS name, e.entity_type AS etype,
                   (SELECT GROUP_CONCAT(o.content, ' ')
                      FROM observations o WHERE o.entity_id = e.id) AS obs
            FROM entities e
            -- Suppression predicate added 2026-08-11; this read was
            -- unfiltered while the primary search paths suppressed.
            WHERE e.archived_at IS NULL AND e.superseded_by IS NULL
              AND COALESCE(e.tier,'') != 'quarantine'
            ORDER BY e.id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    finally:
        conn.close()

    items: List[Dict[str, Any]] = []
    for r in rows:
        name = r["name"] or ""
        obs = r["obs"] or ""
        # Verify comparisons must run against the RAW fact text. Prepending
        # the entity name and keeping the "[Context: ...]" enrichment suffix
        # dragged an exact duplicate's cosine to 0.81 (< 0.95 duplicate
        # threshold), so IGNORE never fired (verified live 2026-07-02).
        # Keywords still derive from name+obs to keep pre-filter recall.
        clean_obs = obs.rsplit(" [Context:", 1)[0].strip()
        text = clean_obs if clean_obs else name
        kws = extract_keywords(f"{name}. {obs}".strip(), max_keywords=8)
        items.append(
            {
                "id": str(r["id"]),
                "name": name,
                "entity_type": r["etype"],
                "text": text[:2000],
                "keywords": kws,
                "people": [],
            }
        )
    return items


def decide_fact(
    db_path: str,
    fact: str,
    max_candidates: int = 60,
    use_llm: bool = True,
    people: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Module-level verify decision (extracted 2026-07-02, Phase 2.2) so the
    bi-temporal write path in server.py (`store_fact_versioned`) can reuse the
    exact logic of the `verify_fact_before_store` MCP tool, which is now a
    thin wrapper over this."""
    from . import embedder

    if not embedder.available():
        return {
            "action": "CREATE",
            "reason": "embedder unavailable; cannot dedup, defaulting to store",
            "store_text": fact,
            "is_residual": False,
            "conflict_with": None,
            "candidates_considered": 0,
            "degraded": True,
        }

    new_kw = extract_keywords(fact, max_keywords=8)
    items = _load_entity_items(db_path, _DEFAULT_ENTITY_SCAN)
    # Cheap keyword pre-filter, then embed only the top candidates.
    ranked = sorted(
        items,
        key=lambda it: jaccard_similarity(new_kw, it.get("keywords", [])),
        reverse=True,
    )
    candidates = [
        it for it in ranked if jaccard_similarity(new_kw, it.get("keywords", [])) > 0
    ][:max_candidates]
    existing = [
        {
            "id": it["id"],
            "fact": it["text"],
            "keywords": it["keywords"],
            "people": it["people"],
        }
        for it in candidates
    ]

    llm = get_llm() if use_llm else None
    verifier = FactVerifier(embed_fn=embedder.embed, llm=llm, config=VerifyConfig())
    new_fact = {"fact": fact, "keywords": new_kw, "people": people or []}
    decision = verifier.verify(new_fact, existing)

    conflict_with = None
    if decision.get("conflict_fact"):
        cf = decision["conflict_fact"]
        name = next(
            (it.get("name") for it in candidates if it["id"] == cf.get("id")), ""
        )
        conflict_with = {"id": cf.get("id"), "name": name}

    return {
        "action": decision["action"],
        "reason": decision["reason"],
        "store_text": decision["store_text"],
        "is_residual": decision["is_residual"],
        "conflict_with": conflict_with,
        "candidates_considered": len(existing),
    }


def register_atommem_tools(app, db_path: str) -> None:
    # Ensure the temporal_profiles table exists (isolated from core schema).
    TemporalProfileStore(db_path=db_path)

    # ---- Δ1: atomic fact extraction --------------------------------------- #
    @app.tool()
    def extract_atomic_facts(
        text: str,
        session_time: str = "",
        speaker: str = "",
        with_metadata: bool = True,
    ) -> Dict[str, Any]:
        """Extract objective, self-contained, coreference-resolved, time-anchored
        atomic facts from raw dialogue/text (AtomMem Δ1, via headless CLI).

        Args:
            text: Raw dialogue or note to decompose into atomic facts.
            session_time: Conversation date (YYYY-MM-DD) used to anchor relative
                times like "yesterday". Optional but improves time accuracy.
            speaker: Name of the speaker, used for coreference (e.g. "I" -> name).
            with_metadata: Also extract people/keywords/time/profile-flag per fact.

        Returns:
            {"facts": [{fact, people, keywords, time, needs_profile, _extracted}],
             "count": int, "llm_available": bool, "providers": [str],
             "degraded": bool, "degrade_reason": dict|None}
            "degraded" is True when the LLM path failed and the result is the
            deterministic passthrough (each fact has _extracted False);
            "degrade_reason" names the provider error. Both are False/None when
            extraction ran for real.
        """
        extractor = AtomicFactExtractor()
        facts = extractor.extract_structured(
            dialogue=text,
            session_time=session_time,
            speaker=speaker,
            with_metadata=with_metadata,
        )
        degraded = bool(facts) and all(not f.get("_extracted") for f in facts)
        return {
            "facts": facts,
            "count": len(facts),
            "llm_available": extractor.llm.available(),
            "providers": extractor.llm.available_providers(),
            "degraded": degraded,
            "degrade_reason": extractor.last_extract_error if degraded else None,
        }

    # ---- Δ2: IDF-weighted keyword graph recall ---------------------------- #
    @app.tool()
    def atommem_graph_recall(
        query: str,
        max_entities: int = _DEFAULT_ENTITY_SCAN,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """IDF-weighted keyword-graph recall (AtomMem Δ2): seed by query keywords,
        run Personalized PageRank over IDF-weighted entity-keyword edges, return
        associatively-related entities (surfaces connections a flat keyword match
        misses). Read-only over the newest `max_entities` entities.

        Returns:
            {"results": [{id, name, entity_type, graph_score, keywords}],
             "query_keywords": [...], "entities_scanned": int}
        """
        items = _load_entity_items(db_path, max_entities)
        graph = IDFKeywordGraph(items)
        q_keywords = extract_keywords(query, max_keywords=8)
        recalled = graph.graph_recall(query_keywords=q_keywords, top_k=top_k)
        results = [
            {
                "id": r["id"],
                "name": r.get("name", ""),
                "entity_type": r.get("entity_type"),
                "graph_score": r.get("graph_score"),
                "keywords": r.get("keywords", []),
            }
            for r in recalled
        ]
        return {
            "results": results,
            "query_keywords": q_keywords,
            "entities_scanned": len(items),
        }

    @app.tool()
    def atommem_keyword_neighbors(
        entity_name: str,
        max_entities: int = _DEFAULT_ENTITY_SCAN,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """Top IDF-weighted keyword neighbors of a named entity (AtomMem Δ2 edges).

        Returns:
            {"entity": str, "neighbors": [{id, name, edge_weight, keywords}]}
            or {"error": ...} if the entity is not in the scanned window.
        """
        items = _load_entity_items(db_path, max_entities)
        target = next(
            (
                it
                for it in items
                if (it.get("name") or "").lower() == entity_name.lower()
            ),
            None,
        )
        if target is None:
            return {
                "error": f"entity '{entity_name}' not found in newest {len(items)} entities"
            }
        graph = IDFKeywordGraph(items)
        neighbors = graph.neighbors(target["id"], top_k=top_k)
        return {
            "entity": entity_name,
            "neighbors": [
                {
                    "id": n["id"],
                    "name": n.get("name", ""),
                    "edge_weight": n.get("edge_weight"),
                    "keywords": n.get("keywords", []),
                }
                for n in neighbors
            ],
        }

    # ---- Δ3: residual-delta verification ---------------------------------- #
    @app.tool()
    def verify_fact_before_store(
        fact: str,
        max_candidates: int = 60,
        use_llm: bool = True,
        people: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Decide whether a new fact should be stored, ignored as a duplicate,
        treated as a conflict, or trimmed to its residual novel content
        (AtomMem Δ3). Uses real embeddings (all-MiniLM-L6-v2) + headless-CLI LLM.

        Returns:
            {"action": "CREATE"|"IGNORE"|"CONFLICT_RESOLVED", "reason": str,
             "store_text": str, "is_residual": bool,
             "conflict_with": {id, name}|None, "candidates_considered": int}
        """
        return decide_fact(db_path, fact, max_candidates, use_llm, people)

    # ---- Δ4: temporal profile --------------------------------------------- #
    @app.tool()
    def upsert_temporal_profile(
        subject: str,
        content: str,
        valid_from: str = "",
        keywords: Optional[List[str]] = None,
        evidence: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Merge a stable attribute observation into a subject's versioned
        timeline (AtomMem Δ4). State changes archive the prior state with a
        valid-time interval rather than overwriting it.

        Args:
            subject: Person/entity the attribute is about (e.g. "Caroline").
            content: The attribute statement (e.g. "Caroline lives in Seattle.").
            valid_from: When this state began (YYYY-MM-DD / YYYY-MM / YYYY).
            keywords: Optional retrieval keywords; derived from content if omitted.
            evidence: Optional supporting fact/entity ids.

        Returns: {"action": "new"|"confirm"|"update_current"|"update_history",
                  "profile_id": str}
        """
        store = TemporalProfileStore(db_path=db_path)
        kw = keywords or extract_keywords(content, max_keywords=5)
        return store.upsert(
            subject=subject,
            content=content,
            valid_from=valid_from,
            keywords=kw,
            evidence=evidence or [],
        )

    @app.tool()
    def query_temporal_profile(
        subject: str = "",
        query_time: str = "",
        keywords: Optional[List[str]] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Retrieve a subject's profile at a point in time (AtomMem Δ4). With
        query_time set, returns the version valid then (e.g. where someone lived
        in 2020), otherwise the current state.

        Returns: {"profiles": [{profile_id, subject, content, valid_from,
                  valid_to?, profile_version_id?, keywords, evidence}]}
        """
        store = TemporalProfileStore(db_path=db_path)
        profiles = store.query(
            subject=subject or None,
            query_time=query_time,
            keywords=keywords,
            top_k=top_k,
        )
        return {"profiles": profiles, "count": len(profiles)}

    # ---- diagnostics ------------------------------------------------------ #
    @app.tool()
    def atommem_status() -> Dict[str, Any]:
        """Report AtomMem-upgrade subsystem health: LLM providers, embedder, and
        temporal-profile count."""
        from . import embedder

        llm = get_llm()
        store = TemporalProfileStore(db_path=db_path)
        with store._conn() as conn:
            n_profiles = conn.execute(
                "SELECT COUNT(*) AS n FROM temporal_profiles"
            ).fetchone()["n"]
        return {
            "llm_available": llm.available(),
            "llm_providers": llm.available_providers(),
            "embedder_available": embedder.available(),
            "embedder_dim": embedder.dimension(),
            "temporal_profiles": n_profiles,
            "deltas": {
                "1_atomic_facts": "extract_atomic_facts",
                "2_idf_graph": "atommem_graph_recall / atommem_keyword_neighbors",
                "3_residual_verify": "verify_fact_before_store",
                "4_temporal_profile": "upsert_temporal_profile / query_temporal_profile",
            },
        }
