"""AtomMem-derived memory upgrades for the enhanced-memory MCP.

Ports four mechanisms from "AtomMem: Building Simple and Effective Memory
System for LLM Agents via Atomic Facts" (Yao et al., arXiv 2606.19847) into our
4-tier memory system. Each module is self-contained and additive; nothing here
mutates existing tool behaviour unless explicitly wired in server.py.

Modules
-------
idf_keyword_graph : Delta 2. IDF-weighted keyword-overlap edges + Personalized
                    PageRank over the entity graph (paper Eq. 2/3, RWR recall).
atomic_facts      : Delta 1. Headless-CLI extraction/normalization of raw text
                    into objective, coreference-resolved, time-anchored facts.
fact_verification : Delta 3. Residual-delta write path: embedding dedup gate +
                    hybrid-similarity conflict gate + LLM logical-conflict check.
temporal_profile  : Delta 4. Versioned per-subject profiles with valid-time
                    intervals and point-in-time query selection.

LLM calls in this package go through the project headless-CLI convention
(claude --print / codex exec / gemini), never a provider SDK, per
rules/intent-engineering.md.
"""

from .idf_keyword_graph import (
    IDFKeywordGraph,
    hybrid_similarity,
    jaccard_similarity,
)

__all__ = [
    "IDFKeywordGraph",
    "hybrid_similarity",
    "jaccard_similarity",
]
