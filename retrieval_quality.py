"""Retrieval-quality helpers shared by the retrieval backends.

Phase G (2026-08-05): give every retrieval path a `confidence` and a
`low_confidence` flag so quiet misses are visible. All backends (keyword/FTS
search_nodes, vector semantic_recall, graph_enhanced_search) report the same
shape; the threshold lives in one place.

The threshold is calibrated for embeddinggemma 768d (local_semantic_recall):
real matches run >= 0.55, nonsense queries sit ~0.46. Override via
MEMORY_LOW_CONF_THRESHOLD.
"""

from local_semantic_recall import LOW_CONFIDENCE_THRESHOLD

VECTOR_LOW_CONFIDENCE_THRESHOLD = LOW_CONFIDENCE_THRESHOLD
GRAPH_LOW_CONFIDENCE_THRESHOLD = 0.45  # combined_score is weighted vector+graph


def vector_low_confidence(top_score: float) -> bool:
    """True when the top vector/combined cosine is below the calibrated floor."""
    return top_score < VECTOR_LOW_CONFIDENCE_THRESHOLD


def keyword_confidence(fts_rank: float) -> float:
    """Per-result confidence for search_nodes.

    A non-zero FTS5 bm25 rank (negative, better) means the query matched the
    observation text (content match); rank 0 means a name-substring-only hit,
    which is a weaker signal.
    """
    if fts_rank:
        return round(min(0.95, max(0.55, 0.8 - fts_rank * 0.001)), 3)
    return 0.5


def keyword_low_confidence(results: list) -> bool:
    """True when no results, or the top result is a name-only (0.5) hit."""
    if not results:
        return True
    return results[0].get("confidence", 0.0) <= 0.5
