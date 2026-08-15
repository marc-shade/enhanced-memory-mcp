#!/usr/bin/env python3
"""
Hybrid search module (BM25 + Vector) for enhanced-memory-mcp.

Combines sparse (BM25) and dense (vector) search for improved recall.

Part of RAG Tier 1 Strategy - Week 1, Day 3-4
Expected improvement: +20-30% recall with minimal latency overhead
"""

import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector

logger = logging.getLogger(__name__)


class HybridSearcher:
    """
    Hybrid search combining BM25 (sparse) and vector (dense) search.

    Uses Qdrant's native hybrid search with Reciprocal Rank Fusion (RRF)
    for combining results from both search methods.
    """

    def __init__(self, qdrant_url: str = "http://localhost:6333"):
        """
        Initialize hybrid searcher.

        Args:
            qdrant_url: Qdrant server URL
        """
        self.qdrant_url = qdrant_url
        self.client = None
        self._client_initialized = False

        logger.info(f"HybridSearcher initialized with Qdrant at: {qdrant_url}")

    def _ensure_client_initialized(self):
        """Lazy initialize Qdrant client."""
        if not self._client_initialized:
            logger.info("Initializing Qdrant client for hybrid search")
            self.client = QdrantClient(url=self.qdrant_url)
            self._client_initialized = True
            logger.info("Qdrant client initialized")

    def _create_sparse_vector_from_text(self, text: str) -> SparseVector:
        """
        Create sparse vector from text using simple term frequency.

        Note: This is a simplified BM25 implementation. For production,
        use a proper BM25 tokenizer and scorer.

        Args:
            text: Input text

        Returns:
            SparseVector for Qdrant
        """
        # Simple tokenization (word-based)
        words = text.lower().split()

        # Count term frequencies
        term_freq = {}
        for word in words:
            term_freq[word] = term_freq.get(word, 0) + 1

        # Convert to sparse vector format
        # Map words to indices (simple hash-based)
        indices = []
        values = []

        for word, freq in term_freq.items():
            # Use hash of word as index (mod to keep reasonable range)
            idx = hash(word) % 100000
            indices.append(idx)
            values.append(float(freq))

        return SparseVector(indices=indices, values=values)

    async def hybrid_search(
        self,
        collection_name: str,
        query_text: str,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Dense vector search. NOT hybrid, despite the name.

        The production collection carries no sparse vector index, so there is
        nothing lexical to combine with; query_text is therefore unused. See
        the comment in the body for the paths that do real fusion.

        Args:
            collection_name: Qdrant collection name
            query_text: Query text for BM25 sparse search
            query_vector: Query embedding for dense vector search
            limit: Number of results to return
            score_threshold: Minimum score threshold

        Returns:
            List of search results with scores
        """
        self._ensure_client_initialized()

        try:
            # NOT hybrid. This runs dense-only, and has since it was written:
            # the production `enhanced_memory` collection has sparse_vectors
            # null, so there is no sparse index to query. Until 2026-07-19 this
            # method built a sparse vector, discarded it unused, ran a dense
            # search, and labelled the output search_type="hybrid" -- so callers
            # could not tell. The useless call is gone and the label now tells
            # the truth.
            #
            # For real lexical+dense fusion today, use
            # MemoryDatabaseV2.search_nodes, which RRF-fuses the name ranking
            # with BM25 over observations_fts. Making THIS path genuinely hybrid
            # requires re-indexing the collection with named sparse vectors;
            # see hybrid_search_tools_nmf.py, which implements Qdrant-native RRF
            # but is currently dead against production because it queries a
            # vector named "text-dense" that the live collection does not have.
            logger.warning(
                "search_hybrid on %s is running DENSE-ONLY: collection has no "
                "sparse vector index. Results are not hybrid.",
                collection_name,
            )
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,  # Dense vector query
                limit=limit,
                score_threshold=score_threshold,
            )

            # Format results
            formatted_results = []
            for hit in results:
                result = {
                    "id": hit.id,
                    "score": hit.score,
                    "payload": hit.payload,
                    "metadata": {
                        # Was "hybrid". It is not, and a caller trusting this
                        # field to tell whether lexical matching happened was
                        # being handed a wrong answer.
                        "search_type": "dense_only_no_sparse_index",
                        "collection": collection_name,
                    },
                }
                formatted_results.append(result)

            logger.debug(f"Hybrid search returned {len(formatted_results)} results")

            return formatted_results

        except Exception as e:
            logger.error(f"Error in hybrid search: {str(e)}")
            # Fallback to vector-only search
            logger.warning("Falling back to vector-only search")
            return await self.vector_only_search(
                collection_name, query_vector, limit, score_threshold
            )

    async def vector_only_search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fallback to vector-only search if hybrid fails.

        Args:
            collection_name: Qdrant collection name
            query_vector: Query embedding
            limit: Number of results
            score_threshold: Minimum score threshold

        Returns:
            List of search results
        """
        self._ensure_client_initialized()

        try:
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
            )

            formatted_results = []
            for hit in results:
                result = {
                    "id": hit.id,
                    "score": hit.score,
                    "payload": hit.payload,
                    "metadata": {
                        "search_type": "vector_only",
                        "collection": collection_name,
                    },
                }
                formatted_results.append(result)

            return formatted_results

        except Exception as e:
            logger.error(f"Error in vector search: {str(e)}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """Get hybrid searcher statistics."""
        return {
            "qdrant_url": self.qdrant_url,
            "client_initialized": self._client_initialized,
            "hybrid_search_enabled": True,
        }


# Global hybrid searcher instance
_hybrid_searcher = None


def get_hybrid_searcher() -> HybridSearcher:
    """Get or create global hybrid searcher instance."""
    global _hybrid_searcher
    if _hybrid_searcher is None:
        _hybrid_searcher = HybridSearcher()
    return _hybrid_searcher
