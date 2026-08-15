"""Deterministic keyword extraction.

Our entities/observations have no keyword field, so the IDF keyword graph
(Delta 2) needs a keyword source for items that lack atomic-fact metadata. This
is a stopword-filtered token extractor (no LLM, no deps) modelled on the
existing agentic_rag_tools._extract_keywords pattern. The LLM-based extractor in
atomic_facts.py produces better keywords; this is the always-available fallback.
"""

from __future__ import annotations

import re
from typing import List

# Compact English stopword list (sufficient for retrieval keyword filtering).
_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "if",
    "then",
    "else",
    "when",
    "while",
    "for",
    "to",
    "of",
    "in",
    "on",
    "at",
    "by",
    "with",
    "from",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "shall",
    "should",
    "can",
    "could",
    "may",
    "might",
    "must",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "he",
    "she",
    "they",
    "them",
    "his",
    "her",
    "their",
    "i",
    "you",
    "we",
    "us",
    "me",
    "my",
    "your",
    "our",
    "him",
    "what",
    "which",
    "who",
    "whom",
    "whose",
    "how",
    "why",
    "where",
    "there",
    "here",
    "all",
    "any",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "just",
    "about",
    "into",
    "over",
    "after",
    "before",
    "up",
    "down",
    "out",
    "off",
    "again",
    "once",
    "also",
    "got",
    "get",
    "getting",
    "really",
    "user",
    "assistant",
    "system",
    "okay",
    "yeah",
    "hey",
    "hi",
    "hello",
    "thanks",
    "thank",
    "please",
    "well",
    "like",
    "going",
    "go",
    "went",
    "said",
    "say",
}

_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9'\-]+")


def _singularize(word: str) -> str:
    """Cheap pluralization stripping (ies->y, es->'', s->'')."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("ses"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def extract_keywords(
    text: str, max_keywords: int = 5, singularize: bool = True
) -> List[str]:
    """Extract up to max_keywords retrieval keywords from text.

    Lowercased, stopword-filtered, length>2, de-duplicated, ranked by frequency
    then first-appearance order. Mirrors AtomMem's "no more than 5, noun-ish,
    singular" guidance from fact_metadata_extraction_prompt.txt.
    """
    if not text:
        return []
    counts: dict[str, int] = {}
    order: dict[str, int] = {}
    idx = 0
    for raw in _TOKEN_RE.findall(text.lower()):
        if raw in _STOPWORDS or len(raw) <= 2:
            continue
        token = _singularize(raw) if singularize else raw
        if len(token) <= 2 or token in _STOPWORDS:
            continue
        counts[token] = counts.get(token, 0) + 1
        if token not in order:
            order[token] = idx
            idx += 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], order[kv[0]]))
    return [w for w, _ in ranked[:max_keywords]]


if __name__ == "__main__":
    samples = [
        "Emma got an A on her first psychology exam last Friday.",
        "Zoe loves plants and has a big indoor plant collection with cacti.",
        "Hey thanks so much, really appreciate it!",
    ]
    for s in samples:
        print(f"{s!r}\n  -> {extract_keywords(s)}")
