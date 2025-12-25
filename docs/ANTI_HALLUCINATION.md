# Anti-Hallucination System

The Anti-Hallucination system detects and prevents AI fabrications through pattern detection, citation validation, and provenance tracking.

## Overview

AI models can hallucinate - producing confident but false information. This system provides multiple layers of protection:

1. **Hallucination Detection** - Pattern matching for common hallucination indicators
2. **Citation Validation** - Verify sources and references
3. **L-Score Provenance** - Track knowledge trustworthiness through derivation chains
4. **Shadow Vector Search** - Find contradicting evidence
5. **Confidence Monitoring** - Real-time confidence tracking

## Hallucination Detection

Detect common hallucination patterns in text.

```python
# Detect hallucinations in text
result = detect_hallucinations(
    text="I believe the study from 2024 showed a 47.3% improvement...",
    context={"topic": "performance optimization"}
)

# Returns detected patterns with severity
# {
#   "hallucinations": [
#     {
#       "pattern": "vague_certainty",
#       "text": "I believe",
#       "severity": "medium",
#       "suggestion": "State facts directly or cite sources"
#     },
#     {
#       "pattern": "over_specific_stats",
#       "text": "47.3%",
#       "severity": "high",
#       "suggestion": "Verify exact statistics with source"
#     }
#   ]
# }
```

### Detection Patterns

| Pattern | Indicators | Severity |
|---------|-----------|----------|
| `vague_certainty` | "I believe", "I think", "typically" | Medium |
| `unsupported_stats` | Precise percentages without source | High |
| `fabricated_refs` | Non-existent papers, URLs | Critical |
| `over_specific` | Unlikely precision ("exactly 47.3%") | High |
| `confident_uncertainty` | "Definitely maybe", contradictions | Medium |

## Citation Validation

Verify the quality and trustworthiness of citations.

```python
# Validate a list of citations
result = validate_citations([
    {"source": "arxiv.org", "title": "Paper Title", "year": 2024},
    {"source": "blog.random.com", "title": "Tutorial", "year": 2023}
])

# Returns validation results
# {
#   "valid_citations": 1,
#   "issues": [
#     {"citation": 1, "issue": "untrusted_source", "source": "blog.random.com"}
#   ],
#   "recommendations": ["Prefer peer-reviewed sources"]
# }

# Add trusted sources
add_trusted_source("nature.com")
add_trusted_source("ieee.org")
```

## L-Score Provenance Tracking

Track the trustworthiness of derived knowledge through the derivation chain.

### L-Score Formula

```
L-Score = geometric_mean(confidence) × average(relevance) / depth_factor
```

- **Confidence**: How confident are we in each source? (0.0-1.0)
- **Relevance**: How relevant is each source? (0.0-1.0)
- **Depth**: How many derivation hops from original source?

### L-Score Thresholds

| Score | Quality | Action |
|-------|---------|--------|
| ≥ 0.7 | High | Accept as trustworthy |
| 0.3 - 0.7 | Medium | Review recommended |
| < 0.3 | Low | Reject or verify manually |

### Usage

```python
# Create entity with provenance
create_entity_with_provenance(
    entity_id=123,
    source_ids=[456, 789],  # Source entities this derives from
    confidence=0.85,
    relevance=0.9,
    derivation_method="inference"  # inference, extraction, synthesis, citation
)

# Get full provenance chain
chain = get_provenance_chain(
    entity_id=123,
    max_depth=5
)

# Validate meets threshold
result = validate_l_score(
    entity_id=123,
    threshold=0.5
)

# Preview L-Score calculation
preview = calculate_l_score_preview(
    confidence_scores=[0.9, 0.85, 0.8],
    relevance_scores=[0.95, 0.9, 0.85],
    depth=2
)
```

### Finding High/Low Provenance Entities

```python
# Get well-sourced knowledge
high_quality = get_high_provenance_entities(
    min_l_score=0.7,
    limit=50
)

# Get entities needing review
needs_review = get_low_provenance_entities(
    max_l_score=0.3,
    limit=50
)
```

## Shadow Vector Search

Find contradicting evidence using inverted embeddings.

```python
# Regular search finds supporting evidence
# Shadow search (inverted embedding) finds contradictions

result = validate_claim(
    claim_text="Feature X improves performance by 50%",
    claim_embedding=embedding_vector,  # 768-dim embedding
    support_threshold=0.7,
    contradict_threshold=0.6
)

# Returns validation report
# {
#   "credibility_score": 0.65,
#   "recommendation": "REVIEW",
#   "supporting_evidence": [...],
#   "contradicting_evidence": [...],
#   "reasoning": "Mixed evidence found"
# }

# Find direct contradictions
contradictions = find_contradictions(
    claim_embedding=embedding_vector,
    threshold=0.6,
    limit=10
)
```

### Credibility Calculation

```python
# Calculate credibility from pre-gathered evidence
result = calculate_claim_credibility(
    supporting_count=5,
    contradicting_count=2,
    supporting_l_scores=[0.8, 0.75, 0.9, 0.7, 0.85],
    contradicting_l_scores=[0.6, 0.5],
    supporting_similarities=[0.9, 0.85, 0.88, 0.82, 0.87],
    contradicting_similarities=[0.75, 0.7]
)
```

## Confidence Monitoring

Real-time monitoring of analysis confidence.

```python
# Monitor confidence levels
result = monitor_confidence({
    "claims": ["Claim 1", "Claim 2"],
    "citations": [{"source": "arxiv.org", "title": "..."}],
    "confidence": 0.75
})

# Returns metrics and issues
# {
#   "overall_confidence": 0.75,
#   "claim_count": 2,
#   "citation_quality": 0.9,
#   "issues": []
# }
```

## Claim Verification

Full claim verification workflow.

```python
# Verify a specific claim
result = verify_claim(
    claim="The system processes 1M requests per second",
    citations=[
        {"source": "benchmark.md", "line": 42, "text": "Throughput: 1.2M req/s"}
    ],
    context={"component": "api_gateway"}
)

# Returns verification result
# {
#   "verified": True,
#   "confidence": 0.85,
#   "issues": [],
#   "supporting_citations": [...]
# }
```

## Get System Status

```python
# Check anti-hallucination system status
status = anti_hallucination_status()

# Returns configuration and statistics
# {
#   "enabled": True,
#   "trusted_sources_count": 15,
#   "validation_history_count": 234,
#   "average_credibility": 0.72
# }

# Get validation history
history = get_validation_history(
    limit=50,
    min_credibility=0.5,
    recommendation_filter="accept"
)
```

## Best Practices

1. **Always verify claims** with citations before storing in memory
2. **Track provenance** for all derived knowledge
3. **Set appropriate L-Score thresholds** for your use case
4. **Use shadow search** for critical claims
5. **Monitor confidence levels** during complex analyses
6. **Add trusted sources** for your domain

## Related Documentation

- [Holographic Memory](HOLOGRAPHIC_MEMORY.md) - Spreading activation
- [Continuous Learning](CONTINUOUS_LEARNING.md) - Learning from corrections
- [Advanced Features](ADVANCED_FEATURES.md) - Causal inference
