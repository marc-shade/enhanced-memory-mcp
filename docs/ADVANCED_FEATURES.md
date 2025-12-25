# Advanced Features

This document covers additional advanced features of the Enhanced Memory MCP server.

## Table of Contents

- [Causal Inference Engine](#causal-inference-engine)
- [Strange Loops Detector](#strange-loops-detector)
- [ReasoningBank](#reasoningbank)
- [Semantic Cache](#semantic-cache)
- [FACT Cache](#fact-cache)
- [Surprise-Based Consolidation](#surprise-based-consolidation)
- [Model Router](#model-router)
- [Hybrid Search](#hybrid-search)
- [Multi-Query RAG](#multi-query-rag)
- [ART (Adaptive Resonance Theory)](#art-adaptive-resonance-theory)

---

## Causal Inference Engine

Validate causal models and identify bias threats in analyses.

### Model Validation

```python
# Validate a causal model
result = ci_validate_causal_model(
    variables=[
        {"name": "treatment", "type": "treatment", "observed": True},
        {"name": "outcome", "type": "outcome", "observed": True},
        {"name": "age", "type": "covariate", "observed": True}
    ],
    relationships=[
        {"from_var": "treatment", "to_var": "outcome", "type": "direct"},
        {"from_var": "age", "to_var": "outcome", "type": "direct"}
    ],
    data={
        "observations": 1000,
        "randomized": False,
        "missingRate": 0.05
    },
    confounders=["age"]
)
```

### Bias Threat Identification

```python
# Identify potential biases
threats = ci_identify_bias_threats(
    variables=[...],
    relationships=[...],
    data={"randomized": False, "missingRate": 0.15}
)

# Returns detected threats
# {
#   "threats": [
#     {"type": "selection_bias", "severity": "high", "mitigation": "..."},
#     {"type": "missing_data_bias", "severity": "medium", "mitigation": "..."}
#   ]
# }
```

### Statistical Testing

```python
# Significance testing
result = ci_significance_test(
    hypothesis="Treatment improves outcome",
    test_type="t-test",  # t-test, chi-square, anova, regression
    test_statistic=2.45,
    p_value=0.014,
    sample_size=500,
    effect_size=0.3
)

# Power analysis
power = ci_power_analysis(
    sample_size=200,
    effect_size=0.5,
    alpha=0.05
)

# Effect estimation
effect = ci_estimate_effect(
    effect_estimate=0.35,
    standard_error=0.12,
    confidence_level=0.95
)
```

---

## Strange Loops Detector

Detect circular reasoning, contradictions, and recursive patterns.

### Circular Reasoning Detection

```python
# Detect cycles in reasoning graph
result = sl_detect_circular(
    graph={
        "A": ["B"],
        "B": ["C"],
        "C": ["A"]  # Creates cycle A→B→C→A
    }
)

# Returns detected cycles with severity
```

### Contradiction Detection

```python
# Detect contradictions between statements
result = sl_detect_contradictions([
    "The system always restarts at midnight",
    "The system never restarts automatically"
])

# Returns contradiction patterns
# {
#   "contradictions": [
#     {"statements": [0, 1], "type": "always_never", "severity": "high"}
#   ]
# }
```

### Causal Chain Validation

```python
# Validate a causal chain
result = sl_validate_chain(
    chain_id="chain_001",
    nodes=[
        {"id": "A", "statement": "Rain falls", "confidence": 0.9},
        {"id": "B", "statement": "Ground is wet", "confidence": 0.95}
    ],
    edges=[
        {"from": "A", "to": "B", "strength": 0.8, "type": "causes"}
    ]
)
```

### Recursive Pattern Detection

```python
# Detect self-references in structures
result = sl_detect_recursive({
    "name": "root",
    "children": [
        {"name": "child1"},
        {"name": "child2", "parent": "root"}  # Potential reference
    ]
})
```

---

## ReasoningBank

Store and retrieve reasoning experiences for learning.

```python
# Retrieve relevant past reasoning
results = rb_retrieve(
    query="authentication implementation",
    domain="security",
    k=5
)

# Learn from task outcome
rb_learn(
    task_id="task_123",
    query="implement JWT auth",
    outcome="success",  # success, failure, partial
    trajectory="1. Analyzed requirements 2. Chose JWT 3. Implemented...",
    domain="security"
)

# Consolidate learnings (deduplicate, detect contradictions)
rb_consolidate()

# Get status
status = rb_status()
```

---

## Semantic Cache

Cache expensive computations based on semantic similarity.

### Basic Usage

```python
# Check cache before expensive operation
cache_result = semantic_cache_get(
    query="How do I implement authentication?",
    context="Python FastAPI project"
)

if cache_result["hit"]:
    response = cache_result["response"]
else:
    # Compute expensive operation
    response = expensive_llm_call(query)

    # Store result
    semantic_cache_store(
        query="How do I implement authentication?",
        response=response,
        context="Python FastAPI project"
    )
```

### AGI-Optimized Caching

Domain-specific caching with optimized thresholds:

```python
# Check AGI cache (domain-specific thresholds)
result = agi_cached_reasoning(
    query="Optimal strategy for memory consolidation",
    domain="reasoning",  # reasoning, consolidation, research, api_calls, embeddings
    context="performance optimization"
)

# Store result
agi_cache_store_result(
    query="Optimal strategy for memory consolidation",
    response={"strategy": "incremental", "rationale": "..."},
    domain="reasoning"
)

# Get metrics
metrics = agi_cache_metrics(domain="reasoning")
```

### Cache Management

```python
# Get cache statistics
stats = semantic_cache_stats()

# Search cache
results = semantic_cache_search(query="authentication", top_k=5)

# Clean up expired entries
semantic_cache_cleanup(force=False)

# Check availability
available = semantic_cache_available()
```

---

## FACT Cache

Fast cache-first retrieval for memory searches.

```python
# FACT-accelerated search (<48ms on hit, <140ms on miss)
results = fact_search(query="optimization patterns", limit=10)

# Get cache status
status = fact_cache_status()

# Clear expired entries
fact_cache_clear_expired()

# Warm cache with common queries
fact_warm_cache(
    queries=["authentication", "performance", "error handling"],
    limit=10
)
```

### Unified Search API

```python
# Intelligent routing between cache and Qdrant
results = unified_search(
    query="memory patterns",
    limit=10,
    backend="fact_cache"  # fact_cache, qdrant, hybrid, semantic
)

# Get metrics
metrics = unified_search_metrics()

# Warm cache
unified_search_warm(queries=["common query 1", "common query 2"])
```

---

## Surprise-Based Consolidation

Titans/MIRAS-inspired consolidation using surprise scores.

### Surprise Scoring

```python
# Calculate surprise for content
score = calculate_surprise_score(
    content="Discovered that parallel tool calls reduce latency by 60%",
    memory_type="semantic",
    context="Recent conversation about optimization"
)

# Returns:
# {
#   "surprise_score": 0.78,
#   "components": {"novelty": 0.8, "salience": 0.7, "temporal": 0.85},
#   "should_store": True,
#   "recommendation": "High novelty - promote to semantic memory"
# }
```

### Consolidation

```python
# Run surprise-based consolidation
result = run_surprise_consolidation(
    time_window_hours=24,
    min_surprise_threshold=0.4
)

# Returns:
# {
#   "memories_evaluated": 150,
#   "memories_promoted": 23,
#   "memories_skipped": 112,
#   "memories_forgotten": 15,
#   "average_surprise_score": 0.45
# }

# Get consolidation stats
stats = get_surprise_consolidation_stats(days=7)
```

### Retention Management

```python
# Get candidates for forgetting
candidates = get_retention_candidates(count_to_remove=10)
```

---

## Model Router

Intelligent routing to different LLM providers.

```python
# Chat with automatic routing
response = router_chat(
    model="claude-3.5-sonnet",  # or "gpt-4o", "auto"
    messages=[{"role": "user", "content": "Complex reasoning task"}],
    temperature=0.7,
    agent_type="researcher"  # Influences routing
)

# Preview provider selection
selection = router_select_provider(
    model="auto",
    agent_type="coder",
    requires_tools=True
)

# Get metrics
metrics = router_metrics()

# Get status
status = router_status()

# Change mode
router_set_mode("cost-optimized")  # manual, rule-based, cost-optimized, performance-optimized

# Add routing rule
router_add_rule(
    provider="anthropic",
    model="claude-3.5-sonnet",
    agent_types=["researcher", "analyst"],
    requires_tools=True
)

# Get uncertainty estimation
uncertainty = router_get_uncertainty()
router_estimate_uncertainty(prediction=0.75)
```

---

## Hybrid Search

Combine BM25 lexical search with vector semantic search.

```python
# Hybrid search with RRF fusion
results = search_hybrid(
    query="authentication best practices",
    limit=10,
    score_threshold=0.5
)

# Get hybrid search stats
stats = get_hybrid_search_stats()
```

### Cross-Encoder Reranking

```python
# Search with reranking (+40-55% precision)
results = search_with_reranking(
    query="memory optimization techniques",
    limit=10,
    over_retrieve_factor=4  # Retrieve 4x, rerank to top 10
)

# Get reranking stats
stats = get_reranking_stats()
```

### Query Expansion

```python
# Search with query expansion (+15-25% recall)
results = search_with_query_expansion(
    query="auth",
    max_expansions=3,
    strategies=["llm", "synonym", "concept"],
    limit=10
)

# Get expansion stats
stats = get_query_expansion_stats()
```

---

## Multi-Query RAG

Generate multiple query perspectives for comprehensive retrieval.

```python
# Search with multiple perspectives
results = search_with_multi_query(
    query="How do I optimize database queries?",
    perspective_types=["technical", "user", "conceptual"],
    max_perspectives=3,
    limit=10
)

# Analyze perspectives without searching
analysis = analyze_query_perspectives(
    query="database optimization",
    max_perspectives=3
)

# Get stats
stats = get_multi_query_stats()
```

---

## ART (Adaptive Resonance Theory)

Online learning without catastrophic forgetting.

### Basic ART

```python
# Learn a pattern
result = art_learn(
    data=[0.1, 0.5, 0.3, 0.8, ...],  # Feature vector
    metadata={"source": "observation_1"},
    vigilance=0.75  # Category granularity
)

# Classify without learning
category = art_classify(
    data=[0.12, 0.48, 0.31, 0.79, ...],
    vigilance=0.75
)

# Adjust vigilance (THE KEY DIAL)
art_adjust_vigilance(
    vigilance=0.8,  # 0.9+ = fine-grained, 0.5 = coarse
    instance="main"
)

# Get categories
categories = art_get_categories()

# Get stats
stats = art_get_stats()

# Reset (warning: deletes learned knowledge)
art_reset(instance="main", confirm=True)
```

### Hybrid ART (with embeddings)

```python
# Learn from embedding
result = art_hybrid_learn(
    embedding=[0.1, 0.2, ...],  # Pre-computed embedding
    content="Original text content",
    metadata={"type": "concept"}
)

# Find similar categories
similar = art_hybrid_find_similar(
    embedding=[0.1, 0.2, ...],
    top_k=5
)
```

---

## Contextual Retrieval

Generate contextual prefixes for better RAG retrieval.

```python
# Generate context for a chunk
context = generate_context_for_chunk(
    chunk="The function returns early if validation fails.",
    document="Full document text...",
    metadata={"title": "API Documentation"},
    max_words=100
)

# Re-index all entities with context
result = reindex_with_context(
    batch_size=10,
    max_workers=10,
    resume=True
)

# Check progress
progress = get_reindexing_progress()

# Get stats
stats = get_contextual_retrieval_stats()
```

---

## Related Documentation

- [Holographic Memory](HOLOGRAPHIC_MEMORY.md) - Spreading activation and associations
- [Anti-Hallucination](ANTI_HALLUCINATION.md) - Truth verification and L-Score
- [Continuous Learning](CONTINUOUS_LEARNING.md) - EWC++ and meta-cognition
