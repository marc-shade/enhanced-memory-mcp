# Holographic Memory System

The Holographic Memory system provides spreading activation across semantically related memories, enabling more intelligent retrieval and learning.

## Overview

Traditional memory systems retrieve exact matches. Holographic Memory uses **spreading activation** - when you access one memory, related memories automatically activate based on semantic similarity.

```
Access "authentication" →
  ├─ "JWT tokens" activates (0.8)
  ├─ "OAuth2" activates (0.7)
  ├─ "session management" activates (0.6)
  └─ "password hashing" activates (0.5)
```

## Four Phases

### Phase 1: Activation Field

Semantic spreading based on embedding similarity. When a memory is accessed, activation spreads to related concepts.

```python
# Spread activation from a source entity
result = spread_activation(
    source_entity_id=123,
    initial_activation=1.0,
    max_hops=3,
    activation_threshold=0.3
)

# Returns activated memories with activation levels
# [{"entity_id": 456, "activation": 0.75, "name": "related_concept"}, ...]
```

### Phase 2: Memory-Influenced Routing

Memory informs which approach, model, or strategy to use. The system learns from past decisions.

```python
# Router uses memory to select optimal provider
result = router_chat(
    model="auto",  # Let memory influence selection
    messages=[{"role": "user", "content": "Complex reasoning task"}],
    agent_type="researcher"
)

# Memory of past successes influences routing
```

### Phase 3: Procedural Evolution

Skills improve through execution tracking. Each skill records success rates and execution times.

```python
# Add a skill
add_skill(
    skill_name="api_design",
    skill_category="development",
    procedure_steps=["Define endpoints", "Add validation", "Write tests"],
    preconditions=["requirements documented"],
    success_criteria=["all tests pass", "docs complete"]
)

# Record execution outcome
record_skill_execution(
    skill_name="api_design",
    success=True,
    execution_time_ms=45000
)

# Skills with higher success rates are preferred
```

### Phase 4: Routing Learning

System learns optimal patterns over time based on outcomes.

```python
# Track reasoning strategy effectiveness
track_reasoning_strategy(
    agent_id="phoenix",
    strategy_name="decomposition",
    strategy_type="hierarchical",
    success=True,
    confidence=0.85
)

# Get most effective strategies
get_effective_reasoning_strategies(
    agent_id="phoenix",
    min_success_rate=0.7,
    min_usage=5
)
```

## Associative Memory

Create and traverse associative links between memories.

```python
# Create association
create_association(
    entity_a_id=123,
    entity_b_id=456,
    association_type="semantic",  # semantic, temporal, causal, emotional, spatial
    association_strength=0.8,
    bidirectional=True
)

# Get associations for an entity
get_associations(
    entity_id=123,
    min_strength=0.5,
    association_type="semantic"
)

# Reinforce when co-activated
reinforce_association(
    entity_a_id=123,
    entity_b_id=456,
    reinforcement=0.1
)
```

## Emotional Memory

Tag memories with emotional metadata for better recall.

```python
# Tag entity with emotion
tag_entity_emotion(
    entity_id=123,
    valence=0.8,        # Positive
    arousal=0.6,        # Moderately excited
    dominance=0.7,      # In control
    primary_emotion="joy",
    salience_score=0.9  # Very important
)

# Search by emotional criteria
search_by_emotion(
    emotion_filter={
        "valence_min": 0.5,      # Positive memories
        "primary_emotion": "joy",
        "min_salience": 0.7
    }
)

# Get high-salience memories
get_high_salience_memories(threshold=0.8)
```

## Attention Mechanism

Focus retrieval on currently relevant memories.

```python
# Set attention weights
set_attention(
    entity_id=123,
    relevance_score=0.9,
    recency_weight=0.3,
    frequency_weight=0.3,
    emotional_weight=0.4
)

# Get currently attended memories
get_attended_memories(threshold=0.5)

# Memories needing review (spaced repetition)
get_memories_needing_review(limit=20)
```

## Memory Strength and Decay

Implements Ebbinghaus forgetting curve.

```python
# Decay memory strength over time
decay_memory_strength(
    entity_id=123,
    time_elapsed_hours=48
)

# Boost on retrieval (spacing effect)
boost_memory_strength(
    entity_id=123,
    boost_amount=0.2
)
```

## Best Practices

1. **Use spreading activation** for exploratory searches
2. **Tag emotional significance** for important learnings
3. **Create associations** between related concepts
4. **Track skill execution** to improve procedural memory
5. **Run consolidation** regularly to strengthen patterns

## Related Documentation

- [Anti-Hallucination](ANTI_HALLUCINATION.md) - Truth verification
- [Continuous Learning](CONTINUOUS_LEARNING.md) - EWC++ and pattern recognition
- [Advanced Features](ADVANCED_FEATURES.md) - Causal inference, caching
