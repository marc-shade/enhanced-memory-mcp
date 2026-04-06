# Enhanced Memory MCP Server

[![MCP](https://img.shields.io/badge/MCP-Compatible-blue)](https://modelcontextprotocol.io)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-green)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Tools](https://img.shields.io/badge/MCP_Tools-200%2B-informational)]()

A high-performance memory management system for AI agents built on the [Model Context Protocol](https://modelcontextprotocol.io/). Provides 200+ tools across compressed SQLite storage, 4-tier memory architecture, Git-like versioning, multi-strategy RAG, AGI cognitive phases, and modular tool loading with profile-based scaling.

## Features

- **4-Tier Memory Architecture**: Core, Working, Reference, and Archive tiers with automatic promotion/demotion
- **200+ MCP Tools**: Modular registration system with profile-based loading (full vs orchestrator mode)
- **Advanced RAG Pipeline**: 4-tier retrieval strategy — hybrid search, re-ranking, query expansion, agentic RAG, GraphRAG
- **Neural Memory Fabric (NMF)**: Letta-style memory blocks with open/edit/close semantics
- **Git-Like Versioning**: Branch, diff, and restore memory states across sessions
- **Real Compression**: 2.4x data reduction with zlib level 9, SHA256 checksums
- **AGI Cognitive Phases**: Identity, temporal reasoning, emotional tagging, meta-cognition (4 phases)
- **Intelligent Router**: Multi-provider LLM routing with uncertainty scoring
- **Anti-Hallucination Engine**: Causal inference, strange loop detection, continuous learning
- **Code Execution Sandbox**: RestrictedPython-based secure execution with PII tokenization
- **Semantic Cache**: LLM reasoning result caching (30-40% hit rate in production)
- **Manifold Working Memory**: High-dimensional working memory with trajectory compression
- **Triple-Signal Search**: Three-way ranking combining BM25, vector similarity, and graph proximity
- **Entropy Scoring**: Information-theoretic importance scoring for memory prioritization
- **Tool Usage Analytics**: Track which tools are invoked to optimize profile loading
- **Cluster Intelligence**: Multi-node coordination via cluster brain and SAFLA remote integration

## Performance

Based on production testing:
- **Write Speed**: ~0.04ms per entity
- **Read Speed**: ~0.01ms per query
- **Compression**: 2.4x average reduction
- **Semantic Cache Hit Rate**: 30-40%
- **Storage**: SQLite database at `~/.claude/enhanced_memories/memory.db`

## Installation

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Quick Start

```bash
git clone https://github.com/marc-shade/enhanced-memory-mcp.git
cd enhanced-memory-mcp
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Configure in Claude Code

Add to your `~/.claude.json`:

```json
{
  "mcpServers": {
    "enhanced-memory": {
      "command": "python3",
      "args": ["/path/to/enhanced-memory-mcp/server.py"]
    }
  }
}
```

### Configure in Claude Desktop

Add to your Claude Desktop MCP configuration:

```json
{
  "mcpServers": {
    "enhanced-memory": {
      "command": "/path/to/enhanced-memory-mcp/.venv/bin/python3",
      "args": ["/path/to/enhanced-memory-mcp/server.py"]
    }
  }
}
```

## Architecture

### Memory Tiers

| Tier | Purpose | Access Pattern |
|------|---------|----------------|
| **Core** | System roles, AI agent library, execution patterns | Pre-loaded on startup, sub-ms access |
| **Working** | Active projects, current context, agent assignments | Session-scoped, frequent read/write |
| **Reference** | Documentation, code patterns, error solutions | Full-text search, lazy loaded |
| **Archive** | Historical data, metrics, decision logs | Maximum compression, date-partitioned |

### Module Architecture

```
server.py                    # Main FastMCP entry point
├── server/                  # Core server modules
│   ├── config.py            # Configuration and logging
│   ├── database.py          # SQLite connection management
│   ├── compression.py       # zlib compression engine
│   ├── compaction.py        # Entity compaction and cleanup
│   ├── integrity.py         # SHA256 integrity verification
│   ├── versioning.py        # Git-like memory versioning
│   └── modules.py           # Profile-based module loader
├── router/                  # Intelligent LLM routing
│   ├── router.py            # Multi-provider router
│   ├── intelligent_router.py # Uncertainty-aware routing
│   ├── uncertainty.py       # Uncertainty scoring
│   └── providers/           # Provider implementations
├── sandbox/                 # Code execution sandbox
│   ├── executor.py          # RestrictedPython execution
│   ├── security.py          # Safety checks
│   ├── pii_tokenizer.py     # PII detection and tokenization
│   ├── lazy_loader.py       # Deferred module loading
│   └── tool_discovery.py    # Dynamic tool discovery
├── agi/                     # AGI cognitive modules (22 files)
│   ├── consolidation.py     # Sleep-like memory consolidation
│   ├── metacognition.py     # Self-awareness tracking
│   ├── belief_tracking.py   # Probabilistic belief states
│   ├── temporal_reasoning.py # Causal chains
│   ├── emotional_memory.py  # Emotional tagging
│   └── ...
├── *_tools.py (31 files)    # MCP tool modules
└── test_*.py (67 files)     # Test suite
```

### RAG Strategy Pipeline

| Tier | Strategy | Tools | File |
|------|----------|-------|------|
| 1 | Hybrid Search (BM25 + Vector) | `search_hybrid` | `hybrid_search_tools_nmf.py` |
| 1 | Re-ranking (Cross-Encoder) | `search_with_reranking` | `reranking_tools_nmf.py` |
| 2 | Query Expansion | `search_with_query_expansion` | `query_expansion_tools.py` |
| 2 | Multi-Query RAG | `search_with_multi_query` | `multi_query_rag_tools.py` |
| 3.1 | Contextual Retrieval | `generate_context_for_chunk` | `contextual_retrieval_tools.py` |
| 3.2 | Context-Aware Chunking | `chunk_document_semantic` | `context_aware_chunking.py` |
| 3.3 | Hierarchical RAG | `search_hierarchical` | `hierarchical_rag_tools.py` |
| 4.1 | Agentic + Self-Reflective RAG | `agentic_retrieve` | `agentic_rag_tools.py` |
| 4.2 | GraphRAG | `graph_enhanced_search` | `graphrag_tools.py` |
| 4.3 | Visual Memory | `store_visual_episode` | `visual_memory_tools.py` |
| -- | Triple-Signal Search | `triple_signal_search` | `triple_signal_tools.py` |
| -- | Semantic Cache | `semantic_cache_get`, `agi_cached_reasoning` | `semantic_cache_tools.py` |
| -- | FACT Cache | `fact_search` | `fact_integration.py` |
| -- | Unified Search | `unified_search` | `unified_search_api.py` |

### Memory Profiles

Control tool loading via the `MEMORY_PROFILE` environment variable:

```bash
# Full mode (default): All 200+ tools loaded
MEMORY_PROFILE=full python3 server.py

# Orchestrator mode: ~15 essential tools for coordination
MEMORY_PROFILE=orchestrator python3 server.py
```

Orchestrator mode loads only: `nmf_tools`, `safla_remote_integration`, `fact_integration`, `unified_search_api`, `semantic_cache_tools`, `reasoning_bank`.

### Database Schema

Primary tables in `~/.claude/enhanced_memories/memory.db`:

```sql
-- Core memory storage with compression and versioning
CREATE TABLE entities (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    entity_type TEXT NOT NULL,
    tier TEXT DEFAULT 'working',
    compressed_data BLOB,
    original_size INTEGER,
    compressed_size INTEGER,
    compression_ratio REAL,
    checksum TEXT,
    created_at TIMESTAMP,
    accessed_at TIMESTAMP,
    access_count INTEGER DEFAULT 0
);

-- Entity relationships with causal tracking
CREATE TABLE relations (
    id INTEGER PRIMARY KEY,
    from_entity TEXT NOT NULL,
    to_entity TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    weight REAL DEFAULT 1.0,
    causal INTEGER DEFAULT 0,
    created_at TIMESTAMP,
    UNIQUE(from_entity, to_entity, relation_type)
);

-- Git-like version history
CREATE TABLE entity_versions (
    id INTEGER PRIMARY KEY,
    entity_name TEXT NOT NULL,
    version INTEGER NOT NULL,
    branch TEXT DEFAULT 'main',
    compressed_data BLOB,
    checksum TEXT,
    created_at TIMESTAMP
);
```

Additional tables: `observations`, `entity_branches`, `working_memory`, `episodic_memory`, `semantic_memory`, `procedural_memory`, `visual_episodes`.

## API Examples

### Create Entities
```python
await create_entities({
    "entities": [
        {
            "name": "project_alpha",
            "entityType": "project",
            "observations": ["Architecture uses microservices", "Deployed on Kubernetes"]
        }
    ]
})
```

### Search Nodes
```python
await search_nodes({
    "query": "microservices architecture",
    "entity_types": ["project"],
    "limit": 10
})
```

### Unified Search (Intelligent Routing)
```python
await unified_search({
    "query": "How does authentication work?",
    "strategy": "auto"  # Automatically selects best RAG strategy
})
```

### Agentic RAG (Self-Reflective Retrieval)
```python
await agentic_retrieve({
    "query": "memory consolidation patterns",
    "max_iterations": 3,
    "quality_threshold": 0.7
})
```

### Neural Memory Fabric
```python
# Open a memory block for editing
await nmf_open_block({"block_id": "working_context"})

# Edit the block
await nmf_edit_block({
    "block_id": "working_context",
    "content": "Current focus: implementing authentication module"
})

# Recall related memories
await nmf_recall({"query": "authentication patterns"})

# Close the block
await nmf_close_block({"block_id": "working_context"})
```

### Semantic Cache
```python
# Cache an LLM reasoning result
await semantic_cache_store({
    "query": "Explain transformer attention mechanisms",
    "result": "Transformers use self-attention to...",
    "ttl_hours": 24
})

# Retrieve cached result (fuzzy match)
await semantic_cache_get({
    "query": "How do transformer attention heads work?"
})
```

### Memory Versioning
```python
# Create a branch
await memory_branch({"branch_name": "experiment-v2"})

# Make changes, then diff
await memory_diff({"branch": "experiment-v2", "base": "main"})

# Revert if needed
await memory_revert({"entity_name": "project_alpha", "version": 3})
```

## Tool Modules

### Core Tools (always loaded)

| Module | Tools | Description |
|--------|-------|-------------|
| `server/` | `create_entities`, `search_nodes`, `get_memory_status` | Core CRUD + versioning |

### AGI Cognitive Tools

| Module | Phase | Description |
|--------|-------|-------------|
| `agi_tools.py` | Phase 1 | Identity, action tracking, agent registry |
| `agi_tools_phase2.py` | Phase 2 | Temporal reasoning, sleep-like consolidation |
| `agi_tools_phase3.py` | Phase 3 | Emotional tagging, associative networks |
| `agi_tools_phase4.py` | Phase 4 | Meta-cognition, self-improvement cycles |

### RAG & Search Tools

| Module | Description |
|--------|-------------|
| `hybrid_search_tools_nmf.py` | BM25 + vector hybrid search |
| `reranking_tools_nmf.py` | Cross-encoder re-ranking (ms-marco-MiniLM) |
| `query_expansion_tools.py` | LLM-powered query expansion |
| `multi_query_rag_tools.py` | Multi-perspective query generation |
| `contextual_retrieval_tools.py` | Context-enhanced chunk retrieval |
| `hierarchical_rag_tools.py` | Multi-level document indexing |
| `agentic_rag_tools.py` | Autonomous self-reflective retrieval |
| `graphrag_tools.py` | Graph-enhanced search |
| `triple_signal_tools.py` | Three-way ranking (BM25 + vector + graph) |
| `visual_memory_tools.py` | Visual episode storage and similarity search |

### Memory Management Tools

| Module | Description |
|--------|-------------|
| `nmf_tools.py` | Neural Memory Fabric (Letta-style blocks) |
| `reasoning_tools.py` | 75/15 rule prioritization |
| `semantic_cache_tools.py` | LLM reasoning result caching |
| `fact_integration.py` | Fast cache-first fact retrieval |
| `unified_search_api.py` | Intelligent search strategy routing |
| `reasoning_bank.py` | Persistent learning from reasoning outcomes |
| `manifold_working_memory_tools.py` | High-dimensional working memory |
| `trajectory_compression.py` | Memory trajectory compression |
| `entropy_scoring.py` | Information-theoretic importance scoring |
| `lru_cache_layer.py` | LRU caching for hot entities |

### Intelligence Tools

| Module | Description |
|--------|-------------|
| `anti_hallucination.py` | Hallucination detection and prevention |
| `causal_inference.py` | Causal relationship discovery |
| `strange_loops.py` | Self-referential loop detection |
| `continuous_learning.py` | Online learning from interactions |
| `model_router.py` | Multi-provider LLM routing |
| `activation_field_tools.py` | Memory activation field dynamics |
| `procedural_evolution_tools.py` | Procedural memory evolution |
| `routing_learning_tools.py` | Learned query routing optimization |
| `surprise_consolidation_tools.py` | Surprise-based memory consolidation |
| `provenance.py` | Provenance tracking and L-Score validation |

### Integration Tools

| Module | Description |
|--------|-------------|
| `safla_tools.py` | SAFLA 4-tier memory integration |
| `safla_remote_integration.py` | Remote SAFLA cluster bridge |
| `cluster_brain_tools.py` | Multi-node cluster intelligence |
| `sleeptime_tools.py` | Letta sleeptime compute integration |
| `tool_usage_logger.py` | Tool invocation analytics |

## Testing

```bash
# Run comprehensive test suite
python3 comprehensive_test.py

# RAG integration tests (22 tests)
python3 test_rag_integration_comprehensive.py

# Test specific subsystems
python3 test_graphrag_integration.py
python3 test_manifold_working_memory.py
python3 test_triple_signal_search.py
python3 test_surprise_consolidation.py
python3 test_trajectory_compression.py
python3 test_anti_hallucination.py
python3 test_causal_inference.py

# AGI phase tests
python3 test_agi_phase1.py
python3 test_agi_phase2.py
python3 test_agi_phase3.py
python3 test_agi_phase4.py

# Code execution sandbox
python3 test_advanced_tool_use.py
```

## Adding New Tools

1. Create `{feature}_tools.py` with the registration pattern:

```python
def register_{feature}_tools(app, *args):
    @app.tool()
    async def my_new_tool(param: str) -> str:
        """Tool description shown in MCP."""
        return result
```

2. Register in `server/modules.py`:

```python
if should_load_module("{feature}_tools"):
    try:
        from {feature}_tools import register_{feature}_tools
        register_{feature}_tools(app)
    except Exception as e:
        logger.warning(f"{feature} integration skipped: {e}")
```

3. Add to `tool_catalog.py` for progressive tool discovery.
4. Write tests in `test_{feature}.py`.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMORY_PROFILE` | `full` | Tool loading profile (`full` or `orchestrator`) |
| `TOOL_USAGE_LOGGING` | `true` | Enable tool invocation analytics |
| `AGENTIC_SYSTEM_PATH` | `~/agentic-system` | Root path for agentic system |
| `OLLAMA_HOST` | `localhost:11434` | Ollama server for LLM operations |
| `QDRANT_HOST` | `localhost` | Qdrant vector database host |
| `QDRANT_PORT` | `6333` | Qdrant vector database port |
| `ANTHROPIC_API_KEY` | -- | For contextual prefix generation |
| `OPENAI_API_KEY` | -- | For query expansion (optional) |

## Dependencies

Key dependencies (see `requirements.txt` for full list):

- `fastmcp` — MCP protocol implementation
- `sentence-transformers` — Cross-encoder re-ranking (ms-marco-MiniLM-L-6-v2)
- `qdrant-client` — Hybrid search with BM25 + vector
- `RestrictedPython` — Secure sandbox code execution
- `anthropic` — Claude API for contextual retrieval
- `numpy` — Vector operations and entropy scoring

## License

MIT
