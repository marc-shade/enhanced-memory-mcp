# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Enhanced Memory MCP Server - a high-performance memory management system for AI agents. Provides compressed SQLite storage with 4-tier architecture, Git-like versioning, and advanced RAG (Retrieval-Augmented Generation) capabilities.

## Build & Test Commands

```bash
# Activate virtual environment
source .venv/bin/activate

# Run MCP server standalone (for testing)
python3 server.py

# Run comprehensive tests
python3 test_comprehensive.py

# Test RAG features
python3 test_graphrag_integration.py
python3 test_letta_integration.py

# Test code execution sandbox
python3 test_advanced_tool_use.py
```

## Architecture

### Core Components

**server.py** - Main FastMCP server with 200+ MCP tools. Architecture:
- Uses `memory_client.py` to connect to memory-db Unix socket service for concurrent access
- Core operations (create_entities, search_nodes) delegated to memory-db service
- Advanced features (versioning, branching, RAG) handled locally
- Tools registered modularly via `register_*_tools()` functions

**memory_db_service.py** - Unix socket service providing centralized SQLite access. Enables multiple Claude Code instances to share memory without corruption.

**memory_client.py** - Client wrapper for memory-db service with automatic fallback to direct SQLite when service unavailable.

### Memory Tiers

| Tier | Purpose | Access Pattern |
|------|---------|----------------|
| Core | System roles, AI agent library | Pre-loaded, sub-ms access |
| Working | Active projects, current context | Session-scoped, frequent r/w |
| Reference | Documentation, code patterns | Full-text search, lazy loaded |
| Archive | Historical data, metrics | Maximum compression, rare access |

### RAG Strategy Implementation

The system implements multiple RAG strategies from the roadmap in `COMPLETE_RAG_ROADMAP.md`:

| Strategy | Status | Tools |
|----------|--------|-------|
| Hybrid Search (BM25+Vector) | ✅ Active | `search_hybrid`, `search_with_reranking` |
| Query Expansion | ✅ Active | `search_with_query_expansion` |
| Multi-Query RAG | ✅ Active | `search_with_multi_query` |
| Contextual Retrieval | ✅ Active | `generate_context_for_chunk`, `reindex_with_context` |
| GraphRAG/Knowledge Graph | ✅ Active | `graph_enhanced_search`, `get_entity_neighbors` |
| Visual Memory | ✅ Active | `store_visual_episode`, `find_similar_visual` |

### Tool Registration Pattern

Tools are registered modularly in `server.py` around line 1370+:

```python
from query_expansion_tools import register_query_expansion_tools
register_query_expansion_tools(app, nmf_instance)
```

To add new tools:
1. Create `{feature}_tools.py` with a `register_{feature}_tools(app, ...)` function
2. Import and call in server.py's tool registration section
3. Tools automatically appear in MCP tool list

### Key Directories

```
agi/                    # AGI-specific modules (22 files)
  consolidation.py      # Sleep-like memory consolidation
  metacognition.py      # Self-awareness tracking
  belief_tracking.py    # Probabilistic belief states
  epistemic_scheduler.py # Epistemic flexibility audit

sandbox/                # Code execution sandbox
  executor.py           # RestrictedPython execution
  security.py           # Safety checks and PII tokenization

docs/                   # Organized documentation
  architecture/         # Technical specs, system design
  guides/               # User guides, API reference
  implementation-reports/  # Status and completion reports
  session-summaries/    # Development session notes
```

### Database Schema

Primary tables in `~/.claude/enhanced_memories/memory.db`:

- `entities` - Core memory storage with compression, versioning
- `observations` - Entity observations/facts
- `relations` - Entity relationships (from → to)
- `entity_versions` - Git-like version history
- `entity_branches` - Branch management
- `working_memory` - TTL-based temporary storage
- `episodic_memory` - Time-bound experiences
- `semantic_memory` - Abstract concepts
- `procedural_memory` - Skills and procedures

### Dependencies

Critical dependencies (from requirements.txt):
- `fastmcp` - MCP protocol implementation
- `sentence-transformers` - Re-ranking with cross-encoder
- `qdrant-client` - Hybrid search with BM25 + Vector
- `RestrictedPython` - Secure sandbox code execution
- `anthropic` - Claude API for contextual prefix generation

### Environment Variables

- `AGENTIC_SYSTEM_PATH` - Root path for agentic system (default: ~/agentic-system)
- `OLLAMA_HOST` - Ollama server for LLM operations
- `QDRANT_HOST` / `QDRANT_PORT` - Vector database connection

### TPU Integration

Real-time importance scoring via Google Coral TPU (when available):
- Entities scored at creation time
- Auto-assigned to appropriate tier based on importance
- Graceful fallback to heuristic scoring when TPU unavailable

### Neural Memory Fabric (NMF)

Alternative memory architecture in `nmf_*.py` files. Provides Letta-style memory blocks with open/edit/close semantics. See `README_NMF.md` for details.

## Common Patterns

### Adding a new memory tier feature

1. Add database table in `init_database()` if needed
2. Create tool file: `{feature}_tools.py`
3. Implement `register_{feature}_tools(app, db_path)`
4. Register in server.py tool registration section
5. Update `tool_catalog.py` for progressive tool discovery

### Testing MCP tools locally

```python
# Direct tool testing (bypasses MCP protocol)
from server import app

# Get tool function directly
tool_fn = app.tools['search_nodes']
result = await tool_fn(query="test", limit=5)
```

### Memory consolidation (sleep-like cycles)

```python
# Run full consolidation: patterns → semantic, causal discovery, compression
result = await run_full_consolidation(time_window_hours=24)
```

## RAG Roadmap Status

The `COMPLETE_RAG_ROADMAP.md` shows planned strategies but is outdated. Actual implementation status:

- **Tier 1** (Hybrid + Reranking): ✅ Complete
- **Tier 2** (Query Expansion, Multi-Query): ✅ Implemented
- **Tier 3** (Contextual Retrieval): ✅ Implemented
- **Tier 4** (GraphRAG, Visual Memory): ✅ Implemented

Remaining from roadmap not yet integrated:
- Context-Aware Chunking
- Hierarchical RAG
- Self-Reflective RAG
- Late Chunking
- Fine-tuned Embeddings
