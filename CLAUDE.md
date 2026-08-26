# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Enhanced Memory MCP Server - persistent, compressed, searchable memory for AI
agents. SQLite storage behind a single-owner Unix-socket daemon, 4-tier
architecture, Git-like versioning, and layered RAG retrieval. **186 tools with
the core install, 204 with the optional backends** (`ENHANCED_MEMORY_SURFACE`
controls how many are register-time visible; counts measured at e9ca30c - any
tool-count claim without its environment stated is underspecified).

Figures in this file were measured at commit `99dc95e`; re-measure before
citing them elsewhere.

## Build & Test Commands

```bash
setup/setup.sh                      # venv, deps, .env, database dir (idempotent)
setup/bin/memory-db-daemon.sh &     # the daemon; nothing works without it
./healthcheck.sh                    # post-install gate: real write/read/cleanup round trip

# Functional suite (needs the daemon). Judge it by EXIT CODE, not pass count:
# with no ENHANCED_MEMORY_*/MEMORY_DB_* vars set it builds its own sandbox and
# runs everything; with them set it runs operator-directed and skips the checks
# that describe a sandbox it did not create. Measured on one machine at one
# commit: 106 isolated / 102 operator-directed, both exit 0. The run prints its
# own mode and names what it skipped.
python3 comprehensive_test.py

# Developer suite (needs: pip install -r dev-requirements.txt).
# 164 passed at 99dc95e.
python -m pytest tests/

# RAG integration suite - registration and class-level, no daemon needed.
# 22/22 at 99dc95e, verified under an isolated ENHANCED_MEMORY_DIR.
python3 test_rag_integration_comprehensive.py
```

**Store-safety warning:** `test_agi_phase2.py`, `test_agi_phase3.py`,
and `test_agi_phase4.py` operate on whatever
database the environment resolves - on a configured machine that is the REAL
store. Run them only with `ENHANCED_MEMORY_DIR` pointed at a scratch
directory. (The pytest suite under `tests/` is isolated by its conftest; these
standalone scripts are not.)

## Architecture

### Core Components

**server.py** (~2,400 lines) - FastMCP server. Core operations
(create_entities, search_nodes, get_memory_status) are delegated over the
Unix socket to the daemon; tools are registered modularly via
`register_*_tools()` functions in the `__main__` block. Entity creation runs
importance scoring (optional `tpu_importance` module; in this standalone
release the heuristic fallback is the path that actually runs - see
Importance scoring below) to auto-assign tiers.

**memory_db_service.py** - the daemon: single owner of the SQLite database
behind `MEMORY_DB_SOCKET_PATH`. Owns the schema: since e9ca30c,
`init_database()` creates every table the API touches (entities,
observations, relations, the `observations_fts` FTS5 index and its sync
triggers), so a database born from any creation path supports the full API.
Refuses to start on an occupied socket (socket_guard.py, probe-before-unlink)
and never unlinks a socket it did not bind.

**memory_client.py** - thin socket client used by server.py. It is
socket-only: **there is no automatic fallback to direct SQLite** when the
daemon is down; calls return a well-formed error object instead (an earlier
revision of this file claimed a fallback that does not exist).

**memory_paths.py** - the one path resolver. Precedence:
`ENHANCED_MEMORY_DB_PATH` / `MEMORY_DB_PATH` (exact file) >
`ENHANCED_MEMORY_DIR` (directory) > `~/.claude/enhanced_memories/`. Every
module resolves through this; do not hand-build database paths.

### Memory Tiers

| Tier | Purpose | Access Pattern |
|------|---------|----------------|
| Core | System roles, AI agent library | Pre-loaded, sub-ms access |
| Working | Active projects, current context | Session-scoped, frequent r/w |
| Reference | Documentation, code patterns | Full-text search, lazy loaded |
| Archive | Historical data, metrics | Maximum compression, rare access |

### RAG Strategies

Registered in server.py's `__main__` block:

| Tier | Strategy | Tools | File |
|------|----------|-------|------|
| 1 | Hybrid Search (BM25+Vector) | `search_hybrid` | `hybrid_search_tools_nmf.py` |
| 1 | Re-ranking (Cross-Encoder) | `search_with_reranking` | `reranking_tools_nmf.py` |
| 2 | Query Expansion | `search_with_query_expansion` | `query_expansion_tools.py` |
| 2 | Multi-Query RAG | `search_with_multi_query` | `multi_query_rag_tools.py` |
| 3.1 | Contextual Retrieval | `generate_context_for_chunk`, `reindex_with_context` | `contextual_retrieval_tools.py` |
| 3.3 | Hierarchical RAG | `index_document_hierarchical`, `search_hierarchical` | `hierarchical_rag_tools.py` |
| 4.1+4.3 | Agentic + Self-Reflective RAG | `agentic_retrieve`, `analyze_query`, `evaluate_results` | `agentic_rag_tools.py` |
| 4 | GraphRAG | `graph_enhanced_search`, `get_entity_neighbors` | `graphrag_tools.py` |
| 4 | Visual Memory | `store_visual_episode`, `find_similar_visual` | `visual_memory_tools.py` |
| - | Semantic Cache | `semantic_cache_get`, `semantic_cache_store`, `agi_cached_reasoning` | `semantic_cache_tools.py` |

22 `*_tools*.py` files at the repo root; `agi/` holds 11 supporting modules
(consolidation, metacognition, temporal reasoning, associative networks,
action tracking, identity, and friends), all path-resolved through
`memory_paths`.

### Tool Registration Pattern

```python
from query_expansion_tools import register_query_expansion_tools
register_query_expansion_tools(app, nmf_instance)
```

**To add new tools**:
1. Create `{feature}_tools.py` with `register_{feature}_tools(app, ...)`
2. Import and call it in server.py's registration section
4. Add tests (`tests/` for pytest, or `test_rag_integration_comprehensive.py`
   for registration-level coverage)

### Response conventions

- `success` is decided from the failure count, never assumed; a partial write
  reports `success: false` with an `error`.
- `create_entities` skips exact-duplicate observations per entity and reports
  the skips as `observations_deduped` (idempotent re-imports, e9ca30c).
  Re-WORDED near-duplicates (simhash, `simhash_dedup.py`) are stored and
  reported in `near_duplicates` by default — corrections must never be
  silently dropped; `ENHANCED_MEMORY_NEAR_DUP_POLICY=skip` opts an import
  pipeline into dropping them (`observations_near_dup_skipped`).
- `search_nodes` responses carry `degraded: "name-only (...)"` when content
  search is unavailable; absence of the key means FTS ran.
- A daemon-down state returns well-formed error objects, not empty successes;
  `./healthcheck.sh` distinguishes "empty" from "deaf".

### Database Schema

Path: `memory_paths.resolve()` - default `~/.claude/enhanced_memories/memory.db`,
overridable via `ENHANCED_MEMORY_DIR` / `ENHANCED_MEMORY_DB_PATH`.

- `entities` - compressed storage, versioning, tier assignment, MAU columns
- `observations` + `observations_fts` (FTS5, trigger-synced) - facts + content search
- `relations` - typed entity relationships
- `memory_versions` / `memory_branches` - Git-like history (an earlier
  revision of this file called these `entity_versions`/`entity_branches`;
  those tables do not exist)
- `memory_conflicts`, `implementation_plans`, `project_handbooks`
- `entity_visibility` / `entity_acl` - fail-closed viewer scoping
- `memory_scope` - per-project filter used by `search_nodes(scope=...)`

### Importance scoring

`server.py` imports `tpu_importance` when present and falls back to
heuristics otherwise. The hardware scoring service this module could reach
does not ship with this repository and is not part of any deployment this
release targets, so **the heuristic fallback is the real path**: score >= 0.8
-> long_term, >= 0.6 -> episodic, else working. Treat the TPU branch as an
optional integration point, not a dependency.

### Environment Variables

| Variable | Effect |
|---|---|
| `ENHANCED_MEMORY_DIR` | Base directory for database + NMF state |
| `ENHANCED_MEMORY_DB_PATH` / `MEMORY_DB_PATH` | Exact database file (outranks DIR) |
| `MEMORY_DB_SOCKET_PATH` | Daemon socket (default `/tmp/memory-db.sock`) |
| `ENHANCED_MEMORY_SURFACE` | `frontdoor` (default) / `consolidated` / `full` tool exposure |
| `ENHANCED_MEMORY_CONSOLIDATE` | Consolidation scheduling knob (see server.py) |
| `MEMORY_QDRANT_URL` | Existing Qdrant for the optional vector path |
| `OLLAMA_HOST` | Local embeddings for that index |
| `ENRICHMENT_OLLAMA_URL` / `ENRICHMENT_OLLAMA_MODEL` | Local ollama that generates the `[Context: ...]` prefix on `create_entities` (default `http://127.0.0.1:11434`, `gemma4:e4b-it-q8_0`); any failure degrades to the template prefix and reports `backend: template` |
| `AGENTIC_SYSTEM_PATH` | OPTIONAL: point at an agentic-system checkout to add its GraphRAG script tools (+7 tools; unset is the supported standalone mode - leave it unset when measuring tool counts) |

### Dependencies

Core (`requirements.txt`): `fastmcp`, `qdrant-client`-free stdlib path.
Optional (`requirements-optional.txt`): `sentence-transformers` (re-ranking),
`qdrant-client` (vector search), `anthropic` (contextual enrichment), torch.
Dev (`dev-requirements.txt`): pytest. The two shipped gates run on the stdlib
alone.

### Neural Memory Fabric (NMF)

`neural_memory_fabric.py` + `nmf_*.py`: Letta-style memory blocks with
open/edit/close semantics; required by the Tier 1-2 RAG tools. State lives
under `ENHANCED_MEMORY_DIR` (shipped-default config never outranks the
environment).

## Common Patterns

### Testing MCP tool registration without a server

```python
class MockFastMCPApp:
    def __init__(self):
        self.tools = {}
    def tool(self):
        def decorator(func):
            self.tools[func.__name__] = func
            return func
        return decorator

mock_app = MockFastMCPApp()
register_your_tools(mock_app, dependencies)
assert "your_tool_name" in mock_app.tools
```

(This is the pattern `test_rag_integration_comprehensive.py` itself uses.)

### Memory consolidation

```python
result = await run_full_consolidation(time_window_hours=24)
```

## Gaps / not covered

- Counts (tool totals, test totals, line counts) rot; each is stamped with
  the commit it was measured at. Re-measure, never trust this file over the
  tree.
- The RAG tier table lists the registration surface, not proof each strategy
  retrieves well; retrieval quality is unverified here.
- The standalone `test_agi_phase*.py` scripts remain unguarded against a
  configured real store (warning above); adding a guard is an open task.
