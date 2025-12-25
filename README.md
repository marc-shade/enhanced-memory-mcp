<img align="center" width="300" height="300" alt="Enhanced Memory MCP Server" src="https://github.com/user-attachments/assets/43639aae-da9b-4bb5-8fb9-3943d623a667" />

# Enhanced Memory MCP Server


[![MCP](https://img.shields.io/badge/MCP-Compatible-blue)](https://modelcontextprotocol.io)
[![Python](https://img.shields.io/badge/Python-3.11+-green)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Claude Code](https://img.shields.io/badge/Claude_Code-Ready-purple)](https://claude.ai/code)
[![Part of Agentic System](https://img.shields.io/badge/Part_of-Agentic_System-brightgreen)](https://github.com/marc-shade/agentic-system-oss)

> **Give your AI agents persistent memory that survives sessions, learns patterns, and actually remembers what you worked on.**

A production-grade memory system for Claude Code and MCP-compatible AI agents. Used by 200+ developers building agentic systems.

## Why Enhanced Memory?

| Feature | Enhanced Memory | Basic File Storage | Vector DB Only |
|---------|----------------|-------------------|----------------|
| Cross-session persistence | Yes | Manual | Yes |
| Sub-millisecond access | Yes (~0.01ms) | No | No |
| 2.4x compression | Yes | No | No |
| Semantic search | Yes | No | Yes |
| Memory tiers (working/archive) | Yes | No | No |
| Causal chains & patterns | Yes | No | No |
| Zero external dependencies | Yes | Yes | No (needs DB) |
| Works offline | Yes | Yes | Sometimes |

## Quick Start (Claude Code)

### 1. Install

```bash
# Clone the repo
git clone https://github.com/marc-shade/enhanced-memory-mcp.git
cd enhanced-memory-mcp

# Install with uv (recommended)
uv venv && uv pip install -r requirements.txt

# Or with pip
pip install -r requirements.txt
```

### 2. Configure Claude Code

Add to your `~/.claude.json`:

```json
{
  "mcpServers": {
    "enhanced-memory-mcp": {
      "command": "python",
      "args": ["/path/to/enhanced-memory-mcp/server.py"],
      "env": {}
    }
  }
}
```

### 3. Use It

```
You: Remember that I prefer TypeScript over JavaScript for new projects

Claude: I'll store that preference in memory.
[Uses create_entities tool]

--- Next session ---

You: What language should we use for this new API?

Claude: Based on your stored preferences, you prefer TypeScript over JavaScript
for new projects. I'll use TypeScript.
```

## Features

### Core Capabilities
- **Persistent Memory**: Survives sessions, restarts, even system reboots
- **Blazing Fast**: ~0.01ms reads, ~0.04ms writes
- **Smart Compression**: 2.4x data reduction with zlib level 9
- **Data Integrity**: SHA256 checksums on all stored data
- **Memory Tiers**: Working, Reference, Archive - automatic lifecycle management

### Advanced Features (100+ MCP Tools)
- **Semantic Search**: Find memories by meaning, not just keywords
- **Causal Chains**: Track cause-effect relationships
- **Pattern Learning**: Automatic pattern extraction from experiences
- **Memory Consolidation**: Sleep-inspired consolidation cycles
- **Episodic Memory**: Time-bound experiences and events
- **Procedural Memory**: Skills that improve with use
- **Theory of Mind**: Model other agents' beliefs and intentions

### NEW: Production-Grade AGI Features

| Feature | Description | Documentation |
|---------|-------------|---------------|
| **Holographic Memory** | Spreading activation across related memories | [docs/HOLOGRAPHIC_MEMORY.md](docs/HOLOGRAPHIC_MEMORY.md) |
| **Anti-Hallucination** | L-Score provenance, shadow vector search, citation validation | [docs/ANTI_HALLUCINATION.md](docs/ANTI_HALLUCINATION.md) |
| **Continuous Learning** | EWC++ to prevent catastrophic forgetting | [docs/CONTINUOUS_LEARNING.md](docs/CONTINUOUS_LEARNING.md) |
| **Advanced Features** | Causal inference, strange loops, ReasoningBank, caching | [docs/ADVANCED_FEATURES.md](docs/ADVANCED_FEATURES.md) |

**Holographic Memory** - 4-phase cognitive architecture:
- Activation Field (spreading activation)
- Memory-Influenced Routing
- Procedural Evolution
- Routing Learning

**Anti-Hallucination System**:
- Pattern detection for common hallucination indicators
- L-Score provenance tracking (confidence × relevance / depth)
- Shadow vector search for contradicting evidence
- Citation validation and trusted source management

**Continuous Learning (EWC++)**:
- Elastic Weight Consolidation prevents forgetting
- Pattern recognition from corrections
- Source reliability tracking
- Self-improvement cycles with meta-cognition

**Additional Advanced Capabilities**:
- Causal Inference Engine with DAG validation
- Strange Loops Detector (circular reasoning, contradictions)
- ReasoningBank for experience-based learning
- Semantic Cache and FACT Cache for performance
- Hybrid Search with cross-encoder reranking
- Multi-Query RAG and query expansion
- ART (Adaptive Resonance Theory) online learning

## Use Cases

### Personal AI Assistant
```python
# Store user preferences
create_entities([{
    "name": "user_preferences",
    "entityType": "preferences",
    "observations": ["Prefers dark mode", "TypeScript > JavaScript", "Uses vim keybindings"]
}])
```

### Multi-Agent Coordination
```python
# Share context between agents
send_coordination_message(
    sender="researcher",
    recipient="coder",
    subject="Found solution",
    content={"approach": "Use connection pooling", "source": "arxiv:2024.12345"}
)
```

### Learning from Experience
```python
# Record action outcomes
add_episode(
    event_type="code_generation",
    episode_data={"task": "API endpoint", "approach": "FastAPI", "result": "success"},
    significance_score=0.8
)

# Later: patterns extracted automatically during consolidation
```

## Architecture

### Memory Tiers

| Tier | Purpose | Access Pattern | Compression |
|------|---------|----------------|-------------|
| **Core** | System roles, critical config | Always loaded | Medium |
| **Working** | Active context, current session | High frequency | Low |
| **Reference** | Documentation, patterns | On-demand | Medium |
| **Archive** | Historical data, old sessions | Rare | Maximum |

### Database

Single SQLite file at `~/.claude/enhanced_memories/memory.db` - no external services required.

```sql
-- Core schema (simplified)
entities(id, name, type, tier, compressed_data, checksum, created_at, accessed_at)
relations(from_entity, to_entity, relation_type)
episodes(event_type, data, significance, emotional_valence)
concepts(name, type, definition, confidence)
skills(name, category, steps, success_rate)
```

## Performance

Benchmarked on production workloads:

| Operation | Latency | Throughput |
|-----------|---------|------------|
| Create entity | 0.04ms | 25,000/sec |
| Search nodes | 0.01ms | 100,000/sec |
| Semantic search | 5ms | 200/sec |
| Consolidation cycle | 2-5s | - |

Storage: ~1KB per entity after compression (varies with content).

## Advanced: Memory Consolidation

Inspired by sleep research on memory consolidation:

```bash
# Run consolidation (extracts patterns, compresses old memories)
# Automatic: runs every 6 hours or 10 Claude sessions
# Manual:
python -c "from server import run_full_consolidation; run_full_consolidation()"
```

What consolidation does:
1. **Pattern Extraction**: Recurring episodic patterns → semantic concepts
2. **Causal Discovery**: Learns cause-effect from action outcomes
3. **Memory Compression**: Archives old, low-access memories
4. **Skill Refinement**: Updates procedural memory from execution stats

## API Reference

See [API_REFERENCE.md](API_REFERENCE.md) for complete documentation of all 40+ tools.

### Most Used Tools

| Tool | Purpose |
|------|---------|
| `create_entities` | Store new memories |
| `search_nodes` | Find memories by query |
| `add_episode` | Record experiences |
| `get_memory_status` | Check system health |
| `run_full_consolidation` | Trigger learning cycle |

## Integration Examples

### With Claude Code Hooks

```yaml
# .claude/hooks/post-task.yaml
- name: "Record task outcome"
  command: |
    curl -X POST http://localhost:8080/record_outcome \
      -d '{"task": "$TASK", "result": "$RESULT"}'
```

### With Other MCP Servers

```python
# Combine with research-paper-mcp
papers = research_paper_search("transformer attention")
for paper in papers:
    create_entities([{
        "name": f"paper_{paper.id}",
        "entityType": "research",
        "observations": [paper.abstract]
    }])
```

## Troubleshooting

### Memory not persisting?
```bash
# Check database exists
ls -la ~/.claude/enhanced_memories/memory.db

# Check permissions
chmod 644 ~/.claude/enhanced_memories/memory.db
```

### Slow searches?
```bash
# Run optimization
python -c "from server import optimize_database; optimize_database()"
```

### Server not starting?
```bash
# Test standalone
python server.py
# Check logs for errors
```

## Contributing

PRs welcome! Areas of interest:
- Additional embedding providers
- Memory visualization tools
- Performance optimizations
- Documentation improvements

## Part of the MCP Ecosystem

This server is part of a comprehensive 24/7 autonomous AI framework. See [agentic-system-oss](https://github.com/marc-shade/agentic-system-oss) for the complete system with all MCP servers pre-configured.

### Related MCP Servers

**Core AGI Infrastructure**
| Server | Purpose |
|--------|---------|
| [agent-runtime-mcp](https://github.com/marc-shade/agent-runtime-mcp) | Persistent task queues, goal decomposition, relay pipelines |
| [agi-mcp](https://github.com/marc-shade/agi-mcp) | Production-ready AGI orchestration |
| [voice-agi-mcp](https://github.com/marc-shade/voice-agi-mcp) | Stateful voice-controlled AGI with Letta-style memory |

**Multi-Agent Coordination**
| Server | Purpose |
|--------|---------|
| [cluster-execution-mcp](https://github.com/marc-shade/cluster-execution-mcp) | Distributed execution across clusters |
| [node-chat-mcp](https://github.com/marc-shade/node-chat-mcp) | Inter-node AI communication |
| [claude-flow-mcp](https://github.com/marc-shade/claude-flow-mcp) | Workflow orchestration and agent coordination |
| [code-execution-mcp](https://github.com/marc-shade/code-execution-mcp) | Sandboxed code execution |

**Knowledge Acquisition**
| Server | Purpose |
|--------|---------|
| [research-paper-mcp](https://github.com/marc-shade/research-paper-mcp) | arXiv/Semantic Scholar paper search |
| [video-transcript-mcp](https://github.com/marc-shade/video-transcript-mcp) | YouTube transcript extraction and concept mining |

**Quality & Security**
| Server | Purpose |
|--------|---------|
| [ember-mcp](https://github.com/marc-shade/ember-mcp) | Production-only policy enforcement |
| [security-scanner-mcp](https://github.com/marc-shade/security-scanner-mcp) | Nuclei vulnerability scanning |
| [security-auditor-mcp](https://github.com/marc-shade/security-auditor-mcp) | AI-powered security vulnerability scanning |

**Visual & Media**
| Server | Purpose |
|--------|---------|
| [image-gen-mcp](https://github.com/marc-shade/image-gen-mcp) | Multi-provider image generation |

**Hardware Integration**
| Server | Purpose |
|--------|---------|
| [coral-tpu-mcp](https://github.com/marc-shade/coral-tpu-mcp) | Coral TPU local ML inference |
| [network-scanner-mcp](https://github.com/marc-shade/network-scanner-mcp) | Network device scanning |

### Other Projects

- [Ollama-Workbench](https://github.com/marc-shade/Ollama-Workbench) - Local LLM management and testing
- [Ollama-Workbench-2](https://github.com/marc-shade/Ollama-Workbench-2) - v2.0 with SvelteKit + Tauri
- [TeamForgeAI](https://github.com/marc-shade/TeamForgeAI) - Multi-agent team collaboration

## License

MIT License - Use freely in personal and commercial projects.

---

**Built for the agentic AI era.** If this helps your project, consider giving it a star!
