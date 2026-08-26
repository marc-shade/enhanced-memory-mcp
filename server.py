#!/usr/bin/env python3
"""
Enhanced Memory MCP Server with Git-like Version Control
Combines existing compression/tiering with version control features

ARCHITECTURE: Uses memory-db Unix socket service for core operations
- create_entities, search_nodes, get_memory_status: Delegated to memory-db
- Versioning, branching, conflicts: Local advanced features
- Concurrent access: Enabled via memory-db central coordinator
"""

import asyncio
import difflib
import hashlib
import inspect
import json
import logging
import logging.handlers
import os
import pickle
import sqlite3

# TPU Importance Scoring - Add to Python path with platform detection
import sys
import zlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

# FastMCP implementation
from fastmcp import FastMCP

# Memory-DB client for concurrent access
from memory_client import MemoryClient

# Code Execution imports
from sandbox.executor import CodeExecutor, create_api_context
from sandbox.security import comprehensive_safety_check, sanitize_output


def _get_storage_base() -> Path:
    """Resolve the optional surrounding agentic-system checkout.

    Only used to put an optional hooks directory on sys.path; the memory server
    runs standalone without it. Set AGENTIC_SYSTEM_PATH to point at a checkout,
    otherwise this resolves relative to the repository.
    """
    env_base = os.environ.get("AGENTIC_SYSTEM_PATH")
    if env_base:
        return Path(os.path.expandvars(os.path.expanduser(env_base)))
    return Path(__file__).resolve().parent.parent.parent


_STORAGE_BASE = _get_storage_base()
_HOOKS_PATH = _STORAGE_BASE / "scripts" / "hooks"
if str(_HOOKS_PATH) not in sys.path:
    sys.path.insert(0, str(_HOOKS_PATH))

try:
    from tpu_importance import is_tpu_available, score_importance

    TPU_SCORING_AVAILABLE = True
except ImportError:
    TPU_SCORING_AVAILABLE = False

    def score_importance(
        text: str, context: str = "memory", source: str = "direct"
    ) -> float:
        """Fallback heuristic scoring when TPU module unavailable."""
        score = 0.3
        text_lower = text.lower()
        high_kw = ["error", "critical", "security", "bug", "important", "urgent"]
        for kw in high_kw:
            if kw in text_lower:
                score += 0.15
        return min(1.0, score)

    def is_tpu_available() -> bool:
        return False


# Set up logging - CRITICAL: Must use stderr for MCP compatibility
# MCP protocol requires stdout is reserved for JSON-RPC messages only
import sys
from pathlib import Path as _LogPath

# MCP servers MUST NOT output to stderr - Claude Code interprets it as errors
# Force-redirect ALL logging to file (basicConfig doesn't work if already configured)
_log_file = _LogPath("/tmp") / "enhanced-memory-mcp.log"
_file_handler = logging.handlers.RotatingFileHandler(
    str(_log_file), mode="a", maxBytes=50 * 1024 * 1024, backupCount=2
)
_file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)

# Remove ALL existing handlers from root logger and add file handler only
logging.root.handlers.clear()
logging.root.addHandler(_file_handler)
logging.root.setLevel(logging.INFO)

# WARNING and above ALSO go to stderr, unless MEMORY_LOG_STDERR=0.
#
# Routine INFO stays in the file, because MCP clients treat stderr chatter as
# errors. But warnings here are not chatter: a skipped tool group, a missing
# dependency, or the database split-brain banner are exactly what an installer
# has to see, and routing them only to a rotating file under /tmp meant nobody
# ever did. The governance group failed to register on every machine without a
# separate private checkout and the sole record was a line in that file.
#
# If this is noisy in your client, the noise is the signal: a healthy install
# skips nothing. Set MEMORY_LOG_STDERR=0 to silence it anyway.
if os.environ.get("MEMORY_LOG_STDERR", "1") != "0":
    _stderr_handler = logging.StreamHandler(sys.stderr)
    _stderr_handler.setLevel(logging.WARNING)
    _stderr_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logging.root.addHandler(_stderr_handler)

logger = logging.getLogger("enhanced-memory-git")


# Resolution order lives in memory_paths so the code-execution API and anything
# else that opens the database directly cannot drift from what the server uses.
from memory_paths import get_memory_paths as _get_memory_paths


# Configuration
MEMORY_DIR, DB_PATH = _get_memory_paths()

# NOTE: MEMORY_DIR is deliberately NOT created here. This used to run at module
# scope, so merely importing this module created the configured memory directory
# -- which meant any test, tooling or introspection that did `import server`
# reached into the operator's real memory location as a side effect of the
# import. exist_ok made it silent. Directory creation now happens in
# init_database(), which the __main__ block calls before anything touches the
# database. Importing this module must have no filesystem effect.

# Initialize FastMCP app
app = FastMCP("enhanced-memory")

# Initialize memory-db client for concurrent access
memory_client = MemoryClient(
    os.environ.get("MEMORY_DB_SOCKET_PATH", "/tmp/memory-db.sock")
)

# Exceptions that mean "the memory-db socket is not there / not accepting", as
# opposed to the daemon answering with an application-level error.
_UNREACHABLE_ERRORS = (
    FileNotFoundError,
    ConnectionRefusedError,
    ConnectionResetError,
    PermissionError,
    TimeoutError,
    asyncio.TimeoutError,
    OSError,
)


def _daemon_failure(error: str, *, unreachable: bool, **echo: Any) -> dict[str, Any]:
    """Build the response for a memory-db call that did not succeed.

    This envelope carries NO data-shaped keys on purpose: no counts, no empty
    result lists, no zeroed totals. A dead daemon and an empty memory system
    are different facts, and a caller must not be able to render one as the
    other. `echo` carries back only the request parameters (e.g. the query),
    never a fabricated result.

    Keys:
        error:  what went wrong.
        daemon: 'UNREACHABLE (<socket>)' when the socket could not be reached,
                'ERROR (<socket>)' when the daemon replied with a failure.
        status: always 'failed', so a truthiness check is not required.
    """
    state = "UNREACHABLE" if unreachable else "ERROR"
    return {
        "status": "failed",
        "error": error,
        "daemon": f"{state} ({memory_client.socket_path})",
        **echo,
    }


def _check_db_path_agreement() -> None:
    """Warn loudly when the server and the daemon resolve DIFFERENT databases.

    Setting ENHANCED_MEMORY_DB_PATH (or MEMORY_DB_PATH / ENHANCED_MEMORY_DIR)
    on the server alone does not move the daemon. The server then creates and
    reports an empty database at the new path while every front-door tool keeps
    serving the daemon's database. Both halves look healthy, and the operator
    concludes the memories are gone.

    The daemon is authoritative: create_entities, search_nodes and
    get_memory_status all go through the socket. This check is advisory and
    never blocks startup.
    """

    async def _probe() -> dict:
        return await asyncio.wait_for(memory_client.get_memory_status(), timeout=5.0)

    try:
        status = asyncio.run(_probe())
    except Exception as e:
        logger.warning(
            "DB-path agreement UNCHECKED: memory-db daemon not reachable at %s (%s: %s). "
            "Server would use %s; the tools will fail until the daemon is running.",
            memory_client.socket_path,
            type(e).__name__,
            e,
            DB_PATH,
        )
        return

    daemon_db = status.get("database_path")
    if not daemon_db:
        logger.warning(
            "DB-path agreement UNCHECKED: daemon at %s did not report a "
            "database_path (keys: %s).",
            memory_client.socket_path,
            sorted(status),
        )
        return

    if Path(daemon_db).resolve() != DB_PATH.resolve():
        logger.warning(
            "\n"
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
            "!!  DATABASE SPLIT-BRAIN: server and daemon disagree          !!\n"
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
            "  server resolved : %s\n"
            "  daemon serves   : %s   <-- the tools read and write HERE\n"
            "  socket          : %s\n"
            "\n"
            "  The server's path is NOT the one your memories live in. Point\n"
            "  the daemon at the same database (set ENHANCED_MEMORY_DB_PATH in\n"
            "  the daemon's environment and restart it), or unset the override\n"
            "  on the server. Until then the server-side path is a decoy: it\n"
            "  will be created empty and stay empty.\n"
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",
            DB_PATH,
            daemon_db,
            memory_client.socket_path,
        )
    else:
        logger.info("DB-path agreement OK: server and daemon both use %s", DB_PATH)


def init_database():
    """Initialize SQLite database with all tables including Git features"""
    # Created here rather than at import time; see the note beside MEMORY_DIR.
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA busy_timeout = 30000")
    try:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
    except sqlite3.OperationalError as e:
        logger.warning("SQLite tuning skipped for %s: %s", DB_PATH, e)
    cursor = conn.cursor()

    # Original entities table with compression
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            entity_type TEXT NOT NULL,
            tier TEXT DEFAULT 'working',
            compressed_data BLOB,
            original_size INTEGER,
            compressed_size INTEGER,
            compression_ratio REAL,
            checksum TEXT,
            access_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            current_version INTEGER DEFAULT 1,
            current_branch TEXT DEFAULT 'main'
        )
    """)

    # Observations table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_id INTEGER,
            content TEXT NOT NULL,
            compressed BLOB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (entity_id) REFERENCES entities (id)
        )
    """)

    # Relations table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS relations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            from_entity_id INTEGER,
            to_entity_id INTEGER,
            relation_type TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (from_entity_id) REFERENCES entities (id),
            FOREIGN KEY (to_entity_id) REFERENCES entities (id)
        )
    """)

    # NEW: Memory versions table for Git-like history
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memory_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_id INTEGER NOT NULL,
            version_number INTEGER NOT NULL,
            compressed_data BLOB NOT NULL,
            diff_from_previous TEXT,
            commit_message TEXT,
            author TEXT DEFAULT 'system',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_current BOOLEAN DEFAULT 0,
            branch_name TEXT DEFAULT 'main',
            parent_version_id INTEGER,
            FOREIGN KEY (entity_id) REFERENCES entities (id),
            FOREIGN KEY (parent_version_id) REFERENCES memory_versions (id),
            UNIQUE(entity_id, version_number, branch_name)
        )
    """)

    # NEW: Memory branches table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memory_branches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_id INTEGER NOT NULL,
            branch_name TEXT NOT NULL,
            base_version_id INTEGER,
            is_active BOOLEAN DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_by TEXT DEFAULT 'system',
            description TEXT,
            FOREIGN KEY (entity_id) REFERENCES entities (id),
            FOREIGN KEY (base_version_id) REFERENCES memory_versions (id),
            UNIQUE(entity_id, branch_name)
        )
    """)

    # NEW: Conflict detection table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memory_conflicts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity1_id INTEGER NOT NULL,
            entity2_id INTEGER NOT NULL,
            conflict_type TEXT NOT NULL,
            similarity_score REAL,
            detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            resolved BOOLEAN DEFAULT 0,
            resolution_notes TEXT,
            FOREIGN KEY (entity1_id) REFERENCES entities (id),
            FOREIGN KEY (entity2_id) REFERENCES entities (id)
        )
    """)

    # NEW: Implementation plans table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS implementation_plans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            description TEXT,
            steps JSON NOT NULL,
            status TEXT DEFAULT 'draft',
            progress JSON,
            entity_id INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (entity_id) REFERENCES entities (id)
        )
    """)

    # NEW: Project handbooks table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS project_handbooks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_name TEXT UNIQUE NOT NULL,
            overview TEXT,
            architecture JSON,
            conventions JSON,
            setup_instructions TEXT,
            entity_id INTEGER,
            version INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (entity_id) REFERENCES entities (id)
        )
    """)

    # Create all indexes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name)")
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_entities_accessed ON entities(last_accessed)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_versions_entity ON memory_versions(entity_id)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_versions_branch ON memory_versions(branch_name)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_versions_current ON memory_versions(is_current)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_conflicts_unresolved ON memory_conflicts(resolved)"
    )

    conn.commit()
    conn.close()


def compress_data(data: Any) -> tuple[bytes, int, int, float]:
    """Compress data using zlib with maximum compression"""
    serialized = pickle.dumps(data)
    original_size = len(serialized)
    compressed = zlib.compress(serialized, level=9)
    compressed_size = len(compressed)
    compression_ratio = compressed_size / original_size if original_size > 0 else 1.0
    return compressed, original_size, compressed_size, compression_ratio


def decompress_data(compressed: bytes) -> Any:
    """Decompress and deserialize data.

    Tolerant of the historical encodings present in entities.compressed_data:
    zlib+pickle (canonical write path), zlib+json, zlib+plain-text (episodic /
    insight entities), and gzip (service_event, magic 1f 8b). Read-path only —
    the write path still uses zlib+pickle. Mirrors
    memory_db_service_v2._decompress_data so all callers see one shape.
    Previously this assumed zlib+pickle and raised on ~76% of entities, which
    broke detect_memory_conflicts and any tool reading compressed_data.
    """
    try:
        decompressed = zlib.decompress(compressed)
    except zlib.error:
        import gzip

        decompressed = gzip.decompress(compressed)
    # SECURITY: pickle source is the system's OWN local DB
    # (~/.claude/enhanced_memories/memory.db), written by this server's compress
    # path — first-party trusted data, not external input. (Pre-existing behavior.)
    try:
        return pickle.loads(decompressed)
    except Exception:
        try:
            return json.loads(decompressed.decode("utf-8"))
        except Exception:
            return {"observations": [decompressed.decode("utf-8", errors="replace")]}


def calculate_checksum(data: bytes) -> str:
    """Calculate SHA256 checksum for data integrity"""
    return hashlib.sha256(data).hexdigest()


def classify_tier(entity_type: str, name: str) -> str:
    """Classify entity into memory tier"""
    if entity_type in ["system_role", "core_system"] or "orchestrator" in name.lower():
        return "core"
    elif entity_type in ["project", "session"] or "current" in name.lower():
        return "working"
    elif "archive" in name.lower() or "historical" in entity_type.lower():
        return "archive"
    else:
        return "reference"


def create_version(
    entity_id: int, data: Any, message: str = None, author: str = "system"
) -> int:
    """Create a new version when entity is updated"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get current branch
    cursor.execute("SELECT current_branch FROM entities WHERE id = ?", (entity_id,))
    branch = cursor.fetchone()[0] or "main"

    # Get current version number
    cursor.execute(
        """
        SELECT MAX(version_number) FROM memory_versions
        WHERE entity_id = ? AND branch_name = ?
    """,
        (entity_id, branch),
    )

    current_version = cursor.fetchone()[0]
    new_version = (current_version or 0) + 1

    # Compress data
    compressed, _, _, _ = compress_data(data)

    # Calculate diff if there's a previous version
    diff_text = None
    if current_version:
        cursor.execute(
            """
            SELECT compressed_data FROM memory_versions
            WHERE entity_id = ? AND version_number = ? AND branch_name = ?
        """,
            (entity_id, current_version, branch),
        )

        prev_data = cursor.fetchone()
        if prev_data:
            prev_decompressed = decompress_data(prev_data[0])
            old_str = json.dumps(prev_decompressed, indent=2, default=str)
            new_str = json.dumps(data, indent=2, default=str)
            diff = difflib.unified_diff(
                old_str.splitlines(keepends=True),
                new_str.splitlines(keepends=True),
                fromfile="previous",
                tofile="current",
            )
            diff_text = "".join(diff)

    # Mark all previous versions as not current
    cursor.execute(
        """
        UPDATE memory_versions SET is_current = 0
        WHERE entity_id = ? AND branch_name = ?
    """,
        (entity_id, branch),
    )

    # Insert new version
    cursor.execute(
        """
        INSERT INTO memory_versions
        (entity_id, version_number, compressed_data, diff_from_previous,
         commit_message, author, is_current, branch_name)
        VALUES (?, ?, ?, ?, ?, ?, 1, ?)
    """,
        (entity_id, new_version, compressed, diff_text, message, author, branch),
    )

    # Update entity's current version
    cursor.execute(
        """
        UPDATE entities SET current_version = ? WHERE id = ?
    """,
        (new_version, entity_id),
    )

    version_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return version_id


# ORIGINAL TOOLS WITH VERSION CONTROL ADDED


@app.tool()
async def create_entities(entities: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Create entities with compression, storage, automatic versioning, and contextual enrichment.

    CONCURRENT ACCESS: Uses memory-db Unix socket service for database operations.
    CONTEXTUAL ENRICHMENT: prepends one "[Context: ...]" observation per entity.
    Since 2026-08-24 the prefix is generated by LOCAL ollama
    (ENRICHMENT_OLLAMA_MODEL, default gemma4:e4b-it-q8_0, think disabled) when
    the local daemon answers, else the TEMPLATE "[Context: This is a {type}
    entity named '{name}'...]". The Anthropic branch stays dead by policy (no
    direct AI SDK calls). Every prefix stored before 2026-08-24 is the
    template (2,429/2,429 measured). The response's
    `contextual_enrichment.backend` ("ollama"|"template") and `using_llm` say
    which branch ran for THIS call; trust those, not this description.

    Args:
        entities: List of entity objects with name, entityType, and observations

    Returns:
        Results with compression statistics and entity details
    """
    try:
        # Delegate to memory-db service for concurrent access
        response = await memory_client.create_entities(entities)

        if response.get("success"):
            # Apply contextual enrichment to newly created entities
            enrichment_stats = await _enrich_new_entities(entities)

            # Score importance and assign memory tiers via TPU
            scoring_stats = await _score_and_tier_entities(entities)

            # Phase 0 spine repair (2026-07-02): index new entities into the
            # vector store immediately. Before this, create_entities never
            # touched Qdrant and semantic_recall was blind to all new memory
            # (audit: 364/9,841 indexed). Best-effort in a worker thread; the
            # background sweeper self-heals anything missed here.
            vector_stats: dict[str, Any] = {}
            try:
                import asyncio as _aio

                from vector_write_indexer import index_entities as _vwi_index

                new_ids = [
                    r.get("id")
                    for r in response.get("results", [])
                    if isinstance(r, dict) and r.get("id") is not None
                ]
                if new_ids:
                    vector_stats = await _aio.to_thread(_vwi_index, new_ids)
            except Exception as _vwi_err:
                vector_stats = {"indexed": 0, "error": str(_vwi_err)}

            return {
                "created": response.get("count", 0),
                "failed": 0,
                "results": response.get("results", []),
                "contextual_enrichment": enrichment_stats,
                "tpu_scoring": scoring_stats,
                "vector_indexing": vector_stats,
            }
        else:
            # A partial write reports what DID land. That is a measured fact,
            # not a zero-fill: `status: failed` and `error` are both present, so
            # this cannot be misread as an empty success.
            # Each key is included only when it carries real content, so a total
            # failure never emits `stored: 0` or `results: []`.
            partial: dict[str, Any] = {}
            stored = (response.get("created") or 0) + (response.get("updated") or 0)
            if stored:
                partial["stored"] = stored
            if response.get("failed"):
                partial["failed"] = response["failed"]
            # WHICH entities landed, not just how many: a caller retrying a
            # partial batch needs the names it does not have to write again.
            if response.get("results"):
                partial["results"] = response["results"]
            if response.get("errors"):
                partial["errors"] = response["errors"]
            return _daemon_failure(
                response.get("error", "Unknown error from memory-db service"),
                unreachable=False,
                requested=len(entities),
                **partial,
            )

    except Exception as e:
        logger.error(f"Error creating entities via memory-db: {str(e)}")
        return _daemon_failure(
            f"Memory-DB service error: {e}",
            unreachable=isinstance(e, _UNREACHABLE_ERRORS),
            requested=len(entities),
        )


@app.tool()
async def store_fact_versioned(
    fact: str,
    name: str,
    entity_type: str = "fact",
    people: list[str] | None = None,
    use_llm: bool = True,
) -> dict[str, Any]:
    """Bi-temporal fact write (Phase 2.2, memory roadmap 2026-07-02): verify
    THEN execute. verify_fact_before_store only ever returned a decision;
    nothing executed it, so contradicting facts accumulated side by side —
    the knowledge-updates weakness every benchmarked memory system shares
    (even LongMemEval leader MemOS falls there).

    Pipeline:
      IGNORE            -> not stored (duplicate), returns the matched fact
      CREATE            -> stored via create_entities (compression +
                           enrichment + vector indexing)
      CONFLICT_RESOLVED -> new fact stored, conflicting entity marked
                           superseded (valid_until=now, superseded_by=<new id>)
                           and evicted from the vector index. History is
                           preserved in sqlite — superseded facts are filtered
                           from current-state retrieval, never deleted.

    Args:
        fact: The fact text to store.
        name: Entity name for the stored fact (must be unique).
        entity_type: Entity type (default "fact").
        people: Optional people the fact is about (conflict gating).
        use_llm: Use the LLM for nuanced conflict judgment (default True,
                 ~5-20s; without it the conflict band falls through to
                 CREATE and contradictions accumulate). Set False only for
                 latency-critical duplicate-only screening.

    Returns:
        {"action": ..., "reason": ..., "stored_id": int|None,
         "superseded": {id, name}|None}
    """
    try:
        from atommem.tools import decide_fact

        decision = await asyncio.to_thread(
            decide_fact, str(DB_PATH), fact, 60, use_llm, people
        )
        action = decision.get("action")

        if action == "IGNORE":
            return {
                "action": "IGNORE",
                "reason": decision.get("reason"),
                "stored_id": None,
                "superseded": None,
                "duplicate_of": decision.get("conflict_with"),
            }

        # create_entities is a FastMCP FunctionTool at module level; the
        # underlying coroutine is .fn ('FunctionTool' object is not callable)
        _create = getattr(create_entities, "fn", create_entities)
        create_res = await _create(
            [
                {
                    "name": name,
                    "entityType": entity_type,
                    "observations": [decision.get("store_text") or fact],
                }
            ]
        )
        results = create_res.get("results") or []
        stored_id = (
            results[0].get("id") if results and isinstance(results[0], dict) else None
        )

        superseded = None
        conflict = decision.get("conflict_with")
        if action == "CONFLICT_RESOLVED" and conflict and stored_id:
            try:
                old_id = int(conflict["id"])
                conn = sqlite3.connect(DB_PATH, timeout=30)
                conn.execute("PRAGMA busy_timeout=30000")
                conn.execute(
                    """UPDATE entities
                       SET valid_until=CURRENT_TIMESTAMP, superseded_by=?
                       WHERE id=? AND superseded_by IS NULL""",
                    (stored_id, old_id),
                )
                conn.commit()
                conn.close()
                try:
                    from qdrant_client import QdrantClient

                    from local_semantic_recall import (
                        DEFAULT_MODEL,
                        QDRANT,
                        collection_for,
                    )

                    QdrantClient(url=QDRANT).delete(
                        collection_for(DEFAULT_MODEL), points_selector=[old_id]
                    )
                except Exception as evict_err:
                    logger.warning(f"superseded-point eviction failed: {evict_err}")
                superseded = conflict
            except Exception as sup_err:
                logger.warning(f"supersede execution failed: {sup_err}")

        return {
            "action": action,
            "reason": decision.get("reason"),
            "stored_id": stored_id,
            "superseded": superseded,
            "is_residual": decision.get("is_residual", False),
        }
    except Exception as e:
        logger.error(f"store_fact_versioned failed: {e}")
        return {
            "action": "ERROR",
            "reason": str(e),
            "stored_id": None,
            "superseded": None,
        }


async def _enrich_new_entities(entities: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Add contextual prefixes to newly created entities.

    Part of RAG Tier 1 Strategy - Contextual Enrichment
    Expected improvement: -35% retrieval failures

    Args:
        entities: List of entity dictionaries

    Returns:
        Statistics about enrichment (enriched count, tokens used, cost)
    """
    try:
        from contextual_llm import get_prefix_generator

        generator = get_prefix_generator()
        enriched_count = 0
        failed_count = 0

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        for entity in entities:
            try:
                entity_name = entity.get("name")
                entity_type = entity.get("entityType", "unknown")
                observations = entity.get("observations", [])

                # Get entity ID
                cursor.execute("SELECT id FROM entities WHERE name = ?", (entity_name,))
                result = cursor.fetchone()
                if not result:
                    logger.warning(f"Entity '{entity_name}' not found for enrichment")
                    failed_count += 1
                    continue

                entity_id = result[0]

                # Generate contextual prefix
                prefix, input_tokens, output_tokens = await generator.generate_prefix(
                    entity_name=entity_name,
                    entity_type=entity_type,
                    observations=observations,
                )

                # Get earliest observation timestamp to insert before it
                cursor.execute(
                    """
                    SELECT MIN(created_at) FROM observations WHERE entity_id = ?
                """,
                    (entity_id,),
                )
                min_created = cursor.fetchone()[0]

                # Use earlier timestamp to ensure prefix is first
                # Use SQL datetime format (YYYY-MM-DD HH:MM:SS) to match database format
                if min_created:
                    # Parse timestamp (handle both ISO and SQL formats)
                    if "T" in min_created:
                        dt = datetime.fromisoformat(min_created.replace("Z", "+00:00"))
                    else:
                        dt = datetime.strptime(min_created, "%Y-%m-%d %H:%M:%S")

                    # Subtract 1 second and format as SQL datetime
                    insert_time = (dt - timedelta(seconds=1)).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                else:
                    insert_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Insert contextual prefix as first observation
                cursor.execute(
                    """
                    INSERT INTO observations (entity_id, content, created_at)
                    VALUES (?, ?, ?)
                """,
                    (entity_id, prefix, insert_time),
                )

                enriched_count += 1

            except Exception as e:
                logger.error(f"Error enriching entity '{entity.get('name')}': {e}")
                failed_count += 1

        conn.commit()
        conn.close()

        # Get enrichment statistics
        stats = generator.get_stats()

        return {
            "enriched": enriched_count,
            "failed": failed_count,
            "tokens": {
                "input": stats.get("total_input_tokens", 0),
                "output": stats.get("total_output_tokens", 0),
            },
            "cost_usd": stats.get("estimated_cost_usd", 0.0),
            "using_llm": not stats.get("using_fallback", False),
            "backend": stats.get("backend", "unknown"),
        }

    except ImportError as e:
        logger.warning(f"Contextual enrichment not available: {e}")
        return {
            "enriched": 0,
            "failed": len(entities),
            "error": "contextual_llm module not available",
        }
    except Exception as e:
        logger.error(f"Error in contextual enrichment: {e}")
        return {"enriched": 0, "failed": len(entities), "error": str(e)}


def _score_meta() -> dict:
    try:
        import tpu_importance as _ti

        return dict(getattr(_ti, "LAST_SCORE_META", {}) or {})
    except Exception:
        return {}


async def _score_and_tier_entities(entities: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Score entity importance via TPU and assign appropriate memory tier.

    Importance scoring via the warm scoring service (port 8780) when reachable,
    else local heuristics. The response names the device that scored
    (warm_service_cpu / warm_service_tpu / heuristic).

    Tier assignment rules:
    - score >= 0.8: long_term (permanent storage, high importance)
    - score >= 0.6: episodic (time-bound experiences)
    - score < 0.6: working (temporary, session-scoped)

    Args:
        entities: List of entity dictionaries with observations

    Returns:
        Statistics about scoring and tier assignments
    """
    scored_count = 0
    tier_changes = {"long_term": 0, "episodic": 0, "working": 0}
    tpu_used = is_tpu_available() if TPU_SCORING_AVAILABLE else False

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        for entity in entities:
            try:
                entity_name = entity.get("name")
                observations = entity.get("observations", [])

                # Combine observations for scoring
                combined_text = f"{entity_name}: " + " ".join(
                    str(obs)
                    for obs in observations[:5]  # Limit for speed
                )

                # Score importance via TPU or heuristics
                importance = score_importance(
                    combined_text, context="memory", source="direct"
                )

                # Determine tier based on score
                if importance >= 0.8:
                    new_tier = "long_term"
                elif importance >= 0.6:
                    new_tier = "episodic"
                else:
                    new_tier = "working"

                # Update entity tier in database
                cursor.execute(
                    """
                    UPDATE entities
                    SET tier = ?
                    WHERE name = ?
                """,
                    (new_tier, entity_name),
                )

                if cursor.rowcount > 0:
                    tier_changes[new_tier] += 1
                    scored_count += 1

            except Exception as e:
                logger.debug(f"Error scoring entity '{entity.get('name')}': {e}")

        conn.commit()
        conn.close()

        return {
            "scored": scored_count,
            "tier_assignments": tier_changes,
            "warm_service_reachable": tpu_used,
            # Named from what actually scored (the warm service reports its
            # device per request; on 2026-08-25 it was "cpu" for every call
            # while advertising tpu_available). "tpu_warm_service" was the
            # old label regardless of device.
            "scoring_method": (
                f"warm_service_{_score_meta().get('device') or 'unknown'}"
                if tpu_used and _score_meta().get("method") != "heuristic"
                else "heuristic"
            ),
        }

    except Exception as e:
        logger.error(f"Error in TPU scoring: {e}")
        return {"scored": 0, "error": str(e), "tpu_available": False}


@app.tool()
async def search_nodes(
    query: str,
    limit: int = 10,
    viewer_agent: Optional[str] = None,
    scope: Optional[str] = None,
) -> dict[str, Any]:
    """
    Search for entities by name or type with automatic version history.

    CONCURRENT ACCESS: Uses memory-db Unix socket service for database operations.

    Args:
        query: Search query string
        limit: Maximum number of results
        viewer_agent: Optional scoped viewer. When set, results are filtered
            fail-closed: only PUBLIC/CLUSTER entities plus PRIVATE ones this
            agent owns or is granted. Omit for the orchestrator/system view.
        scope: Optional project filter. Restricts results to memories filed
            under one project folder: 'cfgi', 'arc-agi3', 'harness', 'hardware',
            'research', 'ops', 'kre', 'business', or 'global' for top-level
            files. Applied in SQL before the limit, so a scoped search of 10
            returns the top 10 WITHIN that project rather than whatever survives
            filtering a global top 10. Errors rather than silently returning
            everything if the scope table is missing.

    Returns:
        List of matching entities with version information
    """
    try:
        # Delegate to memory-db service for concurrent access
        response = await memory_client.search_nodes(query, limit, viewer_agent, scope)

        if response.get("success"):
            entities = response.get("results", response.get("entities", []))
            return {
                "query": query,
                "count": len(entities),
                "confidence": response.get("confidence", 0.0),
                "low_confidence": response.get("low_confidence", True),
                "results": entities,
            }
        else:
            return _daemon_failure(
                response.get("error", "Unknown error from memory-db service"),
                unreachable=False,
                query=query,
            )

    except Exception as e:
        logger.error(f"Error searching nodes via memory-db: {str(e)}")
        return _daemon_failure(
            f"Memory-DB service error: {e}",
            unreachable=isinstance(e, _UNREACHABLE_ERRORS),
            query=query,
        )


# NEW GIT-LIKE TOOLS


@app.tool()
async def memory_diff(
    entity_name: str, version1: int = None, version2: int = None
) -> dict:
    """
    Get diff between two versions of a memory.

    Args:
        entity_name: Name of the entity
        version1: First version number (default: current-1)
        version2: Second version number (default: current)

    Returns:
        Diff information between versions
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT id, current_branch FROM entities WHERE name = ?", (entity_name,)
    )
    entity = cursor.fetchone()
    if not entity:
        return {"error": "Entity not found"}

    entity_id, branch = entity

    if version2 is None:
        cursor.execute(
            """
            SELECT MAX(version_number) FROM memory_versions
            WHERE entity_id = ? AND branch_name = ?
        """,
            (entity_id, branch),
        )
        version2 = cursor.fetchone()[0]

    if version1 is None:
        version1 = max(1, version2 - 1)

    cursor.execute(
        """
        SELECT compressed_data, version_number, commit_message, created_at
        FROM memory_versions
        WHERE entity_id = ? AND version_number IN (?, ?) AND branch_name = ?
        ORDER BY version_number
    """,
        (entity_id, version1, version2, branch),
    )

    versions = cursor.fetchall()
    conn.close()

    if len(versions) != 2:
        return {"error": "Could not find both versions"}

    data1 = decompress_data(versions[0][0])
    data2 = decompress_data(versions[1][0])

    old_str = json.dumps(data1, indent=2, default=str)
    new_str = json.dumps(data2, indent=2, default=str)
    diff = difflib.unified_diff(
        old_str.splitlines(keepends=True),
        new_str.splitlines(keepends=True),
        fromfile=f"version_{versions[0][1]}",
        tofile=f"version_{versions[1][1]}",
    )

    return {
        "entity": entity_name,
        "branch": branch,
        "version1": {
            "number": versions[0][1],
            "message": versions[0][2],
            "timestamp": versions[0][3],
        },
        "version2": {
            "number": versions[1][1],
            "message": versions[1][2],
            "timestamp": versions[1][3],
        },
        "diff": "".join(diff),
    }


@app.tool()
async def memory_revert(entity_name: str, version: int) -> dict:
    """
    Revert a memory to a specific version.

    Args:
        entity_name: Name of the entity
        version: Version number to revert to

    Returns:
        Result of the revert operation
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT id, current_branch FROM entities WHERE name = ?", (entity_name,)
    )
    entity = cursor.fetchone()
    if not entity:
        return {"error": "Entity not found"}

    entity_id, branch = entity

    # Get the version data
    cursor.execute(
        """
        SELECT compressed_data FROM memory_versions
        WHERE entity_id = ? AND version_number = ? AND branch_name = ?
    """,
        (entity_id, version, branch),
    )

    version_data = cursor.fetchone()
    if not version_data:
        conn.close()
        return {"error": f"Version {version} not found"}

    # Update entity with old data
    cursor.execute(
        """
        UPDATE entities SET
            compressed_data = ?,
            last_accessed = CURRENT_TIMESTAMP,
            current_version = ?
        WHERE id = ?
    """,
        (version_data[0], version, entity_id),
    )

    # Create a new version entry for the revert
    data = decompress_data(version_data[0])
    create_version(entity_id, data, message=f"Reverted to version {version}")

    conn.commit()
    conn.close()

    return {
        "success": True,
        "entity": entity_name,
        "reverted_to": version,
        "branch": branch,
        "message": f"Successfully reverted to version {version}",
    }


@app.tool()
async def memory_branch(
    entity_name: str, branch_name: str, description: str = None
) -> dict:
    """
    Create a branch of a memory for experimentation.

    Args:
        entity_name: Name of the entity to branch
        branch_name: Name for the new branch
        description: Optional description of the branch purpose

    Returns:
        Result of the branch creation
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT id, current_branch, compressed_data FROM entities WHERE name = ?",
        (entity_name,),
    )
    entity = cursor.fetchone()
    if not entity:
        return {"error": "Entity not found"}

    entity_id, base_branch, current_data = entity

    # Get current version from base branch
    cursor.execute(
        """
        SELECT id FROM memory_versions
        WHERE entity_id = ? AND branch_name = ? AND is_current = 1
    """,
        (entity_id, base_branch),
    )

    base_version = cursor.fetchone()
    if not base_version:
        conn.close()
        return {"error": "No current version found"}

    # Create branch record
    cursor.execute(
        """
        INSERT INTO memory_branches (entity_id, branch_name, base_version_id, description)
        VALUES (?, ?, ?, ?)
    """,
        (entity_id, branch_name, base_version[0], description),
    )

    # Copy current version to new branch
    cursor.execute(
        """
        INSERT INTO memory_versions
        (entity_id, version_number, compressed_data, commit_message,
         author, is_current, branch_name, parent_version_id)
        VALUES (?, 1, ?, ?, 'system', 1, ?, ?)
    """,
        (
            entity_id,
            current_data,
            f"Branch created from {base_branch}",
            branch_name,
            base_version[0],
        ),
    )

    conn.commit()
    conn.close()

    return {
        "success": True,
        "entity": entity_name,
        "branch": branch_name,
        "base_branch": base_branch,
        "description": description,
        "message": f"Branch '{branch_name}' created successfully",
    }


@app.tool()
async def detect_memory_conflicts(
    threshold: float = 0.85,
    entity_type_prefix: str = None,
    max_entities: int = 800,
) -> dict:
    """
    Detect duplicate or near-duplicate memories via pairwise text similarity.

    Args:
        threshold: Similarity threshold (0.0 to 1.0)
        entity_type_prefix: Optional filter, e.g. 'auto_memory/' to dedup only
            curated memories (recommended — the full store has ~9.5k entities).
        max_entities: Cap on the newest-N entities scanned (default 1500) to keep
            runtime bounded.

    Returns:
        Detected conflicts.

    Fixes vs the original: (1) decompresses each entity ONCE instead of n times in
    the inner loop; (2) tolerates all compressed_data encodings via decompress_data
    (the original raised on ~76% of entities); (3) bounds the scan + uses difflib
    real_quick_ratio/quick_ratio pre-filters so it returns quickly instead of an
    unbounded O(n^2) full-ratio over the whole store (which would stall the daemon).
    """
    from difflib import SequenceMatcher

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    if entity_type_prefix:
        cursor.execute(
            "SELECT id, name, compressed_data FROM entities WHERE entity_type LIKE ? ORDER BY id DESC",
            (entity_type_prefix + "%",),
        )
    else:
        cursor.execute(
            "SELECT id, name, compressed_data FROM entities ORDER BY id DESC"
        )
    rows = cursor.fetchall()
    truncated = len(rows) > max_entities
    rows = rows[:max_entities]

    # Decompress each entity ONCE (the original re-decompressed inside the loop),
    # and compare only the first 400 chars (name + description + start of body):
    # SequenceMatcher is O(len^2), so full-text over hundreds of entities took ~44s
    # vs ~5s at 400 chars with identical strong-duplicate detection.
    CMP_LEN = 400
    items = []  # (id, name, text)
    for eid, name, blob in rows:
        if not blob:
            continue
        try:
            items.append((eid, name, str(decompress_data(blob))[:CMP_LEN]))
        except Exception:
            continue  # skip unreadable rather than crash the whole scan

    conflicts = []
    for i in range(len(items)):
        id1, name1, t1 = items[i]
        for j in range(i + 1, len(items)):
            id2, name2, t2 = items[j]
            sm = SequenceMatcher(None, t1, t2)
            # Cheap pre-filters (O(1) then O(n)) before the O(n*m) full ratio.
            if sm.real_quick_ratio() < threshold or sm.quick_ratio() < threshold:
                continue
            similarity = sm.ratio()
            if similarity > threshold:
                cursor.execute(
                    """
                    INSERT OR IGNORE INTO memory_conflicts
                    (entity1_id, entity2_id, conflict_type, similarity_score)
                    VALUES (?, ?, 'duplicate', ?)
                """,
                    (id1, id2, similarity),
                )
                conflicts.append(
                    {
                        "entity1": {"id": id1, "name": name1},
                        "entity2": {"id": id2, "name": name2},
                        "similarity": f"{similarity:.2%}",
                        "type": "duplicate" if similarity > 0.95 else "overlap",
                    }
                )

    conn.commit()
    conn.close()

    return {
        "conflicts_detected": len(conflicts),
        "threshold": threshold,
        "scanned": len(items),
        "scope": entity_type_prefix or "all",
        "truncated": truncated,
        "conflicts": conflicts[:50],
        "recommendation": (
            "Review conflicts and consider merging or removing duplicates."
            + (
                f" NOTE: scanned the newest {max_entities} entities; narrow with "
                f"entity_type_prefix='auto_memory/' or raise max_entities to widen."
                if truncated
                else ""
            )
        ),
    }


@app.tool()
async def save_implementation_plan(
    name: str, steps: list[dict], description: str = None
) -> dict:
    """
    Save a structured implementation plan.

    Args:
        name: Plan name
        steps: List of step dictionaries
        description: Optional plan description

    Returns:
        Result of the save operation
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create entity for the plan
    entity_name = f"plan_{name}"

    # Create as entity with versioning. create_entities is @app.tool()-wrapped
    # (a FunctionTool, not callable — calling it directly raised
    # "'FunctionTool' object is not callable" on every invocation of this
    # tool); .fn is the underlying coroutine.
    await create_entities.fn(
        [
            {
                "name": entity_name,
                "entityType": "implementation_plan",
                "observations": [
                    f"Step {i + 1}: {step}" for i, step in enumerate(steps)
                ],
            }
        ]
    )

    # Also save in specialized table
    cursor.execute("SELECT id FROM entities WHERE name = ?", (entity_name,))
    entity_id = cursor.fetchone()[0]

    cursor.execute(
        """
        INSERT INTO implementation_plans (name, description, steps, entity_id)
        VALUES (?, ?, ?, ?)
    """,
        (name, description, json.dumps(steps), entity_id),
    )

    conn.commit()
    conn.close()

    return {
        "success": True,
        "name": name,
        "step_count": len(steps),
        "entity_name": entity_name,
        "versioned": True,
        "message": f"Implementation plan '{name}' saved with version control",
    }


@app.tool()
async def get_memory_status() -> dict:
    """
    Get overall memory system status and statistics.

    CONCURRENT ACCESS: Uses memory-db Unix socket service for core stats.

    Returns:
        System statistics and health information
    """
    try:
        # Get basic stats from memory-db service
        response = await memory_client.get_memory_status()

        if response.get("success"):
            # Return the stats from memory-db
            return response
        else:
            return _daemon_failure(
                response.get("error", "Unknown error from memory-db service"),
                unreachable=False,
            )

    except Exception as e:
        logger.error(f"Error getting memory status via memory-db: {str(e)}")
        return _daemon_failure(
            f"Memory-DB service error: {e}",
            unreachable=isinstance(e, _UNREACHABLE_ERRORS),
        )


# === CODE EXECUTION TOOL ===
@app.tool()
async def execute_code(
    code: str, context_vars: dict[str, Any] | None = None
) -> dict[str, Any]:
    """
        Execute Python code in secure sandbox with API access.

        Implements Anthropic's code execution pattern for massive token reduction.
        Agents write code using APIs instead of calling tools directly.

        Token Savings:
        - Progressive disclosure: 2,000 → 200 tokens (90% reduction)
        - Local processing: 50,000 → 500 tokens (99% reduction)
        - Average: 96.6% token reduction

        Security Features:
        - RestrictedPython compilation
        - 30-second timeout
        - 500MB memory limit
        - Dangerous import blocking
        - PII tokenization

        Available APIs in code:
        - Memory: create_entities, search_nodes, get_status, update_entity
        - Versioning: diff, revert, branch, history, commit
        - Analysis: detect_conflicts, analyze_patterns, classify_content, find_related
        - Utils: filter_by_confidence, summarize_results, aggregate_stats, format_output
        - Filesystem: workspace, list_files, read_file, write_file, delete_file
        - Skills: save_skill, load_skill, list_skills

        Example Code:
            # Basic search and filter
            results = search_nodes("optimization", limit=100)
            high_conf = filter_by_confidence(results, 0.8)
            summary = summarize_results(high_conf)
            result = summary  # Return this

            # Save intermediate state
            write_file("results.json", json.dumps(results))

            # Save working code as skill
            code = '''
    def filter_high_confidence(query, threshold=0.8):
        results = search_nodes(query, limit=1000)
        return [r for r in results if r.confidence > threshold]
    '''
            save_skill("filter_high_confidence", code, "Filter memories by confidence")

        Args:
            code: Python code to execute
            context_vars: Additional variables to make available

        Returns:
            Execution result with success status, result data, and any errors
    """
    logger.info("🔧 Code execution requested")

    # Security check
    is_safe, safety_issues = comprehensive_safety_check(code)
    if not is_safe:
        logger.warning(f"⚠️  Code safety check failed: {safety_issues}")
        return {
            "success": False,
            "error": "Code safety check failed",
            "issues": safety_issues,
        }

    # Create executor FIRST (so it can create workspace)
    executor = CodeExecutor(timeout_seconds=30, memory_limit_bytes=500 * 1024 * 1024)

    # Create API context with all available functions (pass executor for filesystem access)
    api_context = create_api_context(executor=executor)

    # Add any additional context variables
    if context_vars:
        api_context.update(context_vars)

    # Execute code in sandbox
    exec_result = executor.execute(code, context=api_context)

    if exec_result.success:
        # Sanitize output (PII tokenization, size limits)
        sanitized_result = sanitize_output(exec_result.result)
        logger.info(
            f"✅ Code executed successfully in {exec_result.execution_time_ms:.2f}ms"
        )
        return {
            "success": True,
            "result": sanitized_result,
            "stdout": exec_result.stdout,
            "execution_time_ms": exec_result.execution_time_ms,
        }
    else:
        logger.error(f"❌ Code execution failed: {exec_result.error}")
        return {
            "success": False,
            "error": exec_result.error,
            "stdout": exec_result.stdout,
            "stderr": exec_result.stderr,
            "execution_time_ms": exec_result.execution_time_ms,
        }


# === REASONING PRIORITIZATION INTEGRATION (75/15 Rule) ===
try:
    from reasoning_tools import register_reasoning_tools
    # Don't register yet - will do in main block after db init
except Exception as e:
    logger.warning(f"⚠️  Reasoning prioritization integration skipped: {e}")

# === NEURAL MEMORY FABRIC INTEGRATION ===
try:
    from nmf_tools import register_nmf_tools
    # Don't register yet - will do in main block after db init
except Exception as e:
    logger.warning(f"⚠️  NMF integration skipped: {e}")


# =============================================================================
# CONSOLIDATED TOOL SURFACE (see docs/mcp/tool-design-standard.md)
#
# When this surface was designed the server registered 142 tools, ~23,334 tokens
# of schema, EAGERLY LOADED into every context window of every session whether or
# not memory is touched. That was 80% of the harness's entire eager tool-schema tax.
#
# MEASURED ON A CLEAN CLONE (stdio tools/list, Python 3.11): 186 tools with the
# core dependency set, 204 with the optional backends. Both rise by 7 if
# AGENTIC_SYSTEM_PATH points at a checkout containing scripts/graph-rag.py,
# which is NOT shipped here. The 142 figure below is the historical decision
# point, not a current count.
#
# If you re-measure, unset AGENTIC_SYSTEM_PATH first. An earlier draft of this
# comment said 188 because it was measured on a machine that had it set, and
# a second machine with the same variable "confirmed" it.
#
# Tool count is a selection-accuracy variable, not just a token-cost one: ~18 of
# those tools were all "retrieve memories given a query" (search_nodes,
# semantic_recall, graph_enhanced_search, omnimem_nhop_retrieve, nmf_recall,
# atommem_graph_recall, find_similar_visual, search_by_emotion, ...). Asking a
# model to pick one of 18 near-synonyms, against 124 other distractors, is a
# selection problem we manufactured for ourselves.
#
# Rather than RENAME every tool (which would break CLAUDE.md, ~15 agent prompts and
# several hooks that call these names by hand), we keep the front-door names the
# harness already instructs and demote the long tail to hidden-but-callable:
#
#   registered : search_nodes, semantic_recall, create_entities,
#                get_memory_status, execute_code
#                + memory_catalog (discover the rest)
#                + memory_call    (invoke any of the rest)
#
# NOTHING IS LOST. Every hidden tool stays reachable through memory_call, and
# memory_catalog makes it discoverable. This is the same allowlist + escape-hatch
# pattern proven on cognitum-bridge (130 -> 14).
#
# Reversible without a code change: ENHANCED_MEMORY_CONSOLIDATE=0 restores all 142.
# =============================================================================

MEMORY_KEEP_TOOLS = {
    # The retrieval + write front doors CLAUDE.md, the agent prompts and the
    # hooks reference by name. Renaming these would buy no accuracy (they are
    # not ambiguous siblings; they are the primary entry points) and would break
    # a lot of callers.
    "search_nodes",
    "semantic_recall",
    "create_entities",
    "get_memory_status",
    # The pre-existing code-execution escape hatch.
    "execute_code",
    # The two facades registered below.
    "memory_catalog",
    "memory_call",
}

_HIDDEN_TOOLS: dict[str, Any] = {}

# The handful of operations the agent reaches for cold, many times a session.
# Everything else is long tail.
FRONT_DOOR_TOOLS = {
    "search_nodes",
    "semantic_recall",
    "create_entities",
    "get_memory_status",
    "execute_code",
}


def _stamp_front_door_meta() -> int:
    """Mark ONLY the front-door tools as eagerly-loaded, per-tool.

    The better surface (see evaluation/mcp_selection_ab/RESULTS.md): register
    every tool, but flag only the front door with `anthropic/alwaysLoad` so the
    client loads those schemas up front and leaves the rest deferred. The long
    tail then costs its NAMES, not its schemas, and the client's own ToolSearch
    reaches any of it in ONE hop.

    That beats hiding the tail behind memory_call, which measured 2-4 hops
    because memory_catalog is a worse reimplementation of ToolSearch.

    Requires the MCP client to honor per-tool `_meta`. Server-level `alwaysLoad`
    in ~/.claude.json is verified to work (it removes the ToolSearch hop); this
    is the per-tool form of the same flag.
    """
    manager = app._tool_manager
    stamped = 0
    for name in FRONT_DOOR_TOOLS:
        tool = manager._tools.get(name)
        if tool is None:
            logger.warning("front-door tool %r not registered; cannot stamp", name)
            continue
        meta = dict(getattr(tool, "meta", None) or {})
        meta["anthropic/alwaysLoad"] = True
        tool.meta = meta
        stamped += 1
    return stamped


def _consolidate_tool_surface() -> None:
    """Demote the long tail of memory tools to hidden-but-callable.

    Runs after every register_*_tools() call, immediately before transport start,
    so it sees the fully-populated registry.
    """
    # ENHANCED_MEMORY_SURFACE selects the tool surface:
    #
    #   frontdoor (DEFAULT) register every tool; stamp ONLY the front door
    #                       with anthropic/alwaysLoad so the client loads those
    #                       schemas eagerly and leaves the tail deferred. The
    #                       client's own ToolSearch reaches the tail in ONE hop.
    #   consolidated        register 7 tools; hide the tail behind memory_call.
    #   full                register everything, stamp nothing (all deferred).
    #
    # `frontdoor` is the default because it MEASURED best. Head-to-head over 14
    # tasks (evaluation/mcp_selection_ab/):
    #
    #                        consolidated   frontdoor
    #   right tool 1st try       5/14        13/14
    #   mean MCP calls           3.43         1.57
    #   long-tail median          3            1
    #   prompt tokens          74,135       76,377
    #
    # i.e. ~2.2k extra tokens (the deferred tool NAMES) halves the number of tool
    # calls. memory_catalog + memory_call turned out to be a worse
    # reimplementation of ToolSearch, which the client already has.
    #
    # NOTE: this only works if enhanced-memory does NOT carry server-level
    # "alwaysLoad": true in ~/.claude.json. That flag forces ALL tools eager and
    # would reinstate the ~23k/turn schema tax it was hiding.
    surface = os.environ.get("ENHANCED_MEMORY_SURFACE", "").lower()
    if not surface:
        # Back-compat: the old boolean still selects the old behavior.
        surface = (
            "consolidated"
            if os.environ.get("ENHANCED_MEMORY_CONSOLIDATE") == "1"
            else "frontdoor"
        )

    if surface != "consolidated":
        total = len(app._tool_manager._tools)
        if surface == "full":
            logger.info(f"Surface=full: all {total} tools registered, none stamped")
            return
        n = _stamp_front_door_meta()
        logger.info(
            f"Surface=frontdoor: {total} tools registered; {n} front-door tools "
            f"stamped anthropic/alwaysLoad, the other {total - n} left deferred "
            f"(ToolSearch reaches them in one hop)"
        )
        return

    manager = app._tool_manager
    before = len(manager._tools)

    for name, tool in list(manager._tools.items()):
        if name in MEMORY_KEEP_TOOLS:
            continue
        _HIDDEN_TOOLS[name] = tool
        manager.remove_tool(name)

    @app.tool()
    async def memory_catalog(query: str = "", limit: int = 40) -> dict:
        """
        Discover the advanced memory tools that are not registered directly.

        The memory server exposes a small front door (search_nodes,
        semantic_recall, create_entities, get_memory_status, execute_code) plus
        ~137 specialist tools kept out of the tool list to preserve selection
        accuracy. This lists those specialists so you can invoke them with
        memory_call.

        Covers: episodic/semantic/procedural tiers, graph + causal + temporal
        reasoning, visual and cross-modal memory, consolidation and decay,
        provenance and L-scores, emotional tagging, ART clustering, NMF blocks,
        semantic cache, and self-improvement cycles.

        Args:
            query: Filter by keyword against tool name and description.
                   Empty returns everything (grouped, name + one-line summary).
            limit: Max tools to return.

        Returns:
            Matching tool names with their one-line descriptions.
        """
        q = (query or "").lower().strip()
        hits = []
        for name, tool in sorted(_HIDDEN_TOOLS.items()):
            desc = (tool.description or "").strip()
            summary = desc.split("\n")[0][:140]
            if q and q not in name.lower() and q not in desc.lower():
                continue
            hits.append({"tool": name, "summary": summary})

        return {
            "matched": len(hits),
            "total_hidden": len(_HIDDEN_TOOLS),
            "showing": min(len(hits), limit),
            "tools": hits[:limit],
            "usage": "memory_call(tool='<name>', arguments={...})",
        }

    @app.tool()
    async def memory_call(tool: str, arguments: dict | None = None) -> Any:
        """
        Invoke any advanced memory tool by name.

        Escape hatch for the ~137 specialist memory tools that are not registered
        directly. Use memory_catalog to find the tool name and its arguments,
        then call it here. Behaves exactly as the tool would if registered.

        Args:
            tool: Tool name, e.g. "run_full_consolidation", "get_causal_chain",
                  "store_fact_versioned", "graph_enhanced_search".
            arguments: That tool's arguments as an object. Omit if it takes none.

        Returns:
            Whatever the underlying tool returns.
        """
        target = _HIDDEN_TOOLS.get(tool)
        if target is None:
            if tool in MEMORY_KEEP_TOOLS:
                return {
                    "error": f"{tool!r} is registered directly; call it as a tool, "
                    f"not through memory_call.",
                }
            return {
                "error": f"unknown memory tool: {tool!r}",
                "suggestion": "Use memory_catalog(query=...) to find the right name.",
            }
        result = target.fn(**(arguments or {}))
        if inspect.isawaitable(result):
            result = await result
        return result

    after = len(manager._tools)
    logger.info(
        f"Tool surface consolidated: {before} -> {after} registered "
        f"({len(_HIDDEN_TOOLS)} hidden, reachable via memory_call)"
    )


if __name__ == "__main__":
    _MEMORY_PROFILE = os.environ.get("MEMORY_PROFILE", "full").lower()
    _MINIMAL_MODE = _MEMORY_PROFILE in ("minimal", "codex", "gemini")
    _LITE_MODE = _MEMORY_PROFILE in (
        "orchestrator",
        "lite",
        "minimal",
        "codex",
        "gemini",
    )
    logger.info("Enhanced Memory MCP Server with Git Features starting...")
    logger.info(f"Database: {DB_PATH}")
    logger.info(f"Memory profile: {_MEMORY_PROFILE} (lite_mode={_LITE_MODE})")

    # Initialize database FIRST, inside main block
    init_database()

    # The daemon owns the data; say so at boot if the two paths disagree.
    _check_db_path_agreement()

    def _run_transport() -> None:
        _consolidate_tool_surface()
        # Transport selection via MCP_TRANSPORT env:
        #   "stdio" (default)         — one subprocess per local client
        #   "sse" | "streamable-http" — HTTP daemon shared across sessions
        transport_mode = os.environ.get("MCP_TRANSPORT", "stdio").lower()
        if transport_mode in ("sse", "streamable-http", "http"):
            host = os.environ.get("MCP_HOST", "127.0.0.1")
            port = int(os.environ.get("MCP_PORT", "9106"))
            try:
                app.settings.host = host
                app.settings.port = port
            except AttributeError as host_err:
                logger.warning(
                    "FastMCP settings unavailable (%s) — using defaults", host_err
                )
            logger.info(
                f"Enhanced-memory HTTP transport on {host}:{port} ({transport_mode})"
            )
            # Arm SIGUSR1 stack dumps + timestamped logging (accept-then-stall
            # self-diagnosis, 2026-08-23). Port-keyed so an operator can tell
            # two HTTP daemons apart in $STACK_DUMP_DIR. The helper is
            # fail-soft by design (see stack_dump_on_signal.py): it never
            # takes the daemon down, so an ImportError here is a packaging
            # defect and is reported loudly rather than swallowed.
            from stack_dump_on_signal import install as _sd_install

            _sd_install(f"port-{port}")
            app.run(
                transport="sse" if transport_mode == "sse" else "streamable-http",
                show_banner=False,
            )
        else:
            # Disable banner to prevent stdout pollution (MCP protocol requirement)
            app.run(transport="stdio", show_banner=False)

    if _MINIMAL_MODE:
        logger.info("Minimal memory profile: skipping optional integrations")
        _run_transport()
        raise SystemExit(0)

    # Register reasoning tools after database is ready
    try:
        from reasoning_tools import register_reasoning_tools

        register_reasoning_tools(app, DB_PATH)
        logger.info("✅ Reasoning Prioritization (75/15 rule) integrated")
    except Exception as e:
        logger.warning(f"⚠️  Reasoning prioritization integration skipped: {e}")

    # Register NMF tools after database is ready
    try:
        from nmf_tools import register_nmf_tools

        register_nmf_tools(app)
        logger.info("✅ Neural Memory Fabric tools integrated")
    except Exception as e:
        logger.warning(f"⚠️  NMF integration skipped: {e}")

    # Register SAFLA 4-tier memory tools
    try:
        from safla_orchestrator import SAFLAOrchestrator

        safla = SAFLAOrchestrator(DB_PATH)
        logger.info("✅ SAFLA 4-tier memory initialized")

        # SAFLA tool registration
        from safla_tools import register_safla_tools

        register_safla_tools(app, safla)
        logger.info("✅ SAFLA tools integrated")
    except Exception as e:
        logger.warning(f"⚠️  SAFLA integration skipped: {e}")

    # Register AGI Memory tools (Phase 1: Cross-session identity & memory-action loop)
    try:
        from agi_tools import register_agi_tools

        register_agi_tools(app, DB_PATH)
        logger.info("✅ AGI Memory tools integrated (Phase 1: Identity & Actions)")
    except Exception as e:
        logger.warning(f"⚠️  AGI Memory Phase 1 integration skipped: {e}")

    # Register AGI Memory Phase 2 tools (Temporal reasoning & consolidation)
    try:
        from agi_tools_phase2 import register_agi_phase2_tools

        register_agi_phase2_tools(app, DB_PATH)
        logger.info(
            "✅ AGI Memory Phase 2 tools integrated (Temporal Reasoning & Consolidation)"
        )
    except Exception as e:
        logger.warning(f"⚠️  AGI Memory Phase 2 integration skipped: {e}")

    # Register AGI Memory Phase 3 tools (Emotional tagging & associative networks)
    try:
        from agi_tools_phase3 import register_agi_phase3_tools

        register_agi_phase3_tools(app, DB_PATH)
        logger.info(
            "✅ AGI Memory Phase 3 tools integrated (Emotional Tagging & Associative Networks)"
        )
    except Exception as e:
        logger.warning(f"⚠️  AGI Memory Phase 3 integration skipped: {e}")

    # Register AGI Memory Phase 4 tools (Meta-cognition & self-improvement)
    try:
        from agi_tools_phase4 import register_agi_phase4_tools

        register_agi_phase4_tools(app, DB_PATH)
        logger.info(
            "✅ AGI Memory Phase 4 tools integrated (Meta-Cognitive Awareness & Self-Improvement)"
        )
    except Exception as e:
        logger.warning(f"⚠️  AGI Memory Phase 4 integration skipped: {e}")

    # Register Provenance & L-Score tools (God Agent integration - Phase 1)
    try:
        from provenance import register_provenance_tools

        register_provenance_tools(app, DB_PATH)
        logger.info(
            "✅ Provenance/L-Score tools integrated (God Agent Phase 1: Source chain tracking)"
        )
    except Exception as e:
        logger.warning(f"⚠️  Provenance/L-Score integration skipped: {e}")

    # Register Shadow Vector tools (God Agent integration - Phase 3: Adversarial Validation)
    if _LITE_MODE:
        logger.info("⏭️  Shadow Vector tools skipped (lite_mode): Qdrant probe disabled")
    else:
        try:
            from shadow_vector import register_shadow_vector_tools

            register_shadow_vector_tools(app, DB_PATH)
            logger.info(
                "✅ Shadow Vector tools integrated (God Agent Phase 3: Adversarial contradiction detection)"
            )
        except Exception as e:
            logger.warning(f"⚠️  Shadow Vector integration skipped: {e}")

    # Surprise-Based Consolidation tools REMOVED 2026-08-25: calculate_surprise_score
    # rated every input (including verbatim copies of stored observations) novelty
    # 1.0 because its embedding lookup always returned [], and run_surprise_consolidation
    # failed on a missing column every call.

    # Register ART (Adaptive Resonance Theory) tools - Online learning without catastrophic forgetting
    try:
        from art_tools import register_art_tools

        register_art_tools(app)
        logger.info(
            "✅ ART tools integrated (Fuzzy ART clustering, vigilance control, hybrid architecture)"
        )
    except Exception as e:
        logger.warning(f"⚠️  ART integration skipped: {e}")

    # Initialize Neural Memory Fabric for RAG tools
    nmf_instance = None
    if _LITE_MODE:
        logger.info(
            "⏭️  NMF RAG init skipped (lite_mode): re-ranking/hybrid/query-expansion/multi-query/contextual tools disabled"
        )
    else:
        try:
            from neural_memory_fabric import get_nmf

            nmf_instance = asyncio.run(get_nmf())
            logger.info("✅ Neural Memory Fabric initialized for RAG")
        except Exception as e:
            logger.warning(f"⚠️  NMF initialization skipped: {e}")

    # Register Re-ranking tools (RAG Tier 1 Strategy) - NMF backend
    if nmf_instance:
        try:
            from reranking_tools_nmf import register_reranking_tools_nmf

            register_reranking_tools_nmf(app, nmf_instance)
            logger.info(
                "✅ Re-ranking (RAG Tier 1) integrated with NMF/Qdrant - Expected +40-55% precision"
            )
        except Exception as e:
            logger.warning(f"⚠️  Re-ranking integration skipped: {e}")
    else:
        logger.warning("⚠️  Re-ranking skipped: NMF not available")

    # Register Hybrid Search tools (RAG Tier 1 Strategy) - NMF backend
    if nmf_instance:
        try:
            from hybrid_search_tools_nmf import register_hybrid_search_tools_nmf

            register_hybrid_search_tools_nmf(app, nmf_instance)
            logger.info(
                "✅ Hybrid Search (RAG Tier 1) integrated with NMF/Qdrant - Expected +20-30% recall"
            )
        except Exception as e:
            logger.warning(f"⚠️  Hybrid search integration skipped: {e}")
    else:
        logger.warning("⚠️  Hybrid search skipped: NMF not available")

    # Register Query Expansion tools (RAG Tier 2 Strategy) - Query Optimization
    if nmf_instance:
        try:
            from query_expansion_tools import register_query_expansion_tools

            register_query_expansion_tools(app, nmf_instance)
            logger.info(
                "✅ Query Expansion (RAG Tier 2) integrated - Expected +15-25% recall"
            )
        except Exception as e:
            logger.warning(f"⚠️  Query expansion integration skipped: {e}")
    else:
        logger.warning("⚠️  Query expansion skipped: NMF not available")

    # Register Multi-Query RAG tools (RAG Tier 2 Strategy) - Query Optimization
    if nmf_instance:
        try:
            from multi_query_rag_tools import register_multi_query_rag_tools

            register_multi_query_rag_tools(app, nmf_instance)
            logger.info(
                "✅ Multi-Query RAG (RAG Tier 2) integrated - Expected +20-30% coverage"
            )
        except Exception as e:
            logger.warning(f"⚠️  Multi-Query RAG integration skipped: {e}")
    else:
        logger.warning("⚠️  Multi-Query RAG skipped: NMF not available")

    # Register Contextual Retrieval tools (RAG Tier 3.1 Strategy) - Context Enhancement
    if nmf_instance:
        try:
            from contextual_retrieval_tools import register_contextual_retrieval_tools

            register_contextual_retrieval_tools(app, nmf_instance)
            logger.info(
                "✅ Contextual Retrieval (RAG Tier 3.1) integrated - Expected +35-49% accuracy"
            )
        except Exception as e:
            logger.warning(f"⚠️  Contextual Retrieval integration skipped: {e}")

    # Register Hierarchical RAG tools (RAG Tier 3.3 Strategy) - Multi-level Indexing
    if nmf_instance:
        try:
            from hierarchical_rag_tools import register_hierarchical_rag_tools

            register_hierarchical_rag_tools(app, nmf_instance)
            logger.info(
                "✅ Hierarchical RAG (RAG Tier 3.3) integrated - document → section → chunk indexing"
            )
        except Exception as e:
            logger.warning(f"⚠️  Hierarchical RAG integration skipped: {e}")

    # Register GraphRAG tools (RAG Tier 4 Strategy) - Knowledge Graph Retrieval
    try:
        from graphrag_tools import register_graphrag_tools

        register_graphrag_tools(app, db_path=DB_PATH)
        logger.info(
            "✅ GraphRAG (RAG Tier 4) integrated - graph-enhanced search + entity neighbors"
        )
    except Exception as e:
        # Optional external integration: not shipped, not required.
        logger.warning(f"⚠️  GraphRAG (optional external integration) skipped: {e}")

    # Register Agentic RAG tools (RAG Tier 4.1+4.3 Strategy) - Self-Reflective Retrieval
    if nmf_instance:
        try:
            from agentic_rag_tools import register_agentic_rag_tools

            register_agentic_rag_tools(app, nmf_instance)
            logger.info(
                "✅ Agentic RAG (RAG Tier 4.1+4.3) integrated - autonomous + self-reflective retrieval"
            )
        except Exception as e:
            logger.warning(f"⚠️  Agentic RAG integration skipped: {e}")

    # Register Visual Memory tools (RAG Tier 4 Strategy) - TPU-Powered Visual Embeddings
    try:
        from visual_memory_tools import register_visual_memory_tools

        register_visual_memory_tools(app, use_tpu=True)
        logger.info(
            "✅ Visual Memory (RAG Tier 4) integrated - TPU embeddings for visual similarity"
        )
    except Exception as e:
        logger.warning(f"⚠️  Visual Memory integration skipped: {e}")

    # Register the vector/semantic honest-degradation health tool + guard proxy.
    # The guard wraps the vector/semantic tool entrypoints so a missing optional
    # dependency (qdrant_client / sentence_transformers) or an unreachable Qdrant
    # returns a STRUCTURED {status, reason, remediation} dict instead of leaking a
    # ModuleNotFoundError / traceback to the caller.
    try:
        from vector_health import (
            GuardedToolApp,
            _semantic_cache_precheck,
            _vector_precheck,
            register_vector_health_tools,
        )

        register_vector_health_tools(app)
        logger.info(
            "✅ Vector/semantic health tool integrated (vector_semantic_health)"
        )
    except Exception as e:
        logger.warning(f"⚠️  Vector/semantic health tool integration skipped: {e}")
        GuardedToolApp = None  # type: ignore

    # Register Semantic Cache tools - LLM reasoning result caching with 30-40% hit rate
    try:
        from semantic_cache_tools import register_semantic_cache_tools

        cache_app = (
            GuardedToolApp(app, _semantic_cache_precheck)
            if GuardedToolApp is not None
            else app
        )
        register_semantic_cache_tools(cache_app)
        logger.info(
            "✅ Semantic Cache integrated - 30-40% hit rate, sub-50ms retrieval (guarded)"
        )
    except Exception as e:
        logger.warning(f"⚠️  Semantic Cache integration skipped: {e}")

    # Register OmniMEM N-hop Graph Retrieval tools (OmniMEM-inspired, April 2026)
    try:
        from omnimem_nhop_tools import register_omnimem_nhop_tools

        register_omnimem_nhop_tools(app, DB_PATH)
        logger.info(
            "✅ OmniMEM N-hop graph retrieval integrated (distance-decaying relevance, set-union fusion)"
        )
    except Exception as e:
        logger.warning(f"⚠️  OmniMEM N-hop integration skipped: {e}")

    # Register native semantic (vector) recall — searches the Qdrant
    # enhanced_memory collection via fedora nomic-embed-text (added 2026-06-07).
    try:
        from semantic_vector_tools import register_semantic_vector_tools

        vector_app = (
            GuardedToolApp(app, _vector_precheck) if GuardedToolApp is not None else app
        )
        register_semantic_vector_tools(vector_app, DB_PATH)
        logger.info(
            "✅ Semantic vector recall integrated (fedora nomic-embed-text, Qdrant enhanced_memory, guarded)"
        )
    except Exception as e:
        logger.warning(f"⚠️  Semantic vector recall integration skipped: {e}")

    # Register AtomMem upgrades (atomic facts, IDF keyword graph, residual-delta
    # verification, versioned temporal profiles — arXiv 2606.19847, added 2026-06-19).
    try:
        from atommem.tools import register_atommem_tools

        register_atommem_tools(app, DB_PATH)
        logger.info(
            "✅ AtomMem upgrades integrated (atomic-fact extraction, IDF keyword graph, "
            "residual-delta verify, versioned temporal profiles)"
        )
    except Exception as e:
        logger.warning(f"⚠️  AtomMem upgrades integration skipped: {e}")

    # Session evidence retention (the "never lose the road back to evidence"
    # layer): full tool outputs on disk, a compact navigable sketch in context.
    try:
        from session_evidence_tools import register_session_evidence_tools

        register_session_evidence_tools(app, DB_PATH)
        logger.info(
            "✅ Session evidence retention integrated (evidence_log/sketch/get/search/prune)"
        )
    except Exception as e:
        logger.warning(f"⚠️  Session evidence retention integration skipped: {e}")

    # Governance: per-agent ACL + visibility tools (Phase D, 2026-08-05).
    try:
        from governance_tools import register_governance_tools

        register_governance_tools(app, DB_PATH)
        logger.info(
            "✅ Governance integrated (set/get visibility, owner, grant/revoke access)"
        )
    except Exception as e:
        logger.warning(f"⚠️  Governance integration skipped: {e}")

    # Retrieval quality diagnostics (Phase G, 2026-08-05): explain quiet misses.
    try:
        from retrieval_diagnostics_tools import register_retrieval_diagnostics_tools

        register_retrieval_diagnostics_tools(app, memory_client)
        logger.info("✅ Retrieval diagnostics integrated (retrieval_diagnostics)")
    except Exception as e:
        logger.warning(f"⚠️  Retrieval diagnostics integration skipped: {e}")

    # Tier health + promotion (Phase H3, 2026-08-05).
    try:
        from tier_health_tools import register_tier_health_tools

        register_tier_health_tools(app, DB_PATH)
        logger.info(
            "✅ Tier health integrated (tier_health, promote_to_semantic/episodic)"
        )
    except Exception as e:
        logger.warning(f"⚠️  Tier health integration skipped: {e}")

    # Knowledge assets: Wiki + CodeGraph (Phase E, 2026-08-05).
    try:
        from knowledge_wiki_tools import register_knowledge_wiki_tools
        from knowledge_codegraph_tools import register_knowledge_codegraph_tools

        register_knowledge_wiki_tools(app, DB_PATH)
        register_knowledge_codegraph_tools(app, DB_PATH)
        logger.info(
            "✅ Knowledge assets integrated (wiki_ingest/search, codegraph_index/callers/callees/impact)"
        )
    except Exception as e:
        logger.warning(f"⚠️  Knowledge assets integration skipped: {e}")

    # Memory-side injection guard (Phase H1, 2026-08-05).
    try:
        from ops.memory_injection_guard_tools import (
            register_memory_injection_guard_tools,
        )

        register_memory_injection_guard_tools(app, DB_PATH)
        logger.info("✅ Memory injection guard integrated (memory_injection_check)")
    except Exception as e:
        logger.warning(f"⚠️  Memory injection guard integration skipped: {e}")

    # Drift curation: temporal profiles backfill + contradiction surfacing (Phase H2).
    try:
        from curate_profiles_tools import register_curate_profiles_tools

        register_curate_profiles_tools(app, DB_PATH)
        logger.info("✅ Drift curation integrated (curate_profiles, profile_summary)")
    except Exception as e:
        logger.warning(f"⚠️  Drift curation integration skipped: {e}")

    # Cold-start backfill (Phase F, 2026-08-05): code + docs + transcripts.
    try:
        from backfill_tools import register_backfill_tools

        register_backfill_tools(app, DB_PATH)
        logger.info("✅ Cold-start backfill integrated (backfill)")
    except Exception as e:
        logger.warning(f"⚠️  Cold-start backfill integration skipped: {e}")

    # Run OmniMEM MAU migration (add modality + raw_data_pointer columns)
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        # Check if columns already exist before migrating
        cursor.execute("PRAGMA table_info(entities)")
        existing_cols = {row[1] for row in cursor.fetchall()}
        if "modality" not in existing_cols:
            migration_path = (
                Path(__file__).parent / "migrations" / "005_omnimem_mau_fields.sql"
            )
            if migration_path.exists():
                cursor.executescript(migration_path.read_text())
                conn.commit()
                logger.info(
                    "✅ OmniMEM MAU migration applied (modality + raw_data_pointer columns)"
                )
            else:
                logger.warning("⚠️  OmniMEM MAU migration file not found")
        else:
            logger.info("✅ OmniMEM MAU fields already present")
        conn.close()
    except Exception as e:
        logger.warning(f"⚠️  OmniMEM MAU migration skipped: {e}")

    # Patch FastMCP to include _meta for large result size support.
    # This allows Claude Code to accept tool results up to 500K characters
    # instead of the default limit. Memory search and retrieval results
    # can be very large.
    _LARGE_RESULT_META = {"anthropic/maxResultSizeChars": 500000}
    # FastMCP 2.x uses _call_tool_mcp(key, arguments) as the MCP-protocol entry point.
    # Older versions used _mcp_call_tool or _call_tool(key, arguments).
    # _call_tool in modern FastMCP takes (context) — wrong signature for this shim,
    # so we prefer _call_tool_mcp when present.
    if hasattr(app, "_mcp_call_tool"):
        _call_tool_attr = "_mcp_call_tool"
    elif hasattr(app, "_call_tool_mcp"):
        _call_tool_attr = "_call_tool_mcp"
    else:
        _call_tool_attr = None

    if _call_tool_attr is not None:
        _original_call_tool = getattr(app, _call_tool_attr)

        async def _patched_mcp_call_tool(key, arguments):
            result = await _original_call_tool(key, arguments)
            if isinstance(result, mcp_types.CallToolResult):
                if result.meta is None:
                    result.meta = _LARGE_RESULT_META
                return result
            if isinstance(result, tuple) and len(result) == 2:
                content, structured = result
                return mcp_types.CallToolResult(
                    content=list(content),
                    structuredContent=structured,
                    meta=_LARGE_RESULT_META,
                )
            if isinstance(result, (list, tuple)):
                return mcp_types.CallToolResult(
                    content=list(result), meta=_LARGE_RESULT_META
                )
            return result

        setattr(app, _call_tool_attr, _patched_mcp_call_tool)
        logger.info(f"✅ Patched {_call_tool_attr} for 500K result meta")
    else:
        logger.warning(
            "⚠️  Skipping result-meta patch: no compatible _call_tool method found"
        )

    # Phase 0 spine repair (2026-07-02): background vector sweeper. Catches
    # writers that bypass create_entities (memory_promotion.py direct sqlite,
    # consolidation jobs) and enforces the working-tier TTL. Fail-soft: an
    # import or startup failure logs and the daemon runs without it.
    try:
        from vector_write_indexer import start_sweeper

        start_sweeper()
    except Exception as _sweep_err:
        logger.warning(f"vector sweeper not started: {_sweep_err}")

    _run_transport()
