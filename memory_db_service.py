#!/usr/bin/env python3
"""
Memory Database Service - Unix Socket Server for Concurrent Access

Provides lock-free concurrent access to the memory database for multiple
MCP servers and subagents via Unix socket.

Architecture:
- Listens on /tmp/memory-db.sock
- Handles JSON-RPC style requests
- Manages SQLite database with proper locking
- Supports create_entities, search_nodes, get_memory_status operations
"""

import asyncio
import json
import logging
import os
import signal
import sqlite3
import sys
import hashlib
import re  # noqa: F401  — used by search_nodes FTS tokenisation
import zlib
import pickle
from pathlib import Path

from simhash_dedup import find_near_duplicate, near_dup_policy
from typing import Dict, List, Any, Optional

from socket_guard import SocketInUseError, claim_socket_path

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("memory-db-service")

# Configuration
SOCKET_PATH = os.environ.get("MEMORY_DB_SOCKET_PATH", "/tmp/memory-db.sock")
_DB_PATH_OVERRIDE = os.environ.get("ENHANCED_MEMORY_DB_PATH") or os.environ.get(
    "MEMORY_DB_PATH"
)
_DIR_OVERRIDE = os.environ.get("ENHANCED_MEMORY_DIR") or os.environ.get("MEMORY_DIR")
if _DB_PATH_OVERRIDE:
    DB_PATH = Path(os.path.expandvars(os.path.expanduser(_DB_PATH_OVERRIDE)))
    MEMORY_DIR = DB_PATH.parent
else:
    MEMORY_DIR = (
        Path(os.path.expandvars(os.path.expanduser(_DIR_OVERRIDE)))
        if _DIR_OVERRIDE
        else Path.home() / ".claude" / "enhanced_memories"
    )
    DB_PATH = MEMORY_DIR / "memory.db"

# Create directories
MEMORY_DIR.mkdir(parents=True, exist_ok=True)


class MemoryDatabase:
    """Central memory database with concurrent access support"""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize SQLite database with all required tables"""
        # Whoever creates the file sets its mode. setup.sh no longer chmods a
        # database it did not create, so a new one has to be born 600 here
        # rather than inheriting the umask (0644 on a default macOS or Linux
        # account) and being tightened later, if anyone re-ran the installer.
        existed = os.path.exists(self.db_path)
        conn = sqlite3.connect(self.db_path)
        if not existed:
            try:
                os.chmod(self.db_path, 0o600)
            except OSError as exc:
                logger.warning("could not chmod 600 %s: %s", self.db_path, exc)
        cursor = conn.cursor()

        # Entities table
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
                current_branch TEXT DEFAULT 'main',
                -- OmniMEM MAU fields. These MUST be here, not only in
                -- migrations/005: create_entities INSERTs both columns, so a
                -- database created without them rejects every write.
                modality TEXT DEFAULT 'text',
                raw_data_pointer TEXT
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

        # Relations table. Historically only server.py created this (issue #6):
        # a database born from this class had no relations table, so the daemon
        # and any script importing MemoryDatabase directly failed with "no such
        # table" the first time relations were touched. Every creation path must
        # yield the same schema; DDL matches server.py exactly.
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

        # Create indexes for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_entity_name ON entities(name)")
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_entity_type ON entities(entity_type)"
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_entity_tier ON entities(tier)")
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_observations_entity "
            "ON observations(entity_id)"
        )

        # FTS5 index over observation content + sync triggers (issue #7). Only
        # migrations/phase0_spine_repair_2026_07.py used to create this, so a
        # database born here answered every content query with a well-formed
        # empty while name queries kept working -- indistinguishable from
        # "nothing stored" for the caller. DDL matches the phase0 migration;
        # the rebuild backfills the index when this runs against an existing
        # database that predates it.
        fts_exists = cursor.execute(
            "SELECT 1 FROM sqlite_master WHERE name='observations_fts'"
        ).fetchone()
        if not fts_exists:
            cursor.execute(
                """CREATE VIRTUAL TABLE observations_fts USING fts5(
                     content, content='observations', content_rowid='id')"""
            )
            cursor.execute(
                """CREATE TRIGGER obs_fts_ai AFTER INSERT ON observations BEGIN
                     INSERT INTO observations_fts(rowid, content) VALUES (new.id, new.content);
                   END"""
            )
            cursor.execute(
                """CREATE TRIGGER obs_fts_ad AFTER DELETE ON observations BEGIN
                     INSERT INTO observations_fts(observations_fts, rowid, content)
                     VALUES('delete', old.id, old.content);
                   END"""
            )
            cursor.execute(
                """CREATE TRIGGER obs_fts_au AFTER UPDATE ON observations BEGIN
                     INSERT INTO observations_fts(observations_fts, rowid, content)
                     VALUES('delete', old.id, old.content);
                     INSERT INTO observations_fts(rowid, content) VALUES (new.id, new.content);
                   END"""
            )
            cursor.execute(
                "INSERT INTO observations_fts(observations_fts) VALUES('rebuild')"
            )

        # Converge databases created before the OmniMEM columns were added to
        # the CREATE TABLE above. CREATE TABLE IF NOT EXISTS does nothing to an
        # existing table, so without this an older database keeps rejecting
        # every write. Previously only server.py ran this migration, which made
        # the daemon's writability depend on whether a server had ever started
        # against the same file -- and the documented start order is daemon
        # first.
        cursor.execute("PRAGMA table_info(entities)")
        existing_cols = {row[1] for row in cursor.fetchall()}
        for column, ddl in (
            (
                "modality",
                "ALTER TABLE entities ADD COLUMN modality TEXT DEFAULT 'text'",
            ),
            (
                "raw_data_pointer",
                "ALTER TABLE entities ADD COLUMN raw_data_pointer TEXT",
            ),
        ):
            if column not in existing_cols:
                cursor.execute(ddl)
                logger.info("Added missing entities.%s column", column)

        conn.commit()
        conn.close()
        logger.info(f"Database initialized at {self.db_path}")

    def _compress_data(self, data: Any) -> bytes:
        """Serialize as JSON, then zlib-compress.

        Was `pickle.dumps`. rules/security.md: "JSON only for new data. Never
        introduce unsafe deserialization." Pickle on the write path meant every
        future read had to unpickle attacker-influenceable-in-principle rows;
        JSON removes the code-execution primitive from the read path entirely
        as old rows age out.

        No migration is required and nothing is rewritten: existing zlib+pickle
        rows stay readable through the tolerant `_decompress_data`, which
        already handled zlib+json because ~76% of the DB was json anyway. This
        changes only what NEW writes produce.

        `default=str` keeps values pickle accepted but JSON does not (datetime,
        Path, sets) writable, degrading them to their string form rather than
        raising on a write that used to succeed.
        """
        serialized = json.dumps(data, default=str).encode("utf-8")
        return zlib.compress(serialized, level=9)

    def _decompress_data(self, compressed: bytes) -> Any:
        """Decompress and deserialize, tolerant of every historical encoding.

        entities.compressed_data holds four shapes accumulated over time:
        zlib+pickle (the current write path), zlib+json, zlib+plain-text
        (episodic / insight entities), and gzip (service_event, magic 1f 8b).

        This previously assumed zlib+pickle unconditionally, so a single
        JSON-encoded row in a result set raised `invalid load key, '{'` and the
        whole search returned zero rows. Because search_nodes reports that as
        `count: 0`, the failure read as "no such memory" rather than "search is
        broken" — a silent recall failure. Mirrors the tolerant readers already
        in server.py and memory_db_service_v2.py so all callers see one shape.

        SECURITY: the pickle source is this system's own local memory.db,
        written by this service's own compress path. First-party data, not
        external input. Read-path only; the write path is unchanged.
        """
        # NULL / empty payload: 261 live rows (plus 18 quarantined) have
        # compressed_data IS NULL — mostly pattern_* rollups and doc stubs that
        # were row-created without a body. zlib.decompress(None) raises
        # TypeError, which is NOT a zlib.error, so it escaped the handler below
        # and took the whole search down with `count: 0` — the same silent
        # zero-results failure as the pickle bug. An empty entity is a real, if
        # useless, row; return it empty rather than failing the query it
        # happened to appear in.
        if not compressed:
            return {"observations": []}
        try:
            decompressed = zlib.decompress(compressed)
        except zlib.error:
            import gzip

            decompressed = gzip.decompress(compressed)
        except TypeError:
            return {"observations": []}
        try:
            return pickle.loads(decompressed)
        except (
            pickle.UnpicklingError,
            EOFError,
            AttributeError,
            ImportError,
            IndexError,
            ValueError,
        ):
            try:
                return json.loads(decompressed.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                return {
                    "observations": [decompressed.decode("utf-8", errors="replace")]
                }

    def _calculate_checksum(self, data: bytes) -> str:
        """Calculate SHA-256 checksum"""
        return hashlib.sha256(data).hexdigest()

    def create_entities(self, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create or update entities in the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # `success` is decided AFTER the loop from the failure count, never
        # assumed here. It previously stayed True even when every entity failed,
        # so a caller was told the write worked while nothing was stored.
        results = {
            "success": False,
            "created": 0,
            "updated": 0,
            "failed": 0,
            "count": 0,
            "observations_deduped": 0,
            "observations_near_dup_stored": 0,
            "observations_near_dup_skipped": 0,
            "near_duplicates": [],
            "results": [],
            "errors": [],
        }

        try:
            for entity in entities:
                try:
                    name = entity.get("name")
                    entity_type = entity.get("entityType", "general")
                    observations = entity.get("observations", [])

                    # MAU fields (OmniMEM: modality + raw data pointer)
                    modality = entity.get("modality", "text")
                    raw_data_pointer = entity.get("raw_data_pointer")

                    # Compress entity data (pickle is used here intentionally for
                    # internal-only serialization of trusted entity data structures)
                    entity_data = {
                        "name": name,
                        "type": entity_type,
                        "observations": observations,
                    }
                    compressed = self._compress_data(entity_data)
                    original_size = len(pickle.dumps(entity_data))
                    compressed_size = len(compressed)
                    compression_ratio = (
                        compressed_size / original_size if original_size > 0 else 1.0
                    )
                    checksum = self._calculate_checksum(compressed)

                    # Check if entity exists
                    cursor.execute("SELECT id FROM entities WHERE name = ?", (name,))
                    existing = cursor.fetchone()

                    if existing:
                        # Update existing entity
                        cursor.execute(
                            """
                            UPDATE entities
                            SET entity_type = ?, compressed_data = ?, original_size = ?,
                                compressed_size = ?, compression_ratio = ?, checksum = ?,
                                modality = ?, raw_data_pointer = ?,
                                access_count = access_count + 1, last_accessed = CURRENT_TIMESTAMP,
                                current_version = current_version + 1
                            WHERE name = ?
                        """,
                            (
                                entity_type,
                                compressed,
                                original_size,
                                compressed_size,
                                compression_ratio,
                                checksum,
                                modality,
                                raw_data_pointer,
                                name,
                            ),
                        )
                        results["updated"] += 1
                        entity_id = existing[0]
                    else:
                        # Create new entity
                        cursor.execute(
                            """
                            INSERT INTO entities
                            (name, entity_type, compressed_data, original_size, compressed_size,
                             compression_ratio, checksum, modality, raw_data_pointer)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                            (
                                name,
                                entity_type,
                                compressed,
                                original_size,
                                compressed_size,
                                compression_ratio,
                                checksum,
                                modality,
                                raw_data_pointer,
                            ),
                        )
                        results["created"] += 1
                        entity_id = cursor.lastrowid

                    # Store observations. Exact-content duplicates are skipped
                    # (issue #8): re-importing an unchanged entity used to
                    # append its observations again, so an idempotent-looking
                    # seed import multiplied rows (measured: 3 imports of a
                    # 32-entity seed = 3 identical copies of every row) and
                    # skewed FTS relevance toward whatever was re-imported.
                    # Genuinely new observations on an existing entity still
                    # append; the check also holds within one batch because
                    # earlier inserts in this transaction are visible to it.
                    # NEAR-duplicates (re-worded re-imports) are detected via
                    # simhash against this entity's existing rows. Default
                    # policy REPORTS them and inserts anyway -- a correction
                    # is indistinguishable from a reword at this layer, and
                    # silently dropping corrections is the one behavior a
                    # memory store must never have (see simhash_dedup.py for
                    # the measured distance bands and the 62Gi precedent).
                    # ENHANCED_MEMORY_NEAR_DUP_POLICY=skip opts an import
                    # pipeline into dropping them.
                    dup_policy = near_dup_policy()
                    existing_texts = [
                        r[0]
                        for r in cursor.execute(
                            "SELECT content FROM observations WHERE entity_id = ?",
                            (entity_id,),
                        )
                    ]
                    for obs in observations:
                        if obs in existing_texts:
                            results["observations_deduped"] += 1
                            continue
                        near = find_near_duplicate(obs, existing_texts)
                        if near is not None:
                            detail = {
                                "entity": name,
                                "new": obs[:120],
                                "resembles": near[0][:120],
                                "distance": near[1],
                                "action": "skipped"
                                if dup_policy == "skip"
                                else "stored",
                            }
                            if len(results["near_duplicates"]) < 20:
                                results["near_duplicates"].append(detail)
                            if dup_policy == "skip":
                                results["observations_near_dup_skipped"] += 1
                                continue
                            results["observations_near_dup_stored"] += 1
                        cursor.execute(
                            """
                            INSERT INTO observations (entity_id, content)
                            VALUES (?, ?)
                        """,
                            (entity_id, obs),
                        )
                        existing_texts.append(obs)

                    results["results"].append(
                        {
                            "name": name,
                            "id": entity_id,
                            "compression_ratio": f"{compression_ratio:.2%}",
                        }
                    )

                except Exception as e:
                    logger.error(f"Failed to create entity: {e}")
                    results["failed"] += 1
                    results["errors"].append(
                        {"name": entity.get("name"), "error": str(e)}
                    )

            results["count"] = results["created"] + results["updated"]
            # A partial write is not a success. Any failure makes the whole call
            # unsuccessful so the caller cannot read `success` and move on.
            results["success"] = results["failed"] == 0
            if not results["success"]:
                results["error"] = (
                    f"{results['failed']} of {len(entities)} entities failed to store"
                )
            conn.commit()
            return results

        except Exception as e:
            conn.rollback()
            logger.error(f"Error in create_entities: {e}")
            return {
                "success": False,
                "error": str(e),
                "created": 0,
                "failed": len(entities),
                "count": 0,
            }
        finally:
            conn.close()

    def _ensure_governance_tables(self) -> None:
        """Create the entity_visibility/entity_acl sidecar tables if absent.

        Self-contained mirror of memory_federation/visibility.py so this socket
        service does not import from intelligent-agents/. The MCP server creates
        the same tables via governance_tools; the CREATE IF NOT EXISTS here makes
        the search-time ACL filter robust even if governance tools never ran.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS entity_visibility (
                    entity_id  INTEGER PRIMARY KEY,
                    visibility TEXT    NOT NULL
                        CHECK (visibility IN ('private', 'cluster', 'public')),
                    owner_agent TEXT,
                    set_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    set_by     TEXT,
                    FOREIGN KEY (entity_id) REFERENCES entities(id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS entity_acl (
                    entity_id  INTEGER NOT NULL,
                    agent_id   TEXT    NOT NULL,
                    access     TEXT    DEFAULT 'read',
                    granted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (entity_id, agent_id),
                    FOREIGN KEY (entity_id) REFERENCES entities(id)
                )
                """
            )
            cols = {
                r[1]
                for r in conn.execute("PRAGMA table_info(entity_visibility)").fetchall()
            }
            if "owner_agent" not in cols:
                conn.execute(
                    "ALTER TABLE entity_visibility ADD COLUMN owner_agent TEXT"
                )
            conn.commit()
        finally:
            conn.close()

    def search_nodes(
        self,
        query: str,
        limit: int = 10,
        viewer_agent: Optional[str] = None,
        scope: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Search for entities matching the query.

        viewer_agent scopes results fail-closed: a viewer sees only PUBLIC/CLUSTER
        entities plus PRIVATE ones they own or are granted. When viewer_agent is
        None, no ACL filtering is applied (orchestrator/system view).

        scope restricts results to one project (memory_scope table, populated by
        hooks/memory_promotion.py from the memory file's subdirectory). Added
        2026-08-09 alongside the folder split: 'cfgi', 'arc-agi3', 'harness',
        'hardware', 'research', 'ops', 'kre', 'business', or 'global' for
        top-level files.

        The filter is applied in SQL, inside both query branches, BEFORE LIMIT.
        Filtering the result set afterwards would have been a few lines shorter
        and wrong: asking for 10 rows and then discarding the out-of-scope ones
        returns 2 results for a small project while claiming to be a top-10, and
        the caller cannot tell a scope with little content from a scope whose
        content sorted below the cut.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            if viewer_agent:
                self._ensure_governance_tables()

            # A filter that cannot filter must say so. If scope is requested and
            # the table is absent, returning unfiltered results would look like a
            # scoped search that found everything, which is the failure mode this
            # whole store keeps hitting: a well-formed answer to a question that
            # was never asked. Error out instead.
            if scope is not None:
                if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", scope):
                    return {
                        "success": False,
                        "error": f"invalid scope {scope!r}: expected [A-Za-z0-9_-]{{1,64}}",
                    }
                if not hasattr(self, "_has_memory_scope"):
                    self._has_memory_scope = bool(
                        cursor.execute(
                            "SELECT 1 FROM sqlite_master "
                            "WHERE type='table' AND name='memory_scope'"
                        ).fetchone()
                    )
                if not self._has_memory_scope:
                    return {
                        "success": False,
                        "error": (
                            "scope filtering requested but the memory_scope table "
                            "does not exist; run hooks/memory_promotion.py to build it"
                        ),
                    }
                # An unknown scope is a typo, not an empty project. Returning
                # zero rows for `scope='arcagi3'` is a well-formed answer to a
                # question nobody asked, and the caller reads it as "this project
                # has nothing about X" rather than "that project does not exist".
                # Cheap to distinguish, so distinguish it.
                known = {
                    r[0]
                    for r in cursor.execute("SELECT DISTINCT scope FROM memory_scope")
                }
                if scope not in known:
                    return {
                        "success": False,
                        "error": f"unknown scope {scope!r}",
                        "known_scopes": sorted(known),
                    }
            # Schema-compat: only filter archived rows when the column exists.
            # Detected once and cached on the instance, so the service still
            # runs against a DB predating the archived_at column.
            if not hasattr(self, "_has_archived_at"):
                self._has_archived_at = any(
                    r[1] == "archived_at"
                    for r in cursor.execute("PRAGMA table_info(entities)").fetchall()
                )
            # Bi-temporal filter (Phase 2.2, 2026-07-02): superseded facts are
            # history, not current state — excluded from search the same way
            # archived rows are. Same schema-compat guard pattern.
            if not hasattr(self, "_has_superseded_by"):
                self._has_superseded_by = any(
                    r[1] == "superseded_by"
                    for r in cursor.execute("PRAGMA table_info(entities)").fetchall()
                )

            # Parenthesize the OR so the AND filter binds to the whole match,
            # not just the entity_type branch (SQL AND > OR precedence).
            #
            # QUARANTINE (added 2026-07-26): the benchmark-fixture-pollution
            # incident (2026-07-24) moved 1,351 fabricated "user facts" to
            # tier='quarantine', but never set archived_at on them, so they
            # still passed every visibility filter here and were returned by
            # search. That went unnoticed only because search itself was broken
            # by the pickle decode bug; repairing search would have re-exposed
            # them. Quarantine now means quarantined.
            #
            # Built with an explicit table prefix rather than a string replace:
            # the FTS branch aliases entities as `e`, and the previous
            # `.replace("archived_at", "e.archived_at")` left superseded_by
            # unqualified, which resolved correctly only by accident of the
            # subquery's column names.
            def _visibility(prefix: str = "", viewer: Optional[str] = None) -> str:
                parts = []
                if scope:
                    # Validated against [A-Za-z0-9_-] above, so it cannot carry a
                    # quote; inlined for the same reason `viewer` is, namely that
                    # this string is spliced into two queries whose positional
                    # parameters differ, and threading a bound parameter through
                    # both is where an off-by-one silently mis-binds.
                    parts.append(
                        f" AND EXISTS (SELECT 1 FROM memory_scope ms "
                        f"WHERE ms.entity_name = {prefix}name AND ms.scope = '{scope}')"
                    )
                if self._has_archived_at:
                    parts.append(f" AND {prefix}archived_at IS NULL")
                if self._has_superseded_by:
                    parts.append(f" AND {prefix}superseded_by IS NULL")
                parts.append(
                    f" AND ({prefix}tier IS NULL OR {prefix}tier != 'quarantine')"
                )
                if viewer:
                    # Fail-closed ACL for scoped viewers (Phase D, 2026-08-05):
                    # visible iff cluster/public tagged, OR private with this
                    # viewer as owner, OR an explicit ACL grant. Untagged
                    # entities default to private. viewer is a sanitized agent
                    # id (validated upstream), so inlining is injection-safe.
                    v = viewer.replace("'", "''")
                    parts.append(
                        f"""
                        AND (
                            EXISTS (
                                SELECT 1 FROM entity_visibility ev
                                WHERE ev.entity_id = {prefix}id
                                  AND ev.visibility IN ('cluster','public')
                            )
                            OR EXISTS (
                                SELECT 1 FROM entity_visibility ev
                                WHERE ev.entity_id = {prefix}id
                                  AND ev.visibility = 'private'
                                  AND ev.owner_agent = '{v}'
                            )
                            OR EXISTS (
                                SELECT 1 FROM entity_acl acl
                                WHERE acl.entity_id = {prefix}id
                                  AND acl.agent_id = '{v}'
                            )
                        )
                        """
                    )
                return "".join(parts)

            archived_clause = _visibility(viewer=viewer_agent)

            # Phase 0 spine repair (2026-07-02): search observation TEXT via
            # FTS5, not just entity name/type. Before this, a query matching
            # only an observation returned nothing (audit: the canary
            # observation "kumquat telescopes" was unfindable). FTS matches
            # rank first (bm25), then legacy name/type matches. Falls back to
            # name/type-only when the FTS table is absent (older DB) or the
            # query breaks FTS syntax (quotes strip most operator issues).
            if not hasattr(self, "_has_obs_fts"):
                self._has_obs_fts = bool(
                    cursor.execute(
                        "SELECT 1 FROM sqlite_master WHERE name='observations_fts'"
                    ).fetchone()
                )

            # A search that cannot see observation content must say so (issue
            # #7, same principle as the scope filter above): without the
            # marker, zero content recall is indistinguishable from "nothing
            # stored". Set on both degraded paths, surfaced in the response.
            degraded = None
            if not self._has_obs_fts:
                degraded = "name-only (observations_fts missing)"

            rows = []
            if self._has_obs_fts:
                # Quote each TOKEN, not the whole query. Wrapping the entire
                # query in one pair of quotes makes it an FTS5 *phrase* query,
                # so any multi-word search only matched documents where the
                # words were adjacent: "permission prohibition" missed an
                # entity reading "permission-gated, not a blanket in-band
                # prohibition" (verified 2026-07-26). Per-token quoting keeps
                # the operator-injection safety that motivated the original
                # quoting (a stray -, *, NEAR or OR can no longer reach the FTS
                # parser) while restoring implicit AND across terms.
                tokens = re.findall(r"[\w']+", query)
                fts_query = " ".join('"' + t.replace('"', '""') + '"' for t in tokens)
                try:
                    # bm25() is not usable under GROUP BY aggregation ("unable
                    # to use function bm25 in the requested context", verified
                    # live); the FTS `rank` auxiliary column aggregated in a
                    # subquery is the working form.
                    cursor.execute(
                        f"""
                        SELECT e.id, e.name, e.entity_type, e.compressed_data,
                               e.compression_ratio, e.access_count, e.created_at,
                               e.last_accessed, e.tier, sub.rank
                        FROM (
                            SELECT o.entity_id AS eid,
                                   MIN(observations_fts.rank) AS rank
                            FROM observations_fts
                            JOIN observations o ON o.id = observations_fts.rowid
                            WHERE observations_fts MATCH ?
                            GROUP BY o.entity_id
                        ) sub
                        JOIN entities e ON e.id = sub.eid
                        WHERE 1=1{_visibility("e.", viewer_agent)}
                        ORDER BY sub.rank
                        LIMIT ?
                    """,
                        (fts_query, limit),
                    )
                    rows = cursor.fetchall()
                except sqlite3.OperationalError:
                    rows = []  # FTS syntax edge case; name/type search still runs
                    degraded = "name-only (FTS query error)"

            seen_ids = {r[0] for r in rows}
            if len(rows) < limit:
                cursor.execute(
                    f"""
                    SELECT id, name, entity_type, compressed_data, compression_ratio,
                           access_count, created_at, last_accessed, tier, 0 AS rank
                    FROM entities
                    WHERE (name LIKE ? OR entity_type LIKE ?){archived_clause}
                    ORDER BY access_count DESC, last_accessed DESC
                    LIMIT ?
                """,
                    (f"%{query}%", f"%{query}%", limit),
                )
                rows.extend(r for r in cursor.fetchall() if r[0] not in seen_ids)
            rows = rows[:limit]

            results = []
            for row in rows:
                (
                    entity_id,
                    name,
                    entity_type,
                    compressed_data,
                    compression_ratio,
                    access_count,
                    created_at,
                    last_accessed,
                    tier,
                    _rank,
                ) = row

                # Decompress entity data
                entity_data = self._decompress_data(compressed_data)

                results.append(
                    {
                        "id": entity_id,
                        "name": name,
                        "entityType": entity_type,
                        "observations": entity_data.get("observations", []),
                        "tier": tier,
                        # 259 rows have NULL compression_ratio; formatting None
                        # raised and killed the whole search result
                        "compression_ratio": (
                            f"{compression_ratio:.2%}"
                            if compression_ratio is not None
                            else "n/a"
                        ),
                        "access_count": access_count,
                        "created_at": created_at,
                        "last_accessed": last_accessed,
                        # Retrieval-quality signal (Phase G, 2026-08-05): FTS
                        # content matches are stronger than name-substring-only
                        # matches; _rank is the FTS5 bm25 rank (<= 0), 0 for
                        # the name-branch fallback.
                        "confidence": (
                            round(min(0.95, max(0.55, 0.8 - _rank * 0.001)), 3)
                            if _rank
                            else 0.5
                        ),
                    }
                )

            # Update access count
            for entity in results:
                cursor.execute(
                    """
                    UPDATE entities
                    SET access_count = access_count + 1, last_accessed = CURRENT_TIMESTAMP
                    WHERE id = ?
                """,
                    (entity["id"],),
                )

            conn.commit()

            # Append-only retrieval telemetry. Swallowed: telemetry must never
            # break a retrieval. session_id is not carried on the request
            # envelope today, so it logs "unknown" (see proposal caveat).
            try:
                from ops.retrieval_log import log_retrieval

                log_retrieval(
                    "unknown",
                    query,
                    [r["id"] for r in results],
                    source="search_nodes",
                )
            except Exception:
                pass

            response = {
                "success": True,
                "query": query,
                "count": len(results),
                "confidence": results[0]["confidence"] if results else 0.0,
                "low_confidence": (not results) or results[0]["confidence"] <= 0.5,
                "results": results,
            }
            if degraded:
                response["degraded"] = degraded
            return response

        except Exception as e:
            logger.error(f"Error in search_nodes: {e}")
            # No "count": 0 / "results": [] here. A failed search and a search
            # that legitimately matched nothing must not serialize alike --
            # server.py hides this behind its own envelope, but memory_client
            # and any direct consumer of the daemon see this dict as-is.
            return {"success": False, "error": str(e), "query": query}
        finally:
            conn.close()

    def get_memory_status(self) -> Dict[str, Any]:
        """Get memory system status and statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Get entity count
            cursor.execute("SELECT COUNT(*) FROM entities")
            total_entities = cursor.fetchone()[0]

            # Get compression stats
            cursor.execute("""
                SELECT
                    AVG(compression_ratio) as avg_ratio,
                    SUM(original_size) as total_original,
                    SUM(compressed_size) as total_compressed
                FROM entities
            """)
            avg_ratio, total_original, total_compressed = cursor.fetchone()

            # Get tier distribution
            cursor.execute("""
                SELECT tier, COUNT(*) as count
                FROM entities
                GROUP BY tier
            """)
            tier_distribution = {row[0]: row[1] for row in cursor.fetchall()}

            return {
                "success": True,
                "entities": {"total": total_entities},
                "compression": {
                    # Savings computed from byte totals; the per-entity
                    # compression_ratio column mixes two unit conventions
                    # (fraction vs legacy percent rows) and must not be averaged.
                    "ratio": f"{(1 - total_compressed / total_original) * 100:.2f}%"
                    if total_original
                    else "N/A",
                    "total_original_kb": round(total_original / 1024, 2)
                    if total_original
                    else 0,
                    "total_compressed_kb": round(total_compressed / 1024, 2)
                    if total_compressed
                    else 0,
                },
                "tiers": tier_distribution,
                "database_path": str(self.db_path),
            }

        except Exception as e:
            logger.error(f"Error in get_memory_status: {e}")
            # No "entities": {"total": 0} here: a failed status query and an
            # empty database must not serialize to the same thing.
            return {
                "success": False,
                "error": str(e),
                "database_path": str(self.db_path),
            }
        finally:
            conn.close()


class MemoryDBServer:
    """Unix socket server for memory database"""

    def __init__(self, socket_path: str, db_path: Path):
        self.socket_path = socket_path
        self.db = MemoryDatabase(db_path)
        self.server = None
        # Only a process that bound the socket itself may remove the file.
        self._owns_socket = False

    async def handle_request(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ):
        """Handle incoming client request"""
        try:
            # Read request
            request_data = await reader.read(10 * 1024 * 1024)  # 10MB max
            if not request_data:
                return

            request = json.loads(request_data.decode())
            method = request.get("method")
            params = request.get("params", {})

            logger.info(f"Received request: {method}")

            # Route to appropriate handler
            if method == "create_entities":
                result = self.db.create_entities(params.get("entities", []))
            elif method == "search_nodes":
                result = self.db.search_nodes(
                    params.get("query", ""),
                    params.get("limit", 10),
                    params.get("viewer_agent"),
                    params.get("scope"),
                )
            elif method == "get_memory_status":
                result = self.db.get_memory_status()
            else:
                result = {"error": f"Unknown method: {method}"}

            # Send response
            response = json.dumps(result).encode()
            writer.write(response)
            await writer.drain()

        except Exception as e:
            logger.error(f"Error handling request: {e}")
            error_response = json.dumps({"error": str(e)}).encode()
            writer.write(error_response)
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    async def start(self):
        """Start the Unix socket server.

        Raises SocketInUseError rather than unlinking a socket another daemon
        is answering on -- see socket_guard for why that takeover is silent.
        """
        if claim_socket_path(self.socket_path):
            logger.warning("removed stale socket file %s", self.socket_path)

        # Start server
        self.server = await asyncio.start_unix_server(
            self.handle_request, path=self.socket_path
        )
        self._owns_socket = True

        # Set socket permissions
        os.chmod(self.socket_path, 0o666)

        logger.info(f"Memory-DB service listening on {self.socket_path}")

        async with self.server:
            await self.server.serve_forever()

    async def stop(self):
        """Stop the server"""
        if self.server:
            self.server.close()
            await self.server.wait_closed()

        # Never unlink a socket this process did not bind: on a refused start
        # the file belongs to the daemon that is still serving it, and removing
        # it would take that daemon's clients down instead of taking them over.
        if self._owns_socket and os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
            self._owns_socket = False

        logger.info("Memory-DB service stopped")


async def main():
    """Main entry point"""
    server = MemoryDBServer(SOCKET_PATH, DB_PATH)

    # Handle shutdown signals
    loop = asyncio.get_event_loop()

    def signal_handler():
        logger.info("Received shutdown signal")
        asyncio.create_task(server.stop())

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    try:
        await server.start()
    except SocketInUseError as exc:
        # Loud and fatal on purpose: the alternative is a silent takeover whose
        # only symptom is that somebody else's memory store reads as empty.
        logger.error("%s", exc)
        return 2
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        await server.stop()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
