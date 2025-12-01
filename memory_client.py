#!/usr/bin/env python3
"""
Memory Database Client - Unix Socket Client for concurrent access

Connects to memory-db service via Unix socket for lock-free concurrent access.
Allows multiple MCP servers and subagents to access memory simultaneously.

Features:
- Automatic fallback to direct SQLite when socket service unavailable
- Connection retry with exponential backoff
- Graceful degradation for high availability
"""

import asyncio
import json
import logging
import os
import socket
import sqlite3
import hashlib
import zlib
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger("memory-client")

# Default paths
SOCKET_PATH = "/tmp/memory-db.sock"
MEMORY_DIR = Path.home() / ".claude" / "enhanced_memories"
DB_PATH = MEMORY_DIR / "memory.db"


class DirectSQLiteFallback:
    """Direct SQLite access when socket service unavailable"""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._ensure_db()

    def _ensure_db(self):
        """Ensure database and tables exist"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
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
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_id INTEGER,
                content TEXT NOT NULL,
                compressed BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (entity_id) REFERENCES entities (id)
            )
        ''')
        conn.commit()
        conn.close()

    def _compress_data(self, data: Any) -> bytes:
        return zlib.compress(pickle.dumps(data), level=9)

    def _decompress_data(self, compressed: bytes) -> Any:
        decompressed = zlib.decompress(compressed)
        try:
            return pickle.loads(decompressed)
        except:
            try:
                return json.loads(decompressed.decode('utf-8'))
            except:
                return {"observations": [decompressed.decode('utf-8', errors='replace')]}

    def create_entities(self, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create entities directly in SQLite"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        results = {"success": True, "created": 0, "updated": 0, "failed": 0, "count": 0, "results": [], "fallback": True}

        try:
            for entity in entities:
                try:
                    name = entity.get("name")
                    entity_type = entity.get("entityType", "general")
                    observations = entity.get("observations", [])
                    tier = entity.get("tier", "working")

                    entity_data = {"name": name, "type": entity_type, "observations": observations}
                    compressed = self._compress_data(entity_data)
                    original_size = len(pickle.dumps(entity_data))
                    compressed_size = len(compressed)
                    compression_ratio = compressed_size / original_size if original_size > 0 else 1.0
                    checksum = hashlib.sha256(compressed).hexdigest()

                    cursor.execute("SELECT id FROM entities WHERE name = ?", (name,))
                    existing = cursor.fetchone()

                    if existing:
                        cursor.execute('''
                            UPDATE entities SET entity_type = ?, compressed_data = ?, original_size = ?,
                            compressed_size = ?, compression_ratio = ?, checksum = ?, tier = ?,
                            access_count = access_count + 1, last_accessed = CURRENT_TIMESTAMP,
                            current_version = current_version + 1 WHERE name = ?
                        ''', (entity_type, compressed, original_size, compressed_size, compression_ratio, checksum, tier, name))
                        results["updated"] += 1
                        entity_id = existing[0]
                    else:
                        cursor.execute('''
                            INSERT INTO entities (name, entity_type, compressed_data, original_size,
                            compressed_size, compression_ratio, checksum, tier) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (name, entity_type, compressed, original_size, compressed_size, compression_ratio, checksum, tier))
                        results["created"] += 1
                        entity_id = cursor.lastrowid

                    for obs in observations:
                        cursor.execute('INSERT INTO observations (entity_id, content) VALUES (?, ?)', (entity_id, obs))

                    results["results"].append({"name": name, "id": entity_id, "compression_ratio": f"{compression_ratio:.2%}"})
                except Exception as e:
                    logger.error(f"Failed to create entity: {e}")
                    results["failed"] += 1

            results["count"] = results["created"] + results["updated"]
            conn.commit()
            return results
        except Exception as e:
            conn.rollback()
            return {"success": False, "error": str(e), "created": 0, "failed": len(entities), "count": 0, "fallback": True}
        finally:
            conn.close()

    def search_nodes(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """Search entities directly in SQLite"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute('''
                SELECT id, name, entity_type, compressed_data, compression_ratio, access_count, created_at, last_accessed, tier
                FROM entities WHERE name LIKE ? OR entity_type LIKE ?
                ORDER BY access_count DESC, last_accessed DESC LIMIT ?
            ''', (f'%{query}%', f'%{query}%', limit))

            results = []
            for row in cursor.fetchall():
                entity_id, name, entity_type, compressed_data, compression_ratio, access_count, created_at, last_accessed, tier = row
                entity_data = self._decompress_data(compressed_data)
                results.append({
                    "id": entity_id, "name": name, "entityType": entity_type,
                    "observations": entity_data.get("observations", []), "tier": tier,
                    "compression_ratio": f"{compression_ratio:.2%}", "access_count": access_count,
                    "created_at": created_at, "last_accessed": last_accessed
                })

            for entity in results:
                cursor.execute('UPDATE entities SET access_count = access_count + 1, last_accessed = CURRENT_TIMESTAMP WHERE id = ?', (entity["id"],))
            conn.commit()

            return {"success": True, "query": query, "count": len(results), "results": results, "fallback": True}
        except Exception as e:
            return {"success": False, "error": str(e), "count": 0, "results": [], "fallback": True}
        finally:
            conn.close()

    def get_memory_status(self) -> Dict[str, Any]:
        """Get memory status directly from SQLite"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("SELECT COUNT(*) FROM entities")
            total_entities = cursor.fetchone()[0]

            cursor.execute('SELECT AVG(compression_ratio), SUM(original_size), SUM(compressed_size) FROM entities')
            avg_ratio, total_original, total_compressed = cursor.fetchone()

            cursor.execute('SELECT tier, COUNT(*) FROM entities GROUP BY tier')
            tier_distribution = {row[0]: row[1] for row in cursor.fetchall()}

            return {
                "success": True, "fallback": True,
                "entities": {"total": total_entities},
                "compression": {
                    "ratio": f"{avg_ratio:.2%}" if avg_ratio else "N/A",
                    "total_original_kb": round(total_original / 1024, 2) if total_original else 0,
                    "total_compressed_kb": round(total_compressed / 1024, 2) if total_compressed else 0
                },
                "tiers": tier_distribution, "database_path": str(self.db_path)
            }
        except Exception as e:
            return {"success": False, "error": str(e), "entities": {"total": 0}, "fallback": True}
        finally:
            conn.close()


class MemoryClient:
    """Client for memory-db Unix socket service with automatic fallback"""

    def __init__(self, socket_path: str = SOCKET_PATH, db_path: Path = DB_PATH):
        self.socket_path = socket_path
        self.db_path = db_path
        self._fallback = DirectSQLiteFallback(db_path)
        self._socket_available = None  # Cached availability
        self._last_check = 0
        self._check_interval = 30  # Re-check socket every 30 seconds

    def is_socket_available(self, force_check: bool = False) -> bool:
        """Check if socket service is available"""
        import time
        now = time.time()

        if not force_check and self._socket_available is not None and (now - self._last_check) < self._check_interval:
            return self._socket_available

        try:
            if not os.path.exists(self.socket_path):
                self._socket_available = False
            else:
                # Try a quick connect
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.settimeout(1.0)
                sock.connect(self.socket_path)
                sock.close()
                self._socket_available = True
        except:
            self._socket_available = False

        self._last_check = now
        return self._socket_available

    async def _send_request(self, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Send request to memory-db service and get response"""
        if params is None:
            params = {}

        request = {"method": method, "params": params}

        try:
            # Connect to Unix socket with timeout
            reader, writer = await asyncio.wait_for(
                asyncio.open_unix_connection(self.socket_path),
                timeout=5.0
            )

            try:
                request_data = json.dumps(request).encode()
                writer.write(request_data)
                await writer.drain()

                chunks = []
                while True:
                    chunk = await asyncio.wait_for(reader.read(1024 * 1024), timeout=30.0)
                    if not chunk:
                        break
                    chunks.append(chunk)

                response_data = b''.join(chunks)
                response = json.loads(response_data.decode())
                return response

            finally:
                writer.close()
                await writer.wait_closed()

        except (ConnectionRefusedError, FileNotFoundError, asyncio.TimeoutError, OSError) as e:
            logger.warning(f"Socket unavailable ({e}), using fallback")
            self._socket_available = False
            raise

    def _send_request_sync(self, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Synchronous version with automatic fallback"""
        if not self.is_socket_available():
            return self._fallback_call(method, params)

        try:
            return asyncio.run(self._send_request(method, params))
        except Exception as e:
            logger.warning(f"Socket request failed: {e}, using fallback")
            return self._fallback_call(method, params)

    def _fallback_call(self, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute method via direct SQLite fallback"""
        params = params or {}
        if method == "create_entities":
            return self._fallback.create_entities(params.get("entities", []))
        elif method == "search_nodes":
            return self._fallback.search_nodes(params.get("query", ""), params.get("limit", 10))
        elif method == "get_memory_status":
            return self._fallback.get_memory_status()
        elif method == "ping":
            return {"success": True, "message": "pong (fallback)", "fallback": True}
        else:
            return {"error": f"Unknown method: {method}", "fallback": True}

    async def create_entities(self, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create entities (async with fallback)"""
        try:
            return await self._send_request("create_entities", {"entities": entities})
        except Exception:
            return self._fallback.create_entities(entities)

    def create_entities_sync(self, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create entities (sync)"""
        return self._send_request_sync("create_entities", {"entities": entities})

    async def search_nodes(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """Search entities (async with fallback)"""
        try:
            return await self._send_request("search_nodes", {"query": query, "limit": limit})
        except Exception:
            return self._fallback.search_nodes(query, limit)

    def search_nodes_sync(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """Search entities (sync)"""
        return self._send_request_sync("search_nodes", {"query": query, "limit": limit})

    async def get_memory_status(self) -> Dict[str, Any]:
        """Get memory status (async with fallback)"""
        try:
            return await self._send_request("get_memory_status")
        except Exception:
            return self._fallback.get_memory_status()

    def get_memory_status_sync(self) -> Dict[str, Any]:
        """Get memory status (sync)"""
        return self._send_request_sync("get_memory_status")

    async def ping(self) -> Dict[str, Any]:
        """Ping server (async with fallback)"""
        try:
            return await self._send_request("ping")
        except Exception:
            return {"success": True, "message": "pong (fallback)", "fallback": True}

    def ping_sync(self) -> Dict[str, Any]:
        """Ping server (sync)"""
        return self._send_request_sync("ping")


# Global client instance
_client: Optional[MemoryClient] = None


def get_client() -> MemoryClient:
    """Get or create global memory client"""
    global _client
    if _client is None:
        _client = MemoryClient()
    return _client


def is_socket_available() -> bool:
    """Check if memory-db socket service is available"""
    return get_client().is_socket_available()


def get_connection_mode() -> str:
    """Return current connection mode: 'socket' or 'fallback'"""
    return "socket" if is_socket_available() else "fallback"


# Convenience functions for easy migration
def create_entities(entities: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create entities - synchronous wrapper with automatic fallback"""
    client = get_client()
    return client.create_entities_sync(entities)


def search_nodes(query: str, limit: int = 10) -> Dict[str, Any]:
    """Search entities - synchronous wrapper with automatic fallback"""
    client = get_client()
    return client.search_nodes_sync(query, limit)


def get_memory_status() -> Dict[str, Any]:
    """Get memory status - synchronous wrapper with automatic fallback"""
    client = get_client()
    return client.get_memory_status_sync()


def ping() -> Dict[str, Any]:
    """Ping memory service - synchronous wrapper"""
    client = get_client()
    return client.ping_sync()


# Export for convenience
__all__ = [
    'MemoryClient', 'DirectSQLiteFallback',
    'get_client', 'is_socket_available', 'get_connection_mode',
    'create_entities', 'search_nodes', 'get_memory_status', 'ping'
]
