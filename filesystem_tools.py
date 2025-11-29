#!/usr/bin/env python3
"""
Filesystem Integration Tools for FastMCP

Simplified filesystem integration for attaching documents to agents.
Based on Letta's filesystem pattern but simplified for initial implementation.

Future enhancements:
- Full Qdrant vector search integration
- Automatic file embedding
- Advanced search (grep, semantic search)
"""

import logging
import sqlite3
from typing import Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)

# Database path
MEMORY_DIR = Path.home() / ".claude" / "enhanced_memories"
DB_PATH = MEMORY_DIR / "memory.db"


def register_filesystem_tools(app, db_path):
    """Register simplified filesystem tools with FastMCP app"""

    def _init_filesystem_tables():
        """Initialize filesystem tables in database"""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS agent_folders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT NOT NULL,
                folder_name TEXT NOT NULL,
                folder_path TEXT NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(agent_id, folder_name)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS agent_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                folder_id INTEGER NOT NULL,
                filename TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_size_bytes INTEGER,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (folder_id) REFERENCES agent_folders (id) ON DELETE CASCADE
            )
        ''')

        conn.commit()
        conn.close()

    _init_filesystem_tables()

    @app.tool()
    async def create_agent_folder(
        agent_id: str,
        folder_name: str,
        folder_path: str,
        description: str = ""
    ) -> Dict[str, Any]:
        """
        Create a folder attachment for an agent (simplified).

        Args:
            agent_id: Agent identifier
            folder_name: Logical name for the folder
            folder_path: Actual filesystem path
            description: Optional folder description

        Returns:
            Folder creation result

        Example:
            create_agent_folder(
                agent_id="macpro51",
                folder_name="architecture_docs",
                folder_path="/mnt/agentic-system/docs",
                description="System architecture documentation"
            )
        """
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        try:
            cursor.execute('''
                INSERT INTO agent_folders (agent_id, folder_name, folder_path, description)
                VALUES (?, ?, ?, ?)
            ''', (agent_id, folder_name, folder_path, description))

            folder_id = cursor.lastrowid
            conn.commit()

            return {
                "success": True,
                "folder_id": folder_id,
                "agent_id": agent_id,
                "folder_name": folder_name,
                "folder_path": folder_path
            }
        except sqlite3.IntegrityError:
            return {
                "success": False,
                "error": f"Folder '{folder_name}' already exists for agent '{agent_id}'"
            }
        finally:
            conn.close()

    @app.tool()
    async def list_agent_folders(agent_id: str) -> Dict[str, Any]:
        """
        List all folders attached to an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            List of attached folders

        Example:
            list_agent_folders(agent_id="macpro51")
        """
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT id, folder_name, folder_path, description, created_at
            FROM agent_folders
            WHERE agent_id = ?
            ORDER BY created_at
        ''', (agent_id,))

        folders = []
        for row in cursor.fetchall():
            # Count files in folder
            cursor.execute('''
                SELECT COUNT(*) FROM agent_files WHERE folder_id = ?
            ''', (row[0],))
            file_count = cursor.fetchone()[0]

            folders.append({
                "folder_id": row[0],
                "folder_name": row[1],
                "folder_path": row[2],
                "description": row[3],
                "created_at": row[4],
                "file_count": file_count
            })

        conn.close()

        return {
            "success": True,
            "agent_id": agent_id,
            "count": len(folders),
            "folders": folders
        }

    @app.tool()
    async def simple_file_search(
        agent_id: str,
        query: str
    ) -> Dict[str, Any]:
        """
        Simple file search in attached folders (filename matching).

        Note: This is a simplified implementation. Full semantic search
        with Qdrant embeddings will be added in future versions.

        Args:
            agent_id: Agent identifier
            query: Search query (filename pattern)

        Returns:
            Matching files

        Example:
            simple_file_search(agent_id="macpro51", query="CLAUDE.md")
        """
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get folders for this agent
        cursor.execute('''
            SELECT id, folder_path FROM agent_folders WHERE agent_id = ?
        ''', (agent_id,))

        folders = cursor.fetchall()
        matches = []

        for folder_id, folder_path in folders:
            # Search files in folder
            folder = Path(folder_path)
            if folder.exists():
                for file in folder.rglob(f"*{query}*"):
                    if file.is_file():
                        matches.append({
                            "filename": file.name,
                            "path": str(file),
                            "size_kb": file.stat().st_size / 1024
                        })

        conn.close()

        return {
            "success": True,
            "agent_id": agent_id,
            "query": query,
            "count": len(matches),
            "matches": matches[:20]  # Limit to 20 results
        }

    logger.info("✅ Filesystem tools registered (3 tools - simplified)")
    logger.info("   Note: Full Qdrant embedding integration pending")
