"""Regression tests for GitHub issues #6, #7, #8 (filed 2026-08-15).

All three were found by a CFGI seed-import exercise against a database born
from MemoryDatabase.init_database() rather than server.py or the phase0
migration -- the creation path a from-scratch team rollout actually uses:

  #6  init_database() did not create the relations table (server.py-only DDL).
  #7  init_database() did not create observations_fts, so content search
      silently degraded to name-only matching with well-formed empty results.
  #8  create_entities() appended exact-duplicate observations on re-import,
      so an idempotent-looking seed import multiplied rows.

Each test here fails against commit 551211b (verified before fixing) and
passes after. They pin the contract that every database creation path yields
a schema the full API works against.

Gaps / not covered: the phase0 migration path and server.py init path are not
re-tested here (covered elsewhere); FTS trigger behavior is asserted via the
triggers' existence and single-writer round-trips, not under concurrent
writers.
"""

import sqlite3

import pytest

from memory_db_service import MemoryDatabase


@pytest.fixture()
def fresh_db(tmp_path):
    """A database born from the service class alone -- the issue-6/7 path."""
    return MemoryDatabase(tmp_path / "fresh.db")


def _table_names(db_path):
    return {
        r[0]
        for r in sqlite3.connect(db_path).execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','trigger')"
        )
    }


def _make_aged_db(path):
    """A pre-FTS database: entities + observations only, as init_database()
    produced them before this fix. Hand-written DDL on purpose -- building it
    with today's init_database() and then removing the index would test the
    removal, not the aged shape."""
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE entities (
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
            modality TEXT DEFAULT 'text',
            raw_data_pointer TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_id INTEGER,
            content TEXT NOT NULL,
            compressed BLOB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (entity_id) REFERENCES entities (id)
        )
        """
    )
    conn.commit()
    conn.close()


class TestIssue6RelationsTable:
    def test_relations_table_exists_on_fresh_db(self, fresh_db):
        rows = (
            sqlite3.connect(fresh_db.db_path)
            .execute("SELECT COUNT(*) FROM relations")
            .fetchone()
        )
        assert rows == (0,)

    def test_relations_schema_matches_server_ddl(self, fresh_db):
        cols = {
            r[1]
            for r in sqlite3.connect(fresh_db.db_path).execute(
                "PRAGMA table_info(relations)"
            )
        }
        assert cols == {
            "id",
            "from_entity_id",
            "to_entity_id",
            "relation_type",
            "created_at",
        }


class TestIssue7ContentSearch:
    def test_fts_index_and_triggers_exist_on_fresh_db(self, fresh_db):
        names = _table_names(fresh_db.db_path)
        assert "observations_fts" in names
        assert {"obs_fts_ai", "obs_fts_ad", "obs_fts_au"} <= names

    def test_content_query_matches_observation_text(self, fresh_db):
        fresh_db.create_entities(
            [
                {
                    "name": "bridge_model",
                    "entityType": "fact",
                    "observations": ["deployment is claude-haiku-4-5"],
                }
            ]
        )
        by_content = fresh_db.search_nodes("haiku")
        assert by_content["success"] is True
        assert by_content["count"] == 1, (
            "content query must match observation text; a well-formed zero "
            "here is exactly issue #7"
        )
        assert "degraded" not in by_content

    def test_degraded_marker_when_fts_absent(self, tmp_path):
        aged_path = tmp_path / "aged.db"
        _make_aged_db(aged_path)
        aged = MemoryDatabase.__new__(MemoryDatabase)
        aged.db_path = aged_path  # bypass init_database: the aged shape is the point
        aged.create_entities(
            [{"name": "e_fts", "entityType": "t", "observations": ["kumquat lens"]}]
        )
        out = aged.search_nodes("kumquat")
        assert out["success"] is True
        assert out["count"] == 0
        assert out.get("degraded") == "name-only (observations_fts missing)", (
            "a search that cannot see content must say so"
        )

    def test_init_database_backfills_fts_on_aged_db(self, tmp_path):
        aged_path = tmp_path / "aged2.db"
        _make_aged_db(aged_path)
        conn = sqlite3.connect(aged_path)
        conn.execute("INSERT INTO entities (name, entity_type) VALUES ('old_e', 't')")
        conn.execute(
            "INSERT INTO observations (entity_id, content) "
            "VALUES (1, 'pre-existing tamarind row')"
        )
        conn.commit()
        conn.close()

        MemoryDatabase(aged_path)  # runs init_database against the aged file
        hits = (
            sqlite3.connect(aged_path)
            .execute(
                "SELECT COUNT(*) FROM observations_fts WHERE observations_fts "
                "MATCH 'tamarind'"
            )
            .fetchone()[0]
        )
        assert hits == 1, "rebuild must index rows that predate the FTS table"


class TestIssue8ObservationDedupe:
    def test_triple_import_is_idempotent(self, fresh_db):
        ent = [{"name": "e1", "entityType": "t", "observations": ["same fact"]}]
        first = fresh_db.create_entities(ent)
        assert first["created"] == 1
        for _ in range(2):
            again = fresh_db.create_entities(ent)
            assert again["updated"] == 1
            assert again["observations_deduped"] == 1
        n = (
            sqlite3.connect(fresh_db.db_path)
            .execute(
                "SELECT COUNT(*) FROM observations o JOIN entities e "
                "ON o.entity_id = e.id WHERE e.name = 'e1'"
            )
            .fetchone()[0]
        )
        assert n == 1

    def test_new_observations_still_append(self, fresh_db):
        fresh_db.create_entities(
            [{"name": "e2", "entityType": "t", "observations": ["fact one"]}]
        )
        out = fresh_db.create_entities(
            [
                {
                    "name": "e2",
                    "entityType": "t",
                    "observations": ["fact one", "fact two"],
                }
            ]
        )
        assert out["observations_deduped"] == 1
        n = (
            sqlite3.connect(fresh_db.db_path)
            .execute(
                "SELECT COUNT(*) FROM observations o JOIN entities e "
                "ON o.entity_id = e.id WHERE e.name = 'e2'"
            )
            .fetchone()[0]
        )
        assert n == 2

    def test_in_batch_duplicates_collapse(self, fresh_db):
        out = fresh_db.create_entities(
            [{"name": "e3", "entityType": "t", "observations": ["x", "x", "x"]}]
        )
        assert out["created"] == 1
        assert out["observations_deduped"] == 2
        n = (
            sqlite3.connect(fresh_db.db_path)
            .execute(
                "SELECT COUNT(*) FROM observations o JOIN entities e "
                "ON o.entity_id = e.id WHERE e.name = 'e3'"
            )
            .fetchone()[0]
        )
        assert n == 1
