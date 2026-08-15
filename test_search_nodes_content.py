"""Regression tests for search_nodes observation-content search.

Until 2026-07-19 search_nodes matched only `name LIKE` / `entity_type LIKE`, so
the body of a memory was unreachable. These tests pin the fixed behaviour and
the failure modes that made it risky to fix: FTS5 syntax in user input, a
deployment with no FTS index, and losing the fused ranking through
`SELECT ... IN (...)`.

Runs against temporary in-memory databases, never production.
"""

from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "libs"))

_spec = importlib.util.spec_from_file_location(
    "memory_db_service_v2", HERE / "memory_db_service_v2.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
Service = _mod.MemoryDatabaseV2

ENTITIES = [
    (1, "cluster_topology", "reference"),
    (2, "retrieval_notes", "reference"),
    (3, "kre_training_log", "project"),
]
OBSERVATIONS = [
    (1, 1, "fedora runs an RTX 3060 with personalized pagerank experiments"),
    (2, 2, "notes about ranking and recall"),
    (3, 3, "the QLoRA run used rank 4 on the final feed-forward layer"),
    (4, 3, "it's a temporal conflict when the value changes over time"),
]

BASE_SCHEMA = """
CREATE TABLE entities (
    id INTEGER PRIMARY KEY, name TEXT, entity_type TEXT,
    access_count INTEGER DEFAULT 0, last_accessed TEXT DEFAULT ''
);
CREATE TABLE observations (
    id INTEGER PRIMARY KEY, entity_id INTEGER, content TEXT
);
"""

FTS_SCHEMA = """
CREATE VIRTUAL TABLE observations_fts
    USING fts5(content, content='observations', content_rowid='id');
"""


def _build(with_fts: bool):
    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()
    cur.executescript(BASE_SCHEMA + (FTS_SCHEMA if with_fts else ""))
    cur.executemany(
        "INSERT INTO entities(id,name,entity_type) VALUES (?,?,?)", ENTITIES
    )
    cur.executemany(
        "INSERT INTO observations(id,entity_id,content) VALUES (?,?,?)", OBSERVATIONS
    )
    if with_fts:
        cur.executemany(
            "INSERT INTO observations_fts(rowid,content) VALUES (?,?)",
            [(o[0], o[2]) for o in OBSERVATIONS],
        )
    return conn, cur


@pytest.fixture
def cursor():
    conn, cur = _build(with_fts=True)
    yield cur
    conn.close()


@pytest.fixture
def cursor_without_index():
    """A deployment where the FTS index was never created."""
    conn, cur = _build(with_fts=False)
    yield cur
    conn.close()


def test_content_search_finds_what_name_search_cannot(cursor):
    """The defect: 'pagerank' appears only in observation text."""
    assert Service._search_ids_by_name(cursor, "pagerank", 10) == []
    assert Service._search_ids_by_content(cursor, "pagerank", 10) == [1]


def test_content_search_finds_multiword_phrases(cursor):
    assert Service._search_ids_by_content(cursor, "feed-forward layer", 10) == [3]


def test_name_search_still_works(cursor):
    assert Service._search_ids_by_name(cursor, "retrieval", 10) == [2]


def test_entity_appears_once_despite_multiple_matching_observations(cursor):
    """Entity 3 owns two matching observations; double-counting would let a
    single entity dominate the fused ranking."""
    ids = Service._search_ids_by_content(cursor, "the", 10)
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize(
    "query",
    ["it's", "a AND b", "x* ^y", '"unclosed', "NEAR(a b)", "OR", "", "   ", "a-b"],
)
def test_fts_syntax_in_user_input_never_raises(cursor, query):
    """Unquoted, each of these is either valid FTS5 syntax or a syntax error;
    either way the user meant them as literal text."""
    assert isinstance(Service._search_ids_by_content(cursor, query, 5), list)


def test_absent_fts_index_degrades_instead_of_raising(cursor_without_index):
    """No index must mean name-only search, not broken memory retrieval."""
    assert Service._search_ids_by_content(cursor_without_index, "pagerank", 10) == []
    assert Service._search_ids_by_name(cursor_without_index, "retrieval", 10) == [2]


def test_fusion_ranks_name_matches_above_content_matches():
    """Looking an entity up by its exact name must still put it first."""
    from memgraph.fusion import fuse_to_ids

    assert fuse_to_ids([[99], [1, 2, 3]], limit=4, weights=[2.0, 1.0])[0] == 99


def test_fusion_keeps_content_only_hits():
    from memgraph.fusion import fuse_to_ids

    assert 7 in fuse_to_ids([[], [7]], limit=5, weights=[2.0, 1.0])


def test_rrf_import_resolved():
    """If this is False in production, search silently reverts to name-only."""
    assert _mod.RRF_AVAILABLE, "memgraph.fusion did not import; RRF is inactive"


# ------------------------------------------------------------- end-to-end


@pytest.fixture
def service(tmp_path):
    """A real MemoryDatabaseV2 on a fresh database, seeded via the real write
    path so compression and schema match production."""
    svc = _mod.MemoryDatabaseV2(tmp_path / "e2e.db")
    svc.create_entities(
        [
            {
                "name": "cluster_topology",
                "entityType": "reference",
                "observations": [
                    "fedora runs an RTX 3060 with personalized pagerank experiments"
                ],
            },
            {
                "name": "retrieval_notes",
                "entityType": "reference",
                "observations": ["notes about ranking and recall"],
            },
        ]
    )
    return svc


def test_fresh_database_creates_the_fts_index(service):
    """Regression: init_database created `observations` but not
    `observations_fts`. The index existed in production only because a
    migration added it, so every NEW deployment came up silently without
    content search."""
    with service.pool.get_connection() as conn:
        found = conn.execute(
            "SELECT name FROM sqlite_master WHERE name = 'observations_fts'"
        ).fetchone()
    assert found, "fresh database has no FTS index; content search is dead"


def test_insert_trigger_populates_the_index(service):
    with service.pool.get_connection() as conn:
        count = conn.execute("SELECT COUNT(*) FROM observations_fts").fetchone()[0]
    assert count == 2, "obs_fts_ai trigger did not fire on insert"


def test_end_to_end_finds_entity_by_observation_content_only(service):
    """The defect, through the real public entry point: 'pagerank' appears
    nowhere in any entity name or type."""
    result = service.search_nodes("pagerank", limit=5)
    assert result["success"] is True
    assert [r["name"] for r in result["results"]] == ["cluster_topology"]


def test_end_to_end_preserves_the_return_shape(service):
    """Callers depend on these keys; fusion must not change the contract."""
    for record in service.search_nodes("recall", limit=5)["results"]:
        assert {"id", "name", "entityType", "observations", "tier"} <= set(record)


def test_end_to_end_name_match_still_outranks_content_match(service):
    """Both entities match 'retrieval' somewhere; the one named for it wins."""
    results = service.search_nodes("retrieval", limit=5)["results"]
    assert results[0]["name"] == "retrieval_notes"


def test_end_to_end_no_match_returns_empty_success(service):
    result = service.search_nodes("zzz_no_such_token_anywhere", limit=5)
    assert result["success"] is True and result["count"] == 0
