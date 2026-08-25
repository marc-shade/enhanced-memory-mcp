"""local_semantic_recall must agree with the SQL search paths on what is live.

Two regressions, both measured on a real store before the fix:

- A backfill that reads every entity resurrects archived and quarantined
  memories into the vector index that injects into prompts (2,048 points
  became 12,017 on 2026-08-24).
- A reader that trusts the Qdrant payload ranks an archived entity first,
  because archiving does not evict the point (2026-08-25).

Both are fixed by one predicate, LIVE_PREDICATE, applied by the backfill
(`_rows`) and the reader (`_drop_dead`). These tests build a throwaway
database with the four ways an entity can be dead and check that neither
path lets one through, and that the template contextual prefix (measured to
hurt vague-query retrieval) is not embedded.

Gaps / not covered: nothing here talks to Qdrant or ollama; `search()` and
`backfill()` themselves are exercised only through the helpers they call.
"""

import sqlite3
from types import SimpleNamespace

import pytest

import local_semantic_recall as lsr

TEMPLATE = "[Context: This is a note entity named 'x' with information about y]"


@pytest.fixture
def store(tmp_path, monkeypatch):
    db = tmp_path / "recall.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE entities (
            id INTEGER PRIMARY KEY, name TEXT, entity_type TEXT, tier TEXT,
            archived_at TEXT, superseded_by INTEGER
        );
        CREATE TABLE observations (entity_id INTEGER, content TEXT);
        """
    )
    rows = [
        (1, "live", "note", "long_term", None, None),
        (2, "archived", "note", "long_term", "2026-08-25T00:00:00", None),
        (3, "superseded", "note", "long_term", None, 1),
        (4, "archive-tier", "note", "archive", None, None),
        (5, "quarantined", "note", "quarantine", None, None),
        (6, "template-only", "note", None, None, None),
        (7, "auto", "auto_memory/project", "", None, None),
    ]
    conn.executemany("INSERT INTO entities VALUES (?,?,?,?,?,?)", rows)
    conn.executemany(
        "INSERT INTO observations VALUES (?,?)",
        [
            (1, TEMPLATE),
            (1, "real content about the live entity"),
            (2, "archived content"),
            (3, "superseded content"),
            (4, "archive tier content"),
            (5, "quarantined content"),
            (6, TEMPLATE),
            (7, "auto memory content"),
        ],
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr(lsr, "DB", str(db))
    return db


def test_drop_dead_keeps_only_live_entities_in_rank_order(store):
    hits = [SimpleNamespace(id=i) for i in (2, 7, 3, 1, 4, 5, 6)]
    kept = [h.id for h in lsr._drop_dead(hits)]
    assert kept == [7, 1, 6]


def test_drop_dead_with_no_hits_is_empty(store):
    assert lsr._drop_dead([]) == []


def test_drop_dead_drops_ids_missing_from_the_store(store):
    kept = lsr._drop_dead([SimpleNamespace(id=999), SimpleNamespace(id=1)])
    assert [h.id for h in kept] == [1]


def test_rows_excludes_every_dead_form_and_the_template_prefix(store):
    rows = lsr._rows(auto_only=False)
    by_id = {r[0]: r for r in rows}
    assert set(by_id) == {1, 7}, "only live entities with real content embed"
    text = by_id[1][3]
    assert "real content about the live entity" in text
    assert "[Context: This is a" not in text, "template prefix must not be embedded"


def test_rows_auto_only_narrows_within_the_live_set(store):
    rows = lsr._rows(auto_only=True)
    assert [r[0] for r in rows] == [7]
