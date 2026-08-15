"""Tests for the schema-health and graph-bridge reports.

Both tools are read-only by design, and that is the property most worth
pinning: a report that quietly mutates the memory store would be far worse
than one that reports the wrong number.
"""

from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "libs"))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


schema_health = _load("schema_health")
bridge_report = _load("graph_bridge_report")


@pytest.fixture
def db(tmp_path):
    """A store with a realistic type distribution: one dominant type, a few
    stable ones, several singletons, and one case-duplicate pair."""
    path = tmp_path / "m.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE entities (id INTEGER PRIMARY KEY, name TEXT, entity_type TEXT);
        CREATE TABLE relations (
            id INTEGER PRIMARY KEY, from_entity_id INTEGER, to_entity_id INTEGER,
            relation_type TEXT
        );
        """
    )
    rows = [
        (1, "insight_a", "platonic_insight"),
        (2, "insight_b", "platonic_insight"),
        (3, "insight_c", "platonic_insight"),
        (4, "node_one", "gpu_node"),
        (5, "node_two", "gpu_node"),
        (6, "oddity", "DOCUMENT"),
        (7, "another", "system_event"),
        (8, "pat_upper", "PATTERN"),
        (9, "pat_lower", "pattern"),
    ]
    conn.executemany("INSERT INTO entities VALUES (?,?,?)", rows)
    conn.execute("INSERT INTO relations VALUES (1, 4, 5, 'related')")
    conn.commit()
    conn.close()
    return path


# --------------------------------------------------------------- schema health


def test_counts_types_and_entities(db):
    report = schema_health.build_report(db, tau=2, limit=10)
    assert report["totals"]["entities"] == 9
    assert report["totals"]["distinct_types"] == 6


def test_stable_and_singleton_split(db):
    report = schema_health.build_report(db, tau=2, limit=10)
    totals = report["totals"]
    # platonic_insight (3) and gpu_node (2) clear tau; PATTERN+pattern merge to 2.
    assert totals["stable_types"] == 3
    assert totals["singleton_types"] == 4


def test_case_duplicates_are_detected(db):
    report = schema_health.build_report(db, tau=2, limit=10)
    assert report["totals"]["case_punctuation_duplicates"] == 1
    assert ["PATTERN", "pattern"] in report["duplicate_types"].values()


def test_tau_controls_promotion(db):
    strict = schema_health.build_report(db, tau=4, limit=10)
    assert strict["totals"]["stable_types"] == 0  # only platonic_insight has 3


def test_consolidation_suggests_a_token_overlapping_type(db):
    """'DOCUMENT' shares no token with any stable type; a forced guess would be
    worse than an empty suggestion."""
    report = schema_health.build_report(db, tau=2, limit=10)
    assert report["consolidation_worklist"]["DOCUMENT"] == []


def test_consolidation_matches_on_shared_tokens():
    merges = schema_health.suggest_merges(
        ["access_strategy"], ["optimization_strategy", "gpu_node"]
    )
    assert merges["access_strategy"] == ["optimization_strategy"]


def test_report_does_not_modify_the_database(db):
    before = db.read_bytes()
    schema_health.build_report(db, tau=2, limit=10)
    assert db.read_bytes() == before, "read-only report mutated the store"


def test_render_never_raises_on_a_real_report(db):
    assert "Schema health" in schema_health.render(
        schema_health.build_report(db, tau=2, limit=10)
    )


# ---------------------------------------------------------------- bridge report


def test_load_graph_reads_entities_and_relations(db):
    entities, relations = bridge_report.load_graph(db)
    assert len(entities) == 9
    assert relations == [("4", "5")]


def test_load_graph_does_not_modify_the_database(db):
    before = db.read_bytes()
    bridge_report.load_graph(db)
    assert db.read_bytes() == before, "read-only load mutated the store"


def test_connectivity_reflects_the_sparse_graph(db):
    from memgraph import bridging

    entities, relations = bridge_report.load_graph(db)
    stats = bridging.connectivity(list(entities), relations)
    assert stats["connected_entities"] == 2
    assert stats["entities"] == 9


def test_fetch_embeddings_degrades_when_qdrant_is_down():
    """The similarity pass must be skippable; an unreachable vector store
    should not take out the whole report."""
    names, vectors = bridge_report.fetch_embeddings(
        "definitely_not_a_collection_xyz", limit=5
    )
    assert names == [] and vectors.shape[0] == 0
