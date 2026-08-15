"""Tests for the hub-suppression seeding term (MemGraphRAG eq. 7).

The property under test: a seed wired to most of the local graph must lose
influence relative to an equally-relevant but specific seed, and turning the
flag off must leave existing ranking byte-identical.

Note on the fixture: IDFKeywordGraph expects each item to carry a precomputed
`keywords` list (the production loader derives it via extract_keywords), and
graph_recall deliberately omits the seed nodes from its output. So the observable
effect is on the seeds' NEIGHBOURS, not on the seeds themselves.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atommem.idf_keyword_graph import IDFGraphConfig, IDFKeywordGraph  # noqa: E402


def _item(fid, keywords):
    return {
        "id": fid,
        "name": fid,
        "entity_type": "t",
        "text": " ".join(keywords),
        "keywords": list(keywords),
        "people": [],
    }


def _graph(**config):
    """Two seeds share the query keyword 'alpha'.

    hub       also shares 'common' with eight filler nodes -> high degree
    specific  shares only 'zeta' with one node -> low degree

    Each seed has a private neighbour whose score we can read: hubfriend and
    specfriend. Suppressing the hub should shift score away from hubfriend.
    """
    items = [
        _item("hub", ["alpha", "common"]),
        _item("specific", ["alpha", "zeta"]),
        _item("hubfriend", ["common", "hubonly"]),
        _item("specfriend", ["zeta", "speconly"]),
    ]
    items += [_item(f"filler{i}", ["common", f"f{i}"]) for i in range(8)]
    return IDFKeywordGraph(items, config=IDFGraphConfig(**config) if config else None)


def _scores(**config):
    return {
        r["id"]: r["graph_score"]
        for r in _graph(**config).graph_recall(query_keywords=["alpha"], top_k=20)
    }


def test_fixture_actually_produces_recall():
    """Guard: an empty result would make every other test here vacuous."""
    scores = _scores()
    assert scores, "fixture produced no recall; the remaining tests prove nothing"
    assert "hubfriend" in scores and "specfriend" in scores, scores


def test_flag_defaults_to_off():
    """Existing ranking must not shift because a new option was added."""
    assert IDFGraphConfig().hub_suppression is False


def test_disabled_matches_previous_behaviour():
    assert _scores() == _scores(hub_suppression=False)


def test_enabling_changes_the_ranking():
    """If scores are identical with the flag on, the term is inert."""
    assert _scores() != _scores(hub_suppression=True), (
        "hub_suppression=True produced identical scores; the term is not wired"
    )


def test_hub_neighbour_loses_ground_to_specific_neighbour():
    """The eq. 7 property. hub and specific are equally query-relevant; only
    hub is wired into the filler cluster, so its downstream neighbour should
    lose relative share once the degree penalty applies."""
    off = _scores()
    on = _scores(hub_suppression=True)
    ratio_off = off["hubfriend"] / max(off["specfriend"], 1e-12)
    ratio_on = on["hubfriend"] / max(on["specfriend"], 1e-12)
    assert ratio_on < ratio_off, (
        f"hub neighbour did not lose ground: {ratio_off:.4f} -> {ratio_on:.4f}"
    )


def test_isolated_seed_does_not_divide_by_zero():
    """log1p(0) == 0. A node with no neighbours is not a hub and must keep its
    weight rather than producing NaN or inf."""
    items = [_item("lonely", ["quokka"]), _item("other", ["unrelated"])]
    graph = IDFKeywordGraph(items, config=IDFGraphConfig(hub_suppression=True))
    for record in graph.graph_recall(query_keywords=["quokka"], top_k=5):
        score = record["graph_score"]
        assert score == score and score not in (float("inf"), float("-inf"))


def test_scores_stay_finite_and_nonnegative():
    for flag in (False, True):
        for fid, score in _scores(hub_suppression=flag).items():
            assert score == score, f"NaN score for {fid}"
            assert score >= 0.0, f"negative score for {fid}: {score}"
