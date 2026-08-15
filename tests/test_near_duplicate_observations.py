"""Near-duplicate observation detection (simhash), follow-up to issue #8.

The exact-content dedupe stops identical re-imports; this layer detects
RE-WORDED re-imports. The load-bearing property tested here is what it must
NOT do: a correction ("62Gi" -> "125Gi") is indistinguishable from a reword
at this layer, so the default policy stores near-duplicates and reports them,
never drops them. `skip` is opt-in via ENHANCED_MEMORY_NEAR_DUP_POLICY for
pipelines that know they are re-importing.

Distance bands measured 2026-08-15 (see simhash_dedup.py docstring):
rewords 0..14, corrections 15..25, unrelated 27..31; threshold 16.

Gaps / not covered: cross-entity near-duplicates (per-entity check only);
threshold stability on a larger corpus than the 11 calibration pairs.
"""

import sqlite3

import pytest

import simhash_dedup
from memory_db_service import MemoryDatabase


@pytest.fixture()
def db(tmp_path):
    return MemoryDatabase(tmp_path / "near.db")


def _obs_count(db, name):
    return (
        sqlite3.connect(db.db_path)
        .execute(
            "SELECT COUNT(*) FROM observations o JOIN entities e "
            "ON o.entity_id = e.id WHERE e.name = ?",
            (name,),
        )
        .fetchone()[0]
    )


def _put(db, name, obs):
    return db.create_entities([{"name": name, "entityType": "t", "observations": obs}])


class TestDefaultReportPolicy:
    def test_reworded_reimport_is_stored_and_reported(self, db):
        _put(db, "e1", ["deployment is claude-haiku-4-5"])
        out = _put(db, "e1", ["the deployment uses claude-haiku-4-5"])
        assert out["observations_near_dup_stored"] == 1
        assert out["observations_near_dup_skipped"] == 0
        d = out["near_duplicates"][0]
        assert d["action"] == "stored"
        assert d["resembles"] == "deployment is claude-haiku-4-5"
        assert 0 < d["distance"] <= simhash_dedup.DEFAULT_MAX_DISTANCE
        assert _obs_count(db, "e1") == 2  # stored, not dropped

    def test_correction_is_never_dropped(self, db):
        """THE property this design exists for. Dropping the second row here
        would have preserved a wrong RAM figure forever."""
        _put(db, "node", ["ai-lab has 62Gi RAM"])
        out = _put(db, "node", ["ai-lab has 125Gi RAM"])
        assert _obs_count(db, "node") == 2, "correction must be stored"
        # Whether it is FLAGGED depends on where the pair falls against the
        # threshold (measured d=15 vs threshold 16: flagged). Assert the
        # flag only to document it; the count assertion above is the contract.
        assert out["observations_near_dup_stored"] == 1

    def test_unrelated_observation_not_flagged(self, db):
        _put(db, "e2", ["Docker is banned on mac-studio per operator directive"])
        out = _put(db, "e2", ["The FTS index backfills rows that predate its creation"])
        assert out["observations_near_dup_stored"] == 0
        assert out["near_duplicates"] == []
        assert _obs_count(db, "e2") == 2

    def test_exact_duplicate_takes_precedence_over_near(self, db):
        _put(db, "e3", ["same fact"])
        out = _put(db, "e3", ["same fact"])
        assert out["observations_deduped"] == 1
        assert out["observations_near_dup_stored"] == 0
        assert out["near_duplicates"] == []
        assert _obs_count(db, "e3") == 1

    def test_pure_reorder_is_flagged(self, db):
        """Bag-of-words simhash: a reorder is distance 0 -- the same fact."""
        _put(db, "e4", ["Docker is banned on mac-studio per operator directive"])
        out = _put(db, "e4", ["per operator directive Docker is banned on mac-studio"])
        assert out["observations_near_dup_stored"] == 1
        assert out["near_duplicates"][0]["distance"] == 0


class TestSkipPolicy:
    def test_skip_drops_reword_keeps_distinct(self, db, monkeypatch):
        monkeypatch.setenv(simhash_dedup.POLICY_ENV, "skip")
        _put(db, "e5", ["deployment is claude-haiku-4-5"])
        out = _put(
            db,
            "e5",
            [
                "the deployment uses claude-haiku-4-5",  # reword -> dropped
                "SSDRAID0 is execution and FILES is backup only",  # distinct -> kept
            ],
        )
        assert out["observations_near_dup_skipped"] == 1
        assert out["near_duplicates"][0]["action"] == "skipped"
        assert _obs_count(db, "e5") == 2  # original + the distinct one

    def test_bogus_policy_value_falls_back_to_report(self, db, monkeypatch):
        monkeypatch.setenv(simhash_dedup.POLICY_ENV, "yolo")
        _put(db, "e6", ["deployment is claude-haiku-4-5"])
        out = _put(db, "e6", ["the deployment uses claude-haiku-4-5"])
        assert out["observations_near_dup_skipped"] == 0
        assert _obs_count(db, "e6") == 2, "unknown policy must fail SAFE (store)"


class TestDetailCap:
    def test_detail_list_caps_at_twenty(self, db):
        base = [f"unique fact number {i} about subsystem alpha beta" for i in range(25)]
        _put(db, "e7", base)
        reworded = [
            f"fact number {i} about the subsystem alpha beta" for i in range(25)
        ]
        out = _put(db, "e7", reworded)
        assert out["observations_near_dup_stored"] == 25
        assert len(out["near_duplicates"]) == 20  # capped; counts stay exact
