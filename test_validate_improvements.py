#!/usr/bin/env python3
"""
Unit tests for SelfImprovement.validate_improvements direction/scale-aware
validation (landed 5bce2d68, 2026-07-02).

Regression guard for the direction-blindness that misreported cycles
48/60/61/62 (metrics where a DECREASE is the improvement scored as
regressions; raw-scale metrics dominated sub-1.0 scores).

Runs against a TEMP database only: SelfImprovement is constructed with an
explicit db_path (2026-07-18 refactor), so the production
~/.claude/enhanced_memories/memory.db is never opened.

Run: python3 test_validate_improvements.py   (or: pytest -q test_validate_improvements.py)
"""

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from agi import self_improvement as si  # noqa: E402

SCHEMA = """
CREATE TABLE self_improvement_cycles (
    cycle_id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT NOT NULL,
    cycle_number INTEGER NOT NULL,
    cycle_type TEXT NOT NULL,
    baseline_performance REAL,
    identified_weaknesses TEXT,
    improvement_goals TEXT,
    strategies_applied TEXT,
    changes_made TEXT,
    experiments_run INTEGER DEFAULT 0,
    new_performance REAL,
    improvement_delta REAL,
    success_criteria_met BOOLEAN DEFAULT FALSE,
    lessons_learned TEXT,
    next_cycle_recommendations TEXT,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    duration_seconds INTEGER,
    baseline_metrics TEXT
);
"""

PROD_DB = Path.home() / ".claude" / "enhanced_memories" / "memory.db"


class ValidateImprovementsTest(unittest.TestCase):
    def setUp(self):
        fd = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        fd.close()
        self.db_path = Path(fd.name)
        conn = sqlite3.connect(self.db_path)
        conn.executescript(SCHEMA)
        conn.commit()
        conn.close()
        self.imp = si.SelfImprovement(db_path=self.db_path)

    def tearDown(self):
        self.db_path.unlink(missing_ok=True)

    def _mk_cycle(self, baseline_performance=None, baseline_metrics=None):
        self.assertNotEqual(self.imp.db_path, PROD_DB, "test must not touch production DB")
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO self_improvement_cycles"
            " (agent_id, cycle_number, cycle_type, baseline_performance, baseline_metrics)"
            " VALUES (?, ?, ?, ?, ?)",
            ("test_agent", 1, "performance", baseline_performance, baseline_metrics),
        )
        conn.commit()
        cycle_id = cur.lastrowid
        conn.close()
        assert cycle_id is not None
        return cycle_id

    def _row(self, cycle_id):
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT new_performance, improvement_delta, success_criteria_met"
            " FROM self_improvement_cycles WHERE cycle_id = ?",
            (cycle_id,),
        ).fetchone()
        conn.close()
        return row

    def test_decrease_direction_counts_reduction_as_improvement(self):
        # stuck_tasks 8 -> 0 with direction=decrease: the cycle-48/62 case
        cid = self._mk_cycle(baseline_metrics='{"stuck_tasks": 8}')
        ok = self.imp.validate_improvements(
            cycle_id=cid,
            new_metrics={"stuck_tasks": 0},
            success_criteria={"stuck_tasks": {"direction": "decrease"}},
        )
        self.assertTrue(ok)
        _, delta, met = self._row(cid)
        self.assertAlmostEqual(delta, 1.0)
        self.assertTrue(met)

    def test_default_direction_is_increase(self):
        # Same numbers WITHOUT the flag must score as a regression: the
        # old silent failure is now an explicit, controllable default.
        cid = self._mk_cycle(baseline_metrics='{"stuck_tasks": 8}')
        ok = self.imp.validate_improvements(
            cycle_id=cid,
            new_metrics={"stuck_tasks": 0},
            success_criteria={},
        )
        self.assertFalse(ok)
        _, delta, _ = self._row(cid)
        self.assertAlmostEqual(delta, -1.0)

    def test_scale_bounded_large_metric_cannot_dominate(self):
        # causal_chain_depth-style raw metric (2.8) next to sub-1.0 scores:
        # each metric's contribution must be bounded to [-1, 1].
        cid = self._mk_cycle(baseline_metrics='{"depth": 2.8, "score": 0.5}')
        ok = self.imp.validate_improvements(
            cycle_id=cid,
            new_metrics={"depth": 2.8, "score": 0.6},
            success_criteria={},
        )
        _, delta, _ = self._row(cid)
        # depth: 0/2.8 = 0.0; score: 0.1/0.6 = 0.16667; mean = 0.08333
        self.assertAlmostEqual(delta, (0.0 + 0.1 / 0.6) / 2, places=5)
        self.assertLessEqual(abs(delta), 1.0)
        self.assertTrue(ok)

    def test_zero_baseline_does_not_explode(self):
        cid = self._mk_cycle(baseline_metrics='{"attribution": 0.0}')
        self.imp.validate_improvements(
            cycle_id=cid,
            new_metrics={"attribution": 1.0},
            success_criteria={},
        )
        _, delta, _ = self._row(cid)
        self.assertAlmostEqual(delta, 1.0)

    def test_explicit_met_flags_take_precedence(self):
        # Great normalized deltas must not override an explicit failed criterion.
        cid = self._mk_cycle(baseline_metrics='{"a": 0.1}')
        ok = self.imp.validate_improvements(
            cycle_id=cid,
            new_metrics={"a": 1.0},
            success_criteria={
                "a": {"met": True},
                "b": {"met": False},
            },
        )
        self.assertFalse(ok)
        _, _, met = self._row(cid)
        self.assertFalse(met)

    def test_hold_direction_excluded_from_delta(self):
        cid = self._mk_cycle(baseline_metrics='{"a": 1.0, "b": 0.5}')
        self.imp.validate_improvements(
            cycle_id=cid,
            new_metrics={"a": 0.0, "b": 1.0},
            success_criteria={"a": {"direction": "hold"}},
        )
        _, delta, _ = self._row(cid)
        # a excluded; b: 0.5/1.0 = 0.5
        self.assertAlmostEqual(delta, 0.5)

    def test_legacy_scalar_path_without_baseline_metrics(self):
        cid = self._mk_cycle(baseline_performance=0.5, baseline_metrics=None)
        ok = self.imp.validate_improvements(
            cycle_id=cid,
            new_metrics={"x": 0.7},
            success_criteria={},
        )
        self.assertTrue(ok)
        new_perf, delta, _ = self._row(cid)
        self.assertAlmostEqual(new_perf, 0.7)
        self.assertAlmostEqual(delta, 0.2)

    def test_min_improvement_threshold(self):
        cid = self._mk_cycle(baseline_metrics='{"score": 0.5}')
        ok = self.imp.validate_improvements(
            cycle_id=cid,
            new_metrics={"score": 0.6},
            success_criteria={"min_improvement": 0.5},
        )
        # normalized delta 0.16667 < 0.5
        self.assertFalse(ok)

    def test_default_db_path_is_production(self):
        # Compatibility: agi_tools_phase4's `SelfImprovement()` singleton must
        # keep pointing at the production DB. Instantiation only, no connect.
        self.assertEqual(si.SelfImprovement().db_path, PROD_DB)

    def test_unknown_metric_names_ignored(self):
        cid = self._mk_cycle(baseline_metrics='{"a": 0.5}')
        self.imp.validate_improvements(
            cycle_id=cid,
            new_metrics={"a": 1.0, "not_in_baseline": 99.0},
            success_criteria={},
        )
        _, delta, _ = self._row(cid)
        # only a contributes: 0.5/1.0 = 0.5
        self.assertAlmostEqual(delta, 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
