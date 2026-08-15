#!/usr/bin/env python3
"""
Unit tests for gap-6 calibration Phases B and C
(docs/plans/gap6-calibration-implementation-plan-2026-07-18.md).

Phase B: reliability bins + Murphy decomposition in
compute_confidence_calibration. Phase C: fit_confidence_scaling temperature
fit (raw rows never mutated, refit guard).

Temp DB only via the db_path parameter; production memory.db never opened.

Run: python3 test_confidence_calibration.py
"""

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from agi.action_tracker import (  # noqa: E402
    DB_PATH as PROD_DB,
    compute_confidence_calibration,
    fit_confidence_scaling,
)

SCHEMA = """
CREATE TABLE action_outcomes (
    action_id INTEGER PRIMARY KEY AUTOINCREMENT,
    action_type TEXT,
    action_description TEXT,
    predicted_confidence REAL,
    goal_outcome REAL,
    executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


class CalibrationTestBase(unittest.TestCase):
    def setUp(self):
        fd = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        fd.close()
        self.db_path = Path(fd.name)
        self.assertNotEqual(self.db_path, PROD_DB)
        conn = sqlite3.connect(self.db_path)
        conn.executescript(SCHEMA)
        conn.commit()
        conn.close()

    def tearDown(self):
        self.db_path.unlink(missing_ok=True)

    def _insert(self, pairs):
        conn = sqlite3.connect(self.db_path)
        conn.executemany(
            "INSERT INTO action_outcomes (predicted_confidence, goal_outcome)"
            " VALUES (?, ?)",
            pairs,
        )
        conn.commit()
        conn.close()


class PhaseBReliabilityTest(CalibrationTestBase):
    def test_refusal_below_min_n(self):
        self._insert([(0.9, 1.0)] * 29)
        r = compute_confidence_calibration(db_path=self.db_path)
        self.assertIsNone(r["brier_score"])
        self.assertEqual(r["n"], 29)
        self.assertIn("insufficient data", r["note"])

    def test_known_answer_overconfident_single_bin(self):
        # All predictions 0.9, outcomes 50/50 (n=40): brier = 0.41,
        # reliability = (0.9-0.5)^2 = 0.16, resolution = 0, uncertainty = 0.25.
        self._insert([(0.9, 1.0)] * 20 + [(0.9, 0.0)] * 20)
        r = compute_confidence_calibration(db_path=self.db_path)
        self.assertAlmostEqual(r["brier_score"], 0.41, places=4)
        m = r["murphy_decomposition"]
        self.assertAlmostEqual(m["reliability"], 0.16, places=4)
        self.assertAlmostEqual(m["resolution"], 0.0, places=4)
        self.assertAlmostEqual(m["uncertainty"], 0.25, places=4)
        self.assertEqual(r["n_bins"], 5)
        self.assertEqual(len(r["reliability_bins"]), 1)
        self.assertEqual(r["reliability_bins"][0]["count"], 40)

    def test_known_answer_perfectly_calibrated(self):
        # 20 rows at 0.2 (4 ones) + 20 rows at 0.8 (16 ones):
        # reliability = 0, resolution = 0.09, uncertainty = 0.25, brier = 0.16.
        self._insert(
            [(0.2, 1.0)] * 4 + [(0.2, 0.0)] * 16 + [(0.8, 1.0)] * 16 + [(0.8, 0.0)] * 4
        )
        r = compute_confidence_calibration(db_path=self.db_path)
        m = r["murphy_decomposition"]
        self.assertAlmostEqual(m["reliability"], 0.0, places=4)
        self.assertAlmostEqual(m["resolution"], 0.09, places=4)
        self.assertAlmostEqual(r["brier_score"], 0.16, places=4)
        # Murphy identity (exact: within-bin-constant confidences)
        self.assertAlmostEqual(
            r["brier_score"],
            m["reliability"] - m["resolution"] + m["uncertainty"],
            places=3,
        )

    def test_ten_bins_at_60(self):
        self._insert([(0.55, 1.0)] * 30 + [(0.45, 0.0)] * 30)
        r = compute_confidence_calibration(db_path=self.db_path)
        self.assertEqual(r["n_bins"], 10)

    def test_scaled_flag_without_fit_reports_none(self):
        self._insert([(0.9, 1.0)] * 30)
        r = compute_confidence_calibration(db_path=self.db_path, scaled=True)
        self.assertIsNone(r["scaled"])
        self.assertIsNotNone(r["brier_score"])


class PhaseCScalingTest(CalibrationTestBase):
    def _overconfident_sample(self):
        # Predictions 0.95 where true rate is 0.7, and 0.05 where true rate
        # is 0.3: systematically overconfident, T > 1 must improve Brier.
        return (
            [(0.95, 1.0)] * 35
            + [(0.95, 0.0)] * 15
            + [(0.05, 1.0)] * 15
            + [(0.05, 0.0)] * 35
        )

    def test_fit_reduces_brier_on_overconfident_sample(self):
        self._insert(self._overconfident_sample())
        fit = fit_confidence_scaling(db_path=self.db_path)
        self.assertTrue(fit["fitted"])
        self.assertGreater(fit["temperature"], 1.0)
        self.assertLess(fit["scaled_brier"], fit["raw_brier"])

    def test_raw_rows_byte_identical_after_fit(self):
        self._insert(self._overconfident_sample())
        conn = sqlite3.connect(self.db_path)
        before = conn.execute(
            "SELECT action_id, predicted_confidence, goal_outcome"
            " FROM action_outcomes ORDER BY action_id"
        ).fetchall()
        conn.close()
        fit_confidence_scaling(db_path=self.db_path)
        conn = sqlite3.connect(self.db_path)
        after = conn.execute(
            "SELECT action_id, predicted_confidence, goal_outcome"
            " FROM action_outcomes ORDER BY action_id"
        ).fetchall()
        conn.close()
        self.assertEqual(before, after)

    def test_fit_refusal_below_min_n(self):
        self._insert([(0.9, 1.0)] * 49)
        fit = fit_confidence_scaling(db_path=self.db_path)
        self.assertFalse(fit["fitted"])
        self.assertIn("insufficient data", fit["note"])

    def test_refit_guard_without_growth(self):
        self._insert(self._overconfident_sample())
        first = fit_confidence_scaling(db_path=self.db_path)
        self.assertTrue(first["fitted"])
        second = fit_confidence_scaling(db_path=self.db_path)
        self.assertFalse(second["fitted"])
        self.assertIn("refit refused", second["note"])
        forced = fit_confidence_scaling(db_path=self.db_path, force=True)
        self.assertTrue(forced["fitted"])

    def test_scaled_report_includes_raw_and_scaled(self):
        self._insert(self._overconfident_sample())
        fit_confidence_scaling(db_path=self.db_path)
        r = compute_confidence_calibration(db_path=self.db_path, scaled=True)
        self.assertIsNotNone(r["brier_score"])
        self.assertIsNotNone(r["scaled"])
        self.assertLess(r["scaled"]["scaled_brier"], r["brier_score"])
        self.assertEqual(
            r["scaled"]["note"], "raw brier_score above remains the ground record"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
