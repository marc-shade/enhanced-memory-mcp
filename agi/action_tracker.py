"""
Action Outcome Tracking Module

Implements memory-action closed loop for AGI learning from experience.

Key Features:
- Track action outcomes (success/failure)
- Learn from past actions to improve future decisions
- Extract learnings automatically
- Query past actions for similar contexts
"""

import sqlite3
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from memory_paths import get_memory_paths

logger = logging.getLogger("action-tracker")

# Configuration. Resolving the path inline here ignored every override, so
# test_agi_phase1.py wrote its rows into the operator's real database while the
# harness around it was pointed at a sandbox (measured 2026-08-14).
MEMORY_DIR, DB_PATH = get_memory_paths()


class ActionTracker:
    """Tracks action outcomes for learning"""

    def __init__(self, agent_id: str = "default_agent"):
        self.agent_id = agent_id

    def record_action(
        self,
        action_type: str,
        action_description: str,
        expected_result: str,
        actual_result: str,
        success_score: float,
        session_id: Optional[str] = None,
        entity_id: Optional[int] = None,
        action_context: Optional[str] = None,
        duration_ms: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Record an action and its outcome

        Args:
            action_type: Type of action ("code_change", "command", "research", etc.)
            action_description: What was done
            expected_result: What we expected to happen
            actual_result: What actually happened
            success_score: 0.0 (failure) to 1.0 (success)
            session_id: Session this action belongs to
            entity_id: Associated memory entity
            action_context: Why this action was taken
            duration_ms: How long it took
            metadata: Additional data

        Returns:
            action_id
        """
        # Determine outcome category from success score
        if success_score >= 0.8:
            outcome_category = "success"
        elif success_score >= 0.5:
            outcome_category = "partial"
        elif success_score >= 0.2:
            outcome_category = "failure"
        else:
            outcome_category = "error"

        # Extract learning from outcome
        learning = self._extract_learning(
            action_type,
            action_description,
            expected_result,
            actual_result,
            success_score,
        )

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO action_outcomes (
                entity_id, session_id,
                action_type, action_description, action_context,
                expected_result, actual_result,
                success_score, outcome_category,
                learning_extracted, will_retry,
                executed_at, duration_ms, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entity_id,
                session_id,
                action_type,
                action_description,
                action_context,
                expected_result,
                actual_result,
                success_score,
                outcome_category,
                learning,
                0 if success_score >= 0.7 else 1,
                datetime.now().isoformat(),
                duration_ms,
                json.dumps(metadata or {}),
            ),
        )

        action_id = cursor.lastrowid
        conn.commit()
        conn.close()

        logger.info(
            f"Recorded {outcome_category} action: {action_type} (score: {success_score})"
        )

        return action_id

    def get_similar_actions(
        self, action_type: str, context: Optional[str] = None, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find similar past actions for learning

        Returns most recent similar actions with outcomes
        """
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if context:
            # Search with context similarity (simple LIKE for now)
            cursor.execute(
                """
                SELECT * FROM action_outcomes
                WHERE action_type = ?
                AND (action_description LIKE ? OR action_context LIKE ?)
                ORDER BY executed_at DESC
                LIMIT ?
                """,
                (action_type, f"%{context}%", f"%{context}%", limit),
            )
        else:
            # Just match action type
            cursor.execute(
                """
                SELECT * FROM action_outcomes
                WHERE action_type = ?
                ORDER BY executed_at DESC
                LIMIT ?
                """,
                (action_type, limit),
            )

        rows = cursor.fetchall()
        conn.close()

        actions = []
        for row in rows:
            action = dict(row)

            # Parse metadata
            if action.get("metadata"):
                try:
                    action["metadata"] = json.loads(action["metadata"])
                except:
                    pass

            actions.append(action)

        return actions

    def get_success_rate(
        self, action_type: str, time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Calculate success rate for an action type

        Returns:
            {
                "action_type": str,
                "total_actions": int,
                "success_count": int,
                "success_rate": float,
                "avg_score": float,
                "time_window_hours": int
            }
        """
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN success_score >= 0.7 THEN 1 ELSE 0 END) as successes,
                AVG(success_score) as avg_score
            FROM action_outcomes
            WHERE action_type = ?
            AND executed_at >= ?
            """,
            (action_type, cutoff_time.isoformat()),
        )

        row = cursor.fetchone()
        conn.close()

        total, successes, avg_score = row

        return {
            "action_type": action_type,
            "total_actions": total or 0,
            "success_count": successes or 0,
            "success_rate": (successes / total) if total > 0 else 0.0,
            "avg_score": avg_score or 0.0,
            "time_window_hours": time_window_hours,
        }

    def get_learnings_for_action(self, action_type: str, limit: int = 5) -> List[str]:
        """
        Get key learnings from past actions of this type

        Returns list of learning strings
        """
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT learning_extracted, success_score
            FROM action_outcomes
            WHERE action_type = ?
            AND learning_extracted IS NOT NULL
            AND learning_extracted != ''
            ORDER BY executed_at DESC
            LIMIT ?
            """,
            (action_type, limit * 2),  # Get more to filter
        )

        rows = cursor.fetchall()
        conn.close()

        # Prioritize learnings from failures (more valuable)
        learnings = []
        for learning, score in rows:
            if score < 0.5:  # Failures first
                learnings.append(learning)

        # Add successes if we need more
        for learning, score in rows:
            if score >= 0.5 and learning not in learnings:
                learnings.append(learning)

        return learnings[:limit]

    def should_retry_action(
        self, original_action_id: int, proposed_changes: str
    ) -> Dict[str, Any]:
        """
        Decide if an action should be retried with changes

        Returns:
            {
                "should_retry": bool,
                "confidence": float,
                "reasoning": str,
                "suggested_changes": List[str]
            }
        """
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Get original action
        cursor.execute(
            "SELECT * FROM action_outcomes WHERE action_id = ?", (original_action_id,)
        )

        row = cursor.fetchone()
        conn.close()

        if not row:
            return {
                "should_retry": False,
                "confidence": 0.0,
                "reasoning": "Original action not found",
            }

        action = dict(row)

        # Simple heuristic for now:
        # - If score < 0.3 and changes proposed: definitely retry
        # - If score 0.3-0.7 and changes proposed: maybe retry
        # - If score > 0.7: probably don't retry (already good)

        score = action["success_score"]

        if score < 0.3:
            return {
                "should_retry": True,
                "confidence": 0.9,
                "reasoning": "Action failed significantly, retry with changes likely to improve",
                "suggested_changes": [proposed_changes],
            }
        elif score < 0.7:
            return {
                "should_retry": True,
                "confidence": 0.6,
                "reasoning": "Partial success, changes might improve outcome",
                "suggested_changes": [proposed_changes],
            }
        else:
            return {
                "should_retry": False,
                "confidence": 0.8,
                "reasoning": "Action already succeeded, retry unnecessary",
                "suggested_changes": [],
            }

    def get_action_statistics(self) -> Dict[str, Any]:
        """Get overall action statistics"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Total actions
        cursor.execute("SELECT COUNT(*) FROM action_outcomes")
        total = cursor.fetchone()[0]

        # By outcome category
        cursor.execute(
            """
            SELECT outcome_category, COUNT(*) as count
            FROM action_outcomes
            GROUP BY outcome_category
            """
        )
        by_category = {row[0]: row[1] for row in cursor.fetchall()}

        # Average success score
        cursor.execute("SELECT AVG(success_score) FROM action_outcomes")
        avg_score = cursor.fetchone()[0] or 0.0

        # Recent trend (last 24h vs previous 24h)
        now = datetime.now()
        recent_cutoff = now - timedelta(hours=24)
        previous_cutoff = now - timedelta(hours=48)

        cursor.execute(
            """
            SELECT AVG(success_score)
            FROM action_outcomes
            WHERE executed_at >= ?
            """,
            (recent_cutoff.isoformat(),),
        )
        recent_avg = cursor.fetchone()[0] or 0.0

        cursor.execute(
            """
            SELECT AVG(success_score)
            FROM action_outcomes
            WHERE executed_at >= ? AND executed_at < ?
            """,
            (previous_cutoff.isoformat(), recent_cutoff.isoformat()),
        )
        previous_avg = cursor.fetchone()[0] or 0.0

        conn.close()

        return {
            "total_actions": total,
            "by_category": by_category,
            "avg_success_score": avg_score,
            "recent_24h_avg": recent_avg,
            "previous_24h_avg": previous_avg,
            "trend": "improving"
            if recent_avg > previous_avg
            else "declining"
            if recent_avg < previous_avg
            else "stable",
        }

    def _extract_learning(
        self,
        action_type: str,
        description: str,
        expected: str,
        actual: str,
        score: float,
    ) -> str:
        """
        Extract learning from action outcome

        Simple rule-based extraction for now
        """
        if score >= 0.8:
            return f"Successful {action_type}: '{description}' worked as expected"
        elif score >= 0.5:
            return f"Partial {action_type}: '{description}' - expected '{expected}' but got '{actual}'"
        else:
            return f"Failed {action_type}: '{description}' - '{actual}' indicates need for different approach"


def _load_calibration_pairs(db_path=None) -> list:
    """(predicted_confidence, goal_outcome) rows with both fields present."""
    import sqlite3

    conn = sqlite3.connect(str(db_path or DB_PATH), timeout=5.0)
    try:
        conn.execute("PRAGMA busy_timeout = 5000")
        return conn.execute(
            """SELECT predicted_confidence, goal_outcome FROM action_outcomes
               WHERE predicted_confidence IS NOT NULL
                 AND goal_outcome IS NOT NULL"""
        ).fetchall()
    finally:
        conn.close()


def _reliability_bins(rows: list, n_bins: int) -> list:
    """Fixed-width reliability bins over [0, 1]: per-bin mean confidence,
    observed outcome rate, and count. Empty bins are omitted."""
    binned = {}
    for p, o in rows:
        idx = min(int(p * n_bins), n_bins - 1)
        binned.setdefault(idx, []).append((p, o))
    out = []
    for idx in sorted(binned):
        members = binned[idx]
        cnt = len(members)
        out.append(
            {
                "bin": f"[{idx / n_bins:.2f},{(idx + 1) / n_bins:.2f})",
                "count": cnt,
                "mean_confidence": round(sum(p for p, _ in members) / cnt, 4),
                "observed_rate": round(sum(o for _, o in members) / cnt, 4),
            }
        )
    return out


def _apply_temperature(p: float, temperature: float) -> float:
    """Logit-scale temperature scaling: sigmoid(logit(p) / T). T > 1 shrinks
    toward 0.5 (fixes overconfidence); T < 1 sharpens. Never mutates stored
    values — applied at read/report time only (arXiv 2410.06707)."""
    import math

    eps = 1e-6
    p = min(max(p, eps), 1.0 - eps)
    return 1.0 / (1.0 + math.exp(-(math.log(p / (1.0 - p)) / temperature)))


def _get_calibration_state(db_path=None) -> Optional[dict]:
    """Stored temperature fit, or None. Table is created by fit; a missing
    table simply means no fit exists yet."""
    import sqlite3

    conn = sqlite3.connect(str(db_path or DB_PATH), timeout=5.0)
    try:
        conn.execute("PRAGMA busy_timeout = 5000")
        row = conn.execute(
            "SELECT value FROM calibration_state WHERE key = 'temperature_fit'"
        ).fetchone()
        return json.loads(row[0]) if row else None
    except sqlite3.OperationalError:
        return None
    finally:
        conn.close()


def compute_confidence_calibration(
    db_path=None, min_n: int = 30, scaled: bool = False
) -> dict:
    """Calibration report over rows where BOTH predicted_confidence and
    goal_outcome exist (gap_id=6). Refuses to emit a score below min_n — a
    calibration number from a handful of rows is noise dressed as measurement.

    At n >= min_n adds fixed-width reliability bins (5 bins; 10 at n >= 60)
    and the Murphy decomposition (brier = reliability - resolution +
    uncertainty; computed against bin-mean confidence, so the identity is
    exact when confidences are constant within bins and approximate
    otherwise — the residual is within-bin confidence variance).

    scaled=True additionally reports the temperature-scaled Brier when a fit
    from fit_confidence_scaling() exists. The scaled number is NEVER emitted
    without the raw one, and stored rows are never modified.

    goal_outcome is TOOL-LEVEL truth (populated by
    agentic-system/scripts/reconcile_action_outcomes.py): "P(action ran
    without error)", not goal achievement. Report it as such.
    """
    rows = _load_calibration_pairs(db_path)

    n = len(rows)
    if n < min_n:
        return {
            "n": n,
            "min_n": min_n,
            "brier_score": None,
            "note": (
                f"insufficient data: {n} paired rows < {min_n} minimum; "
                "emit [conf=0.NN] markers on real actions (including ones "
                "that may fail) and run reconcile_action_outcomes.py"
            ),
        }

    brier = sum((p - o) ** 2 for p, o in rows) / n
    base_rate = sum(o for _, o in rows) / n

    n_bins = 10 if n >= 60 else 5
    bins = _reliability_bins(rows, n_bins)
    reliability = (
        sum(b["count"] * (b["mean_confidence"] - b["observed_rate"]) ** 2 for b in bins)
        / n
    )
    resolution = (
        sum(b["count"] * (b["observed_rate"] - base_rate) ** 2 for b in bins) / n
    )
    uncertainty = base_rate * (1.0 - base_rate)

    result = {
        "n": n,
        "min_n": min_n,
        "brier_score": round(brier, 4),
        "base_rate": round(base_rate, 4),
        "n_bins": n_bins,
        "reliability_bins": bins,
        "murphy_decomposition": {
            "reliability": round(reliability, 4),
            "resolution": round(resolution, 4),
            "uncertainty": round(uncertainty, 4),
        },
        "scope": "tool-level calibration (action ran without error), NOT goal achievement",
    }

    if scaled:
        fit = _get_calibration_state(db_path)
        if fit:
            t = fit["temperature"]
            scaled_brier = sum((_apply_temperature(p, t) - o) ** 2 for p, o in rows) / n
            result["scaled"] = {
                "temperature": t,
                "fitted_at": fit.get("fitted_at"),
                "n_at_fit": fit.get("n_at_fit"),
                "scaled_brier": round(scaled_brier, 4),
                "note": "raw brier_score above remains the ground record",
            }
        else:
            result["scaled"] = None

    return result


def fit_confidence_scaling(db_path=None, min_n: int = 50, force: bool = False) -> dict:
    """Fit a single-parameter temperature correction to the verbalized
    confidence scores (arXiv 2410.06707) by minimizing Brier over the paired
    rows. Stores {temperature, fitted_at, n_at_fit, raw_brier, scaled_brier}
    in calibration_state; NEVER mutates action_outcomes rows.

    Refuses below min_n, and refuses to refit until n has grown >= 25% past
    the previous fit's n_at_fit (force=True overrides) — prevents thrash on
    small increments.
    """
    import sqlite3

    rows = _load_calibration_pairs(db_path)
    n = len(rows)
    if n < min_n:
        return {
            "fitted": False,
            "n": n,
            "min_n": min_n,
            "note": f"insufficient data: {n} paired rows < {min_n} minimum",
        }

    prior = _get_calibration_state(db_path)
    if prior and not force and n < prior.get("n_at_fit", 0) * 1.25:
        return {
            "fitted": False,
            "n": n,
            "note": (
                f"refit refused: n={n} has not grown >=25% past previous fit "
                f"(n_at_fit={prior.get('n_at_fit')}); pass force=True to override"
            ),
            "existing_fit": prior,
        }

    def brier_at(t: float) -> float:
        return sum((_apply_temperature(p, t) - o) ** 2 for p, o in rows) / n

    # Deterministic 1-D search: coarse log-space grid, then golden-section
    # refinement around the best cell. No scipy dependency.
    import math

    grid = [
        math.exp(math.log(0.1) + i * (math.log(10.0) - math.log(0.1)) / 199)
        for i in range(200)
    ]
    best_t = min(grid, key=brier_at)
    i = grid.index(best_t)
    lo = grid[max(0, i - 1)]
    hi = grid[min(len(grid) - 1, i + 1)]
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    for _ in range(60):
        a = hi - phi * (hi - lo)
        b = lo + phi * (hi - lo)
        if brier_at(a) < brier_at(b):
            hi = b
        else:
            lo = a
    best_t = (lo + hi) / 2.0

    raw_brier = sum((p - o) ** 2 for p, o in rows) / n
    scaled_brier = brier_at(best_t)
    fit = {
        "temperature": round(best_t, 4),
        "fitted_at": datetime.now().isoformat(),
        "n_at_fit": n,
        "raw_brier": round(raw_brier, 4),
        "scaled_brier": round(scaled_brier, 4),
    }

    conn = sqlite3.connect(str(db_path or DB_PATH), timeout=5.0)
    try:
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.execute(
            """CREATE TABLE IF NOT EXISTS calibration_state (
                   key TEXT PRIMARY KEY,
                   value TEXT NOT NULL,
                   updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
               )"""
        )
        conn.execute(
            """INSERT INTO calibration_state (key, value, updated_at)
               VALUES ('temperature_fit', ?, CURRENT_TIMESTAMP)
               ON CONFLICT(key) DO UPDATE
               SET value = excluded.value, updated_at = CURRENT_TIMESTAMP""",
            (json.dumps(fit),),
        )
        conn.commit()
    finally:
        conn.close()

    return {"fitted": True, **fit}
