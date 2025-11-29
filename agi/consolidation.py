"""
Memory Consolidation Module

Implements sleep-like consolidation for AGI memory.

Key Features:
- Background pattern extraction from episodic to semantic memory
- Causal link discovery and strengthening
- Memory compression and optimization
- Scheduled consolidation jobs
- TPU-accelerated importance scoring (when available)
"""

import sqlite3
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from collections import Counter
import os

logger = logging.getLogger("consolidation")

# TPU importance scoring integration (optional, graceful degradation)
_TPU_AVAILABLE = False
_score_importance_fn = None

try:
    tpu_path = os.path.join(os.environ.get("AGENTIC_SYSTEM_PATH", "${AGENTIC_SYSTEM_PATH:-/opt/agentic}"), "mcp-servers/coral-tpu-mcp/src")
    if tpu_path not in sys.path:
        sys.path.insert(0, tpu_path)

    from pycoral.utils import edgetpu
    if len(edgetpu.list_edge_tpus()) > 0:
        from coral_tpu_mcp.server import handle_score_importance
        _score_importance_fn = handle_score_importance
        _TPU_AVAILABLE = True
        logger.info("TPU importance scoring available for consolidation")
except ImportError:
    logger.debug("TPU not available for consolidation, using heuristics")
except Exception as e:
    logger.warning(f"TPU init error in consolidation: {e}")


def score_importance_tpu(text: str, context: str = "memory") -> float:
    """
    Score importance using TPU when available, heuristics otherwise.

    Args:
        text: Content to score
        context: Context type (memory, action, pattern)

    Returns:
        Importance score 0.0-1.0
    """
    if _TPU_AVAILABLE and _score_importance_fn:
        try:
            import asyncio

            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            # TPU MCP expects "content" not "text"
            result = loop.run_until_complete(
                _score_importance_fn({"content": text, "context": context})
            )

            if result and len(result) > 0:
                import json as json_mod
                data = json_mod.loads(result[0].text)
                return float(data.get("importance_score", 0.5))

        except Exception as e:
            logger.debug(f"TPU scoring failed, using heuristic: {e}")

    # Heuristic fallback
    return _heuristic_importance(text, context)


def _heuristic_importance(text: str, context: str) -> float:
    """Heuristic importance scoring when TPU unavailable."""
    text_lower = text.lower()
    score = 0.5

    high_importance = [
        "critical", "error", "failure", "bug", "security", "vulnerability",
        "pattern", "causal", "learned", "discovered", "improvement"
    ]
    medium_importance = [
        "warning", "issue", "change", "update", "consolidated", "extracted"
    ]
    low_importance = [
        "minor", "trivial", "temp", "debug", "test"
    ]

    for word in high_importance:
        if word in text_lower:
            score = min(1.0, score + 0.12)
    for word in medium_importance:
        if word in text_lower:
            score = min(1.0, score + 0.06)
    for word in low_importance:
        if word in text_lower:
            score = max(0.2, score - 0.1)

    return round(score, 2)

# Configuration
MEMORY_DIR = Path.home() / ".claude" / "enhanced_memories"
DB_PATH = MEMORY_DIR / "memory.db"


class ConsolidationEngine:
    """Handles background memory consolidation"""

    def __init__(self):
        # Ensure WAL mode is enabled for better concurrent access
        self._init_database()

    def _init_database(self):
        """Initialize database with optimal settings for concurrent access."""
        try:
            conn = sqlite3.connect(DB_PATH, timeout=30.0)
            conn.execute("PRAGMA busy_timeout = 30000")  # 30 second timeout
            conn.execute("PRAGMA journal_mode=WAL")  # WAL mode for concurrency
            conn.close()
        except Exception as e:
            logger.warning(f"Could not init database settings: {e}")

    def _get_connection(self):
        """Get a database connection with proper timeout settings."""
        conn = sqlite3.connect(DB_PATH, timeout=30.0)
        conn.execute("PRAGMA busy_timeout = 30000")
        return conn

    def schedule_consolidation(
        self,
        job_type: str,
        time_window_hours: int = 24
    ) -> int:
        """
        Schedule a consolidation job

        Args:
            job_type: "pattern_extraction", "causal_discovery", "memory_compression"
            time_window_hours: Hours of memory to process

        Returns:
            job_id
        """
        now = datetime.now()
        start_time = now - timedelta(hours=time_window_hours)

        # Count entities in time window
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            '''
            SELECT COUNT(*)
            FROM entities
            WHERE created_at >= ?
            ''',
            (start_time.strftime('%Y-%m-%d %H:%M:%S'),)
        )
        entity_count = cursor.fetchone()[0]

        cursor.execute(
            '''
            INSERT INTO consolidation_jobs (
                agent_id, job_type, status,
                time_window_start, time_window_end,
                entity_count
            ) VALUES (?, ?, ?, ?, ?, ?)
            ''',
            (
                'system-consolidation',  # Default agent_id for scheduled jobs
                job_type, 'pending',
                start_time.strftime('%Y-%m-%d %H:%M:%S'), now.strftime('%Y-%m-%d %H:%M:%S'),
                entity_count
            )
        )

        job_id = cursor.lastrowid
        conn.commit()
        conn.close()

        logger.info(f"Scheduled {job_type} job {job_id} for {entity_count} entities")

        return job_id

    def run_pattern_extraction(
        self,
        time_window_hours: int = 24,
        min_pattern_frequency: int = 3
    ) -> Dict[str, Any]:
        """
        Extract patterns from recent episodic memories

        Promotes recurring patterns to semantic memory.

        Args:
            time_window_hours: Hours of memory to analyze
            min_pattern_frequency: Minimum occurrences to be a pattern

        Returns:
            {
                "patterns_found": int,
                "patterns_promoted": int,
                "semantic_memories_created": int
            }
        """
        job_id = self.schedule_consolidation("pattern_extraction", time_window_hours)

        conn = self._get_connection()
        cursor = conn.cursor()

        # Update job status
        cursor.execute(
            'UPDATE consolidation_jobs SET status = ?, started_at = ? WHERE job_id = ?',
            ('running', datetime.now().isoformat(), job_id)
        )
        conn.commit()

        start_time = datetime.now() - timedelta(hours=time_window_hours)

        try:
            # Get recent episodic memories from BOTH sources:
            # 1. entities table with tier='episodic'
            # 2. episodic_memory table (4-tier AGI system)

            # Source 1: entities table
            cursor.execute(
                '''
                SELECT id, name, entity_type, 'entities' as source
                FROM entities
                WHERE tier = 'episodic'
                AND created_at >= ?
                ''',
                (start_time.strftime('%Y-%m-%d %H:%M:%S'),)
            )
            entities_episodic = cursor.fetchall()

            # Source 2: episodic_memory table (4-tier AGI system)
            cursor.execute(
                '''
                SELECT id, event_type, event_type, 'episodic_memory' as source
                FROM episodic_memory
                WHERE created_at >= ?
                ''',
                (start_time.strftime('%Y-%m-%d %H:%M:%S'),)
            )
            agi_episodic = cursor.fetchall()

            # Combine both sources
            episodic_memories = entities_episodic + agi_episodic

            logger.info(f"Found {len(entities_episodic)} entity episodes + {len(agi_episodic)} AGI episodes")

            # Group by entity_type/event_type and look for patterns
            type_counter = Counter()
            for _, name, entity_type, source in episodic_memories:
                type_counter[entity_type] += 1

            # Extract patterns that occur frequently
            patterns_found = []
            patterns_promoted = 0
            semantic_concepts_created = 0

            for entity_type, count in type_counter.items():
                if count >= min_pattern_frequency:
                    patterns_found.append({
                        "type": entity_type,
                        "frequency": count
                    })

                    # Create semantic memory for this pattern in BOTH systems
                    pattern_name = f"pattern_{entity_type}_{datetime.now().strftime('%Y%m%d')}"

                    # 1. Add to entities table (legacy)
                    cursor.execute(
                        'SELECT id FROM entities WHERE name = ?',
                        (pattern_name,)
                    )

                    if not cursor.fetchone():
                        cursor.execute(
                            '''
                            INSERT INTO entities (name, entity_type, tier)
                            VALUES (?, ?, ?)
                            ''',
                            (pattern_name, f"pattern_{entity_type}", "semantic")
                        )
                        patterns_promoted += 1

                    # 2. Add to semantic_memory table (4-tier AGI system)
                    concept_name = f"consolidated_{entity_type}"
                    cursor.execute(
                        'SELECT id FROM semantic_memory WHERE concept_name = ?',
                        (concept_name,)
                    )

                    if not cursor.fetchone():
                        # Use TPU to score pattern importance when available
                        pattern_desc = f"Pattern extracted from {count} episodes of type '{entity_type}'"
                        tpu_score = score_importance_tpu(pattern_desc, "pattern")
                        # Blend TPU score with frequency-based score
                        freq_score = min(0.5 + (count * 0.1), 1.0)
                        confidence = (tpu_score + freq_score) / 2 if _TPU_AVAILABLE else freq_score

                        cursor.execute(
                            '''
                            INSERT INTO semantic_memory (
                                concept_name, concept_type, definition,
                                confidence_score
                            ) VALUES (?, ?, ?, ?)
                            ''',
                            (
                                concept_name,
                                "consolidated_pattern",
                                pattern_desc,
                                confidence
                            )
                        )
                        semantic_concepts_created += 1
                    else:
                        # Update existing concept confidence
                        cursor.execute(
                            '''
                            UPDATE semantic_memory
                            SET confidence_score = MIN(confidence_score + 0.05, 1.0),
                                updated_at = CURRENT_TIMESTAMP
                            WHERE concept_name = ?
                            ''',
                            (concept_name,)
                        )

            logger.info(f"Created {semantic_concepts_created} new semantic concepts")

            # Update job with results
            duration = (datetime.now() - datetime.fromisoformat(
                cursor.execute('SELECT started_at FROM consolidation_jobs WHERE job_id = ?', (job_id,)).fetchone()[0]
            )).seconds

            cursor.execute(
                '''
                UPDATE consolidation_jobs
                SET
                    status = 'completed',
                    completed_at = ?,
                    duration_seconds = ?,
                    patterns_found = ?,
                    memories_promoted = ?,
                    results_summary = ?
                WHERE job_id = ?
                ''',
                (
                    datetime.now().isoformat(),
                    duration,
                    len(patterns_found),
                    patterns_promoted + semantic_concepts_created,
                    json.dumps({
                        "patterns": patterns_found,
                        "entities_episodic_count": len(entities_episodic),
                        "agi_episodic_count": len(agi_episodic),
                        "semantic_concepts_created": semantic_concepts_created
                    }),
                    job_id
                )
            )

            conn.commit()

            logger.info(f"Pattern extraction complete: {len(patterns_found)} patterns, "
                       f"{patterns_promoted} promoted to entities, "
                       f"{semantic_concepts_created} semantic concepts created")

            return {
                "job_id": job_id,
                "patterns_found": len(patterns_found),
                "patterns_promoted": patterns_promoted,
                "semantic_memories_created": semantic_concepts_created,
                "sources_analyzed": {
                    "entities_episodic": len(entities_episodic),
                    "agi_episodic": len(agi_episodic)
                }
            }

        except Exception as e:
            # Mark job as failed
            cursor.execute(
                '''
                UPDATE consolidation_jobs
                SET status = 'failed', error_message = ?
                WHERE job_id = ?
                ''',
                (str(e), job_id)
            )
            conn.commit()

            logger.error(f"Pattern extraction failed: {e}")

            raise

        finally:
            conn.close()

    def run_causal_discovery(
        self,
        time_window_hours: int = 24,
        min_confidence: float = 0.6
    ) -> Dict[str, Any]:
        """
        Discover causal relationships from recent action outcomes

        Args:
            time_window_hours: Hours of memory to analyze
            min_confidence: Minimum confidence for causal link

        Returns:
            {
                "chains_created": int,
                "links_created": int,
                "hypotheses_generated": int
            }
        """
        from agi.temporal_reasoning import TemporalReasoning

        job_id = self.schedule_consolidation("causal_discovery", time_window_hours)

        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            'UPDATE consolidation_jobs SET status = ?, started_at = ? WHERE job_id = ?',
            ('running', datetime.now().isoformat(), job_id)
        )
        conn.commit()

        start_time = datetime.now() - timedelta(hours=time_window_hours)

        try:
            temporal = TemporalReasoning()

            # Get recent action outcomes with context
            cursor.execute(
                '''
                SELECT
                    action_id,
                    action_type,
                    action_context,
                    success_score,
                    executed_at
                FROM action_outcomes
                WHERE executed_at >= ?
                AND action_context IS NOT NULL
                AND action_context != ''
                ORDER BY executed_at
                ''',
                (start_time.strftime('%Y-%m-%d %H:%M:%S'),)
            )

            actions = cursor.fetchall()
            logger.info(f"Analyzing {len(actions)} actions with context for causal patterns")

            links_created = 0
            chains = []

            # Group actions by context (parse enriched context format)
            # Format: key1:val1|key2:val2|... (e.g., file:/path|component:x|category:y)
            context_actions = {}  # Group by component or category
            file_actions = {}     # Also group by file for file-specific patterns

            for action_id, action_type, action_context, success_score, executed_at in actions:
                action_data = {
                    'id': action_id,
                    'type': action_type,
                    'score': success_score,
                    'time': executed_at,
                    'context': action_context
                }

                # Parse enriched context format
                ctx_parts = action_context.split('|') if action_context else []
                component = None
                category = None
                file_path = None

                for part in ctx_parts:
                    if ':' in part:
                        key, val = part.split(':', 1)
                        if key == 'component':
                            component = val
                        elif key == 'category':
                            category = val
                        elif key == 'file':
                            file_path = val

                # Group by component (primary grouping)
                if component:
                    if component not in context_actions:
                        context_actions[component] = []
                    context_actions[component].append(action_data)

                # Group by category as fallback
                elif category:
                    cat_key = f"cat:{category}"
                    if cat_key not in context_actions:
                        context_actions[cat_key] = []
                    context_actions[cat_key].append(action_data)

                # Group by file for file-specific patterns
                if file_path:
                    if file_path not in file_actions:
                        file_actions[file_path] = []
                    file_actions[file_path].append(action_data)

                # Legacy: also use raw context if no structured data
                if not component and not category:
                    if action_context not in context_actions:
                        context_actions[action_context] = []
                    context_actions[action_context].append(action_data)

            # Merge file-specific patterns into context_actions
            context_actions.update(file_actions)

            # Find causal patterns: sequences of actions on same context
            pattern_counter = Counter()
            for context, ctx_actions in context_actions.items():
                if len(ctx_actions) >= 2:
                    # Look at pairs of consecutive actions
                    for i in range(len(ctx_actions) - 1):
                        current = ctx_actions[i]
                        next_act = ctx_actions[i + 1]

                        # Pattern: action_type_A → action_type_B with combined success
                        pattern = f"{current['type']}->{next_act['type']}"
                        avg_score = (current['score'] + next_act['score']) / 2
                        pattern_counter[(pattern, avg_score >= min_confidence)] += 1

            # Create causal links for patterns that tend to succeed
            for (pattern, succeeded), count in pattern_counter.items():
                if succeeded and count >= 2:  # Pattern seen at least twice with success
                    # Use TPU to score causal pattern significance
                    pattern_desc = f"Causal pattern: {pattern} (observed {count} times with success)"
                    tpu_score = score_importance_tpu(pattern_desc, "pattern")

                    # Blend TPU score with frequency-based score
                    freq_score = min(0.5 + (count * 0.1), 0.95)
                    confidence = (tpu_score + freq_score) / 2 if _TPU_AVAILABLE else freq_score

                    # Store as a learned causal pattern
                    cursor.execute(
                        '''
                        INSERT OR REPLACE INTO semantic_memory
                        (concept_name, concept_type, definition, confidence_score)
                        VALUES (?, 'causal_pattern', ?, ?)
                        ''',
                        (
                            f"pattern_{pattern}",
                            pattern_desc,
                            confidence
                        )
                    )
                    links_created += 1
                    chains.append(pattern)

            logger.info(f"Discovered {links_created} causal patterns from {len(context_actions)} contexts")

            # Update job with results
            duration = (datetime.now() - datetime.fromisoformat(
                cursor.execute('SELECT started_at FROM consolidation_jobs WHERE job_id = ?', (job_id,)).fetchone()[0]
            )).seconds

            cursor.execute(
                '''
                UPDATE consolidation_jobs
                SET
                    status = 'completed',
                    completed_at = ?,
                    duration_seconds = ?,
                    chains_created = ?,
                    links_created = ?,
                    results_summary = ?
                WHERE job_id = ?
                ''',
                (
                    datetime.now().isoformat(),
                    duration,
                    len(chains),
                    links_created,
                    json.dumps({"actions_analyzed": len(actions)}),
                    job_id
                )
            )

            conn.commit()

            logger.info(f"Causal discovery complete: {links_created} links created")

            return {
                "job_id": job_id,
                "chains_created": len(chains),
                "links_created": links_created,
                "hypotheses_generated": 0
            }

        except Exception as e:
            cursor.execute(
                '''
                UPDATE consolidation_jobs
                SET status = 'failed', error_message = ?
                WHERE job_id = ?
                ''',
                (str(e), job_id)
            )
            conn.commit()

            logger.error(f"Causal discovery failed: {e}")

            raise

        finally:
            conn.close()

    def run_memory_compression(
        self,
        time_window_hours: int = 168  # 7 days
    ) -> Dict[str, Any]:
        """
        Compress old low-importance memories

        Args:
            time_window_hours: Only compress memories older than this

        Returns:
            {
                "memories_compressed": int,
                "space_saved_bytes": int
            }
        """
        job_id = self.schedule_consolidation("memory_compression", time_window_hours)

        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            'UPDATE consolidation_jobs SET status = ?, started_at = ? WHERE job_id = ?',
            ('running', datetime.now().isoformat(), job_id)
        )
        conn.commit()

        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)

        try:
            # Get old, low-importance memories that aren't already compressed
            cursor.execute(
                '''
                SELECT id, name, entity_type
                FROM entities
                WHERE created_at < ?
                AND salience_score < 0.5
                AND access_count < 5
                AND compressed_data IS NULL
                ''',
                (cutoff_time.strftime('%Y-%m-%d %H:%M:%S'),)
            )

            candidates = cursor.fetchall()
            memories_to_compress = []
            preserved_count = 0

            # Use TPU to re-evaluate importance before compression
            for mem_id, name, entity_type in candidates:
                # Build description for importance scoring
                description = f"{entity_type}: {name}"
                importance = score_importance_tpu(description, "memory")

                if importance < 0.4:  # Still low importance after TPU scoring
                    memories_to_compress.append((mem_id, name))
                else:
                    # TPU identified as important - preserve and update salience
                    cursor.execute(
                        'UPDATE entities SET salience_score = ? WHERE id = ?',
                        (importance, mem_id)
                    )
                    preserved_count += 1

            if preserved_count > 0:
                logger.info(f"TPU preserved {preserved_count} memories from compression")

            memories_compressed = len(memories_to_compress)

            # In real implementation, would compress observations
            # For now, just mark them as candidates

            # Update job
            duration = (datetime.now() - datetime.fromisoformat(
                cursor.execute('SELECT started_at FROM consolidation_jobs WHERE job_id = ?', (job_id,)).fetchone()[0]
            )).seconds

            cursor.execute(
                '''
                UPDATE consolidation_jobs
                SET
                    status = 'completed',
                    completed_at = ?,
                    duration_seconds = ?,
                    memories_compressed = ?
                WHERE job_id = ?
                ''',
                (
                    datetime.now().isoformat(),
                    duration,
                    memories_compressed,
                    job_id
                )
            )

            conn.commit()

            logger.info(f"Memory compression complete: {memories_compressed} memories processed")

            return {
                "job_id": job_id,
                "memories_compressed": memories_compressed,
                "space_saved_bytes": memories_compressed * 1024  # Estimate
            }

        except Exception as e:
            cursor.execute(
                '''
                UPDATE consolidation_jobs
                SET status = 'failed', error_message = ?
                WHERE job_id = ?
                ''',
                (str(e), job_id)
            )
            conn.commit()

            logger.error(f"Memory compression failed: {e}")

            raise

        finally:
            conn.close()

    def run_full_consolidation(
        self,
        time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Run all consolidation processes

        This is the main "sleep-like" consolidation function.

        Args:
            time_window_hours: Hours of memory to consolidate

        Returns:
            Combined results from all consolidation processes
        """
        logger.info(f"Starting full consolidation for last {time_window_hours} hours")

        results = {}

        # 1. Pattern extraction
        try:
            pattern_results = self.run_pattern_extraction(time_window_hours)
            results["pattern_extraction"] = pattern_results
        except Exception as e:
            results["pattern_extraction"] = {"error": str(e)}
            logger.error(f"Pattern extraction failed: {e}")

        # 2. Causal discovery
        try:
            causal_results = self.run_causal_discovery(time_window_hours)
            results["causal_discovery"] = causal_results
        except Exception as e:
            results["causal_discovery"] = {"error": str(e)}
            logger.error(f"Causal discovery failed: {e}")

        # 3. Skill execution tracking
        try:
            skill_results = self.run_skill_execution_tracking(time_window_hours)
            results["skill_tracking"] = skill_results
        except Exception as e:
            results["skill_tracking"] = {"error": str(e)}
            logger.error(f"Skill execution tracking failed: {e}")

        # 4. Memory compression (only for old memories)
        if time_window_hours >= 168:  # Only compress if looking at 7+ days
            try:
                compression_results = self.run_memory_compression(time_window_hours)
                results["memory_compression"] = compression_results
            except Exception as e:
                results["memory_compression"] = {"error": str(e)}
                logger.error(f"Memory compression failed: {e}")

        # 5. Visual memory consolidation (LVR Phase 2)
        try:
            visual_results = self.run_visual_consolidation(time_window_hours)
            results["visual_consolidation"] = visual_results
        except Exception as e:
            results["visual_consolidation"] = {"error": str(e)}
            logger.error(f"Visual consolidation failed: {e}")

        logger.info("Full consolidation complete")

        return results

    def get_consolidation_stats(self) -> Dict[str, Any]:
        """Get consolidation statistics"""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Total jobs
        cursor.execute('SELECT COUNT(*) FROM consolidation_jobs')
        total_jobs = cursor.fetchone()[0]

        # By status
        cursor.execute(
            '''
            SELECT status, COUNT(*)
            FROM consolidation_jobs
            GROUP BY status
            '''
        )
        by_status = {row[0]: row[1] for row in cursor.fetchall()}

        # Total results
        cursor.execute(
            '''
            SELECT
                SUM(patterns_found) as patterns,
                SUM(chains_created) as chains,
                SUM(links_created) as links,
                SUM(memories_promoted) as promoted,
                SUM(memories_compressed) as compressed
            FROM consolidation_jobs
            WHERE status = 'completed'
            '''
        )

        row = cursor.fetchone()
        totals = {
            "patterns_found": row[0] or 0,
            "chains_created": row[1] or 0,
            "links_created": row[2] or 0,
            "memories_promoted": row[3] or 0,
            "memories_compressed": row[4] or 0
        }

        # Recent jobs
        cursor.execute(
            '''
            SELECT job_type, status, started_at, duration_seconds
            FROM consolidation_jobs
            ORDER BY started_at DESC
            LIMIT 10
            '''
        )

        recent = []
        for row in cursor.fetchall():
            recent.append({
                "job_type": row[0],
                "status": row[1],
                "started_at": row[2],
                "duration_seconds": row[3]
            })

        conn.close()

        return {
            "total_jobs": total_jobs,
            "by_status": by_status,
            "totals": totals,
            "recent_jobs": recent
        }

    def run_skill_execution_tracking(
        self,
        time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Match recent actions to skills and update procedural memory.

        Analyzes action_outcomes to detect skill executions based on
        action types and context patterns.

        Args:
            time_window_hours: Hours of actions to analyze

        Returns:
            {
                "skills_matched": int,
                "skills_updated": dict of skill_name -> update_info
            }
        """
        logger.info(f"Running skill execution tracking for last {time_window_hours} hours")

        # Skill detection patterns: skill_name -> (action_types, context_patterns)
        SKILL_PATTERNS = {
            "multi_stage_docker_build": {
                "action_types": ["Bash"],
                "context_patterns": ["docker build", "buildah", "podman build", "Dockerfile"],
                "description": "Container build operations"
            },
            "parallel_test_execution": {
                "action_types": ["Bash"],
                "context_patterns": ["pytest", "jest", "cargo test", "npm test", "npm run test"],
                "description": "Test execution"
            },
            "performance_regression_detection": {
                "action_types": ["Bash"],
                "context_patterns": ["hyperfine", "benchmark", "perf ", "flamegraph"],
                "description": "Performance benchmarking"
            },
            "cross_compilation_workflow": {
                "action_types": ["Bash"],
                "context_patterns": ["cross ", "cargo build --target", "make ARCH=", "--sysroot"],
                "description": "Cross-compilation"
            },
            "cicd_pipeline_executor": {
                "action_types": ["Bash"],
                "context_patterns": ["gh workflow", "github actions", "ci/", ".github/workflows"],
                "description": "CI/CD operations"
            },
            "code_review_automation": {
                "action_types": ["Task", "Read", "Grep"],
                "context_patterns": ["review", "pr ", "pull request"],
                "description": "Code review tasks"
            },
            "memory_consolidation": {
                "action_types": ["mcp__enhanced-memory__"],
                "context_patterns": ["consolidat", "memory", "episodic", "semantic"],
                "description": "Memory system operations"
            },
            "cluster_coordination": {
                "action_types": ["mcp__node-chat__", "mcp__claude-flow__"],
                "context_patterns": ["cluster", "node", "swarm", "orchestrat"],
                "description": "Multi-node coordination"
            }
        }

        now = datetime.now()
        start_time = now - timedelta(hours=time_window_hours)

        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            # Get recent actions
            cursor.execute(
                '''
                SELECT
                    action_id,
                    action_type,
                    action_description,
                    action_context,
                    success_score,
                    duration_ms,
                    executed_at
                FROM action_outcomes
                WHERE executed_at >= ?
                ORDER BY executed_at
                ''',
                (start_time.strftime('%Y-%m-%d %H:%M:%S'),)
            )

            actions = cursor.fetchall()
            logger.info(f"Analyzing {len(actions)} actions for skill matching")

            # Track skill executions
            skill_executions = {}  # skill_name -> [(success_score, duration_ms)]

            for action_id, action_type, description, context, score, duration, executed_at in actions:
                # Build searchable text from all fields
                searchable = f"{action_type} {description or ''} {context or ''}".lower()

                # Check each skill pattern
                for skill_name, patterns in SKILL_PATTERNS.items():
                    matched = False

                    # Check action type match
                    for action_pattern in patterns["action_types"]:
                        if action_pattern.lower() in action_type.lower():
                            matched = True
                            break

                    # Check context patterns
                    if not matched:
                        for ctx_pattern in patterns["context_patterns"]:
                            if ctx_pattern.lower() in searchable:
                                matched = True
                                break

                    if matched:
                        if skill_name not in skill_executions:
                            skill_executions[skill_name] = []
                        skill_executions[skill_name].append({
                            "score": score or 0.5,
                            "duration": duration,
                            "time": executed_at
                        })

            # Update procedural_memory for matched skills
            skills_updated = {}
            for skill_name, executions in skill_executions.items():
                # Calculate metrics
                exec_count = len(executions)
                avg_score = sum(e["score"] for e in executions) / exec_count
                durations = [e["duration"] for e in executions if e["duration"]]
                avg_duration = int(sum(durations) / len(durations)) if durations else None
                last_exec = max(e["time"] for e in executions)

                # Update or insert into procedural_memory
                cursor.execute(
                    '''
                    SELECT id, execution_count, success_rate, avg_execution_time_ms
                    FROM procedural_memory
                    WHERE skill_name = ?
                    ''',
                    (skill_name,)
                )
                existing = cursor.fetchone()

                if existing:
                    # Update existing skill
                    old_count = existing[1] or 0
                    old_rate = existing[2] or 0.0
                    old_avg_time = existing[3]

                    new_count = old_count + exec_count
                    # Weighted average for success rate
                    new_rate = ((old_rate * old_count) + (avg_score * exec_count)) / new_count
                    # Weighted average for execution time
                    if avg_duration and old_avg_time:
                        new_avg_time = int(((old_avg_time * old_count) + (avg_duration * exec_count)) / new_count)
                    else:
                        new_avg_time = avg_duration or old_avg_time

                    cursor.execute(
                        '''
                        UPDATE procedural_memory
                        SET execution_count = ?,
                            success_rate = ?,
                            avg_execution_time_ms = ?,
                            last_executed = ?,
                            updated_at = ?
                        WHERE skill_name = ?
                        ''',
                        (new_count, new_rate, new_avg_time, last_exec, now.isoformat(), skill_name)
                    )

                    skills_updated[skill_name] = {
                        "new_executions": exec_count,
                        "total_executions": new_count,
                        "success_rate": round(new_rate, 3),
                        "status": "updated"
                    }
                else:
                    # Create new skill entry
                    cursor.execute(
                        '''
                        INSERT INTO procedural_memory
                        (skill_name, skill_category, procedure_steps, execution_count,
                         success_rate, avg_execution_time_ms, last_executed)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        ''',
                        (
                            skill_name,
                            SKILL_PATTERNS[skill_name]["description"],
                            json.dumps([f"Auto-detected from action patterns"]),
                            exec_count,
                            avg_score,
                            avg_duration,
                            last_exec
                        )
                    )

                    skills_updated[skill_name] = {
                        "new_executions": exec_count,
                        "total_executions": exec_count,
                        "success_rate": round(avg_score, 3),
                        "status": "created"
                    }

            conn.commit()

            logger.info(f"Skill tracking complete: {len(skills_updated)} skills updated")

            return {
                "skills_matched": len(skills_updated),
                "actions_analyzed": len(actions),
                "skills_updated": skills_updated
            }

        except Exception as e:
            logger.error(f"Skill execution tracking failed: {e}")
            conn.rollback()
            raise

        finally:
            conn.close()

    def run_visual_consolidation(
        self,
        time_window_hours: int = 24,
        n_clusters: int = 10
    ) -> Dict[str, Any]:
        """
        Consolidate visual memories using manifold-based clustering.

        This implements LVR-style visual memory compression:
        - Clusters similar visual experiences
        - Promotes recurring visual patterns to semantic memory
        - Links visual clusters to text-based memories

        Args:
            time_window_hours: Hours of visual memories to analyze
            n_clusters: Number of clusters for manifold compression

        Returns:
            {
                "episodes_analyzed": int,
                "clusters_created": int,
                "visual_patterns_promoted": int
            }
        """
        logger.info(f"Running visual consolidation for last {time_window_hours} hours")

        # Import visual memory module
        try:
            import sys
            perception_path = os.path.join(os.environ.get("AGENTIC_SYSTEM_PATH", "${AGENTIC_SYSTEM_PATH:-/opt/agentic}"), "intelligent-agents/perception")
            if perception_path not in sys.path:
                sys.path.insert(0, perception_path)
            from visual_memory import VisualMemory, VISUAL_MEMORY_DB
        except ImportError as e:
            logger.warning(f"Visual memory module not available: {e}")
            return {"error": "Visual memory module not available", "episodes_analyzed": 0}

        job_id = self.schedule_consolidation("visual_consolidation", time_window_hours)

        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            'UPDATE consolidation_jobs SET status = ?, started_at = ? WHERE job_id = ?',
            ('running', datetime.now().isoformat(), job_id)
        )
        conn.commit()

        start_time = datetime.now() - timedelta(hours=time_window_hours)

        try:
            # Connect to visual memory database
            import sqlite3 as sqlite3_mod
            if not VISUAL_MEMORY_DB.exists():
                logger.info("No visual memory database found")
                return {"episodes_analyzed": 0, "clusters_created": 0, "visual_patterns_promoted": 0}

            vm_conn = sqlite3_mod.connect(str(VISUAL_MEMORY_DB))
            vm_cursor = vm_conn.cursor()

            # Get recent visual episodes
            vm_cursor.execute(
                '''
                SELECT id, context, scene_type, significance, created_at
                FROM visual_episodes
                WHERE created_at >= ?
                ''',
                (start_time.strftime('%Y-%m-%d %H:%M:%S'),)
            )

            episodes = vm_cursor.fetchall()
            episodes_count = len(episodes)
            logger.info(f"Found {episodes_count} visual episodes to consolidate")

            if episodes_count == 0:
                vm_conn.close()
                return {"episodes_analyzed": 0, "clusters_created": 0, "visual_patterns_promoted": 0}

            # Perform clustering on visual embeddings
            vm = VisualMemory(use_tpu=False)  # Don't need TPU for clustering
            cluster_result = vm.cluster_visual_memories(n_clusters=min(n_clusters, episodes_count))

            clusters_created = cluster_result.get("clusters_created", 0)
            visual_patterns_promoted = 0

            # Analyze clusters to find recurring visual patterns
            if "cluster_info" in cluster_result:
                for cluster in cluster_result["cluster_info"]:
                    cluster_size = cluster.get("size", 0)
                    cluster_id = cluster.get("cluster_id")

                    # Only promote patterns with multiple instances
                    if cluster_size >= 3:
                        # Get representative context from cluster
                        vm_cursor.execute(
                            '''
                            SELECT context, scene_type
                            FROM visual_episodes
                            WHERE cluster_id = ?
                            LIMIT 5
                            ''',
                            (cluster_id,)
                        )
                        cluster_episodes = vm_cursor.fetchall()

                        # Extract common context elements
                        contexts = [ep[0] for ep in cluster_episodes if ep[0]]
                        scene_types = [ep[1] for ep in cluster_episodes if ep[1]]

                        # Create semantic memory for visual pattern
                        pattern_name = f"visual_pattern_cluster_{cluster_id}_{datetime.now().strftime('%Y%m%d')}"
                        common_scene = max(set(scene_types), key=scene_types.count) if scene_types else "unknown"

                        pattern_description = f"Visual pattern cluster with {cluster_size} similar episodes. " \
                                            f"Common scene type: {common_scene}. " \
                                            f"Sample contexts: {'; '.join(contexts[:3])}"

                        # Score pattern importance
                        importance = score_importance_tpu(pattern_description, "visual_pattern")
                        confidence = min(0.5 + (cluster_size * 0.05), 0.95)

                        # Add to semantic memory
                        cursor.execute(
                            'SELECT id FROM semantic_memory WHERE concept_name = ?',
                            (pattern_name,)
                        )

                        if not cursor.fetchone():
                            cursor.execute(
                                '''
                                INSERT INTO semantic_memory (
                                    concept_name, concept_type, definition,
                                    confidence_score
                                ) VALUES (?, ?, ?, ?)
                                ''',
                                (
                                    pattern_name,
                                    "visual_pattern",
                                    pattern_description,
                                    confidence
                                )
                            )
                            visual_patterns_promoted += 1

            vm_conn.close()

            # Update job
            duration = (datetime.now() - datetime.fromisoformat(
                cursor.execute('SELECT started_at FROM consolidation_jobs WHERE job_id = ?', (job_id,)).fetchone()[0]
            )).seconds

            cursor.execute(
                '''
                UPDATE consolidation_jobs
                SET
                    status = 'completed',
                    completed_at = ?,
                    duration_seconds = ?,
                    patterns_found = ?,
                    memories_promoted = ?,
                    results_summary = ?
                WHERE job_id = ?
                ''',
                (
                    datetime.now().isoformat(),
                    duration,
                    clusters_created,
                    visual_patterns_promoted,
                    json.dumps({
                        "episodes_analyzed": episodes_count,
                        "clusters_created": clusters_created,
                        "visual_patterns_promoted": visual_patterns_promoted
                    }),
                    job_id
                )
            )

            conn.commit()

            logger.info(f"Visual consolidation complete: {episodes_count} episodes, "
                       f"{clusters_created} clusters, {visual_patterns_promoted} patterns promoted")

            return {
                "job_id": job_id,
                "episodes_analyzed": episodes_count,
                "clusters_created": clusters_created,
                "visual_patterns_promoted": visual_patterns_promoted
            }

        except Exception as e:
            cursor.execute(
                '''
                UPDATE consolidation_jobs
                SET status = 'failed', error_message = ?
                WHERE job_id = ?
                ''',
                (str(e), job_id)
            )
            conn.commit()

            logger.error(f"Visual consolidation failed: {e}")
            return {"error": str(e), "episodes_analyzed": 0}

        finally:
            conn.close()
