"""
AGI Memory Tools - MCP Server Integration

Exposes Phase 1 AGI capabilities (cross-session identity & memory-action loop) via MCP tools.
"""

import logging
from typing import Dict, List, Any, Optional
from fastmcp import FastMCP

from agi import (
    AgentIdentity, SessionManager, ActionTracker,
    BeliefTracker, get_all_agent_beliefs, get_epistemic_flexibility_summary,
    CounterfactualTester, run_flexibility_audit, create_standard_counterfactuals,
    ClusterBeliefManager, get_cluster_belief_summary
)

logger = logging.getLogger("agi-tools")


def register_agi_tools(app: FastMCP, db_path: str):
    """Register AGI memory tools with FastMCP app"""

    # ============================================================================
    # AGENT IDENTITY TOOLS
    # ============================================================================

    @app.tool()
    def get_agent_identity(agent_id: str = "default_agent") -> Dict[str, Any]:
        """
        Get persistent agent identity

        Returns complete identity including skills, traits, beliefs, and stats.
        """
        identity = AgentIdentity(agent_id)
        return identity.get_identity()

    @app.tool()
    def update_agent_skills(
        skill_updates: Dict[str, float],
        agent_id: str = "default_agent"
    ) -> Dict[str, str]:
        """
        Update agent skill levels (0.0 to 1.0)

        Example: {"coding": 0.85, "research": 0.92, "debugging": 0.78}

        Skills track learned capabilities that improve over time.
        """
        identity = AgentIdentity(agent_id)
        identity.update_skills(skill_updates)
        return {"status": "success", "updated": list(skill_updates.keys())}

    @app.tool()
    def add_agent_belief(
        belief: str,
        agent_id: str = "default_agent"
    ) -> Dict[str, str]:
        """
        Add a core belief/knowledge to agent identity

        Example: "async/await is better than threads for I/O-bound tasks"

        Beliefs are persistent truths the agent has learned.
        """
        identity = AgentIdentity(agent_id)
        identity.add_belief(belief)
        return {"status": "success", "belief": belief}

    @app.tool()
    def update_agent_personality(
        trait_updates: Dict[str, float],
        agent_id: str = "default_agent"
    ) -> Dict[str, str]:
        """
        Update agent personality traits (0.0 to 1.0)

        Example: {"curiosity": 0.8, "caution": 0.6, "creativity": 0.9}

        Personality traits evolve based on experiences and preferences.
        """
        identity = AgentIdentity(agent_id)
        identity.update_personality(trait_updates)
        return {"status": "success", "updated": list(trait_updates.keys())}

    @app.tool()
    def set_agent_preference(
        key: str,
        value: Any,
        agent_id: str = "default_agent"
    ) -> Dict[str, str]:
        """
        Set an agent preference

        Example: set_agent_preference("preferred_editor", "vim")

        Preferences customize agent behavior and choices.
        """
        identity = AgentIdentity(agent_id)
        identity.set_preference(key, value)
        return {"status": "success", "preference": key, "value": value}

    # ============================================================================
    # SESSION MANAGEMENT TOOLS
    # ============================================================================

    @app.tool()
    def start_session(
        context_summary: Optional[str] = None,
        agent_id: str = "default_agent"
    ) -> Dict[str, str]:
        """
        Start a new session with context from previous session

        Returns session_id for tracking work in this session.

        Sessions are automatically linked to maintain continuity.
        """
        manager = SessionManager(agent_id)
        session_id = manager.start_session(context_summary)
        return {
            "status": "success",
            "session_id": session_id,
            "agent_id": agent_id
        }

    @app.tool()
    def end_session(
        session_id: str,
        key_learnings: Optional[List[str]] = None,
        unfinished_work: Optional[Dict[str, Any]] = None,
        performance_metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        End a session and record outcomes

        Args:
            session_id: Session to end
            key_learnings: What was learned this session
            unfinished_work: Tasks/goals not completed
            performance_metrics: Success rates, error counts, etc.
        """
        manager = SessionManager()
        manager.end_session(session_id, key_learnings, unfinished_work, performance_metrics)
        return {"status": "success", "session_id": session_id}

    @app.tool()
    def get_session_context(session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get complete context for a session

        Includes learnings, unfinished work, and performance metrics.
        """
        manager = SessionManager()
        return manager.get_session_context(session_id)

    @app.tool()
    def get_recent_sessions(
        limit: int = 10,
        agent_id: str = "default_agent"
    ) -> List[Dict[str, Any]]:
        """
        Get recent sessions for continuity

        Returns sessions from most recent to oldest with all context.
        """
        manager = SessionManager(agent_id)
        return manager.get_recent_sessions(limit)

    @app.tool()
    def get_session_chain(
        session_id: str,
        depth: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get chain of linked sessions going backwards

        Shows session history and how work evolved across sessions.
        """
        manager = SessionManager()
        return manager.get_session_chain(session_id, depth)

    # ============================================================================
    # ACTION OUTCOME TRACKING TOOLS
    # ============================================================================

    @app.tool()
    def record_action_outcome(
        action_type: str,
        action_description: str,
        expected_result: str,
        actual_result: str,
        success_score: float,
        agent_id: str = "default_agent",
        session_id: Optional[str] = None,
        entity_id: Optional[int] = None,
        action_context: Optional[str] = None,
        duration_ms: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Record an action and its outcome for learning

        Args:
            action_type: Type of action ("code_change", "command", "research", etc.)
            action_description: What was done
            expected_result: What we expected
            actual_result: What actually happened
            success_score: 0.0 (failure) to 1.0 (success)
            agent_id: Agent ID recording this action (default: "default_agent")
            session_id: Optional session ID
            entity_id: Optional memory entity ID
            action_context: Why this action was taken
            duration_ms: How long it took
            metadata: Additional data

        Returns:
            action_id for reference
        """
        tracker = ActionTracker(agent_id)
        action_id = tracker.record_action(
            action_type, action_description,
            expected_result, actual_result,
            success_score, session_id, entity_id,
            action_context, duration_ms, metadata
        )
        return {
            "status": "success",
            "action_id": action_id,
            "agent_id": agent_id,
            "success_score": success_score
        }

    @app.tool()
    def get_similar_actions(
        action_type: str,
        context: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find similar past actions to learn from

        Returns actions with outcomes, sorted by recency.

        Use this before taking action to see what worked before.
        """
        tracker = ActionTracker()
        return tracker.get_similar_actions(action_type, context, limit)

    @app.tool()
    def get_action_success_rate(
        action_type: str,
        time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Get success rate for an action type

        Returns:
            {
                "action_type": str,
                "total_actions": int,
                "success_count": int,
                "success_rate": float,
                "avg_score": float,
                "time_window_hours": int
            }

        Use this to know if you're improving at specific actions.
        """
        tracker = ActionTracker()
        return tracker.get_success_rate(action_type, time_window_hours)

    @app.tool()
    def get_learnings_for_action(
        action_type: str,
        limit: int = 5
    ) -> List[str]:
        """
        Get key learnings from past actions

        Returns lessons learned from successes and failures.

        Use this to avoid repeating mistakes.
        """
        tracker = ActionTracker()
        return tracker.get_learnings_for_action(action_type, limit)

    @app.tool()
    def should_retry_action(
        original_action_id: int,
        proposed_changes: str
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

        Use this to decide if failure warrants a retry with modifications.
        """
        tracker = ActionTracker()
        return tracker.should_retry_action(original_action_id, proposed_changes)

    @app.tool()
    def get_action_statistics() -> Dict[str, Any]:
        """
        Get overall action outcome statistics

        Returns comprehensive stats on all actions, success rates,
        trends, and performance over time.

        Use this to monitor agent learning progress.
        """
        tracker = ActionTracker()
        return tracker.get_action_statistics()

    logger.info("✅ AGI Memory tools registered (Phase 1: Identity & Actions)")

    # ============================================================================
    # PHASE 5: EPISTEMIC FLEXIBILITY TOOLS (Stanford Research)
    # ============================================================================

    @app.tool()
    def record_belief_state(
        belief_statement: str,
        probability: float,
        belief_category: str = "fact",
        confidence: float = 0.5,
        supporting_evidence: Optional[List[Dict]] = None,
        contradicting_evidence: Optional[List[Dict]] = None,
        layer_context: Optional[Dict[str, str]] = None,
        agent_id: str = "default_agent",
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Record an agent's belief state with probability and evidence.

        Stanford Research: Tracks beliefs with probability (0.0-1.0) and
        four-layer contextual instantiation for debugging.

        Args:
            belief_statement: What is believed
            probability: Belief strength 0.0-1.0
            belief_category: "identity", "fact", "strategy", "goal"
            confidence: Confidence in probability estimate
            supporting_evidence: List of supporting evidence [{source, weight}]
            contradicting_evidence: List of contradicting evidence
            layer_context: Four-layer context:
                - instruction: Agent identity/role influence
                - conversation: Conversation history influence
                - observations: Tool results/observations
                - constraints: System prompt constraints

        Returns:
            belief_id and status
        """
        tracker = BeliefTracker(agent_id)
        belief_id = tracker.record_belief(
            belief_statement=belief_statement,
            probability=probability,
            belief_category=belief_category,
            confidence=confidence,
            supporting_evidence=supporting_evidence,
            contradicting_evidence=contradicting_evidence,
            layer_context=layer_context,
            session_id=session_id
        )
        return {
            "status": "success",
            "belief_id": belief_id,
            "agent_id": agent_id,
            "probability": probability,
            "category": belief_category
        }

    @app.tool()
    def update_belief_probability(
        belief_id: int,
        new_probability: float,
        revision_trigger: str,
        evidence_provided: str,
        reasoning: Optional[str] = None,
        session_id: Optional[str] = None,
        agent_id: str = "default_agent"
    ) -> Dict[str, Any]:
        """
        Update belief probability based on new evidence.

        Records revision in history for tracking belief flexibility.

        Args:
            belief_id: Belief to update
            new_probability: New probability value (0.0-1.0)
            revision_trigger: "new_evidence", "contradiction", "counterfactual"
            evidence_provided: Description of evidence causing revision
            reasoning: Agent's explanation for revision

        Returns:
            Revision details including probability_delta
        """
        tracker = BeliefTracker(agent_id)
        return tracker.update_probability(
            belief_id=belief_id,
            new_probability=new_probability,
            revision_trigger=revision_trigger,
            evidence_provided=evidence_provided,
            reasoning=reasoning,
            session_id=session_id
        )

    @app.tool()
    def get_agent_beliefs(
        agent_id: str = "default_agent",
        category: Optional[str] = None,
        min_probability: float = 0.0,
        include_evidence: bool = True,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Retrieve agent's current belief states.

        Returns list of beliefs with probabilities and evidence.
        """
        tracker = BeliefTracker(agent_id)
        return tracker.get_beliefs(
            category=category,
            min_probability=min_probability,
            include_evidence=include_evidence,
            limit=limit
        )

    @app.tool()
    def get_belief_revision_history(
        belief_id: int,
        limit: int = 20,
        agent_id: str = "default_agent"
    ) -> List[Dict[str, Any]]:
        """
        Get revision history for a belief.

        Shows how belief probability changed over time.
        """
        tracker = BeliefTracker(agent_id)
        return tracker.get_revision_history(belief_id, limit)

    @app.tool()
    def analyze_belief_rigidity(
        belief_id: Optional[int] = None,
        time_window_hours: int = 24,
        agent_id: str = "default_agent"
    ) -> Dict[str, Any]:
        """
        Analyze how rigid or flexible an agent's beliefs are.

        Returns rigidity metrics including:
        - Average revision magnitude
        - Revision frequency
        - Evidence sensitivity
        - Flexibility score (0.0=rigid, 1.0=flexible)

        High rigidity = potential narrative overfitting risk.
        """
        tracker = BeliefTracker(agent_id)
        return tracker.analyze_rigidity(belief_id, time_window_hours)

    @app.tool()
    def create_counterfactual_scenario(
        scenario_name: str,
        scenario_description: str,
        target_belief_id: int,
        original_facts: Dict[str, Any],
        counterfactual_facts: Dict[str, Any],
        expected_revision: Optional[float] = None,
        agent_id: str = "default_agent"
    ) -> Dict[str, Any]:
        """
        Create a counterfactual scenario to test belief flexibility.

        Stanford Research: Tests if agent can revise beliefs when
        presented with alternative evidence.

        Args:
            scenario_name: Short name for scenario
            scenario_description: What the scenario tests
            target_belief_id: Belief to test
            original_facts: Baseline facts
            counterfactual_facts: Alternative facts that contradict belief
            expected_revision: Expected probability change (optional)

        Returns:
            scenario_id for execution
        """
        tester = CounterfactualTester(agent_id)
        scenario_id = tester.create_scenario(
            scenario_name=scenario_name,
            scenario_description=scenario_description,
            target_belief_id=target_belief_id,
            original_facts=original_facts,
            counterfactual_facts=counterfactual_facts,
            expected_revision=expected_revision
        )
        return {
            "status": "success",
            "scenario_id": scenario_id,
            "message": f"Counterfactual scenario '{scenario_name}' created"
        }

    @app.tool()
    def execute_counterfactual_test(
        scenario_id: int,
        new_belief_probability: float,
        agent_reasoning: Optional[str] = None,
        agent_id: str = "default_agent"
    ) -> Dict[str, Any]:
        """
        Execute counterfactual test and measure agent response.

        After agent considers counterfactual evidence, provide:
        - new_belief_probability: Revised probability after counterfactual
        - agent_reasoning: Explanation for revision (or lack thereof)

        Returns:
            Test results including flexibility_score
        """
        tester = CounterfactualTester(agent_id)
        return tester.execute_test(
            scenario_id=scenario_id,
            new_belief_probability=new_belief_probability,
            agent_reasoning=agent_reasoning
        )

    @app.tool()
    def get_epistemic_flexibility_score(
        include_historical: bool = True,
        time_window_hours: int = 168,
        agent_id: str = "default_agent"
    ) -> Dict[str, Any]:
        """
        Calculate overall epistemic flexibility for an agent.

        Composite score based on:
        - Belief revision frequency
        - Average revision magnitude
        - Response to counterfactual evidence
        - Balance of evidence consideration

        Returns:
            Flexibility score (0.0 = rigid, 1.0 = flexible) with breakdown
        """
        tester = CounterfactualTester(agent_id)
        return tester.get_epistemic_flexibility_score(
            include_historical=include_historical,
            time_window_hours=time_window_hours
        )

    @app.tool()
    def run_epistemic_flexibility_audit(
        agent_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run epistemic flexibility audit across agents.

        Returns audit report with per-agent scores and recommendations.
        Identifies agents at risk of narrative overfitting.
        """
        return run_flexibility_audit(agent_ids)

    @app.tool()
    def create_cluster_belief_block(
        belief_domain: str,
        initial_beliefs: Dict[str, float],
        description: Optional[str] = None,
        consensus_threshold: float = 0.6,
        cluster_id: str = "default_cluster"
    ) -> Dict[str, Any]:
        """
        Create shared memory block for cluster-wide beliefs.

        All agents in cluster can read and propose updates.
        Prevents echo chambers by making divergent beliefs visible.

        Args:
            belief_domain: Domain category (goals, strategies, constraints, facts)
            initial_beliefs: Dict of {statement: probability}
            description: Description of this belief block
            consensus_threshold: Required agreement level for updates (0.5-1.0)

        Returns:
            block_id
        """
        manager = ClusterBeliefManager(cluster_id)
        block_id = manager.create_belief_block(
            belief_domain=belief_domain,
            initial_beliefs=initial_beliefs,
            description=description,
            consensus_threshold=consensus_threshold
        )
        return {
            "status": "success",
            "block_id": block_id,
            "cluster_id": cluster_id,
            "domain": belief_domain
        }

    @app.tool()
    def propose_cluster_belief_update(
        block_id: int,
        agent_id: str,
        belief_statement: str,
        proposed_probability: float,
        evidence: List[Dict[str, Any]],
        reasoning: Optional[str] = None,
        cluster_id: str = "default_cluster"
    ) -> Dict[str, Any]:
        """
        Propose updating a cluster belief.

        Updates require consensus based on threshold set for block.

        Args:
            block_id: Belief block to update
            agent_id: Agent proposing update
            belief_statement: Belief to update
            proposed_probability: New probability value
            evidence: Supporting evidence for change
            reasoning: Agent's explanation

        Returns:
            Proposal status and whether it was applied
        """
        manager = ClusterBeliefManager(cluster_id)
        return manager.propose_belief_update(
            block_id=block_id,
            agent_id=agent_id,
            belief_statement=belief_statement,
            proposed_probability=proposed_probability,
            evidence=evidence,
            reasoning=reasoning
        )

    @app.tool()
    def detect_cluster_belief_divergence(
        threshold: float = 0.3,
        cluster_id: str = "default_cluster"
    ) -> List[Dict[str, Any]]:
        """
        Detect when agents hold significantly different beliefs.

        Compares agent personal beliefs against cluster shared beliefs.

        Args:
            threshold: Minimum probability difference to flag as divergent

        Returns:
            List of divergent beliefs requiring moderator attention
        """
        manager = ClusterBeliefManager(cluster_id)
        return manager.detect_belief_divergence(threshold)

    @app.tool()
    def check_cluster_belief_consistency(
        belief_category: Optional[str] = None,
        cluster_id: str = "default_cluster"
    ) -> Dict[str, Any]:
        """
        Check if agents in cluster have consistent beliefs.

        Returns:
            - Consistent beliefs (all agents agree)
            - Divergent beliefs (agents disagree)
            - Unshared beliefs (only one agent holds)
            - Contradiction pairs (mutually exclusive beliefs)
        """
        manager = ClusterBeliefManager(cluster_id)
        return manager.check_belief_consistency(belief_category)

    @app.tool()
    def reconcile_cluster_belief_conflict(
        belief_statements: List[str],
        strategy: str = "evidence_weighted",
        cluster_id: str = "default_cluster"
    ) -> Dict[str, Any]:
        """
        Attempt to reconcile conflicting beliefs using:
        - evidence_weighted: Weight by evidence strength
        - probability_fusion: Bayesian fusion of probabilities
        - majority_voting: Simple majority wins
        - dempster_shafer: Dempster-Shafer combination rule

        Returns:
            Reconciled belief with confidence
        """
        manager = ClusterBeliefManager(cluster_id)
        return manager.reconcile_belief_conflict(belief_statements, strategy)

    @app.tool()
    def get_cluster_belief_summary_report(
        cluster_id: str = "default_cluster"
    ) -> Dict[str, Any]:
        """
        Get summary of cluster belief state.

        Returns:
            - Belief block count
            - Unresolved divergences
            - Consistency score
            - Overall health status
        """
        return get_cluster_belief_summary(cluster_id)

    logger.info("✅ Epistemic Flexibility tools registered (Phase 5: Stanford Research)")
