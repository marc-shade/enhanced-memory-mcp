"""
AGI Memory Enhancement Module

Provides AGI-level memory capabilities:
- Phase 1: Cross-session identity & memory-action loop
- Phase 2: Temporal reasoning & consolidation
- Phase 3: Emotional tagging & associative networks
- Phase 4: Meta-cognitive awareness & self-improvement
- Phase 5: Epistemic flexibility & belief tracking (Stanford Research)
- Phase 6: Adaptive granularity & RL retrieval (Mem0/RMM Research)
"""

from .agent_identity import AgentIdentity, SessionManager
from .action_tracker import ActionTracker
from .temporal_reasoning import TemporalReasoning
from .consolidation import ConsolidationEngine
from .emotional_memory import EmotionalMemory
from .associative_network import AssociativeNetwork, AttentionMechanism, ForgettingCurve
from .metacognition import MetaCognition, PerformanceTracker
from .self_improvement import SelfImprovement, CoordinationManager
from .belief_tracking import BeliefTracker, get_all_agent_beliefs, get_epistemic_flexibility_summary
from .counterfactual_testing import CounterfactualTester, run_flexibility_audit, create_standard_counterfactuals
from .cluster_beliefs import ClusterBeliefManager, get_cluster_belief_summary
from .adaptive_granularity import (
    AdaptiveGranularityManager,
    GranularityLevel,
    GranularityMetrics,
    get_adaptive_granularity_manager
)
from .rl_retrieval import (
    RLRetrievalOptimizer,
    RetrievalFeedback,
    RetrievalQTable,
    get_rl_retrieval_optimizer
)
from .sharpening_engine import (
    SharpeningEngine,
    SharpeningCandidate,
    VerificationResult,
    get_sharpening_engine
)
from .recursive_improvement import (
    RecursiveImprovementEngine,
    MetaStrategy,
    RecursiveImprovementCycle,
    get_recursive_improvement_engine
)
from .agi_orchestrator import (
    AGIOrchestrator,
    CognitiveState,
    CognitiveCycleResult,
    get_agi_orchestrator
)

__all__ = [
    # Phase 1: Identity & Actions
    'AgentIdentity',
    'SessionManager',
    'ActionTracker',
    # Phase 2: Temporal Reasoning
    'TemporalReasoning',
    'ConsolidationEngine',
    # Phase 3: Emotional & Associative
    'EmotionalMemory',
    'AssociativeNetwork',
    'AttentionMechanism',
    'ForgettingCurve',
    # Phase 4: Meta-cognitive
    'MetaCognition',
    'PerformanceTracker',
    'SelfImprovement',
    'CoordinationManager',
    # Phase 5: Epistemic Flexibility
    'BeliefTracker',
    'get_all_agent_beliefs',
    'get_epistemic_flexibility_summary',
    'CounterfactualTester',
    'run_flexibility_audit',
    'create_standard_counterfactuals',
    'ClusterBeliefManager',
    'get_cluster_belief_summary',
    # Phase 6: Adaptive Granularity & RL Retrieval (Mem0/RMM)
    'AdaptiveGranularityManager',
    'GranularityLevel',
    'GranularityMetrics',
    'get_adaptive_granularity_manager',
    'RLRetrievalOptimizer',
    'RetrievalFeedback',
    'RetrievalQTable',
    'get_rl_retrieval_optimizer',
    # Phase 7: Self-Improvement via Sharpening
    'SharpeningEngine',
    'SharpeningCandidate',
    'VerificationResult',
    'get_sharpening_engine',
    # Phase 8: Recursive Self-Improvement
    'RecursiveImprovementEngine',
    'MetaStrategy',
    'RecursiveImprovementCycle',
    'get_recursive_improvement_engine',
    # Stage 6: AGI Orchestrator (Unified Intelligence)
    'AGIOrchestrator',
    'CognitiveState',
    'CognitiveCycleResult',
    'get_agi_orchestrator'
]
