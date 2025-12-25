#!/usr/bin/env python3
"""
ModelRouter - Multi-Provider LLM Routing System

Comprehensive routing system ported from ruvnet/agentic-flow TypeScript implementation.
Provides intelligent routing across multiple LLM providers with:

1. Multiple Providers: Anthropic, OpenAI, OpenRouter, Ollama, Exo, Gemini, LiteLLM
2. 4 Routing Modes: manual, rule-based, cost-optimized, performance-optimized
3. Provider Fallback Chains: Automatic failover on errors
4. Metrics Tracking: Per-provider and per-agent-type statistics
5. Streaming Support: Async generators for streaming responses
6. Integration: MCP tools for enhanced-memory integration

Also includes legacy IntelligentModelRouter for backward compatibility.

https://github.com/ruvnet/agentic-flow
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    Any, AsyncGenerator, Callable, Dict, List, Optional,
    Tuple, Union, Literal
)
from datetime import datetime, timedelta

try:
    import httpx
except ImportError:
    httpx = None  # Will raise at runtime if needed

logger = logging.getLogger("model_router")


class IntelligentModelRouter:
    """
    Routes AI tasks to optimal models based on complexity and requirements.

    Supports both local Ollama models and cloud-based Ollama models.
    Tracks performance statistics for continuous optimization.
    """

    def __init__(self, stats_file: Optional[str] = None):
        """
        Initialize the intelligent model router.

        Args:
            stats_file: Path to JSON file for tracking routing statistics
        """
        self.stats_file = Path(stats_file) if stats_file else None
        self.ollama_base_url = "http://localhost:11434"

        # Model configuration
        self.models = {
            "local_reasoning": "deepseek-r1:32b-qwen-distill-fp16",
            "local_powerful": "gpt-oss:120b",
            "cloud_powerful": "gpt-oss:20b-cloud"
        }

        # Routing thresholds
        self.complexity_thresholds = {
            "simple": 40,      # < 40: basic local model
            "moderate": 70,    # 40-70: powerful local model
            "complex": 100     # > 70: cloud or reasoning model
        }

        # Initialize stats
        self.stats = self._load_stats()

        logger.info(f"IntelligentModelRouter initialized with {len(self.models)} models")

    def _load_stats(self) -> Dict[str, Any]:
        """Load routing statistics from file."""
        if self.stats_file and self.stats_file.exists():
            try:
                with open(self.stats_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load stats from {self.stats_file}: {e}")

        return {
            "total_requests": 0,
            "local_requests": 0,
            "cloud_requests": 0,
            "total_duration": 0.0,
            "model_usage": {},
            "complexity_distribution": {
                "simple": 0,
                "moderate": 0,
                "complex": 0
            }
        }

    def _save_stats(self):
        """Save routing statistics to file."""
        if self.stats_file:
            try:
                self.stats_file.parent.mkdir(parents=True, exist_ok=True)
                with open(self.stats_file, 'w') as f:
                    json.dump(self.stats, f, indent=2)
            except Exception as e:
                logger.error(f"Failed to save stats to {self.stats_file}: {e}")

    def _select_model(self, context: Dict[str, Any]) -> tuple[str, str]:
        """
        Select the optimal model based on task context.

        Args:
            context: Task context with complexity, requires_reasoning, etc.

        Returns:
            Tuple of (model_name, location) where location is 'local' or 'cloud'
        """
        complexity = context.get("complexity", 50)
        requires_reasoning = context.get("requires_reasoning", False)
        multi_step = context.get("multi_step", False)

        # Reasoning tasks always use reasoning model
        if requires_reasoning or multi_step:
            logger.info(f"Routing to reasoning model (reasoning={requires_reasoning}, multi_step={multi_step})")
            return self.models["local_reasoning"], "local"

        # Route based on complexity
        if complexity < self.complexity_thresholds["simple"]:
            # Simple tasks - use local powerful model
            return self.models["local_powerful"], "local"
        elif complexity < self.complexity_thresholds["moderate"]:
            # Moderate tasks - use local powerful model
            return self.models["local_powerful"], "local"
        else:
            # Complex tasks - prefer cloud for best results
            logger.info(f"High complexity ({complexity}) - routing to cloud model")
            return self.models["cloud_powerful"], "cloud"

    async def _execute_ollama_request(
        self,
        model: str,
        prompt: str,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute request to Ollama API.

        Args:
            model: Model name to use
            prompt: Prompt text
            options: Optional model parameters

        Returns:
            Response dictionary from Ollama
        """
        url = f"{self.ollama_base_url}/api/generate"

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }

        if options:
            payload["options"] = options

        async with httpx.AsyncClient(timeout=300.0) as client:
            try:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                logger.error(f"Ollama API error for model {model}: {e}")
                raise

    async def execute_with_routing(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute task with intelligent model routing.

        Args:
            task: Task description/prompt
            context: Task context (complexity, requires_reasoning, etc.)
            **kwargs: Additional parameters passed to model

        Returns:
            Dictionary containing:
                - response: Model's response text
                - model_used: Name of model that processed the task
                - location: 'local' or 'cloud'
                - duration_seconds: Processing time
                - complexity: Task complexity level
        """
        start_time = time.time()
        context = context or {}

        # Select optimal model
        model_name, location = self._select_model(context)

        logger.info(f"Executing task with {model_name} ({location})")

        try:
            # Execute the task
            result = await self._execute_ollama_request(
                model=model_name,
                prompt=task,
                options=kwargs.get("options")
            )

            duration = time.time() - start_time

            # Update statistics
            self._update_stats(
                model=model_name,
                location=location,
                complexity=context.get("complexity", 50),
                duration=duration
            )

            return {
                "response": result.get("response", ""),
                "model_used": model_name,
                "location": location,
                "duration_seconds": duration,
                "complexity": context.get("complexity", 50),
                "tokens_generated": result.get("eval_count", 0),
                "tokens_prompt": result.get("prompt_eval_count", 0)
            }

        except Exception as e:
            logger.error(f"Error executing task with {model_name}: {e}")
            # Return error response
            return {
                "response": f"Error: Failed to process task - {str(e)}",
                "model_used": model_name,
                "location": location,
                "duration_seconds": time.time() - start_time,
                "complexity": context.get("complexity", 50),
                "error": str(e)
            }

    def _update_stats(
        self,
        model: str,
        location: str,
        complexity: int,
        duration: float
    ):
        """Update routing statistics."""
        self.stats["total_requests"] += 1
        self.stats["total_duration"] += duration

        if location == "local":
            self.stats["local_requests"] += 1
        else:
            self.stats["cloud_requests"] += 1

        # Track model usage
        if model not in self.stats["model_usage"]:
            self.stats["model_usage"][model] = {
                "count": 0,
                "total_duration": 0.0,
                "avg_duration": 0.0
            }

        model_stats = self.stats["model_usage"][model]
        model_stats["count"] += 1
        model_stats["total_duration"] += duration
        model_stats["avg_duration"] = model_stats["total_duration"] / model_stats["count"]

        # Track complexity distribution
        if complexity < self.complexity_thresholds["simple"]:
            self.stats["complexity_distribution"]["simple"] += 1
        elif complexity < self.complexity_thresholds["moderate"]:
            self.stats["complexity_distribution"]["moderate"] += 1
        else:
            self.stats["complexity_distribution"]["complex"] += 1

        # Save updated stats
        self._save_stats()

    def get_stats(self) -> Dict[str, Any]:
        """Get current routing statistics."""
        return self.stats.copy()

    async def health_check(self) -> Dict[str, Any]:
        """
        Check health of available models.

        Returns:
            Dictionary with model availability status
        """
        health = {
            "ollama_available": False,
            "models_available": {},
            "timestamp": time.time()
        }

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                # Check Ollama availability
                response = await client.get(f"{self.ollama_base_url}/api/tags")
                response.raise_for_status()
                health["ollama_available"] = True

                # Check each configured model
                available_models = response.json().get("models", [])
                available_names = [m["name"] for m in available_models]

                for model_type, model_name in self.models.items():
                    health["models_available"][model_type] = {
                        "name": model_name,
                        "available": model_name in available_names
                    }

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            health["error"] = str(e)

        return health


# =============================================================================
# Type Definitions (from ruvnet/agentic-flow types.ts)
# =============================================================================

class ProviderType(str, Enum):
    """Supported LLM provider types."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OPENROUTER = "openrouter"
    OLLAMA = "ollama"
    LITELLM = "litellm"
    GEMINI = "gemini"
    ONNX = "onnx"
    EXO = "exo"  # Distributed inference cluster
    CUSTOM = "custom"


class RoutingMode(str, Enum):
    """Available routing strategies."""
    MANUAL = "manual"
    RULE_BASED = "rule-based"
    COST_OPTIMIZED = "cost-optimized"
    PERFORMANCE_OPTIMIZED = "performance-optimized"
    QUALITY_OPTIMIZED = "quality-optimized"
    MEMORY_INFLUENCED = "memory-influenced"  # Phase 2: Holographic memory routing


class StopReason(str, Enum):
    """Reasons for completion."""
    END_TURN = "end_turn"
    MAX_TOKENS = "max_tokens"
    TOOL_USE = "tool_use"
    STOP_SEQUENCE = "stop_sequence"


@dataclass
class ContentBlock:
    """Content block in a message."""
    type: Literal["text", "tool_use", "tool_result"]
    text: Optional[str] = None
    id: Optional[str] = None
    name: Optional[str] = None
    input: Optional[Any] = None
    content: Optional[Any] = None
    is_error: bool = False


@dataclass
class Message:
    """Chat message."""
    role: Literal["user", "assistant", "system"]
    content: Union[str, List[ContentBlock]]


@dataclass
class Tool:
    """Tool definition."""
    name: str
    description: str
    input_schema: Dict[str, Any]


@dataclass
class ChatParams:
    """Parameters for chat completion."""
    model: str
    messages: List[Message]
    temperature: float = 0.7
    max_tokens: int = 4096
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    stream: bool = False
    metadata: Optional[Dict[str, Any]] = None
    provider: Optional[str] = None  # Force specific provider


@dataclass
class UsageStats:
    """Token usage statistics."""
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class ResponseMetadata:
    """Response metadata."""
    provider: str
    model: Optional[str] = None
    cost: float = 0.0
    latency: float = 0.0
    execution_providers: Optional[List[str]] = None


@dataclass
class ChatResponse:
    """Chat completion response."""
    id: str
    model: str
    content: List[ContentBlock]
    stop_reason: Optional[StopReason] = None
    usage: Optional[UsageStats] = None
    metadata: Optional[ResponseMetadata] = None


@dataclass
class StreamChunk:
    """Streaming response chunk."""
    type: str
    delta: Optional[Dict[str, Any]] = None
    content_block: Optional[ContentBlock] = None
    message: Optional[Dict[str, Any]] = None
    usage: Optional[UsageStats] = None


@dataclass
class ProviderConfig:
    """Provider configuration."""
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    organization: Optional[str] = None
    models: Optional[Dict[str, str]] = None  # default, fast, advanced
    timeout: int = 120
    max_retries: int = 3
    retry_delay: float = 1.0
    rate_limit: Optional[Dict[str, int]] = None
    preferences: Optional[Dict[str, Any]] = None
    # ONNX/Local specific
    model_path: Optional[str] = None
    execution_providers: Optional[List[str]] = None
    local_inference: bool = False
    gpu_acceleration: bool = False


@dataclass
class RoutingRule:
    """Routing rule definition."""
    condition: Dict[str, Any]  # agent_type, requires_tools, complexity, privacy
    action: Dict[str, Any]     # provider, model, temperature, max_tokens
    reason: Optional[str] = None


@dataclass
class RoutingConfig:
    """Routing configuration."""
    mode: RoutingMode = RoutingMode.MANUAL
    rules: List[RoutingRule] = field(default_factory=list)
    cost_optimization: Optional[Dict[str, Any]] = None
    performance: Optional[Dict[str, Any]] = None


@dataclass
class MonitoringConfig:
    """Monitoring configuration."""
    enabled: bool = True
    log_level: str = "info"
    track_cost: bool = True
    track_latency: bool = True
    track_tokens: bool = True
    track_errors: bool = True
    alerts: Optional[Dict[str, float]] = None


@dataclass
class CacheConfig:
    """Cache configuration."""
    enabled: bool = False
    ttl: int = 3600
    max_size: int = 1000
    strategy: str = "lru"


# =============================================================================
# Uncertainty Estimation (Ported from ruvector/tiny-dancer-core)
# =============================================================================

@dataclass
class UncertaintyConfig:
    """Configuration for uncertainty estimation.

    Ported from ruvector/crates/ruvector-tiny-dancer-core/src/uncertainty.rs
    Uses conformal prediction concepts for reliable uncertainty quantification.
    """
    calibration_quantile: float = 0.9  # 90% confidence default
    min_samples_for_calibration: int = 30
    boundary_threshold: float = 0.5  # Decision boundary
    enable_calibration: bool = True


class UncertaintyEstimator:
    """
    Uncertainty estimator for routing decisions.

    Ported from ruvector/crates/ruvector-tiny-dancer-core/src/uncertainty.rs

    Uses conformal prediction concepts:
    - Boundary distance uncertainty: Higher uncertainty near decision boundary
    - Calibration from historical predictions vs outcomes
    - Statistical guarantees via conformal prediction

    Key insight: Uncertainty = 1.0 - 2 * |prediction - boundary|
    - At boundary (0.5): uncertainty = 1.0 (maximum)
    - At extremes (0 or 1): uncertainty = 0.0 (minimum)
    """

    def __init__(self, config: Optional[UncertaintyConfig] = None):
        self.config = config or UncertaintyConfig()
        self.calibration_scores: List[float] = []
        self.prediction_history: List[Tuple[float, bool]] = []  # (prediction, was_correct)
        self._calibration_threshold: Optional[float] = None

    def estimate(self, features: Optional[List[float]], prediction: float) -> float:
        """
        Estimate uncertainty for a prediction.

        Uses boundary distance heuristic:
        - Predictions near 0.5 (decision boundary) have high uncertainty
        - Predictions near 0 or 1 have low uncertainty

        Args:
            features: Input features (reserved for future feature-based uncertainty)
            prediction: Model prediction or confidence score [0, 1]

        Returns:
            Uncertainty score [0, 1] where 1 = maximum uncertainty
        """
        # Distance from decision boundary (0.5)
        boundary_distance = abs(prediction - self.config.boundary_threshold)

        # Higher uncertainty when close to boundary
        # uncertainty = 1 - 2*distance maps:
        #   distance=0 (at boundary) -> uncertainty=1
        #   distance=0.5 (at extremes) -> uncertainty=0
        boundary_uncertainty = 1.0 - (boundary_distance * 2.0)

        # Clip to [0, 1]
        return max(0.0, min(1.0, boundary_uncertainty))

    def calibrate(self, predictions: List[float], outcomes: List[bool]) -> float:
        """
        Calibrate the estimator using historical predictions and outcomes.

        Implements conformal prediction calibration:
        1. Compute non-conformity scores for each (prediction, outcome) pair
        2. Find the quantile threshold that achieves desired coverage

        Args:
            predictions: Historical prediction scores [0, 1]
            outcomes: Actual outcomes (True = success, False = failure)

        Returns:
            Calibration score (1.0 = perfectly calibrated)
        """
        if len(predictions) < self.config.min_samples_for_calibration:
            return 0.5  # Not enough data, return neutral score

        if len(predictions) != len(outcomes):
            raise ValueError("predictions and outcomes must have same length")

        # Compute non-conformity scores
        # For each prediction, score = |prediction - actual_outcome|
        nonconformity_scores = []
        for pred, outcome in zip(predictions, outcomes):
            actual = 1.0 if outcome else 0.0
            score = abs(pred - actual)
            nonconformity_scores.append(score)

        # Store for calibration
        self.calibration_scores = nonconformity_scores

        # Compute calibration threshold (quantile of non-conformity scores)
        sorted_scores = sorted(nonconformity_scores)
        quantile_idx = int(len(sorted_scores) * self.config.calibration_quantile)
        quantile_idx = min(quantile_idx, len(sorted_scores) - 1)
        self._calibration_threshold = sorted_scores[quantile_idx]

        # Update prediction history
        self.prediction_history.extend(zip(predictions, outcomes))

        # Return calibration quality (1 - mean non-conformity)
        mean_nonconformity = sum(nonconformity_scores) / len(nonconformity_scores)
        return 1.0 - mean_nonconformity

    def record_outcome(self, prediction: float, was_correct: bool):
        """Record a prediction outcome for future calibration."""
        self.prediction_history.append((prediction, was_correct))

        # Auto-recalibrate if we have enough new samples
        if len(self.prediction_history) >= self.config.min_samples_for_calibration:
            if len(self.prediction_history) % self.config.min_samples_for_calibration == 0:
                preds = [p for p, _ in self.prediction_history[-100:]]
                outcomes = [o for _, o in self.prediction_history[-100:]]
                self.calibrate(preds, outcomes)

    def get_calibrated_uncertainty(self, prediction: float) -> Tuple[float, float]:
        """
        Get uncertainty with calibration adjustment.

        Returns:
            Tuple of (raw_uncertainty, calibrated_uncertainty)
        """
        raw = self.estimate(None, prediction)

        if self._calibration_threshold is None:
            return (raw, raw)

        # Adjust uncertainty based on calibration
        # If calibration shows model is overconfident, increase uncertainty
        calibration_factor = self._calibration_threshold / 0.5  # 0.5 = neutral
        calibrated = raw * calibration_factor
        calibrated = max(0.0, min(1.0, calibrated))

        return (raw, calibrated)

    def get_statistics(self) -> Dict[str, Any]:
        """Get uncertainty estimation statistics."""
        recent_preds = self.prediction_history[-100:] if self.prediction_history else []

        if recent_preds:
            recent_accuracy = sum(1 for _, o in recent_preds if o) / len(recent_preds)
            avg_uncertainty = sum(self.estimate(None, p) for p, _ in recent_preds) / len(recent_preds)
        else:
            recent_accuracy = 0.0
            avg_uncertainty = 0.5

        return {
            "calibration_quantile": self.config.calibration_quantile,
            "calibration_threshold": self._calibration_threshold,
            "total_predictions_tracked": len(self.prediction_history),
            "recent_accuracy": round(recent_accuracy, 4),
            "average_uncertainty": round(avg_uncertainty, 4),
            "is_calibrated": self._calibration_threshold is not None,
            "calibration_samples": len(self.calibration_scores)
        }

    def reset(self):
        """Reset calibration state."""
        self.calibration_scores.clear()
        self.prediction_history.clear()
        self._calibration_threshold = None


@dataclass
class RouterConfig:
    """Full router configuration."""
    version: str = "1.0.0"
    default_provider: ProviderType = ProviderType.ANTHROPIC
    fallback_chain: List[ProviderType] = field(default_factory=list)
    providers: Dict[str, ProviderConfig] = field(default_factory=dict)
    routing: Optional[RoutingConfig] = None
    monitoring: Optional[MonitoringConfig] = None
    cache: Optional[CacheConfig] = None


@dataclass
class ProviderMetrics:
    """Metrics for a single provider."""
    requests: int = 0
    cost: float = 0.0
    avg_latency: float = 0.0
    errors: int = 0
    total_latency: float = 0.0  # For calculating average


@dataclass
class RouterMetrics:
    """Overall router metrics."""
    total_requests: int = 0
    total_cost: float = 0.0
    total_tokens: UsageStats = field(default_factory=UsageStats)
    provider_breakdown: Dict[str, ProviderMetrics] = field(default_factory=dict)
    agent_breakdown: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class ProviderError(Exception):
    """Provider-specific error."""
    def __init__(
        self,
        message: str,
        provider: str,
        status_code: Optional[int] = None,
        retryable: bool = False
    ):
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.retryable = retryable


# =============================================================================
# Model Mapping (from model-mapping.ts)
# =============================================================================

CLAUDE_MODELS = {
    "claude-sonnet-4.5": {
        "anthropic": "claude-sonnet-4-5-20250929",
        "openrouter": "anthropic/claude-sonnet-4.5",
        "bedrock": "anthropic.claude-sonnet-4-5-v2:0",
        "canonical": "Claude Sonnet 4.5"
    },
    "claude-opus-4.5": {
        "anthropic": "claude-opus-4-5-20251101",
        "openrouter": "anthropic/claude-opus-4.5",
        "canonical": "Claude Opus 4.5"
    },
    "claude-3.5-sonnet": {
        "anthropic": "claude-3-5-sonnet-20241022",
        "openrouter": "anthropic/claude-3.5-sonnet-20241022",
        "bedrock": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "canonical": "Claude 3.5 Sonnet"
    },
    "claude-3.5-haiku": {
        "anthropic": "claude-3-5-haiku-20241022",
        "openrouter": "anthropic/claude-3.5-haiku-20241022",
        "canonical": "Claude 3.5 Haiku"
    }
}

OPENAI_MODELS = {
    "gpt-4o": {
        "openai": "gpt-4o",
        "openrouter": "openai/gpt-4o",
        "canonical": "GPT-4o"
    },
    "gpt-4o-mini": {
        "openai": "gpt-4o-mini",
        "openrouter": "openai/gpt-4o-mini",
        "canonical": "GPT-4o Mini"
    },
    "o1": {
        "openai": "o1",
        "openrouter": "openai/o1",
        "canonical": "O1"
    },
    "o1-mini": {
        "openai": "o1-mini",
        "openrouter": "openai/o1-mini",
        "canonical": "O1 Mini"
    }
}

# Pricing per 1M tokens (input, output)
MODEL_PRICING = {
    "claude-sonnet-4.5": (3.0, 15.0),
    "claude-opus-4.5": (15.0, 75.0),
    "claude-3.5-sonnet": (3.0, 15.0),
    "claude-3.5-haiku": (0.25, 1.25),
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.6),
    "o1": (15.0, 60.0),
    "o1-mini": (3.0, 12.0),
    # Ollama/local models are free
    "llama3.2": (0.0, 0.0),
    "mistral": (0.0, 0.0),
    "gemma2": (0.0, 0.0),
    "deepseek-r1": (0.0, 0.0),
}


def map_model_id(model_id: str, target_provider: str) -> str:
    """Map model ID between providers."""
    # Check Claude models
    for canonical, mapping in CLAUDE_MODELS.items():
        if model_id in [mapping.get("anthropic"), mapping.get("openrouter"),
                        mapping.get("bedrock"), canonical]:
            return mapping.get(target_provider, model_id)

    # Check OpenAI models
    for canonical, mapping in OPENAI_MODELS.items():
        if model_id in [mapping.get("openai"), mapping.get("openrouter"), canonical]:
            return mapping.get(target_provider, model_id)

    return model_id


def get_model_pricing(model_id: str) -> Tuple[float, float]:
    """Get pricing for model (input $/M, output $/M)."""
    normalized = model_id.lower()
    for key, pricing in MODEL_PRICING.items():
        if key in normalized:
            return pricing
    return (0.0, 0.0)  # Unknown model, assume free


# =============================================================================
# Abstract Provider Interface
# =============================================================================

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    name: str
    type: ProviderType
    supports_streaming: bool = True
    supports_tools: bool = True
    supports_mcp: bool = False

    @abstractmethod
    async def chat(self, params: ChatParams) -> ChatResponse:
        """Send chat completion request."""
        pass

    async def stream(self, params: ChatParams) -> AsyncGenerator[StreamChunk, None]:
        """Stream chat completion. Default implementation wraps chat()."""
        response = await self.chat(params)
        yield StreamChunk(
            type="message_start",
            message={"id": response.id, "model": response.model}
        )
        for block in response.content:
            yield StreamChunk(type="content_block_start", content_block=block)
        yield StreamChunk(type="message_stop", usage=response.usage)

    @abstractmethod
    def validate_capabilities(self, features: List[str]) -> bool:
        """Validate that provider supports required features."""
        pass

    def calculate_cost(self, usage: UsageStats, model: str) -> float:
        """Calculate cost for usage."""
        input_price, output_price = get_model_pricing(model)
        return (usage.input_tokens / 1_000_000 * input_price +
                usage.output_tokens / 1_000_000 * output_price)


# =============================================================================
# Provider Implementations
# =============================================================================

class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider."""

    name = "anthropic"
    type = ProviderType.ANTHROPIC
    supports_streaming = True
    supports_tools = True
    supports_mcp = True

    def __init__(self, config: ProviderConfig):
        self.config = config
        self._client = None

    async def chat(self, params: ChatParams) -> ChatResponse:
        """Send chat request to Anthropic."""
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package required: pip install anthropic")

        # Extract system message
        system_msg = None
        messages = []
        for msg in params.messages:
            if msg.role == "system":
                system_msg = msg.content if isinstance(msg.content, str) else str(msg.content)
            else:
                messages.append({
                    "role": msg.role,
                    "content": msg.content if isinstance(msg.content, str) else [
                        {"type": b.type, "text": b.text} if b.type == "text" else
                        {"type": "tool_use", "id": b.id, "name": b.name, "input": b.input}
                        for b in msg.content
                    ]
                })

        kwargs = {
            "model": params.model,
            "messages": messages,
            "max_tokens": params.max_tokens,
        }
        if params.temperature is not None:
            kwargs["temperature"] = params.temperature
        if system_msg:
            kwargs["system"] = system_msg
        if params.tools:
            kwargs["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.input_schema}
                for t in params.tools
            ]
        if params.tool_choice:
            kwargs["tool_choice"] = params.tool_choice

        async_client = anthropic.AsyncAnthropic(
            api_key=self.config.api_key or os.getenv("ANTHROPIC_API_KEY"),
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            max_retries=self.config.max_retries
        )

        try:
            response = await async_client.messages.create(**kwargs)
        except Exception as e:
            raise ProviderError(
                str(e), provider="anthropic",
                retryable="rate" in str(e).lower() or "timeout" in str(e).lower()
            )

        content_blocks = []
        for block in response.content:
            if hasattr(block, 'text'):
                content_blocks.append(ContentBlock(type="text", text=block.text))
            elif hasattr(block, 'name'):
                content_blocks.append(ContentBlock(
                    type="tool_use", id=block.id, name=block.name, input=block.input
                ))

        usage = UsageStats(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens
        )

        return ChatResponse(
            id=response.id,
            model=response.model,
            content=content_blocks,
            stop_reason=StopReason(response.stop_reason) if response.stop_reason else None,
            usage=usage,
            metadata=ResponseMetadata(
                provider="anthropic", model=response.model,
                cost=self.calculate_cost(usage, response.model)
            )
        )

    def validate_capabilities(self, features: List[str]) -> bool:
        return all(f in ["chat", "streaming", "tools", "mcp"] for f in features)


class OpenAIProvider(LLMProvider):
    """OpenAI provider."""

    name = "openai"
    type = ProviderType.OPENAI
    supports_streaming = True
    supports_tools = True
    supports_mcp = False

    def __init__(self, config: ProviderConfig):
        self.config = config

    async def chat(self, params: ChatParams) -> ChatResponse:
        """Send chat request to OpenAI."""
        try:
            import openai
        except ImportError:
            raise ImportError("openai package required: pip install openai")

        messages = [
            {"role": m.role, "content": m.content if isinstance(m.content, str) else str(m.content)}
            for m in params.messages
        ]

        kwargs = {
            "model": params.model,
            "messages": messages,
            "temperature": params.temperature,
            "max_tokens": params.max_tokens,
        }

        if params.tools:
            kwargs["tools"] = [
                {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.input_schema}}
                for t in params.tools
            ]

        client = openai.AsyncOpenAI(
            api_key=self.config.api_key or os.getenv("OPENAI_API_KEY"),
            base_url=self.config.base_url,
            organization=self.config.organization,
            timeout=self.config.timeout,
            max_retries=self.config.max_retries
        )

        try:
            response = await client.chat.completions.create(**kwargs)
        except Exception as e:
            raise ProviderError(
                str(e), provider="openai",
                retryable="rate" in str(e).lower() or "timeout" in str(e).lower()
            )

        choice = response.choices[0]
        content_blocks = []
        if choice.message.content:
            content_blocks.append(ContentBlock(type="text", text=choice.message.content))
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                content_blocks.append(ContentBlock(
                    type="tool_use", id=tc.id, name=tc.function.name,
                    input=json.loads(tc.function.arguments)
                ))

        usage = UsageStats(
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens
        ) if response.usage else UsageStats()

        return ChatResponse(
            id=response.id,
            model=response.model,
            content=content_blocks,
            stop_reason=StopReason.END_TURN if choice.finish_reason == "stop" else
                       StopReason.TOOL_USE if choice.finish_reason == "tool_calls" else
                       StopReason.MAX_TOKENS if choice.finish_reason == "length" else None,
            usage=usage,
            metadata=ResponseMetadata(
                provider="openai", model=response.model,
                cost=self.calculate_cost(usage, response.model)
            )
        )

    def validate_capabilities(self, features: List[str]) -> bool:
        return all(f in ["chat", "streaming", "tools"] for f in features)


class OllamaProvider(LLMProvider):
    """Ollama local inference provider."""

    name = "ollama"
    type = ProviderType.OLLAMA
    supports_streaming = True
    supports_tools = True
    supports_mcp = False

    def __init__(self, config: ProviderConfig):
        self.config = config
        self.base_url = config.base_url or os.getenv("OLLAMA_HOST", "http://localhost:11434")

    async def chat(self, params: ChatParams) -> ChatResponse:
        """Send chat request to Ollama."""
        if httpx is None:
            raise ImportError("httpx package required: pip install httpx")

        messages = [
            {"role": m.role, "content": m.content if isinstance(m.content, str) else str(m.content)}
            for m in params.messages
        ]

        async with httpx.AsyncClient(timeout=httpx.Timeout(self.config.timeout)) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": params.model,
                        "messages": messages,
                        "stream": False,
                        "options": {"temperature": params.temperature, "num_predict": params.max_tokens}
                    }
                )
                response.raise_for_status()
                data = response.json()
            except Exception as e:
                raise ProviderError(str(e), provider="ollama", retryable=True)

        content_blocks = [ContentBlock(type="text", text=data.get("message", {}).get("content", ""))]
        usage = UsageStats(
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0)
        )

        return ChatResponse(
            id=f"ollama-{int(time.time())}",
            model=params.model,
            content=content_blocks,
            stop_reason=StopReason.END_TURN,
            usage=usage,
            metadata=ResponseMetadata(provider="ollama", model=params.model, cost=0.0)
        )

    def validate_capabilities(self, features: List[str]) -> bool:
        return all(f in ["chat", "streaming"] for f in features)


class ExoProvider(LLMProvider):
    """Exo distributed inference cluster provider."""

    name = "exo"
    type = ProviderType.EXO
    supports_streaming = True
    supports_tools = False
    supports_mcp = False

    def __init__(self, config: ProviderConfig):
        self.config = config
        self.base_url = config.base_url or "http://localhost:52415"

    async def chat(self, params: ChatParams) -> ChatResponse:
        """Send chat request to Exo cluster."""
        if httpx is None:
            raise ImportError("httpx package required: pip install httpx")

        messages = [
            {"role": m.role, "content": m.content if isinstance(m.content, str) else str(m.content)}
            for m in params.messages
        ]

        async with httpx.AsyncClient(timeout=httpx.Timeout(self.config.timeout)) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json={
                        "model": params.model,
                        "messages": messages,
                        "temperature": params.temperature,
                        "max_tokens": params.max_tokens,
                        "stream": False
                    }
                )
                response.raise_for_status()
                data = response.json()
            except Exception as e:
                raise ProviderError(str(e), provider="exo", retryable=True)

        choice = data["choices"][0]
        content_blocks = [ContentBlock(type="text", text=choice["message"]["content"])]
        usage = UsageStats(
            input_tokens=data.get("usage", {}).get("prompt_tokens", 0),
            output_tokens=data.get("usage", {}).get("completion_tokens", 0)
        )

        return ChatResponse(
            id=data.get("id", f"exo-{int(time.time())}"),
            model=params.model,
            content=content_blocks,
            stop_reason=StopReason.END_TURN,
            usage=usage,
            metadata=ResponseMetadata(provider="exo", model=params.model, cost=0.0)
        )

    def validate_capabilities(self, features: List[str]) -> bool:
        return all(f in ["chat", "streaming"] for f in features)


# =============================================================================
# Model Router
# =============================================================================

class ModelRouter:
    """
    Multi-provider LLM router with intelligent routing strategies.

    Ported from ruvnet/agentic-flow TypeScript implementation.

    Features:
    - Manual provider selection
    - Rule-based routing (by agent type, complexity, privacy)
    - Cost-optimized routing (prefer cheaper providers)
    - Performance-optimized routing (prefer faster providers)
    - Fallback chains for reliability
    - Metrics tracking per provider and agent type
    """

    def __init__(self, config: Optional[RouterConfig] = None, config_path: Optional[str] = None):
        """
        Initialize router with config or config file path.

        Args:
            config: RouterConfig object
            config_path: Path to JSON config file
        """
        if config:
            self.config = config
        elif config_path:
            self.config = self._load_config(config_path)
        else:
            self.config = self._load_from_env()

        self.providers: Dict[ProviderType, LLMProvider] = {}
        self.metrics = RouterMetrics()
        self.uncertainty_estimator = UncertaintyEstimator()
        self._init_providers()

    def _load_config(self, path: str) -> RouterConfig:
        """Load config from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)

        providers = {}
        for name, pconfig in data.get("providers", {}).items():
            providers[name] = ProviderConfig(**pconfig)

        routing = None
        if "routing" in data:
            rules = [RoutingRule(**r) for r in data["routing"].get("rules", [])]
            routing = RoutingConfig(
                mode=RoutingMode(data["routing"].get("mode", "manual")),
                rules=rules,
                cost_optimization=data["routing"].get("costOptimization"),
                performance=data["routing"].get("performance")
            )

        return RouterConfig(
            version=data.get("version", "1.0.0"),
            default_provider=ProviderType(data.get("defaultProvider", "anthropic")),
            fallback_chain=[ProviderType(p) for p in data.get("fallbackChain", [])],
            providers=providers,
            routing=routing,
            monitoring=MonitoringConfig(**data.get("monitoring", {})) if "monitoring" in data else None,
            cache=CacheConfig(**data.get("cache", {})) if "cache" in data else None
        )

    def _load_from_env(self) -> RouterConfig:
        """Load config from environment variables."""
        providers = {}

        if os.getenv("ANTHROPIC_API_KEY"):
            providers["anthropic"] = ProviderConfig(api_key=os.getenv("ANTHROPIC_API_KEY"))
        if os.getenv("OPENAI_API_KEY"):
            providers["openai"] = ProviderConfig(api_key=os.getenv("OPENAI_API_KEY"))
        if os.getenv("OLLAMA_HOST") or True:
            providers["ollama"] = ProviderConfig(base_url=os.getenv("OLLAMA_HOST"))

        default = ProviderType.ANTHROPIC
        if "anthropic" not in providers and "openai" in providers:
            default = ProviderType.OPENAI
        elif "anthropic" not in providers and "openai" not in providers:
            default = ProviderType.OLLAMA

        fallback = []
        for p in [ProviderType.ANTHROPIC, ProviderType.OPENAI, ProviderType.OLLAMA]:
            if p.value in providers and p != default:
                fallback.append(p)

        return RouterConfig(
            default_provider=default,
            fallback_chain=fallback,
            providers=providers,
            routing=RoutingConfig(mode=RoutingMode.MANUAL),
            monitoring=MonitoringConfig(enabled=True)
        )

    def _init_providers(self):
        """Initialize configured providers."""
        provider_classes = {
            "anthropic": AnthropicProvider,
            "openai": OpenAIProvider,
            "ollama": OllamaProvider,
            "exo": ExoProvider,
        }

        for name, config in self.config.providers.items():
            if name in provider_classes:
                try:
                    self.providers[ProviderType(name)] = provider_classes[name](config)
                    logger.info(f"Initialized provider: {name}")
                except Exception as e:
                    logger.warning(f"Failed to initialize {name}: {e}")

    async def chat(
        self,
        params: ChatParams,
        agent_type: Optional[str] = None
    ) -> ChatResponse:
        """
        Send chat completion with intelligent routing.

        Args:
            params: Chat parameters
            agent_type: Optional agent type for rule-based routing

        Returns:
            ChatResponse from selected provider
        """
        start_time = time.time()

        provider = await self._select_provider(params, agent_type)

        try:
            response = await provider.chat(params)

            latency = time.time() - start_time
            self._track_metrics(provider.type.value, response, latency, agent_type)

            if response.metadata:
                response.metadata.latency = latency

            return response

        except ProviderError as e:
            if e.retryable and self.config.fallback_chain:
                return await self._handle_provider_error(e, params, agent_type)
            raise

    async def stream(
        self,
        params: ChatParams,
        agent_type: Optional[str] = None
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream chat completion."""
        provider = await self._select_provider(params, agent_type)
        async for chunk in provider.stream(params):
            yield chunk

    async def _select_provider(
        self,
        params: ChatParams,
        agent_type: Optional[str] = None
    ) -> LLMProvider:
        """Select appropriate provider based on routing mode."""
        if params.provider:
            provider_type = ProviderType(params.provider)
            if provider_type in self.providers:
                return self.providers[provider_type]

        routing = self.config.routing or RoutingConfig()

        if routing.mode == RoutingMode.RULE_BASED and routing.rules:
            return self._select_by_rules(params, agent_type)
        elif routing.mode == RoutingMode.COST_OPTIMIZED:
            return self._select_by_cost(params)
        elif routing.mode == RoutingMode.PERFORMANCE_OPTIMIZED:
            return self._select_by_performance(params)
        elif routing.mode == RoutingMode.MEMORY_INFLUENCED:
            # Phase 2: Holographic memory routing
            return self._select_by_memory(params, agent_type)
        else:
            if self.config.default_provider in self.providers:
                return self.providers[self.config.default_provider]
            return next(iter(self.providers.values()))

    def _select_by_rules(
        self,
        params: ChatParams,
        agent_type: Optional[str] = None
    ) -> LLMProvider:
        """Select provider based on routing rules."""
        routing = self.config.routing
        if not routing or not routing.rules:
            return self.providers[self.config.default_provider]

        for rule in routing.rules:
            condition = rule.condition

            if "agent_type" in condition:
                if agent_type not in condition["agent_type"]:
                    continue

            if "requires_tools" in condition:
                has_tools = params.tools is not None and len(params.tools) > 0
                if condition["requires_tools"] != has_tools:
                    continue

            if condition.get("local_only"):
                if ProviderType.OLLAMA in self.providers:
                    return self.providers[ProviderType.OLLAMA]
                if ProviderType.EXO in self.providers:
                    return self.providers[ProviderType.EXO]
                continue

            provider_type = ProviderType(rule.action["provider"])
            if provider_type in self.providers:
                logger.debug(f"Rule matched: {rule.reason or 'unnamed rule'}")
                return self.providers[provider_type]

        return self.providers[self.config.default_provider]

    def _select_by_cost(self, params: ChatParams) -> LLMProvider:
        """Select cheapest available provider."""
        costs = []
        for ptype, provider in self.providers.items():
            input_price, output_price = get_model_pricing(params.model)
            estimated_cost = (params.max_tokens / 1_000_000 * output_price +
                             1000 / 1_000_000 * input_price)
            costs.append((estimated_cost, ptype, provider))

        costs.sort(key=lambda x: x[0])

        for cost, ptype, provider in costs:
            return provider

        return self.providers[self.config.default_provider]

    def _select_by_performance(self, params: ChatParams) -> LLMProvider:
        """Select fastest provider based on historical latency."""
        latencies = []
        for ptype, provider in self.providers.items():
            ptype_str = ptype.value
            if ptype_str in self.metrics.provider_breakdown:
                avg_latency = self.metrics.provider_breakdown[ptype_str].avg_latency
            else:
                defaults = {"ollama": 0.5, "exo": 1.0, "anthropic": 2.0, "openai": 1.5}
                avg_latency = defaults.get(ptype_str, 5.0)
            latencies.append((avg_latency, ptype, provider))

        latencies.sort(key=lambda x: x[0])

        for latency, ptype, provider in latencies:
            return provider

        return self.providers[self.config.default_provider]

    def _select_by_memory(
        self,
        params: ChatParams,
        agent_type: Optional[str] = None
    ) -> LLMProvider:
        """
        Select provider based on holographic memory activation field.

        Phase 2 of holographic memory implementation - memory automatically
        influences model selection without explicit retrieval.

        Routing bias mapping:
        - simple: Fast/cheap models (Ollama, Haiku, GPT-4o-mini)
        - balanced: Mid-tier models (Sonnet, GPT-4o)
        - complex: Powerful models (Opus, O1)
        - local: Local inference only (Ollama, Exo)
        """
        try:
            from agi.activation_field import get_activation_field

            field = get_activation_field()

            # Extract query from messages for activation computation
            query = ""
            for msg in params.messages:
                if msg.role == "user":
                    query = msg.content if isinstance(msg.content, str) else str(msg.content)
                    break

            # Compute activation field from query context
            session_context = {"agent_type": agent_type} if agent_type else {}
            state = field.compute_from_context(
                query=query,
                session_context=session_context
            )

            # Get routing bias from activation field
            routing_bias = state.routing_bias

            # Determine dominant bias
            max_bias_key = max(routing_bias, key=routing_bias.get) if routing_bias else "balanced"
            max_bias_value = routing_bias.get(max_bias_key, 0.0) if routing_bias else 0.0

            logger.info(f"Memory-influenced routing: {max_bias_key}={max_bias_value:.2f}, "
                       f"confidence_modifier={state.confidence_modifier:.2f}")

            # Apply confidence modifier to uncertainty estimation
            if state.confidence_modifier != 1.0:
                # High confidence (>1) = lower uncertainty, low confidence (<1) = higher uncertainty
                self.uncertainty_estimator.config.boundary_threshold = (
                    0.5 / state.confidence_modifier
                )

            # Route based on dominant bias
            if max_bias_key == "local" or routing_bias.get("local", 0) > 0.5:
                # Prefer local inference
                if ProviderType.OLLAMA in self.providers:
                    return self.providers[ProviderType.OLLAMA]
                if ProviderType.EXO in self.providers:
                    return self.providers[ProviderType.EXO]

            elif max_bias_key == "simple":
                # Prefer fast/cheap models
                # Priority: Ollama > OpenAI (mini) > Anthropic (Haiku)
                if ProviderType.OLLAMA in self.providers:
                    return self.providers[ProviderType.OLLAMA]
                if ProviderType.OPENAI in self.providers:
                    # OpenAI with mini model preference
                    return self.providers[ProviderType.OPENAI]
                if ProviderType.ANTHROPIC in self.providers:
                    return self.providers[ProviderType.ANTHROPIC]

            elif max_bias_key == "complex":
                # Prefer powerful models
                # Priority: Anthropic (Opus) > OpenAI (O1) > others
                if ProviderType.ANTHROPIC in self.providers:
                    return self.providers[ProviderType.ANTHROPIC]
                if ProviderType.OPENAI in self.providers:
                    return self.providers[ProviderType.OPENAI]

            else:  # balanced
                # Default balanced selection
                if ProviderType.ANTHROPIC in self.providers:
                    return self.providers[ProviderType.ANTHROPIC]
                if ProviderType.OPENAI in self.providers:
                    return self.providers[ProviderType.OPENAI]

            # Fallback to default
            return self.providers.get(
                self.config.default_provider,
                next(iter(self.providers.values()))
            )

        except ImportError as e:
            logger.warning(f"Activation field not available: {e}, falling back to default")
            return self.providers.get(
                self.config.default_provider,
                next(iter(self.providers.values()))
            )
        except Exception as e:
            logger.error(f"Error in memory-influenced routing: {e}, falling back to default")
            return self.providers.get(
                self.config.default_provider,
                next(iter(self.providers.values()))
            )

    async def _handle_provider_error(
        self,
        error: ProviderError,
        params: ChatParams,
        agent_type: Optional[str]
    ) -> ChatResponse:
        """Handle provider error with fallback chain."""
        logger.warning(f"Provider {error.provider} failed: {error}")

        if error.provider in self.metrics.provider_breakdown:
            self.metrics.provider_breakdown[error.provider].errors += 1

        for fallback_type in self.config.fallback_chain:
            if fallback_type in self.providers:
                fallback = self.providers[fallback_type]
                if fallback_type.value == error.provider:
                    continue

                logger.info(f"Trying fallback: {fallback_type.value}")

                try:
                    fallback_params = ChatParams(
                        model=map_model_id(params.model, fallback_type.value),
                        messages=params.messages,
                        temperature=params.temperature,
                        max_tokens=params.max_tokens,
                        tools=params.tools,
                        tool_choice=params.tool_choice,
                        stream=params.stream,
                        metadata=params.metadata
                    )
                    return await fallback.chat(fallback_params)
                except Exception as e:
                    logger.warning(f"Fallback {fallback_type.value} also failed: {e}")
                    continue

        raise error

    def _track_metrics(
        self,
        provider: str,
        response: ChatResponse,
        latency: float,
        agent_type: Optional[str] = None
    ):
        """Track request metrics."""
        self.metrics.total_requests += 1
        if response.metadata:
            self.metrics.total_cost += response.metadata.cost
        if response.usage:
            self.metrics.total_tokens.input_tokens += response.usage.input_tokens
            self.metrics.total_tokens.output_tokens += response.usage.output_tokens

        if provider not in self.metrics.provider_breakdown:
            self.metrics.provider_breakdown[provider] = ProviderMetrics()

        pm = self.metrics.provider_breakdown[provider]
        pm.requests += 1
        pm.cost += response.metadata.cost if response.metadata else 0
        pm.total_latency += latency
        pm.avg_latency = pm.total_latency / pm.requests

        if agent_type:
            if agent_type not in self.metrics.agent_breakdown:
                self.metrics.agent_breakdown[agent_type] = {"requests": 0, "cost": 0.0}
            self.metrics.agent_breakdown[agent_type]["requests"] += 1
            if response.metadata:
                self.metrics.agent_breakdown[agent_type]["cost"] += response.metadata.cost

        # Record for uncertainty calibration
        # Success = non-empty content response with reasonable latency
        was_successful = (
            response.content and
            len(response.content) > 0 and
            latency < 30.0  # Within reasonable time
        )
        self.record_routing_outcome(provider, was_successful, latency)

    def get_metrics(self) -> RouterMetrics:
        """Get current metrics."""
        return self.metrics

    def get_config(self) -> RouterConfig:
        """Get router configuration."""
        return self.config

    def get_providers(self) -> Dict[ProviderType, LLMProvider]:
        """Get initialized providers."""
        return self.providers

    def reset_metrics(self):
        """Reset all metrics."""
        self.metrics = RouterMetrics()

    def record_routing_outcome(self, provider: str, was_successful: bool, latency: float):
        """
        Record routing outcome for uncertainty calibration.

        Args:
            provider: Provider that was used
            was_successful: Whether the request succeeded without errors
            latency: Request latency in seconds
        """
        # Use latency as a proxy for prediction confidence
        # Lower latency = higher confidence, normalized to [0, 1]
        max_latency = 30.0  # 30 seconds max expected
        prediction = 1.0 - min(latency / max_latency, 1.0)

        self.uncertainty_estimator.record_outcome(prediction, was_successful)

    def get_uncertainty_stats(self) -> Dict[str, Any]:
        """Get uncertainty estimation statistics."""
        return self.uncertainty_estimator.get_statistics()

    def get_routing_uncertainty(self, prediction: float) -> Dict[str, float]:
        """
        Get uncertainty for a routing prediction.

        Args:
            prediction: Routing confidence score [0, 1]

        Returns:
            Dict with raw and calibrated uncertainty
        """
        raw, calibrated = self.uncertainty_estimator.get_calibrated_uncertainty(prediction)
        return {"raw_uncertainty": raw, "calibrated_uncertainty": calibrated}


# =============================================================================
# MCP Tool Registration
# =============================================================================

def register_model_router_tools(app, router: Optional[ModelRouter] = None):
    """
    Register ModelRouter tools with FastMCP app.

    Tools:
    - router_chat: Send chat completion
    - router_select_provider: Get provider selection without calling
    - router_metrics: Get routing metrics
    - router_status: Get router status
    - router_set_mode: Change routing mode
    - router_add_rule: Add routing rule
    """

    if router is None:
        router = ModelRouter()

    @app.tool()
    async def router_chat(
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        provider: Optional[str] = None,
        agent_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send chat completion through ModelRouter.

        Intelligently routes to appropriate provider based on routing mode.

        Args:
            model: Model ID (e.g., "claude-3.5-sonnet", "gpt-4o")
            messages: List of message dicts with role and content
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            provider: Force specific provider (anthropic, openai, ollama)
            agent_type: Agent type for rule-based routing

        Returns:
            Response with content, usage, metadata
        """
        params = ChatParams(
            model=model,
            messages=[Message(role=m["role"], content=m["content"]) for m in messages],
            temperature=temperature,
            max_tokens=max_tokens,
            provider=provider
        )

        response = await router.chat(params, agent_type=agent_type)

        return {
            "id": response.id,
            "model": response.model,
            "content": [{"type": b.type, "text": b.text} for b in response.content if b.type == "text"],
            "stop_reason": response.stop_reason.value if response.stop_reason else None,
            "usage": {
                "input_tokens": response.usage.input_tokens if response.usage else 0,
                "output_tokens": response.usage.output_tokens if response.usage else 0
            },
            "metadata": {
                "provider": response.metadata.provider if response.metadata else "unknown",
                "cost": response.metadata.cost if response.metadata else 0.0,
                "latency": response.metadata.latency if response.metadata else 0.0
            }
        }

    @app.tool()
    async def router_select_provider(
        model: str,
        agent_type: Optional[str] = None,
        requires_tools: bool = False
    ) -> Dict[str, Any]:
        """
        Preview which provider would be selected for a request.

        Args:
            model: Model ID to route
            agent_type: Optional agent type for rule-based routing
            requires_tools: Whether request needs tool support

        Returns:
            Provider selection details
        """
        params = ChatParams(
            model=model,
            messages=[],
            tools=[Tool(name="dummy", description="", input_schema={"type": "object", "properties": {}})] if requires_tools else None
        )

        provider = await router._select_provider(params, agent_type)

        return {
            "selected_provider": provider.type.value,
            "provider_name": provider.name,
            "supports_streaming": provider.supports_streaming,
            "supports_tools": provider.supports_tools,
            "supports_mcp": provider.supports_mcp,
            "routing_mode": router.config.routing.mode.value if router.config.routing else "manual"
        }

    @app.tool()
    async def router_metrics() -> Dict[str, Any]:
        """
        Get routing metrics.

        Returns comprehensive metrics including per-provider breakdown.
        """
        m = router.get_metrics()

        return {
            "total_requests": m.total_requests,
            "total_cost": round(m.total_cost, 6),
            "total_tokens": {
                "input": m.total_tokens.input_tokens,
                "output": m.total_tokens.output_tokens
            },
            "provider_breakdown": {
                name: {
                    "requests": pm.requests,
                    "cost": round(pm.cost, 6),
                    "avg_latency_ms": round(pm.avg_latency * 1000, 2),
                    "errors": pm.errors
                }
                for name, pm in m.provider_breakdown.items()
            },
            "agent_breakdown": m.agent_breakdown
        }

    @app.tool()
    async def router_status() -> Dict[str, Any]:
        """Get router status and configuration."""
        return {
            "version": router.config.version,
            "routing_mode": router.config.routing.mode.value if router.config.routing else "manual",
            "default_provider": router.config.default_provider.value,
            "fallback_chain": [p.value for p in router.config.fallback_chain],
            "available_providers": [p.value for p in router.providers.keys()],
            "rules_count": len(router.config.routing.rules) if router.config.routing else 0,
            "monitoring_enabled": router.config.monitoring.enabled if router.config.monitoring else False
        }

    @app.tool()
    async def router_set_mode(mode: str) -> Dict[str, Any]:
        """
        Change routing mode.

        Args:
            mode: One of:
                - "manual": Direct provider selection
                - "rule-based": Match agent types to providers
                - "cost-optimized": Prefer cheaper providers
                - "performance-optimized": Prefer faster providers
                - "memory-influenced": Use holographic memory activation field
                  (Phase 2: Memory automatically influences model selection)
        """
        try:
            new_mode = RoutingMode(mode)
        except ValueError:
            return {"error": f"Invalid mode. Must be one of: {[m.value for m in RoutingMode]}"}

        if router.config.routing is None:
            router.config.routing = RoutingConfig()

        router.config.routing.mode = new_mode

        return {"success": True, "new_mode": new_mode.value}

    @app.tool()
    async def router_add_rule(
        provider: str,
        model: str,
        agent_types: Optional[List[str]] = None,
        requires_tools: Optional[bool] = None,
        local_only: bool = False,
        reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Add a routing rule.

        Args:
            provider: Target provider (anthropic, openai, ollama)
            model: Model to use
            agent_types: List of agent types this rule applies to
            requires_tools: Whether request must require tools
            local_only: Whether to force local inference
            reason: Description of rule
        """
        condition = {}
        if agent_types:
            condition["agent_type"] = agent_types
        if requires_tools is not None:
            condition["requires_tools"] = requires_tools
        if local_only:
            condition["local_only"] = True

        rule = RoutingRule(
            condition=condition,
            action={"provider": provider, "model": model},
            reason=reason
        )

        if router.config.routing is None:
            router.config.routing = RoutingConfig()

        router.config.routing.rules.append(rule)
        router.config.routing.mode = RoutingMode.RULE_BASED

        return {"success": True, "rules_count": len(router.config.routing.rules)}

    @app.tool()
    async def router_get_uncertainty() -> Dict[str, Any]:
        """
        Get uncertainty estimation statistics for routing decisions.

        Returns statistics from conformal prediction-based uncertainty estimation:
        - calibration_quantile: Configured confidence level (default 0.9 = 90%)
        - calibration_threshold: Current calibration threshold from non-conformity scores
        - total_predictions_tracked: Number of routing decisions tracked
        - recent_accuracy: Accuracy of recent routing predictions
        - average_uncertainty: Mean uncertainty across recent predictions
        - is_calibrated: Whether the estimator has been calibrated
        - calibration_samples: Number of samples used for calibration

        Ported from ruvector/tiny-dancer-core uncertainty.rs
        """
        stats = router.get_uncertainty_stats()
        return {
            "success": True,
            "uncertainty_stats": stats,
            "description": "Conformal prediction-based uncertainty estimation for routing decisions"
        }

    @app.tool()
    async def router_estimate_uncertainty(prediction: float) -> Dict[str, Any]:
        """
        Estimate uncertainty for a routing prediction score.

        Args:
            prediction: Routing confidence score [0.0-1.0]
                       0.5 = maximum uncertainty (at decision boundary)
                       0.0 or 1.0 = minimum uncertainty (confident decision)

        Returns:
            raw_uncertainty: Boundary distance uncertainty
            calibrated_uncertainty: Adjusted by calibration threshold (if available)
        """
        if prediction < 0.0 or prediction > 1.0:
            return {"error": "prediction must be between 0.0 and 1.0"}

        result = router.get_routing_uncertainty(prediction)
        return {
            "success": True,
            "prediction": prediction,
            "raw_uncertainty": round(result["raw_uncertainty"], 4),
            "calibrated_uncertainty": round(result["calibrated_uncertainty"], 4)
        }

    @app.tool()
    async def router_get_memory_state() -> Dict[str, Any]:
        """
        Get current memory-influenced routing state from activation field.

        Returns the holographic memory activation state that influences
        routing decisions when mode is "memory-influenced".

        Returns:
            routing_bias: Dict with simple/balanced/complex/local weights
            confidence_modifier: Memory familiarity scaling (>1 = familiar)
            primed_concepts: Subconsciously activated concepts
            emotional_context: Valence/arousal/dominance from memory
            recommendation: Suggested routing tier
        """
        try:
            from agi.activation_field import get_activation_field

            field = get_activation_field()
            state = field.current_state

            if state:
                return {
                    "success": True,
                    "has_state": True,
                    "routing_bias": state.routing_bias,
                    "confidence_modifier": round(state.confidence_modifier, 3),
                    "primed_concepts": list(state.primed_concepts),
                    "emotional_context": state.emotional_context,
                    "recommendation": field.get_routing_recommendation(),
                    "should_elaborate": field.should_elaborate()
                }
            else:
                return {
                    "success": True,
                    "has_state": False,
                    "message": "No activation field computed yet. Use router_chat or compute_activation_field first."
                }

        except ImportError:
            return {
                "success": False,
                "error": "Activation field module not available"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    @app.tool()
    async def router_enable_memory_routing() -> Dict[str, Any]:
        """
        Enable memory-influenced routing mode.

        This activates Phase 2 holographic memory routing, where the
        activation field automatically influences model selection based on:
        - Query familiarity (confidence modifier)
        - Emotional context from activated memories
        - Primed concepts and associations
        - Routing bias (simple/balanced/complex/local)

        Returns:
            Confirmation of mode change
        """
        if router.config.routing is None:
            router.config.routing = RoutingConfig()

        router.config.routing.mode = RoutingMode.MEMORY_INFLUENCED

        return {
            "success": True,
            "mode": "memory-influenced",
            "message": "Memory-influenced routing enabled. Activation field will now influence model selection."
        }

    return router


# =============================================================================
# Convenience Functions
# =============================================================================

_default_router: Optional[ModelRouter] = None


def get_router() -> ModelRouter:
    """Get or create default router instance."""
    global _default_router
    if _default_router is None:
        _default_router = ModelRouter()
    return _default_router


async def chat(
    model: str,
    messages: List[Dict[str, str]],
    **kwargs
) -> ChatResponse:
    """
    Quick chat completion using default router.

    Example:
        response = await chat(
            "claude-3.5-sonnet",
            [{"role": "user", "content": "Hello!"}]
        )
    """
    router = get_router()
    params = ChatParams(
        model=model,
        messages=[Message(role=m["role"], content=m["content"]) for m in messages],
        **kwargs
    )
    return await router.chat(params)
