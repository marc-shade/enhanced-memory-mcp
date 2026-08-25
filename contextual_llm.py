#!/usr/bin/env python3
"""
LLM integration for contextual prefix generation.

Uses Claude API (Anthropic) to generate concise contextual prefixes
that help improve retrieval accuracy.

Part of RAG Tier 1 Strategy - Week 1, Day 5-7
"""

import asyncio
import json
import logging
import os
import urllib.error
import urllib.request
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Local ollama backend (2026-08-24). The Anthropic branch below has been dead
# by policy since creation (rules/intent-engineering.md bans direct AI SDK
# calls, and no env supplies ANTHROPIC_API_KEY), so every one of the 2,429
# prefixes ever stored was the template. Local ollama is the compliant LLM
# path: plain HTTP to the same daemon that already serves embeddings.
OLLAMA_URL = os.environ.get("ENRICHMENT_OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("ENRICHMENT_OLLAMA_MODEL", "gemma4:e4b-it-q8_0")

# Try to import Anthropic SDK
try:
    from anthropic import AsyncAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("Anthropic SDK not installed. Install with: pip install anthropic")


class ContextualPrefixGenerator:
    """
    Generates contextual prefixes for memory chunks using Claude.

    Based on Anthropic's Contextual Retrieval research:
    https://www.anthropic.com/news/contextual-retrieval

    Example:
        Original: "The cross-encoder achieved 45% precision improvement"
        Enriched: "[Context: RAG optimization study comparing re-ranking methods]
                   The cross-encoder achieved 45% precision improvement"
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4.5-20250929",
        max_tokens: int = 200,
        temperature: float = 0.0
    ):
        """
        Initialize contextual prefix generator.

        Args:
            model: Claude model to use (haiku for speed and cost)
            max_tokens: Maximum tokens for prefix (keep concise)
            temperature: Temperature for generation (0.0 for deterministic)
        """
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Initialize client
        self.client = None
        if ANTHROPIC_AVAILABLE:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                self.client = AsyncAnthropic(api_key=api_key)
                logger.info(f"✅ Contextual prefix generator initialized with {model}")
            else:
                logger.warning("ANTHROPIC_API_KEY not set - using fallback mode")
        else:
            logger.warning("Anthropic SDK not available - using fallback mode")

        # Local ollama fallback chain: anthropic (dead by policy) -> ollama
        # -> template. Instance attrs so tests can point at a dead port.
        self.ollama_url = OLLAMA_URL
        self.ollama_model = OLLAMA_MODEL
        self._backend = None  # set by generate_prefix: anthropic|ollama|template

        # Token tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    async def generate_prefix(
        self,
        entity_name: str,
        entity_type: str,
        observations: list,
        metadata: Optional[Dict[str, Any]] = None
    ) -> tuple[str, int, int]:
        """
        Generate contextual prefix for an entity.

        Args:
            entity_name: Name of the entity
            entity_type: Type of the entity
            observations: List of observations about the entity
            metadata: Optional metadata

        Returns:
            Tuple of (prefix, input_tokens, output_tokens)
        """
        if not self.client:
            # Compliant LLM path: local ollama over HTTP, run OFF the event
            # loop (urllib blocks; sync work on the loop is the accept-then-
            # stall mechanism proven on agent-runtime the same day this was
            # written). Any failure degrades to the template, reported as
            # backend="template", never as an LLM result.
            try:
                prefix, ti, to = await asyncio.to_thread(
                    self._ollama_prefix, entity_name, entity_type, observations
                )
                self._backend = "ollama"
                self.total_input_tokens += ti
                self.total_output_tokens += to
                return prefix, ti, to
            except Exception as e:
                logger.warning(f"ollama enrichment unavailable ({e}); template fallback")
                return self._generate_fallback_prefix(entity_name, entity_type, observations)

        try:
            # Prepare context for LLM
            context_lines = []
            context_lines.append(f"Entity Name: {entity_name}")
            context_lines.append(f"Entity Type: {entity_type}")

            if observations:
                context_lines.append(f"Observations: {len(observations)}")
                # Include first few observations
                for i, obs in enumerate(observations[:3]):
                    obs_str = str(obs)
                    if len(obs_str) > 200:
                        obs_str = obs_str[:200] + "..."
                    context_lines.append(f"  {i + 1}. {obs_str}")

            context = "\n".join(context_lines)

            # Generate prefix using Claude
            message = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{
                    "role": "user",
                    "content": f"""Given the following entity information, generate a concise contextual prefix (1-2 sentences) that would help a retrieval system understand what this entity is about.

{context}

Format the prefix as: [Context: <your concise description>]

Keep it under 50 words and focus on the key purpose/domain of this entity.

Respond with ONLY the contextual prefix, nothing else."""
                }]
            )

            # Extract prefix and token counts
            prefix = message.content[0].text.strip()

            input_tokens = message.usage.input_tokens
            output_tokens = message.usage.output_tokens

            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens

            return prefix, input_tokens, output_tokens

        except Exception as e:
            logger.error(f"Error generating prefix with LLM: {e}")
            # Fall back to heuristic
            return self._generate_fallback_prefix(entity_name, entity_type, observations)

    def _ollama_prefix(
        self, entity_name: str, entity_type: str, observations: list
    ) -> tuple[str, int, int]:
        """One-shot local-ollama generation. Raises on any failure."""
        obs_lines = []
        for i, obs in enumerate(observations[:3]):
            text = str(obs)
            obs_lines.append(f"  {i + 1}. {text[:200]}")
        prompt = (
            "Given this entity from a technical memory store, write ONE concise "
            "sentence (under 40 words) describing what it is about, for a "
            "retrieval system.\n"
            f"Name: {entity_name}\nType: {entity_type}\n"
            "Observations:\n" + "\n".join(obs_lines) + "\n"
            "Respond with ONLY the sentence, no preamble."
        )
        req = urllib.request.Request(
            self.ollama_url + "/api/generate",
            data=json.dumps(
                {
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    # gemma4 is a thinking model: with think enabled it spent
                    # the whole num_predict budget reasoning and returned an
                    # EMPTY response (done_reason=length, eval_count=80,
                    # measured 2026-08-24). think:false is ignored by
                    # non-thinking models, so it is safe to send always.
                    "think": False,
                    "options": {"temperature": 0, "num_predict": 120},
                }
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        text = (data.get("response") or "").strip().strip('"')
        if not text:
            raise ValueError("empty ollama response")
        text = " ".join(text.split())[:300]
        prefix = text if text.startswith("[Context:") else f"[Context: {text}]"
        return prefix, int(data.get("prompt_eval_count") or 0), int(data.get("eval_count") or 0)

    def _generate_fallback_prefix(
        self,
        entity_name: str,
        entity_type: str,
        observations: list
    ) -> tuple[str, int, int]:
        """
        Generate prefix using simple heuristics (fallback mode).

        Args:
            entity_name: Name of the entity
            entity_type: Type of the entity
            observations: List of observations

        Returns:
            Tuple of (prefix, 0, 0) - no tokens used in fallback mode
        """
        self._backend = "template"
        prefix = f"[Context: This is a {entity_type} entity named '{entity_name}'"

        if observations:
            first_obs = observations[0] if isinstance(observations, list) else str(observations)
            first_obs_str = str(first_obs)
            if len(first_obs_str) > 50:
                first_obs_str = first_obs_str[:50] + "..."
            prefix += f" with information about {first_obs_str}"

        prefix += "] "

        return prefix, 0, 0

    def get_cost_estimate(self) -> float:
        """
        Estimate cost based on tokens used.

        Returns:
            Total cost in USD
        """
        # Haiku pricing (as of 2024)
        COST_PER_1K_INPUT = 0.00025
        COST_PER_1K_OUTPUT = 0.00125

        input_cost = (self.total_input_tokens / 1000) * COST_PER_1K_INPUT
        output_cost = (self.total_output_tokens / 1000) * COST_PER_1K_OUTPUT

        return input_cost + output_cost

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "estimated_cost_usd": self.get_cost_estimate(),
            "model": self.model,
            "anthropic_available": ANTHROPIC_AVAILABLE,
            "backend": self._backend or ("anthropic" if self.client else "untried"),
            "using_fallback": (self._backend or "template") == "template",
        }


# Global instance
_generator = None


def get_prefix_generator() -> ContextualPrefixGenerator:
    """Get or create global prefix generator instance."""
    global _generator
    if _generator is None:
        _generator = ContextualPrefixGenerator()
    return _generator
