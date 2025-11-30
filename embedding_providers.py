#!/usr/bin/env python3
"""
Multi-Provider Embedding Module
Supports Google, OpenAI, MLX, Ollama, and other embedding providers
with automatic fallback, benchmarking, and provider comparison
"""

import asyncio
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import Enum

logger = logging.getLogger("embedding-providers")


class ProviderType(Enum):
    """Embedding provider types"""
    GOOGLE = "google"
    OPENAI = "openai"
    MLX = "mlx"
    OLLAMA = "ollama"
    VOYAGE = "voyage"
    COHERE = "cohere"
    TPU = "tpu"


@dataclass
class EmbeddingResult:
    """Result from embedding generation"""
    embedding: List[float]
    provider: str
    model: str
    dimensions: int
    latency_ms: float
    token_count: Optional[int] = None
    cost_estimate: Optional[float] = None


class EmbeddingProvider(ABC):
    """Base class for embedding providers"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.provider_name = self.__class__.__name__.lower().replace("provider", "")

    @abstractmethod
    async def generate(self, text: str) -> Optional[EmbeddingResult]:
        """Generate embedding for text"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available"""
        pass

    def _truncate_text(self, text: str, max_length: int = 8000) -> str:
        """Truncate text to maximum length"""
        return text[:max_length] if len(text) > max_length else text


class GoogleProvider(EmbeddingProvider):
    """Google Gemini embedding provider"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = config.get("model", "models/text-embedding-004")
        self.dimensions = config.get("dimensions", 768)
        self.task_type = config.get("task_type", "RETRIEVAL_DOCUMENT")

    def is_available(self) -> bool:
        try:
            import google.generativeai as genai
            api_key = os.getenv(self.config.get("api_key_env", "GOOGLE_API_KEY"))
            return api_key is not None
        except ImportError:
            return False

    async def generate(self, text: str) -> Optional[EmbeddingResult]:
        start_time = time.time()

        try:
            import google.generativeai as genai

            api_key = os.getenv(self.config.get("api_key_env", "GOOGLE_API_KEY"))
            if not api_key:
                return None

            genai.configure(api_key=api_key)

            text = self._truncate_text(text)
            result = genai.embed_content(
                model=self.model,
                content=text,
                task_type=self.task_type
            )

            latency_ms = (time.time() - start_time) * 1000
            embedding = result['embedding']

            logger.debug(f"Google embedding generated: {len(embedding)} dims in {latency_ms:.2f}ms")

            return EmbeddingResult(
                embedding=embedding,
                provider="google",
                model=self.model,
                dimensions=len(embedding),
                latency_ms=latency_ms
            )

        except Exception as e:
            logger.warning(f"Google embedding failed: {e}")
            return None


class OpenAIProvider(EmbeddingProvider):
    """OpenAI embedding provider"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = config.get("model", "text-embedding-3-small")
        self.dimensions = config.get("dimensions", 768)

    def is_available(self) -> bool:
        try:
            import openai
            api_key = os.getenv(self.config.get("api_key_env", "OPENAI_API_KEY"))
            return api_key is not None
        except ImportError:
            return False

    async def generate(self, text: str) -> Optional[EmbeddingResult]:
        start_time = time.time()

        try:
            import openai

            api_key = os.getenv(self.config.get("api_key_env", "OPENAI_API_KEY"))
            if not api_key:
                return None

            client = openai.OpenAI(api_key=api_key)

            text = self._truncate_text(text)
            response = client.embeddings.create(
                model=self.model,
                input=text,
                dimensions=self.dimensions
            )

            latency_ms = (time.time() - start_time) * 1000
            embedding = response.data[0].embedding

            # Cost estimate (text-embedding-3-small: $0.02 per 1M tokens)
            token_count = response.usage.total_tokens if hasattr(response, 'usage') else None
            cost = (token_count / 1_000_000 * 0.02) if token_count else None

            logger.debug(f"OpenAI embedding generated: {len(embedding)} dims in {latency_ms:.2f}ms")

            return EmbeddingResult(
                embedding=embedding,
                provider="openai",
                model=self.model,
                dimensions=len(embedding),
                latency_ms=latency_ms,
                token_count=token_count,
                cost_estimate=cost
            )

        except Exception as e:
            logger.warning(f"OpenAI embedding failed: {e}")
            return None


class MLXProvider(EmbeddingProvider):
    """Apple MLX local embedding provider"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = config.get("model", "sentence-transformers/all-MiniLM-L6-v2")
        self.dimensions = config.get("dimensions", 384)
        self.device = config.get("device", "mps")
        self.batch_size = config.get("batch_size", 32)
        self._model = None
        self._tokenizer = None

    def is_available(self) -> bool:
        try:
            import mlx.core as mx
            import platform
            # MLX only works on Apple Silicon
            return platform.processor() == 'arm'
        except ImportError:
            return False

    def _load_model(self):
        """Lazy load model on first use"""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                import mlx.core as mx

                # Load model with MLX backend
                self._model = SentenceTransformer(self.model, device=self.device)
                logger.info(f"MLX model loaded: {self.model}")
            except Exception as e:
                logger.error(f"Failed to load MLX model: {e}")
                raise

    async def generate(self, text: str) -> Optional[EmbeddingResult]:
        start_time = time.time()

        try:
            self._load_model()

            text = self._truncate_text(text, 512)  # MLX models have shorter context
            embedding = self._model.encode([text], batch_size=1)[0].tolist()

            latency_ms = (time.time() - start_time) * 1000

            logger.debug(f"MLX embedding generated: {len(embedding)} dims in {latency_ms:.2f}ms")

            return EmbeddingResult(
                embedding=embedding,
                provider="mlx",
                model=self.model,
                dimensions=len(embedding),
                latency_ms=latency_ms,
                cost_estimate=0.0  # Local is free
            )

        except Exception as e:
            logger.warning(f"MLX embedding failed: {e}")
            return None


class OllamaProvider(EmbeddingProvider):
    """Ollama embedding provider (cloud-first for GPU acceleration)"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = config.get("model", "mxbai-embed-large")
        self.dimensions = config.get("dimensions", 1024)  # mxbai-embed-large uses 1024 dimensions
        # Cloud-first Ollama for embeddings (prefer GPU nodes)
        # Set OLLAMA_HOST to your inference node (e.g., http://your-gpu-node:11434)
        default_url = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')
        self.base_url = config.get("base_url", default_url)

    def is_available(self) -> bool:
        try:
            import httpx

            # Check if Ollama is running (sync version for is_available)
            with httpx.Client() as client:
                try:
                    response = client.get(f"{self.base_url}/api/tags", timeout=2.0)
                    return response.status_code == 200
                except:
                    return False
        except:
            return False

    async def generate(self, text: str) -> Optional[EmbeddingResult]:
        start_time = time.time()

        try:
            import httpx

            text = self._truncate_text(text)

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/embeddings",
                    json={
                        "model": self.model,
                        "prompt": text
                    }
                )

                if response.status_code != 200:
                    logger.warning(f"Ollama returned status {response.status_code}")
                    return None

                data = response.json()
                embedding = data.get("embedding", [])

                latency_ms = (time.time() - start_time) * 1000

                logger.debug(f"Ollama embedding generated: {len(embedding)} dims in {latency_ms:.2f}ms")

                return EmbeddingResult(
                    embedding=embedding,
                    provider="ollama",
                    model=self.model,
                    dimensions=len(embedding),
                    latency_ms=latency_ms,
                    cost_estimate=0.0  # Local is free
                )

        except Exception as e:
            logger.warning(f"Ollama embedding failed: {e}")
            return None


class VoyageProvider(EmbeddingProvider):
    """Voyage AI embedding provider"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = config.get("model", "voyage-2")
        self.dimensions = config.get("dimensions", 1024)

    def is_available(self) -> bool:
        try:
            import voyageai
            api_key = os.getenv(self.config.get("api_key_env", "VOYAGE_API_KEY"))
            return api_key is not None
        except ImportError:
            return False

    async def generate(self, text: str) -> Optional[EmbeddingResult]:
        start_time = time.time()

        try:
            import voyageai

            api_key = os.getenv(self.config.get("api_key_env", "VOYAGE_API_KEY"))
            if not api_key:
                return None

            client = voyageai.Client(api_key=api_key)

            text = self._truncate_text(text)
            result = client.embed([text], model=self.model)

            latency_ms = (time.time() - start_time) * 1000
            embedding = result.embeddings[0]

            logger.debug(f"Voyage embedding generated: {len(embedding)} dims in {latency_ms:.2f}ms")

            return EmbeddingResult(
                embedding=embedding,
                provider="voyage",
                model=self.model,
                dimensions=len(embedding),
                latency_ms=latency_ms
            )

        except Exception as e:
            logger.warning(f"Voyage embedding failed: {e}")
            return None


class CohereProvider(EmbeddingProvider):
    """Cohere embedding provider"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = config.get("model", "embed-english-v3.0")
        self.dimensions = config.get("dimensions", 1024)
        self.input_type = config.get("input_type", "search_document")

    def is_available(self) -> bool:
        try:
            import cohere
            api_key = os.getenv(self.config.get("api_key_env", "COHERE_API_KEY"))
            return api_key is not None
        except ImportError:
            return False

    async def generate(self, text: str) -> Optional[EmbeddingResult]:
        start_time = time.time()

        try:
            import cohere

            api_key = os.getenv(self.config.get("api_key_env", "COHERE_API_KEY"))
            if not api_key:
                return None

            client = cohere.Client(api_key=api_key)

            text = self._truncate_text(text)
            response = client.embed(
                texts=[text],
                model=self.model,
                input_type=self.input_type
            )

            latency_ms = (time.time() - start_time) * 1000
            embedding = response.embeddings[0]

            logger.debug(f"Cohere embedding generated: {len(embedding)} dims in {latency_ms:.2f}ms")

            return EmbeddingResult(
                embedding=embedding,
                provider="cohere",
                model=self.model,
                dimensions=len(embedding),
                latency_ms=latency_ms
            )

        except Exception as e:
            logger.warning(f"Cohere embedding failed: {e}")
            return None


class TPUProvider(EmbeddingProvider):
    """
    Google Coral Edge TPU embedding provider (local hardware acceleration)

    NOTE: This provider produces 384-dim embeddings (MiniLM-L6-v2) which are
    INCOMPATIBLE with the default 768-dim vector store. Use only for:
    - Specialized tasks requiring smaller embeddings
    - Comparison/benchmarking purposes
    - Systems configured for 384-dim vectors

    For standard enhanced-memory operations, prefer Google or OpenAI providers
    which produce 768-dim embeddings matching the vector store.

    Text embeddings actually run on CPU via sentence-transformers in coral-venv.
    TPU hardware is used for IMAGE inference (MobileNet, EfficientNet, etc.)
    """

    # Path to coral-venv Python which has pycoral and sentence-transformers
    CORAL_VENV_PYTHON = os.path.join(os.environ.get("AGENTIC_SYSTEM_PATH", "${AGENTIC_SYSTEM_PATH:-/opt/agentic}"), "coral-venv/bin/python")
    CORAL_TPU_SRC = os.path.join(os.environ.get("AGENTIC_SYSTEM_PATH", "${AGENTIC_SYSTEM_PATH:-/opt/agentic}"), "mcp-servers/coral-tpu-mcp/src")

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = config.get("model", "MiniLM-L6-v2")
        self.dimensions = config.get("dimensions", 384)
        self._tpu_checked = False
        self._tpu_available = False

    def is_available(self) -> bool:
        """Check if TPU/coral-venv is available via subprocess"""
        if self._tpu_checked:
            return self._tpu_available

        self._tpu_checked = True

        import subprocess
        from pathlib import Path

        if not Path(self.CORAL_VENV_PYTHON).exists():
            logger.info("coral-venv not found for TPU embeddings")
            self._tpu_available = False
            return False

        try:
            # Check if sentence-transformers is available in coral-venv
            result = subprocess.run(
                [self.CORAL_VENV_PYTHON, "-c",
                 "from sentence_transformers import SentenceTransformer; print('ok')"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0 and "ok" in result.stdout:
                self._tpu_available = True
                logger.info("TPU/coral-venv embeddings available (384-dim, CPU-based)")
                return True
        except Exception as e:
            logger.warning(f"TPU availability check failed: {e}")

        self._tpu_available = False
        return False

    async def generate(self, text: str) -> Optional[EmbeddingResult]:
        """Generate embedding via subprocess to coral-venv"""
        start_time = time.time()

        if not self.is_available():
            return None

        try:
            import subprocess
            import json

            text = self._truncate_text(text, 512)  # Shorter context for MiniLM

            # Python code to run in coral-venv
            code = f'''
import json
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
text = {json.dumps(text)}
embedding = model.encode([text], batch_size=1)[0].tolist()
print(json.dumps({{"embedding": embedding, "dimensions": len(embedding)}}))
'''

            result = subprocess.run(
                [self.CORAL_VENV_PYTHON, "-c", code],
                capture_output=True, text=True, timeout=30
            )

            if result.returncode != 0:
                logger.warning(f"TPU embedding subprocess failed: {result.stderr[:200]}")
                return None

            data = json.loads(result.stdout.strip())
            embedding = data.get("embedding", [])

            if not embedding:
                logger.warning("TPU embed_text returned empty embedding")
                return None

            latency_ms = (time.time() - start_time) * 1000

            logger.debug(f"TPU embedding generated: {len(embedding)} dims in {latency_ms:.2f}ms")

            return EmbeddingResult(
                embedding=embedding,
                provider="tpu",
                model=self.model,
                dimensions=len(embedding),
                latency_ms=latency_ms,
                cost_estimate=0.0  # Local is free
            )

        except subprocess.TimeoutExpired:
            logger.warning("TPU embedding timed out")
            return None
        except json.JSONDecodeError as e:
            logger.warning(f"TPU embedding invalid JSON: {e}")
            return None
        except Exception as e:
            logger.warning(f"TPU embedding failed: {e}")
            return None


class EmbeddingManager:
    """
    Manages multiple embedding providers with automatic fallback
    and benchmarking capabilities
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.providers: Dict[str, EmbeddingProvider] = {}
        # Prefer TPU as primary when available (local, fast, free)
        self.primary_provider = config.get("primary", "tpu")
        # Fallback chain: TPU first (local hardware), then cloud services
        self.fallback_chain = config.get("fallback", ["tpu", "ollama", "google", "openai", "mlx"])

        # Initialize providers
        self._init_providers()

    def _init_providers(self):
        """Initialize all configured providers"""
        provider_classes = {
            "google": GoogleProvider,
            "openai": OpenAIProvider,
            "mlx": MLXProvider,
            "ollama": OllamaProvider,
            "voyage": VoyageProvider,
            "cohere": CohereProvider,
            "tpu": TPUProvider
        }

        providers_config = self.config.get("providers", {})

        for provider_name, provider_config in providers_config.items():
            if provider_name in provider_classes:
                try:
                    provider = provider_classes[provider_name](provider_config)
                    if provider.is_available():
                        self.providers[provider_name] = provider
                        logger.info(f"✅ {provider_name} provider initialized")
                    else:
                        logger.info(f"⚠️  {provider_name} provider not available")
                except Exception as e:
                    logger.warning(f"Failed to initialize {provider_name}: {e}")

    async def generate_embedding(
        self,
        text: str,
        provider: Optional[str] = None
    ) -> Optional[EmbeddingResult]:
        """
        Generate embedding with automatic fallback

        Args:
            text: Text to embed
            provider: Specific provider to use (or None for primary + fallback)

        Returns:
            EmbeddingResult or None if all providers fail
        """
        # Try specific provider if requested
        if provider and provider in self.providers:
            result = await self.providers[provider].generate(text)
            if result:
                return result

        # Try primary provider
        if self.primary_provider in self.providers:
            result = await self.providers[self.primary_provider].generate(text)
            if result:
                return result

        # Try fallback chain
        for fallback_provider in self.fallback_chain:
            if fallback_provider in self.providers:
                result = await self.providers[fallback_provider].generate(text)
                if result:
                    logger.info(f"Fallback to {fallback_provider} succeeded")
                    return result

        logger.error("All embedding providers failed")
        return None

    async def benchmark_providers(
        self,
        test_texts: List[str],
        providers: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Benchmark multiple providers on test texts

        Args:
            test_texts: List of test strings
            providers: Providers to benchmark (None = all available)

        Returns:
            Dictionary with benchmark results
        """
        if providers is None:
            providers = list(self.providers.keys())

        results = {
            "providers": {},
            "test_count": len(test_texts),
            "timestamp": time.time()
        }

        for provider_name in providers:
            if provider_name not in self.providers:
                continue

            provider = self.providers[provider_name]
            provider_results = {
                "successful": 0,
                "failed": 0,
                "latencies_ms": [],
                "dimensions": None,
                "total_cost": 0.0
            }

            for text in test_texts:
                result = await provider.generate(text)
                if result:
                    provider_results["successful"] += 1
                    provider_results["latencies_ms"].append(result.latency_ms)
                    provider_results["dimensions"] = result.dimensions
                    if result.cost_estimate:
                        provider_results["total_cost"] += result.cost_estimate
                else:
                    provider_results["failed"] += 1

            # Calculate statistics
            if provider_results["latencies_ms"]:
                latencies = provider_results["latencies_ms"]
                provider_results["avg_latency_ms"] = sum(latencies) / len(latencies)
                provider_results["min_latency_ms"] = min(latencies)
                provider_results["max_latency_ms"] = max(latencies)
                provider_results["median_latency_ms"] = sorted(latencies)[len(latencies) // 2]

            results["providers"][provider_name] = provider_results
            logger.info(f"Benchmark {provider_name}: {provider_results['successful']}/{len(test_texts)} successful")

        return results

    def get_available_providers(self) -> List[str]:
        """Get list of available providers"""
        return list(self.providers.keys())

    def get_provider_info(self, provider_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific provider"""
        if provider_name not in self.providers:
            return None

        provider = self.providers[provider_name]
        return {
            "name": provider_name,
            "model": provider.config.get("model"),
            "dimensions": provider.config.get("dimensions"),
            "available": provider.is_available(),
            "local": provider.config.get("local", False)
        }
