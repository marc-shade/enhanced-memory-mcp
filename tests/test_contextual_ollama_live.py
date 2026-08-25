"""Live contextual enrichment against a real ollama daemon (opt-in).

The unit tests in test_contextual_ollama.py stub the HTTP call. This one
does not: it asks the daemon at ENRICHMENT_OLLAMA_URL for a real prefix and
checks the contract on a real answer. It is opt-in so the default suite stays
hermetic; run it with

    ENHANCED_MEMORY_LIVE_OLLAMA=1 python -m pytest tests/test_contextual_ollama_live.py

When opted in, an unreachable daemon or a missing model is a FAILURE, not a
skip: the point of opting in is to learn that.

Gaps / not covered: prefix quality is not judged, only shape and backend.
"""

import json
import os
import urllib.error
import urllib.request

import pytest

import contextual_llm
from contextual_llm import ContextualPrefixGenerator

LIVE = os.environ.get("ENHANCED_MEMORY_LIVE_OLLAMA") == "1"

pytestmark = pytest.mark.skipif(
    not LIVE, reason="set ENHANCED_MEMORY_LIVE_OLLAMA=1 to run against a real ollama"
)


def _models(url: str) -> list:
    with urllib.request.urlopen(url + "/api/tags", timeout=10) as resp:
        return [m["name"] for m in json.loads(resp.read())["models"]]


@pytest.mark.asyncio
async def test_live_ollama_generates_a_context_prefix(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    gen = ContextualPrefixGenerator()
    gen.client = None
    try:
        models = _models(gen.ollama_url)
    except (urllib.error.URLError, OSError) as exc:
        pytest.fail(f"ollama unreachable at {gen.ollama_url}: {exc}")
    assert gen.ollama_model in models, (
        f"{gen.ollama_model} is not pulled on {gen.ollama_url}; available: {models}"
    )

    prefix, tin, tout = await gen.generate_prefix(
        "memory-db-socket-guard",
        "auto_memory/project",
        [
            "socket_guard.py probes a live socket before unlinking it",
            "a refused daemon never unlinks a socket it did not bind",
        ],
    )

    assert gen._backend == "ollama", f"fell back to template: {prefix}"
    assert prefix.startswith("[Context: ") and prefix.endswith("]")
    assert 20 < len(prefix) <= 310
    assert not prefix.startswith("[Context: This is a"), "that is the template skeleton"
    assert tin > 0 and tout > 0
    stats = gen.get_stats()
    assert stats["backend"] == "ollama" and stats["using_fallback"] is False
    assert contextual_llm.OLLAMA_MODEL == gen.ollama_model
