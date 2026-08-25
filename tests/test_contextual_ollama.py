"""The contextual prefix comes from local ollama, or honestly from the template.

`create_entities` prepends one `[Context: ...]` observation per entity. The
Anthropic branch is dead in every deployment that follows the no-SDK policy,
so before 2026-08-24 every prefix ever stored was the template while the
stats claimed nothing about which branch ran. The ollama branch is the
compliant LLM path; these tests pin its contract:

- ollama answers -> prefix is the model's sentence wrapped as `[Context: ...]`,
  `backend == "ollama"`, token counts come from the response.
- ollama unreachable, or answers with nothing -> the template prefix,
  `backend == "template"`, `using_fallback` true. Never an exception, never
  an LLM claim.

The HTTP call is stubbed at `urllib.request.urlopen`; the dead-port case
binds and releases a real local port so the refusal is real.

Gaps / not covered: no live ollama is contacted, so model availability and
the 30 s timeout are not exercised here; `comprehensive_test.py` reports the
backend that actually ran on the machine it runs on.
"""

import io
import json
import socket
import urllib.request

import pytest

import contextual_llm
from contextual_llm import ContextualPrefixGenerator


@pytest.fixture
def gen(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    g = ContextualPrefixGenerator()
    g.client = None  # the SDK branch must not run, key or no key
    return g


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _fake_urlopen(payload):
    calls = []

    def urlopen(req, timeout=None):
        calls.append(json.loads(req.data))
        return _Resp(json.dumps(payload).encode())

    return urlopen, calls


@pytest.mark.asyncio
async def test_ollama_answer_becomes_the_context_prefix(gen, monkeypatch):
    urlopen, calls = _fake_urlopen(
        {
            "response": ' "Notes on the retry policy of the job runner." ',
            "prompt_eval_count": 57,
            "eval_count": 11,
        }
    )
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)

    prefix, tin, tout = await gen.generate_prefix(
        "job-runner-retries",
        "note",
        ["retries are capped at 3", "backoff is exponential"],
    )

    assert prefix == "[Context: Notes on the retry policy of the job runner.]"
    assert (tin, tout) == (57, 11)
    assert gen._backend == "ollama"
    stats = gen.get_stats()
    assert stats["backend"] == "ollama"
    assert stats["using_fallback"] is False
    assert stats["total_input_tokens"] == 57
    sent = calls[0]
    assert sent["model"] == gen.ollama_model
    assert sent["think"] is False, "thinking models return empty text otherwise"
    assert "job-runner-retries" in sent["prompt"]


@pytest.mark.asyncio
async def test_prefix_already_in_context_form_is_not_double_wrapped(gen, monkeypatch):
    urlopen, _ = _fake_urlopen({"response": "[Context: already wrapped]"})
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    prefix, _, _ = await gen.generate_prefix("e", "note", ["x"])
    assert prefix == "[Context: already wrapped]"


@pytest.mark.asyncio
async def test_empty_ollama_response_falls_back_to_template(gen, monkeypatch):
    urlopen, _ = _fake_urlopen({"response": "", "eval_count": 80})
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)

    prefix, tin, tout = await gen.generate_prefix("e", "note", ["one observation"])

    assert prefix.startswith("[Context: This is a note entity named 'e'")
    assert (tin, tout) == (0, 0)
    assert gen._backend == "template"
    assert gen.get_stats()["using_fallback"] is True


@pytest.mark.asyncio
async def test_unreachable_ollama_falls_back_to_template(gen):
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()  # nothing listens here now; the refusal is real
    gen.ollama_url = f"http://127.0.0.1:{port}"

    prefix, _, _ = await gen.generate_prefix("e", "note", ["one observation"])

    assert prefix.startswith("[Context: This is a note entity named 'e'")
    assert gen.get_stats()["backend"] == "template"


def test_module_defaults_come_from_the_environment(monkeypatch):
    assert contextual_llm.OLLAMA_URL.startswith("http")
    assert contextual_llm.OLLAMA_MODEL
    g = ContextualPrefixGenerator()
    assert g.ollama_url == contextual_llm.OLLAMA_URL
    assert g.ollama_model == contextual_llm.OLLAMA_MODEL
    assert g.get_stats()["backend"] in ("untried", "anthropic")
