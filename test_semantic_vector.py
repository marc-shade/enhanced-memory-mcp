#!/usr/bin/env python3
"""Regression test for the native semantic_recall MCP tool + honest-degradation
guard.

Two contracts, depending on the running interpreter's environment:

  (A) Deps present + Qdrant reachable: `semantic_recall` returns REAL vector
      results from the Qdrant `enhanced_memory` collection (proves vector search,
      not substring). Fails loudly if it no-ops or stops surfacing matches.

  (B) Deps missing OR Qdrant unreachable: the GuardedToolApp wrapper makes
      `semantic_recall` return a STRUCTURED {status, reason, remediation} dict
      instead of leaking a ModuleNotFoundError / traceback / legacy {error}
      string. This is the honest-degradation contract.

Run: <interpreter> test_semantic_vector.py
"""

import asyncio
import json
import sys

from semantic_vector_tools import register_semantic_vector_tools
from vector_health import (
    GuardedToolApp,
    _vector_precheck,
    check_vector_semantic_health,
)


class _MockApp:
    def __init__(self):
        self.tools = {}

    def tool(self):
        def deco(fn):
            self.tools[fn.__name__] = fn
            return fn

        return deco


def test_registration_through_guard():
    """Tools registered THROUGH the guard proxy still register by their real
    name (functools.wraps preserves identity)."""
    app = _MockApp()
    register_semantic_vector_tools(GuardedToolApp(app, _vector_precheck))
    assert "semantic_recall" in app.tools, "semantic_recall not registered via guard"
    return app


def test_returns_semantic_match(app):
    # A query with NO lexical overlap with the target filename — must still
    # surface it by meaning (proves vector search, not substring).
    out = asyncio.run(
        app.tools["semantic_recall"]("rule against using em dashes in prose", 3)
    )
    # Healthy path returns the original JSON string from the underlying tool.
    data = json.loads(out)
    assert "error" not in data, f"tool errored: {data.get('error')}"
    assert data["count"] > 0, (
        "no results — vector index empty or embedding backend down"
    )
    names = [r["name"] for r in data["results"]]
    assert any(
        "em-dash" in (n or "").lower() or "em_dash" in (n or "").lower() for n in names
    ), f"expected the em-dashes memory in results, got {names}"
    return data


def test_returns_structured_degraded(app):
    """When the backend is unavailable the guard must return a STRUCTURED dict
    (not raise, not a legacy {error} JSON string, not a bare traceback)."""
    out = asyncio.run(app.tools["semantic_recall"]("anything", 3))
    assert isinstance(out, dict), (
        f"degraded contract must return a dict, got {type(out).__name__}: {out!r}"
    )
    assert out.get("status") in ("unavailable", "degraded"), (
        f"missing/invalid status: {out}"
    )
    assert out.get("reason"), "degraded payload missing 'reason'"
    assert out.get("remediation"), "degraded payload missing 'remediation'"
    assert "pip install" in out["remediation"], (
        f"remediation should name the install command, got {out['remediation']}"
    )
    return out


if __name__ == "__main__":
    app = test_registration_through_guard()
    print("PASS: registration (through guard)")

    health = check_vector_semantic_health()
    print(
        f"interpreter={health['interpreter']} status={health['status']} "
        f"vector_ready={health['vector_recall_ready']}"
    )

    if health["vector_recall_ready"]:
        data = test_returns_semantic_match(app)
        print(
            f"PASS: semantic match (count={data['count']}, "
            f"top={data['results'][0]['name']} score={data['results'][0]['score']})"
        )
    else:
        payload = test_returns_structured_degraded(app)
        print(
            f"PASS: honest degradation (status={payload['status']}, "
            f"missing={payload.get('missing')})"
        )
    sys.exit(0)
