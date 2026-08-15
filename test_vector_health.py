#!/usr/bin/env python3
"""Health-helper test for the vector/semantic honest-degradation guard.

Asserts that `check_vector_semantic_health()` reports the REAL state of the
running interpreter (it must agree with a direct import probe), that the
structured-degradation contract is well-formed, and that the GuardedToolApp
short-circuits a tool when its precheck reports the backend unavailable while
delegating (and preserving the real return) when the precheck passes.

Run: <interpreter> test_vector_health.py
Exit code = number of failures.
"""

import asyncio
import importlib.util
import sys

import vector_health as vh

FAILURES = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global FAILURES
    if not cond:
        FAILURES += 1
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('' if cond else ' :: ' + detail)}")


def _truly_importable(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except Exception:
        return False


def test_health_matches_reality() -> None:
    print("== health report matches real interpreter state ==")
    h = vh.check_vector_semantic_health()

    check(
        "status is a known value",
        h["status"] in ("available", "degraded", "unavailable"),
        str(h["status"]),
    )
    check(
        "interpreter is this process",
        h["interpreter"] == sys.executable,
        h["interpreter"],
    )

    # The helper's importability verdict must match a direct find_spec probe —
    # this is the anti-fabrication assertion: the report reflects reality.
    real_qc = _truly_importable("qdrant_client")
    real_st = _truly_importable("sentence_transformers")
    check(
        "qdrant_client verdict matches find_spec",
        h["checks"]["qdrant_client_importable"]["available"] == real_qc,
        f"reported={h['checks']['qdrant_client_importable']['available']} real={real_qc}",
    )
    check(
        "sentence_transformers verdict matches find_spec",
        h["checks"]["sentence_transformers_importable"]["available"] == real_st,
        f"reported={h['checks']['sentence_transformers_importable']['available']} real={real_st}",
    )

    # Overall status must be internally consistent with the sub-checks.
    if real_qc and real_st and h["checks"]["qdrant_reachable"]["available"]:
        check(
            "status available when all core deps+server up",
            h["status"] == "available",
            str(h["status"]),
        )
    elif not real_qc and not real_st:
        check(
            "status unavailable when no deps",
            h["status"] == "unavailable",
            str(h["status"]),
        )
    else:
        check(
            "status degraded on partial availability",
            h["status"] == "degraded",
            str(h["status"]),
        )


def test_degraded_payload_shape() -> None:
    print("\n== degraded payload shape ==")
    p = vh.degraded_payload(
        "test reason", status="unavailable", missing=["qdrant-client"]
    )
    check("has status", p.get("status") == "unavailable")
    check("has reason", p.get("reason") == "test reason")
    check("remediation names pip install", "pip install" in p.get("remediation", ""))
    check("remediation names interpreter", sys.executable in p.get("remediation", ""))
    check("missing list normalized", p.get("missing") == ["qdrant-client"])


def test_guard_short_circuits_and_delegates() -> None:
    print("\n== GuardedToolApp behaviour ==")

    class MockApp:
        def __init__(self):
            self.tools = {}

        def tool(self):
            def d(fn):
                self.tools[fn.__name__] = fn
                return fn

            return d

    # Precheck that always blocks -> tool must NOT execute, must return the dict.
    blocked_payload = vh.degraded_payload("forced", status="unavailable")

    def always_block():
        return blocked_payload

    app = MockApp()
    guarded = vh.GuardedToolApp(app, always_block)

    executed = {"hit": False}

    @guarded.tool()
    async def sample_tool(x: int = 1):
        "doc"
        executed["hit"] = True
        return {"real": x}

    out = asyncio.run(app.tools["sample_tool"](5))
    check("blocked precheck short-circuits", out is blocked_payload, str(out))
    check("real tool body NOT executed when blocked", executed["hit"] is False)

    # Precheck that passes -> tool executes and its real return is preserved.
    app2 = MockApp()
    guarded2 = vh.GuardedToolApp(app2, lambda: None)

    @guarded2.tool()
    async def sample_tool2(x: int = 1):
        "doc"
        return {"real": x}

    out2 = asyncio.run(app2.tools["sample_tool2"](7))
    check("passing precheck delegates to real tool", out2 == {"real": 7}, str(out2))

    # functools.wraps identity preserved (needed for FastMCP schema gen).
    check("wrapped name preserved", app.tools["sample_tool"].__name__ == "sample_tool")
    check("wrapped doc preserved", app.tools["sample_tool"].__doc__ == "doc")


def main() -> int:
    test_health_matches_reality()
    test_degraded_payload_shape()
    test_guard_short_circuits_and_delegates()
    print(f"\n{'=' * 50}")
    print(
        f"{'ALL PASS' if FAILURES == 0 else str(FAILURES) + ' FAILURE(S)'} — vector_health suite"
    )
    return FAILURES


if __name__ == "__main__":
    sys.exit(main())
