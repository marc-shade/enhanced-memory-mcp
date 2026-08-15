#!/usr/bin/env python3
"""Regression tests for the ART tools.

Three ART tools (art_learn, art_get_categories, art_get_stats) raised
AttributeError on EVERY call, for months, because they read attributes FuzzyART
does not have:

    art.get_stats()   -> the method is get_statistics()
    art.stats         -> no such attribute; the data is in get_statistics()

Nothing caught it because nothing called them, and the MCP layer turns the
AttributeError into a tool-call error string rather than a crash. These tests
exercise the real objects so the failure is loud next time.

Run: python3 test_art_tools_regression.py
"""

import asyncio
import pathlib
import shutil
import sys
import tempfile

import numpy as np

from art_core import FuzzyART
import art_tools


class MockApp:
    """Minimal FastMCP stand-in: capture the registered tool functions."""

    def __init__(self):
        self.tools = {}

    def tool(self):
        def deco(fn):
            self.tools[fn.__name__] = fn
            return fn

        return deco


def test_fuzzyart_api_surface():
    """The API the tools depend on must exist. This is the root cause."""
    art = FuzzyART(vigilance=0.75, learning_rate=1.0)

    assert hasattr(art, "get_statistics"), "FuzzyART lost get_statistics()"
    stats = art.get_statistics()
    assert isinstance(stats, dict)
    assert "total_inputs" in stats, f"get_statistics() missing total_inputs: {stats}"

    # The two attributes the broken code assumed. If either ever appears, the
    # tools may be rewritten to use it, but today they must NOT be referenced.
    assert not hasattr(art, "get_stats"), (
        "FuzzyART now has get_stats(); art_tools was fixed to use get_statistics()"
    )
    assert not hasattr(art, "stats"), (
        "FuzzyART now has .stats; art_tools was fixed to use get_statistics()"
    )


def test_art_tools_do_not_reference_missing_attrs():
    """Guard the exact regression: no `.stats` / `.get_stats()` on an ART object."""
    src = open("art_tools.py").read()
    # Strip comments so the explanatory notes don't trip the check.
    code = "\n".join(
        line for line in src.splitlines() if not line.lstrip().startswith("#")
    )
    for bad in ("art.stats", "_art_instance.stats", ".get_stats()"):
        assert bad not in code, (
            f"art_tools.py references {bad!r}; FuzzyART has no such member"
        )


def test_art_tools_actually_run():
    """Call every ART tool that touches statistics. These all used to raise."""
    # Isolate from the production ART state under databases/art/. Without this
    # the tools load a persisted network whose input_dim is fixed by whatever
    # ran last, and the test both depends on and mutates live state.
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="art-regression-"))
    original_dir = art_tools.ART_STORAGE_DIR
    art_tools.ART_STORAGE_DIR = tmp
    art_tools._art_instance = None
    art_tools._art_hybrid_instance = None

    try:
        _run_art_tool_calls()
    finally:
        art_tools.ART_STORAGE_DIR = original_dir
        art_tools._art_instance = None
        art_tools._art_hybrid_instance = None
        shutil.rmtree(tmp, ignore_errors=True)


def _run_art_tool_calls():
    app = MockApp()
    art_tools.register_art_tools(app)

    for name in ("art_learn", "art_classify", "art_get_categories", "art_get_stats"):
        assert name in app.tools, f"{name} not registered"

    vec = list(np.random.RandomState(0).rand(8).astype(float))

    learned = asyncio.run(app.tools["art_learn"](data=vec))
    assert learned["success"] is True
    assert "total_patterns_learned" in learned
    assert learned["total_patterns_learned"] >= 1, learned

    classified = asyncio.run(app.tools["art_classify"](data=vec))
    assert classified["success"] is True

    cats = asyncio.run(app.tools["art_get_categories"]())
    assert cats["success"] is True
    assert cats["total_categories"] >= 1
    assert "total_inputs" in cats["stats"], cats["stats"]

    stats = asyncio.run(app.tools["art_get_stats"]())
    assert stats["success"] is True
    main = stats["instances"]["main"]
    assert main["initialized"] is True
    assert "total_inputs" in main["stats"], main["stats"]


def main():
    tests = [
        test_fuzzyart_api_surface,
        test_art_tools_do_not_reference_missing_attrs,
        test_art_tools_actually_run,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"  FAIL {t.__name__}: {type(e).__name__}: {e}")

    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
