#!/usr/bin/env python3
"""
Integration Tests for Code Execution Pattern

Tests end-to-end functionality:
1. Code execution in sandbox
2. API access from executed code
3. Security validation
4. Token savings
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sandbox.executor import CodeExecutor, create_api_context
from sandbox.security import comprehensive_safety_check, sanitize_output


def test_basic_execution():
    """Test basic code execution"""
    print("\n=== Test 1: Basic Execution ===")

    executor = CodeExecutor()

    code = """
result = sum(range(100))
result = result
    """

    result = executor.execute(code)

    assert result.success, f"Execution failed: {result.error}"
    assert result.result == 4950, f"Expected 4950, got {result.result}"

    print(f"✅ Basic execution: {result.result}")
    print(f"   Execution time: {result.execution_time_ms:.2f}ms")


def test_api_access(live_memory):
    """Test API access from executed code.

    Runs against the throwaway daemon from the `live_memory` fixture. Without
    it, `get_status()` reaches whatever daemon owns /tmp/memory-db.sock, which
    on a developer machine is the operator's real memory store -- this test
    used to read and report on 11,952 production entities.
    """
    print("\n=== Test 2: API Access ===")

    executor = CodeExecutor()
    context = create_api_context()

    code = """
status = get_status()
result = {
    "entity_total": status["entities"]["total"],
    "database_path": status["database_path"],
    "test": "api_access_works"
}
    """

    result = executor.execute(code, context=context)

    assert result.success, f"Execution failed: {result.error}"
    assert result.result["test"] == "api_access_works"
    # The status has to describe THIS database, not some other one.
    assert result.result["database_path"] == str(live_memory), (
        f"api reached {result.result['database_path']}, expected {live_memory}"
    )
    assert result.result["entity_total"] == 0, (
        f"throwaway database should be empty, got {result.result['entity_total']}"
    )

    print(f"✅ API access: {result.result}")
    print(f"   Execution time: {result.execution_time_ms:.2f}ms")


def test_api_search_filters(live_memory):
    """search_nodes must apply entity_type and min_confidence filters.

    api.memory.search_nodes treated the daemon's response envelope
    ({"success":..., "results":[...]}) as a bare list. Passing either filter
    iterated the envelope's string keys and raised
    "'str' object has no attribute 'get'". The unfiltered path returned the
    envelope instead of the rows. Neither was reachable from the older test,
    which only called get_status().
    """
    from api import memory

    memory.create_entities(
        [
            {
                "name": "filter_probe_project",
                "entityType": "project",
                "observations": ["filter probe: a project entity"],
            },
            {
                "name": "filter_probe_person",
                "entityType": "person",
                "observations": ["filter probe: a person entity"],
            },
        ]
    )

    unfiltered = memory.search_nodes("filter_probe")
    assert isinstance(unfiltered, list), (
        f"expected the rows, got {type(unfiltered).__name__}: {unfiltered!r}"
    )
    assert {row["name"] for row in unfiltered} == {
        "filter_probe_project",
        "filter_probe_person",
    }

    by_type = memory.search_nodes("filter_probe", entity_type="project")
    assert [row["name"] for row in by_type] == ["filter_probe_project"], (
        f"entity_type filter returned {by_type!r}"
    )

    # Every row carries a confidence; a threshold above all of them empties the
    # list, which is a real empty result rather than an error.
    assert memory.search_nodes("filter_probe", min_confidence=1.1) == []

    print("✅ search filters: entity_type and min_confidence both applied")


def test_update_entity_honours_configured_database(live_memory, monkeypatch):
    """update_entity must write where the configuration points, not to ~/.claude.

    This is the one memory path that opens SQLite directly instead of going
    through the daemon socket, so the socket isolation in conftest cannot
    contain it. Until 2026-08-14 it resolved
    Path.home()/".claude"/"enhanced_memories"/"memory.db" itself, ignoring every
    override, which made it a WRITE into the operator's real store from any
    test, second instance, or container with a database mounted elsewhere.

    The production-database guard in conftest would also fail this test if the
    write went to the real path, but that guard raises on a symptom. This
    asserts the intended behaviour: the row lands in the configured database.
    """
    import sqlite3

    from api import memory

    monkeypatch.setenv("ENHANCED_MEMORY_DB_PATH", str(live_memory))

    memory.create_entities(
        [
            {
                "name": "update_probe",
                "entityType": "test",
                "observations": ["original observation"],
            }
        ]
    )
    result = memory.update_entity("update_probe", ["added by update_entity"])
    assert result["observations_added"] == 1

    conn = sqlite3.connect(live_memory)
    try:
        rows = conn.execute(
            "SELECT content FROM observations o JOIN entities e ON e.id = o.entity_id "
            "WHERE e.name = ?",
            ("update_probe",),
        ).fetchall()
    finally:
        conn.close()

    contents = [row[0] for row in rows]
    assert "added by update_entity" in contents, (
        f"observation did not land in the configured database: {contents}"
    )

    print(f"✅ update_entity wrote to the configured database: {live_memory}")


def test_security_blocking():
    """Test security checks block dangerous code"""
    print("\n=== Test 3: Security Blocking ===")

    dangerous_codes = [
        ("import os", "Dangerous import"),
        ("eval('1+1')", "eval() usage"),
        ("open('/etc/passwd')", "File access"),
    ]

    for code, description in dangerous_codes:
        is_safe, issues = comprehensive_safety_check(code)
        assert not is_safe, f"Security check should have blocked: {description}"
        print(f"✅ Blocked: {description}")


def test_timeout():
    """Test timeout enforcement.

    The work has to be a busy loop, not `import time; time.sleep(10)`. The
    sandbox blocks __import__, so the import version failed with
    "ImportError: __import__ not found" in well under a millisecond and the
    old assertion accepted it because the word "time" appears in
    "ImportError: __import__ not found".lower(). The timeout was never
    exercised; the test would have passed with timeout enforcement removed
    entirely.
    """
    print("\n=== Test 4: Timeout Enforcement ===")

    timeout_seconds = 2
    executor = CodeExecutor(timeout_seconds=timeout_seconds)

    code = """
total = 0
for i in range(10 ** 9):
    total = total + i
result = total
    """

    start = time.monotonic()
    result = executor.execute(code)
    elapsed = time.monotonic() - start

    assert not result.success, "Should have timed out"
    assert "Timeout" in result.error, f"Expected a timeout, got: {result.error}"
    # The interruption has to be the timeout firing, not the work finishing:
    # the loop must actually have been cut off at roughly the limit.
    assert elapsed >= timeout_seconds, (
        f"Returned after {elapsed:.2f}s, before the limit"
    )
    assert elapsed < timeout_seconds + 5, (
        f"Timeout did not interrupt promptly ({elapsed:.2f}s)"
    )

    print(f"✅ Timeout enforced after {elapsed:.2f}s: {result.error}")


def test_token_savings():
    """Test token savings through local filtering"""
    print("\n=== Test 5: Token Savings ===")

    executor = CodeExecutor()
    context = create_api_context()

    # Confidences are i/100, so they span [0.0, 0.99] as a confidence score
    # actually does. The fixture used i/10, which runs to 9.9 and put 92 of the
    # 100 items over the 0.8 threshold -- the test then asserted 20 and failed.
    # With a real 0-1 range, the threshold selects i >= 80, which is 20 items.
    code = """
results = [
    {"name": f"item_{i}", "confidence": i/100, "entityType": "test"}
    for i in range(100)
]

high_conf = filter_by_confidence(results, 0.8)

summary = summarize_results(high_conf)

result = {
    "filtered_count": summary["count"],
    "avg_confidence": summary["avg_confidence"]
}
    """

    result = executor.execute(code, context=context)

    assert result.success, f"Execution failed: {result.error}"
    assert result.result["filtered_count"] == 20
    assert result.result["avg_confidence"] > 0.8

    before_tokens = 100 * 500
    after_tokens = len(str(result.result))
    savings_pct = ((before_tokens - after_tokens) / before_tokens) * 100

    print("✅ Token savings demonstration:")
    print(f"   Before: ~{before_tokens:,} tokens (100 full results)")
    print(f"   After: ~{after_tokens} tokens (summary only)")
    print(f"   Savings: {savings_pct:.1f}%")


def test_pii_tokenization():
    """Test PII tokenization in output"""
    print("\n=== Test 6: PII Tokenization ===")

    executor = CodeExecutor()

    code = """
data = {
    "email": "user@example.com",
    "ssn": "123-45-6789",
    "phone": "555-123-4567"
}
result = data
    """

    result = executor.execute(code)

    sanitized = sanitize_output(result.result)

    assert "[EMAIL]" in str(sanitized.values())
    assert "[SSN]" in str(sanitized.values())
    assert "[PHONE]" in str(sanitized.values())

    print("✅ PII tokenization:")
    print(f"   Original: {result.result}")
    print(f"   Sanitized: {sanitized}")


def test_error_handling():
    """Test error handling in executed code"""
    print("\n=== Test 7: Error Handling ===")

    executor = CodeExecutor()

    code = """
result = 1 / 0
result = result
    """

    result = executor.execute(code)

    assert not result.success, "Should have failed with division by zero"
    assert "ZeroDivisionError" in result.error

    print(f"✅ Error handled: {result.error}")


def run_all_tests():
    """Run all integration tests"""
    print("=" * 60)
    print("Enhanced Memory MCP - Code Execution Integration Tests")
    print("=" * 60)

    tests = [
        test_basic_execution,
        test_api_access,
        test_security_blocking,
        test_timeout,
        test_token_savings,
        test_pii_tokenization,
        test_error_handling,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} failed: {str(e)}")
            failed += 1
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
