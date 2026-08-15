#!/usr/bin/env python3
"""
Tests for server.py utility functions and core logic.

Tests compression, checksums, tier classification, database initialization,
and version management WITHOUT direct pickle usage (mocked for safety).
"""

import pytest
import sqlite3
import tempfile
import sys
import hashlib
from pathlib import Path
from unittest.mock import patch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestTierClassification:
    """Test classify_tier function for memory tier assignment."""

    def test_core_tier_system_role(self):
        """System role entities should be classified as core."""
        from server import classify_tier

        assert classify_tier("system_role", "any_name") == "core"

    def test_core_tier_core_system(self):
        """Core system entities should be classified as core."""
        from server import classify_tier

        assert classify_tier("core_system", "any_name") == "core"

    def test_core_tier_orchestrator_name(self):
        """Names containing 'orchestrator' should be classified as core."""
        from server import classify_tier

        assert classify_tier("agent", "main_orchestrator") == "core"
        assert classify_tier("agent", "ORCHESTRATOR_V2") == "core"

    def test_working_tier_project(self):
        """Project entities should be classified as working."""
        from server import classify_tier

        assert classify_tier("project", "my_project") == "working"

    def test_working_tier_session(self):
        """Session entities should be classified as working."""
        from server import classify_tier

        assert classify_tier("session", "session_123") == "working"

    def test_working_tier_current_name(self):
        """Names containing 'current' should be classified as working."""
        from server import classify_tier

        assert classify_tier("task", "current_task") == "working"

    def test_archive_tier_archive_name(self):
        """Names containing 'archive' should be classified as archive."""
        from server import classify_tier

        assert classify_tier("memory", "archive_2024") == "archive"

    def test_archive_tier_historical_type(self):
        """Historical entity types should be classified as archive."""
        from server import classify_tier

        assert classify_tier("historical_data", "old_records") == "archive"

    def test_reference_tier_default(self):
        """Unmatched entities should be classified as reference."""
        from server import classify_tier

        assert classify_tier("knowledge", "api_docs") == "reference"
        assert classify_tier("skill", "python_tips") == "reference"


class TestChecksum:
    """Test calculate_checksum function for data integrity."""

    def test_checksum_consistency(self):
        """Same data should produce same checksum."""
        from server import calculate_checksum

        data = b"test data for checksum"
        checksum1 = calculate_checksum(data)
        checksum2 = calculate_checksum(data)
        assert checksum1 == checksum2

    def test_checksum_format(self):
        """Checksum should be a valid SHA256 hex string."""
        from server import calculate_checksum

        checksum = calculate_checksum(b"test")
        assert len(checksum) == 64  # SHA256 produces 64 hex chars
        assert all(c in "0123456789abcdef" for c in checksum)

    def test_checksum_different_data(self):
        """Different data should produce different checksums."""
        from server import calculate_checksum

        checksum1 = calculate_checksum(b"data1")
        checksum2 = calculate_checksum(b"data2")
        assert checksum1 != checksum2

    def test_checksum_empty_data(self):
        """Empty data should produce valid checksum."""
        from server import calculate_checksum

        checksum = calculate_checksum(b"")
        assert len(checksum) == 64
        # Known SHA256 of empty string
        expected = hashlib.sha256(b"").hexdigest()
        assert checksum == expected

    def test_checksum_unicode_bytes(self):
        """Unicode encoded as bytes should work."""
        from server import calculate_checksum

        data = "Unicode: café ñ 日本語".encode("utf-8")
        checksum = calculate_checksum(data)
        assert len(checksum) == 64


class TestCompression:
    """Test compression functions with mocked serialization."""

    def test_compress_returns_tuple(self):
        """compress_data should return (compressed, original_size, compressed_size, ratio)."""
        from server import compress_data

        result = compress_data({"key": "value"})
        assert isinstance(result, tuple)
        assert len(result) == 4
        compressed, original_size, compressed_size, ratio = result
        assert isinstance(compressed, bytes)
        assert isinstance(original_size, int)
        assert isinstance(compressed_size, int)
        assert isinstance(ratio, float)

    def test_compress_ratio_calculation(self):
        """Compression ratio should be compressed_size / original_size."""
        from server import compress_data

        # Use larger data for better compression
        data = {"content": "x" * 1000}
        compressed, original_size, compressed_size, ratio = compress_data(data)
        expected_ratio = compressed_size / original_size
        assert abs(ratio - expected_ratio) < 0.001

    def test_compress_decompress_roundtrip(self):
        """Data should survive compress -> decompress cycle."""
        from server import compress_data, decompress_data

        original = {"key": "value", "numbers": [1, 2, 3], "nested": {"a": 1}}
        compressed, _, _, _ = compress_data(original)
        decompressed = decompress_data(compressed)
        assert decompressed == original

    def test_decompress_invalid_data(self):
        """Decompressing invalid data should raise error."""
        from server import decompress_data

        with pytest.raises(Exception):  # zlib.error or other
            decompress_data(b"not valid compressed data")


class TestStoragePath:
    """Test _get_storage_base, which locates an optional surrounding checkout.

    This replaced a pair of tests that mocked platform.system() and then
    asserted nothing at all -- they passed no matter what the function did, and
    they described a platform-detection branch the function no longer has.
    """

    def test_storage_path_from_environment(self, monkeypatch):
        """AGENTIC_SYSTEM_PATH wins when set."""
        from server import _get_storage_base

        monkeypatch.setenv("AGENTIC_SYSTEM_PATH", "/opt/some/checkout")
        assert _get_storage_base() == Path("/opt/some/checkout")

    def test_storage_path_expands_user_and_vars(self, monkeypatch):
        """The override is expanded, so ~ and $VAR in a config are usable."""
        from server import _get_storage_base

        monkeypatch.setenv("AGENTIC_SYSTEM_PATH", "~/checkout")
        assert _get_storage_base() == Path.home() / "checkout"

        monkeypatch.setenv("EM_TEST_BASE", "/expanded")
        monkeypatch.setenv("AGENTIC_SYSTEM_PATH", "$EM_TEST_BASE/checkout")
        assert _get_storage_base() == Path("/expanded/checkout")

    def test_storage_path_falls_back_to_repo_relative(self, monkeypatch):
        """With no override, the path is derived from the source file.

        Nothing platform-specific, no baked-in path: a checkout in any
        directory resolves relative to server.py.
        """
        import server
        from server import _get_storage_base

        monkeypatch.delenv("AGENTIC_SYSTEM_PATH", raising=False)
        expected = Path(server.__file__).resolve().parent.parent.parent
        assert _get_storage_base() == expected


class TestDatabaseInit:
    """Test database initialization and schema."""

    def test_init_creates_tables(self):
        """init_database should create all required tables."""
        from server import init_database

        # Use temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_db = Path(tmpdir) / "test_memory.db"

            with (
                patch("server.DB_PATH", tmp_db),
                patch("server.MEMORY_DIR", Path(tmpdir)),
            ):
                # Create parent directory
                Path(tmpdir).mkdir(parents=True, exist_ok=True)

                # Initialize
                init_database()

                # Verify tables exist
                conn = sqlite3.connect(tmp_db)
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT name FROM sqlite_master
                    WHERE type='table'
                    ORDER BY name
                """)
                tables = [row[0] for row in cursor.fetchall()]

                expected_tables = [
                    "entities",
                    "implementation_plans",
                    "memory_branches",
                    "memory_conflicts",
                    "memory_versions",
                    "observations",
                    "project_handbooks",
                    "relations",
                ]

                for table in expected_tables:
                    assert table in tables, f"Missing table: {table}"

                conn.close()

    def test_init_creates_indexes(self):
        """init_database should create performance indexes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_db = Path(tmpdir) / "test_memory.db"

            with (
                patch("server.DB_PATH", tmp_db),
                patch("server.MEMORY_DIR", Path(tmpdir)),
            ):
                from server import init_database

                init_database()

                conn = sqlite3.connect(tmp_db)
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT name FROM sqlite_master
                    WHERE type='index' AND name LIKE 'idx_%'
                """)
                indexes = [row[0] for row in cursor.fetchall()]

                expected_indexes = [
                    "idx_entities_name",
                    "idx_entities_type",
                    "idx_entities_accessed",
                    "idx_versions_entity",
                    "idx_versions_branch",
                ]

                for index in expected_indexes:
                    assert index in indexes, f"Missing index: {index}"

                conn.close()

    def test_init_idempotent(self):
        """Multiple init_database calls should be safe."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_db = Path(tmpdir) / "test_memory.db"

            with (
                patch("server.DB_PATH", tmp_db),
                patch("server.MEMORY_DIR", Path(tmpdir)),
            ):
                from server import init_database

                # Call twice - should not error
                init_database()
                init_database()

                # Tables should still exist
                conn = sqlite3.connect(tmp_db)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
                count = cursor.fetchone()[0]
                assert count >= 6  # At least our main tables
                conn.close()


class TestVersionManagement:
    """Test version creation and management."""

    def test_create_version_increments(self):
        """Each new version should increment version number."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_db = Path(tmpdir) / "test_memory.db"

            with patch("server.DB_PATH", tmp_db):
                from server import init_database, create_version

                init_database()

                # Create test entity
                conn = sqlite3.connect(tmp_db)
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO entities (name, entity_type, tier)
                    VALUES (?, ?, ?)
                """,
                    ("test_entity", "test", "working"),
                )
                entity_id = cursor.lastrowid
                conn.commit()
                conn.close()

                # Create versions
                v1 = create_version(entity_id, {"v": 1}, "First version")
                v2 = create_version(entity_id, {"v": 2}, "Second version")

                # Check version numbers
                conn = sqlite3.connect(tmp_db)
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT version_number FROM memory_versions
                    WHERE entity_id = ? ORDER BY version_number
                """,
                    (entity_id,),
                )
                versions = [row[0] for row in cursor.fetchall()]
                conn.close()

                assert versions == [1, 2]

    def test_create_version_marks_current(self):
        """Newest version should be marked as current."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_db = Path(tmpdir) / "test_memory.db"

            with patch("server.DB_PATH", tmp_db):
                from server import init_database, create_version

                init_database()

                # Create test entity
                conn = sqlite3.connect(tmp_db)
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO entities (name, entity_type, tier)
                    VALUES (?, ?, ?)
                """,
                    ("test_entity", "test", "working"),
                )
                entity_id = cursor.lastrowid
                conn.commit()
                conn.close()

                # Create multiple versions
                create_version(entity_id, {"v": 1}, "First")
                create_version(entity_id, {"v": 2}, "Second")

                # Check only latest is current
                conn = sqlite3.connect(tmp_db)
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT version_number, is_current FROM memory_versions
                    WHERE entity_id = ? ORDER BY version_number
                """,
                    (entity_id,),
                )
                results = cursor.fetchall()
                conn.close()

                assert results[0] == (1, 0)  # First version not current
                assert results[1] == (2, 1)  # Second version is current


class TestFallbackScoring:
    """Test fallback scoring when TPU unavailable."""

    def test_fallback_score_importance_base(self):
        """Fallback should return base score for normal text."""
        # Import the fallback function
        import server

        if not server.TPU_SCORING_AVAILABLE:
            score = server.score_importance("normal text content")
            assert 0.0 <= score <= 1.0
            assert score == 0.3  # Base score

    def test_fallback_score_importance_keywords(self):
        """Fallback should boost score for important keywords."""
        import server

        if not server.TPU_SCORING_AVAILABLE:
            score = server.score_importance("critical error detected")
            assert score > 0.3  # Should be higher than base
            assert score <= 1.0

    def test_fallback_is_tpu_available(self):
        """Fallback is_tpu_available should return False."""
        import server

        if not server.TPU_SCORING_AVAILABLE:
            assert server.is_tpu_available() == False


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_compress_empty_dict(self):
        """Should handle empty dictionary."""
        from server import compress_data, decompress_data

        original = {}
        compressed, orig_size, comp_size, ratio = compress_data(original)
        assert decompress_data(compressed) == original

    def test_compress_nested_structure(self):
        """Should handle deeply nested data."""
        from server import compress_data, decompress_data

        original = {"level1": {"level2": {"level3": {"data": [1, 2, 3]}}}}
        compressed, _, _, _ = compress_data(original)
        assert decompress_data(compressed) == original

    def test_compress_special_characters(self):
        """Should handle special characters."""
        from server import compress_data, decompress_data

        original = {
            "unicode": "日本語 中文 한국어",
            "symbols": '!@#$%^&*(){}[]|\\:";<>?,./',
            "newlines": "line1\nline2\rline3\r\n",
        }
        compressed, _, _, _ = compress_data(original)
        assert decompress_data(compressed) == original

    def test_classify_tier_case_insensitive(self):
        """Tier classification should be case insensitive for keywords."""
        from server import classify_tier

        assert classify_tier("agent", "ORCHESTRATOR") == "core"
        assert classify_tier("task", "CURRENT_WORK") == "working"
        assert classify_tier("data", "ARCHIVE_2024") == "archive"

    def test_checksum_large_data(self):
        """Should handle large data efficiently."""
        from server import calculate_checksum

        large_data = b"x" * (1024 * 1024)  # 1MB
        checksum = calculate_checksum(large_data)
        assert len(checksum) == 64


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_entity_lifecycle(self):
        """Test creating entity, versioning, and tier classification."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_db = Path(tmpdir) / "test_memory.db"

            with patch("server.DB_PATH", tmp_db):
                from server import (
                    init_database,
                    compress_data,
                    calculate_checksum,
                    classify_tier,
                    create_version,
                )

                init_database()

                # Create entity data
                entity_data = {
                    "name": "test_project",
                    "content": "Important project information",
                    "metadata": {"created": "2024-01-01"},
                }

                # Compress data
                compressed, orig_size, comp_size, ratio = compress_data(entity_data)

                # Calculate checksum
                checksum = calculate_checksum(compressed)

                # Classify tier
                tier = classify_tier("project", "current_test_project")

                # Store in database
                conn = sqlite3.connect(tmp_db)
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO entities
                    (name, entity_type, tier, compressed_data,
                     original_size, compressed_size, compression_ratio, checksum)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        "test_project",
                        "project",
                        tier,
                        compressed,
                        orig_size,
                        comp_size,
                        ratio,
                        checksum,
                    ),
                )
                entity_id = cursor.lastrowid
                conn.commit()
                conn.close()

                # Create versions
                v1 = create_version(entity_id, entity_data, "Initial version")
                entity_data["content"] = "Updated project information"
                v2 = create_version(entity_id, entity_data, "Content update")

                # Verify
                conn = sqlite3.connect(tmp_db)
                cursor = conn.cursor()

                cursor.execute("SELECT tier FROM entities WHERE id = ?", (entity_id,))
                assert cursor.fetchone()[0] == "working"

                cursor.execute(
                    "SELECT COUNT(*) FROM memory_versions WHERE entity_id = ?",
                    (entity_id,),
                )
                assert cursor.fetchone()[0] == 2

                conn.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
