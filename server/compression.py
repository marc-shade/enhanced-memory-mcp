"""
Compression and data utilities for Enhanced Memory MCP Server.

Extracted from server.py for better organization.

SECURITY: New data is serialized with JSON (format marker 0x01) to eliminate
pickle deserialization as an attack vector. Legacy pickle data (marker 0x00 or
no marker) is still readable for backward compatibility, but only when it passes
an HMAC-SHA256 integrity check (or when no HMAC exists yet for pre-migration
data, in which case a deprecation warning is logged).

Format of compressed blob:
    [zlib-compressed payload]
    -- where payload is:
       0x01 + JSON bytes   (new format, safe)
       0x00 + pickle bytes (legacy with HMAC appended after zlib block)
       raw pickle bytes    (oldest legacy, no marker, no HMAC)

When legacy pickle data is written, an HMAC-SHA256 tag (32 bytes) is appended
AFTER the zlib-compressed block so that the zlib stream itself is untouched.
"""

import hashlib
import hmac
import json
import logging
import os
import pickle  # Only used for reading legacy data -- nosec B403
import secrets
import stat
import zlib
from pathlib import Path
from typing import Any, Tuple

logger = logging.getLogger("enhanced-memory.compression")

# --- Format markers (first byte after zlib decompression) ---
_FORMAT_JSON: bytes = b'\x01'
_FORMAT_PICKLE_LEGACY: bytes = b'\x00'

# --- HMAC key management ---
_HMAC_KEY_PATH = Path.home() / ".claude" / "enhanced_memories" / ".compression_hmac_key"
_HMAC_TAG_LENGTH = 32  # SHA-256 produces 32 bytes

_hmac_key: bytes | None = None


def _get_hmac_key() -> bytes:
    """
    Load or generate the HMAC-SHA256 key used to authenticate pickle blobs.

    The key is stored at ~/.claude/enhanced_memories/.compression_hmac_key
    with mode 0600. Created with secrets.token_bytes(32) on first use.
    """
    global _hmac_key
    if _hmac_key is not None:
        return _hmac_key

    key_path = _HMAC_KEY_PATH
    key_path.parent.mkdir(parents=True, exist_ok=True)

    if key_path.exists():
        _hmac_key = key_path.read_bytes()
        if len(_hmac_key) != 32:
            logger.warning("HMAC key file has unexpected length, regenerating")
            _hmac_key = None

    if _hmac_key is None:
        _hmac_key = secrets.token_bytes(32)
        key_path.write_bytes(_hmac_key)
        os.chmod(key_path, stat.S_IRUSR | stat.S_IWUSR)  # 0600
        logger.info("Generated new HMAC key for compression integrity")

    return _hmac_key


def _compute_hmac(data: bytes) -> bytes:
    """Compute HMAC-SHA256 over *data* using the module key."""
    return hmac.new(_get_hmac_key(), data, hashlib.sha256).digest()


def _verify_hmac(data: bytes, expected_tag: bytes) -> bool:
    """Constant-time verification of an HMAC tag."""
    return hmac.compare_digest(_compute_hmac(data), expected_tag)


# ---- Public API ----

def compress_data(data: Any) -> Tuple[bytes, int, int, float]:
    """
    Compress data using JSON serialization + zlib (maximum compression).

    New data is always written as JSON with a 0x01 format marker prepended
    before compression. This eliminates pickle deserialization on read.

    Args:
        data: A JSON-serializable Python object (dict, list, str, int, etc.)

    Returns:
        Tuple of (compressed_bytes, original_size, compressed_size, compression_ratio)
    """
    json_bytes = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
    payload = _FORMAT_JSON + json_bytes
    original_size = len(payload)
    compressed = zlib.compress(payload, level=9)
    compressed_size = len(compressed)
    compression_ratio = compressed_size / original_size if original_size > 0 else 1.0
    return compressed, original_size, compressed_size, compression_ratio


def decompress_data(compressed: bytes) -> Any:
    """
    Decompress and deserialize data, handling both JSON and legacy pickle formats.

    Format detection (after zlib decompression):
      - First byte 0x01: JSON payload (new, safe).
      - First byte 0x00: Legacy pickle with explicit marker.
      - Otherwise: Oldest legacy pickle (no marker at all).

    For legacy pickle data:
      - If an HMAC tag is appended to the compressed blob, it is verified.
        Invalid HMAC raises ValueError (refuses to deserialize).
      - If no HMAC tag is present (pre-migration data), the data is
        deserialized with a deprecation warning logged.

    Args:
        compressed: Bytes from compress_data (zlib-compressed, possibly with HMAC suffix)

    Returns:
        Original Python object

    Raises:
        ValueError: If HMAC verification fails on legacy pickle data.
    """
    # Use decompressobj to detect trailing bytes (possible HMAC suffix).
    # zlib.decompress() silently ignores trailing data, but decompressobj
    # exposes unconsumed_tail so we can detect appended HMAC tags.
    hmac_tag: bytes | None = None

    dobj = zlib.decompressobj()
    decompressed = dobj.decompress(compressed)
    trailing = dobj.unused_data

    if len(trailing) == _HMAC_TAG_LENGTH:
        # Exactly 32 bytes trailing -- this is an HMAC tag
        hmac_tag = trailing
        zlib_blob = compressed[: len(compressed) - _HMAC_TAG_LENGTH]
    elif len(trailing) > 0:
        # Unexpected trailing data (not exactly HMAC length) -- treat as no HMAC
        zlib_blob = compressed[: len(compressed) - len(trailing)]
    else:
        zlib_blob = compressed

    # --- Detect format ---
    if len(decompressed) == 0:
        return None

    first_byte = decompressed[0:1]

    if first_byte == _FORMAT_JSON:
        # New JSON format -- safe, no pickle involved
        return json.loads(decompressed[1:].decode("utf-8"))

    # Legacy pickle path (marker 0x00 or no marker)
    if first_byte == _FORMAT_PICKLE_LEGACY:
        pickle_bytes = decompressed[1:]
    else:
        pickle_bytes = decompressed

    # HMAC integrity check for pickle data
    if hmac_tag is not None:
        if not _verify_hmac(zlib_blob, hmac_tag):
            raise ValueError(
                "HMAC verification failed on legacy pickle data. "
                "The database blob may have been tampered with. "
                "Refusing to deserialize."
            )
        logger.debug("Legacy pickle data passed HMAC verification")
    else:
        logger.warning(
            "DEPRECATION: Deserializing legacy pickle data without HMAC. "
            "This entity should be re-saved to migrate to JSON format."
        )

    return pickle.loads(pickle_bytes)  # nosec B301 -- guarded by HMAC or deprecation warning


def calculate_checksum(data: bytes) -> str:
    """
    Calculate SHA256 checksum for data integrity.

    Args:
        data: Bytes to checksum

    Returns:
        Hex-encoded SHA256 hash
    """
    return hashlib.sha256(data).hexdigest()


def classify_tier(entity_type: str, name: str) -> str:
    """
    Classify entity into memory tier based on type and name.

    Tiers:
    - core: System roles, orchestrator-related
    - working: Projects, sessions, current items
    - archive: Historical, archived items
    - reference: Default for everything else

    Args:
        entity_type: Type of entity
        name: Name of entity

    Returns:
        Tier classification string
    """
    if entity_type in ["system_role", "core_system"] or "orchestrator" in name.lower():
        return "core"
    elif entity_type in ["project", "session"] or "current" in name.lower():
        return "working"
    elif "archive" in name.lower() or "historical" in entity_type.lower():
        return "archive"
    else:
        return "reference"
