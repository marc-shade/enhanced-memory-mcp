"""
Memory Integrity System
=======================

Cryptographic signing and verification for memory entities.
Protects against data poisoning, tampering, and unauthorized modifications.

Security Model:
- HMAC-SHA256 signatures for all memory content
- Per-entity provenance tracking (who created/modified)
- Anomaly detection for suspicious modification patterns
- Integration with threat-intel for IOC scanning in memories

Key Management:
- System key stored in MEMORY_DIR/.integrity_key
- Auto-generated on first use with cryptographically secure random
- Key rotation supported via rotate_integrity_key()
"""

import hashlib
import hmac
import json
import os
import secrets
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import DB_PATH, MEMORY_DIR, logger


# Key file location
INTEGRITY_KEY_PATH = Path(MEMORY_DIR) / ".integrity_key"

# Signature algorithm identifier (for future algorithm upgrades)
SIGNATURE_ALGORITHM = "hmac-sha256-v1"


class IntegrityStatus(Enum):
    """Status of memory integrity verification."""
    VERIFIED = "verified"           # Signature valid, content intact
    TAMPERED = "tampered"           # Signature invalid, content modified
    UNSIGNED = "unsigned"           # No signature present
    KEY_MISMATCH = "key_mismatch"   # Signed with different key
    EXPIRED = "expired"             # Signature too old (optional policy)
    PENDING = "pending"             # Not yet verified


class ThreatLevel(Enum):
    """Threat level for detected anomalies."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class IntegrityResult:
    """Result of integrity verification."""
    entity_name: str
    status: str
    verified: bool
    signature_valid: bool
    signed_at: Optional[str]
    signed_by: Optional[str]
    algorithm: Optional[str]
    threat_level: str
    anomalies: List[str]
    recommendation: str


@dataclass
class AnomalyReport:
    """Report of detected anomalies in memory."""
    total_entities: int
    verified: int
    tampered: int
    unsigned: int
    anomalies_detected: List[Dict[str, Any]]
    threat_level: str
    scan_time: str
    recommendations: List[str]


def _get_or_create_key() -> bytes:
    """
    Get or create the integrity signing key.

    Prefers encrypted vault storage, falls back to plaintext.
    """
    # Try vault first
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        from security.key_vault import KeyVault
        vault = KeyVault()
        key = vault.load_key("memory_integrity")
        if key:
            return key

        # Migrate existing plaintext key into vault
        if INTEGRITY_KEY_PATH.exists():
            if vault.migrate_plaintext_key("memory_integrity", INTEGRITY_KEY_PATH):
                loaded = vault.load_key("memory_integrity")
                if loaded:
                    return loaded

        # Generate new key in vault
        key = secrets.token_bytes(32)
        vault.store_key("memory_integrity", key)
        return key
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Vault error, falling back to plaintext: {e}")

    # Fallback: plaintext key storage
    if INTEGRITY_KEY_PATH.exists():
        with open(INTEGRITY_KEY_PATH, "rb") as f:
            key = f.read()
            if len(key) >= 32:
                return key

    key = secrets.token_bytes(32)
    INTEGRITY_KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(INTEGRITY_KEY_PATH, "wb") as f:
        f.write(key)
    os.chmod(INTEGRITY_KEY_PATH, 0o600)
    logger.info(f"Generated new integrity key at {INTEGRITY_KEY_PATH}")
    return key


def _compute_content_hash(content: Any) -> str:
    """Compute deterministic hash of content."""
    # Serialize to JSON with sorted keys for determinism
    if isinstance(content, (dict, list)):
        serialized = json.dumps(content, sort_keys=True, default=str)
    elif isinstance(content, bytes):
        serialized = content.decode('utf-8', errors='replace')
    else:
        serialized = str(content)

    return hashlib.sha256(serialized.encode('utf-8')).hexdigest()


def sign_content(content: Any, signer: str = "system") -> Dict[str, str]:
    """
    Sign content with HMAC-SHA256.

    Args:
        content: The content to sign (dict, list, str, or bytes)
        signer: Identifier of who is signing (agent, user, system)

    Returns:
        Signature metadata dict with signature, algorithm, timestamp, signer
    """
    key = _get_or_create_key()
    content_hash = _compute_content_hash(content)

    # Create signature over: algorithm + content_hash + timestamp + signer
    timestamp = datetime.now(timezone.utc).isoformat()
    message = f"{SIGNATURE_ALGORITHM}:{content_hash}:{timestamp}:{signer}"

    signature = hmac.new(
        key,
        message.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()

    return {
        "signature": signature,
        "algorithm": SIGNATURE_ALGORITHM,
        "content_hash": content_hash,
        "signed_at": timestamp,
        "signed_by": signer,
    }


def verify_signature(
    content: Any,
    signature_data: Dict[str, str],
    max_age_hours: Optional[int] = None
) -> Tuple[bool, str]:
    """
    Verify content signature.

    Args:
        content: The content to verify
        signature_data: The signature metadata from sign_content()
        max_age_hours: Optional maximum signature age in hours

    Returns:
        Tuple of (is_valid, reason)
    """
    if not signature_data:
        return False, "No signature data provided"

    required_fields = ["signature", "algorithm", "content_hash", "signed_at", "signed_by"]
    for field in required_fields:
        if field not in signature_data:
            return False, f"Missing required field: {field}"

    # Check algorithm
    if signature_data["algorithm"] != SIGNATURE_ALGORITHM:
        return False, f"Unsupported algorithm: {signature_data['algorithm']}"

    # Check signature age if policy set
    if max_age_hours:
        try:
            signed_at = datetime.fromisoformat(signature_data["signed_at"].replace('Z', '+00:00'))
            age = datetime.now(timezone.utc) - signed_at
            if age > timedelta(hours=max_age_hours):
                return False, f"Signature expired ({age.total_seconds() / 3600:.1f} hours old)"
        except (ValueError, TypeError) as e:
            return False, f"Invalid timestamp: {e}"

    # Verify content hash
    current_hash = _compute_content_hash(content)
    if current_hash != signature_data["content_hash"]:
        return False, "Content hash mismatch - content has been modified"

    # Verify HMAC signature
    key = _get_or_create_key()
    message = f"{signature_data['algorithm']}:{signature_data['content_hash']}:{signature_data['signed_at']}:{signature_data['signed_by']}"

    expected_signature = hmac.new(
        key,
        message.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()

    if not hmac.compare_digest(expected_signature, signature_data["signature"]):
        return False, "Signature verification failed - possible tampering or key mismatch"

    return True, "Signature verified successfully"


def rotate_integrity_key() -> bool:
    """
    Rotate the integrity signing key.

    WARNING: This invalidates all existing signatures!
    Should be followed by re-signing all entities.

    Returns:
        True if rotation successful
    """
    # Backup old key
    if INTEGRITY_KEY_PATH.exists():
        backup_path = INTEGRITY_KEY_PATH.with_suffix(f".key.{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}.bak")
        INTEGRITY_KEY_PATH.rename(backup_path)
        logger.warning(f"Old key backed up to {backup_path}")

    # Generate new key
    _get_or_create_key()
    logger.info("Integrity key rotated - all existing signatures are now invalid")

    return True


# =============================================================================
# Database Integration
# =============================================================================

def init_integrity_tables() -> None:
    """Initialize integrity tracking tables in the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Add integrity fields to entities table
    integrity_columns = [
        ("integrity_signature", "TEXT"),
        ("integrity_algorithm", "TEXT"),
        ("integrity_signed_at", "TIMESTAMP"),
        ("integrity_signed_by", "TEXT"),
        ("integrity_status", "TEXT DEFAULT 'unsigned'"),
        ("integrity_verified_at", "TIMESTAMP"),
    ]

    for col_name, col_type in integrity_columns:
        try:
            cursor.execute(f'ALTER TABLE entities ADD COLUMN {col_name} {col_type}')
            logger.info(f"Added integrity column: {col_name}")
        except sqlite3.OperationalError:
            pass  # Column already exists

    # Create integrity verification history table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS integrity_verifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_id INTEGER NOT NULL,
            entity_name TEXT NOT NULL,
            verification_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT NOT NULL,
            signature_valid BOOLEAN,
            content_hash TEXT,
            anomalies TEXT,
            threat_level TEXT,
            verifier TEXT DEFAULT 'system',
            FOREIGN KEY (entity_id) REFERENCES entities (id)
        )
    ''')

    # Create anomaly tracking table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS integrity_anomalies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_id INTEGER,
            entity_name TEXT,
            anomaly_type TEXT NOT NULL,
            description TEXT,
            threat_level TEXT,
            detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            resolved BOOLEAN DEFAULT 0,
            resolution_notes TEXT,
            FOREIGN KEY (entity_id) REFERENCES entities (id)
        )
    ''')

    # Create indexes
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_integrity_status ON entities(integrity_status)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_verifications_entity ON integrity_verifications(entity_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_anomalies_unresolved ON integrity_anomalies(resolved)')

    conn.commit()
    conn.close()
    logger.info("Integrity tables initialized")


def sign_entity(entity_name: str, signer: str = "system") -> bool:
    """
    Sign an entity's content in the database.

    Args:
        entity_name: Name of the entity to sign
        signer: Who is signing (system, agent name, user)

    Returns:
        True if signing successful
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        # Get entity content
        cursor.execute('''
            SELECT id, name, entity_type, compressed_data, tier
            FROM entities WHERE name = ?
        ''', (entity_name,))

        row = cursor.fetchone()
        if not row:
            logger.warning(f"Entity not found: {entity_name}")
            return False

        entity_id, name, entity_type, compressed_data, tier = row

        # Get observations
        cursor.execute('SELECT content FROM observations WHERE entity_id = ?', (entity_id,))
        observations = [r[0] for r in cursor.fetchall()]

        # Create content dict for signing
        content = {
            "name": name,
            "entity_type": entity_type,
            "tier": tier,
            "compressed_data_hash": hashlib.sha256(compressed_data).hexdigest() if compressed_data else None,
            "observations": observations,
        }

        # Sign content
        sig_data = sign_content(content, signer)

        # Update entity
        cursor.execute('''
            UPDATE entities SET
                integrity_signature = ?,
                integrity_algorithm = ?,
                integrity_signed_at = ?,
                integrity_signed_by = ?,
                integrity_status = 'verified'
            WHERE id = ?
        ''', (
            sig_data["signature"],
            sig_data["algorithm"],
            sig_data["signed_at"],
            sig_data["signed_by"],
            entity_id
        ))

        conn.commit()
        logger.debug(f"Signed entity: {entity_name}")
        return True

    except Exception as e:
        logger.error(f"Failed to sign entity {entity_name}: {e}")
        return False
    finally:
        conn.close()


def verify_entity(entity_name: str, record_result: bool = True) -> IntegrityResult:
    """
    Verify an entity's integrity.

    Args:
        entity_name: Name of the entity to verify
        record_result: Whether to record verification in history

    Returns:
        IntegrityResult with verification details
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        # Get entity with signature
        cursor.execute('''
            SELECT id, name, entity_type, compressed_data, tier,
                   integrity_signature, integrity_algorithm,
                   integrity_signed_at, integrity_signed_by
            FROM entities WHERE name = ?
        ''', (entity_name,))

        row = cursor.fetchone()
        if not row:
            return IntegrityResult(
                entity_name=entity_name,
                status=IntegrityStatus.UNSIGNED.value,
                verified=False,
                signature_valid=False,
                signed_at=None,
                signed_by=None,
                algorithm=None,
                threat_level=ThreatLevel.MEDIUM.value,
                anomalies=["Entity not found"],
                recommendation="Entity does not exist in database"
            )

        (entity_id, name, entity_type, compressed_data, tier,
         signature, algorithm, signed_at, signed_by) = row

        # Check if unsigned
        if not signature:
            result = IntegrityResult(
                entity_name=entity_name,
                status=IntegrityStatus.UNSIGNED.value,
                verified=False,
                signature_valid=False,
                signed_at=None,
                signed_by=None,
                algorithm=None,
                threat_level=ThreatLevel.LOW.value,
                anomalies=["Entity has no integrity signature"],
                recommendation="Sign entity with sign_entity() for integrity protection"
            )
        else:
            # Get observations
            cursor.execute('SELECT content FROM observations WHERE entity_id = ?', (entity_id,))
            observations = [r[0] for r in cursor.fetchall()]

            # Recreate content dict
            content = {
                "name": name,
                "entity_type": entity_type,
                "tier": tier,
                "compressed_data_hash": hashlib.sha256(compressed_data).hexdigest() if compressed_data else None,
                "observations": observations,
            }

            # Prepare signature data
            sig_data = {
                "signature": signature,
                "algorithm": algorithm,
                "content_hash": _compute_content_hash(content),
                "signed_at": signed_at,
                "signed_by": signed_by,
            }

            # Verify
            is_valid, reason = verify_signature(content, sig_data)

            if is_valid:
                result = IntegrityResult(
                    entity_name=entity_name,
                    status=IntegrityStatus.VERIFIED.value,
                    verified=True,
                    signature_valid=True,
                    signed_at=signed_at,
                    signed_by=signed_by,
                    algorithm=algorithm,
                    threat_level=ThreatLevel.NONE.value,
                    anomalies=[],
                    recommendation="Entity integrity verified - content is authentic"
                )
            else:
                result = IntegrityResult(
                    entity_name=entity_name,
                    status=IntegrityStatus.TAMPERED.value,
                    verified=False,
                    signature_valid=False,
                    signed_at=signed_at,
                    signed_by=signed_by,
                    algorithm=algorithm,
                    threat_level=ThreatLevel.HIGH.value,
                    anomalies=[reason],
                    recommendation="ALERT: Entity content has been modified! Investigate immediately."
                )

        # Record verification result
        if record_result:
            cursor.execute('''
                INSERT INTO integrity_verifications
                (entity_id, entity_name, status, signature_valid, anomalies, threat_level)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                entity_id,
                entity_name,
                result.status,
                result.signature_valid,
                json.dumps(result.anomalies),
                result.threat_level
            ))

            # Update entity status
            cursor.execute('''
                UPDATE entities SET
                    integrity_status = ?,
                    integrity_verified_at = ?
                WHERE id = ?
            ''', (result.status, datetime.now(timezone.utc).isoformat(), entity_id))

            conn.commit()

        return result

    except Exception as e:
        logger.error(f"Failed to verify entity {entity_name}: {e}")
        return IntegrityResult(
            entity_name=entity_name,
            status="error",
            verified=False,
            signature_valid=False,
            signed_at=None,
            signed_by=None,
            algorithm=None,
            threat_level=ThreatLevel.MEDIUM.value,
            anomalies=[str(e)],
            recommendation="Verification failed due to error"
        )
    finally:
        conn.close()


def scan_all_integrity(limit: int = 1000) -> AnomalyReport:
    """
    Scan all entities for integrity issues.

    Args:
        limit: Maximum entities to scan

    Returns:
        AnomalyReport with scan results
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        cursor.execute('SELECT name FROM entities LIMIT ?', (limit,))
        entities = [r[0] for r in cursor.fetchall()]

        results = {
            "verified": 0,
            "tampered": 0,
            "unsigned": 0,
            "error": 0,
        }
        anomalies = []

        for entity_name in entities:
            result = verify_entity(entity_name, record_result=False)

            if result.status == IntegrityStatus.VERIFIED.value:
                results["verified"] += 1
            elif result.status == IntegrityStatus.TAMPERED.value:
                results["tampered"] += 1
                anomalies.append({
                    "entity": entity_name,
                    "status": result.status,
                    "threat_level": result.threat_level,
                    "anomalies": result.anomalies
                })
            elif result.status == IntegrityStatus.UNSIGNED.value:
                results["unsigned"] += 1
            else:
                results["error"] += 1

        # Determine overall threat level
        if results["tampered"] > 0:
            threat_level = ThreatLevel.CRITICAL.value if results["tampered"] > 5 else ThreatLevel.HIGH.value
        elif results["unsigned"] > len(entities) * 0.5:
            threat_level = ThreatLevel.MEDIUM.value
        else:
            threat_level = ThreatLevel.LOW.value

        # Generate recommendations
        recommendations = []
        if results["tampered"] > 0:
            recommendations.append(f"CRITICAL: {results['tampered']} entities show signs of tampering. Investigate immediately!")
        if results["unsigned"] > 0:
            recommendations.append(f"Sign {results['unsigned']} unsigned entities with bulk_sign_entities()")
        if results["verified"] == len(entities):
            recommendations.append("All entities verified - memory integrity intact")

        return AnomalyReport(
            total_entities=len(entities),
            verified=results["verified"],
            tampered=results["tampered"],
            unsigned=results["unsigned"],
            anomalies_detected=anomalies,
            threat_level=threat_level,
            scan_time=datetime.now(timezone.utc).isoformat(),
            recommendations=recommendations
        )

    finally:
        conn.close()


def bulk_sign_entities(entity_type: Optional[str] = None, signer: str = "system") -> Dict[str, int]:
    """
    Sign multiple entities in bulk.

    Args:
        entity_type: Optional filter by entity type
        signer: Who is signing

    Returns:
        Dict with counts of signed, failed, skipped
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        if entity_type:
            cursor.execute('''
                SELECT name FROM entities
                WHERE entity_type = ? AND (integrity_signature IS NULL OR integrity_signature = '')
            ''', (entity_type,))
        else:
            cursor.execute('''
                SELECT name FROM entities
                WHERE integrity_signature IS NULL OR integrity_signature = ''
            ''')

        unsigned_entities = [r[0] for r in cursor.fetchall()]

        results = {"signed": 0, "failed": 0, "skipped": 0}

        for entity_name in unsigned_entities:
            if sign_entity(entity_name, signer):
                results["signed"] += 1
            else:
                results["failed"] += 1

        logger.info(f"Bulk sign complete: {results}")
        return results

    finally:
        conn.close()


def get_integrity_stats() -> Dict[str, Any]:
    """Get statistics about memory integrity status."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        stats = {}

        # Total entities
        cursor.execute('SELECT COUNT(*) FROM entities')
        stats["total_entities"] = cursor.fetchone()[0]

        # By integrity status
        cursor.execute('''
            SELECT integrity_status, COUNT(*)
            FROM entities
            GROUP BY integrity_status
        ''')
        stats["by_status"] = dict(cursor.fetchall())

        # Recent verifications
        cursor.execute('''
            SELECT COUNT(*) FROM integrity_verifications
            WHERE verification_time > datetime('now', '-24 hours')
        ''')
        stats["verifications_24h"] = cursor.fetchone()[0]

        # Unresolved anomalies
        cursor.execute('''
            SELECT COUNT(*) FROM integrity_anomalies
            WHERE resolved = 0
        ''')
        stats["unresolved_anomalies"] = cursor.fetchone()[0]

        # Recent anomalies
        cursor.execute('''
            SELECT entity_name, anomaly_type, threat_level, detected_at
            FROM integrity_anomalies
            WHERE resolved = 0
            ORDER BY detected_at DESC
            LIMIT 10
        ''')
        stats["recent_anomalies"] = [
            {"entity": r[0], "type": r[1], "threat_level": r[2], "detected_at": r[3]}
            for r in cursor.fetchall()
        ]

        return stats

    finally:
        conn.close()


# Initialize integrity tables on module import
try:
    init_integrity_tables()
except Exception as e:
    logger.warning(f"Could not initialize integrity tables: {e}")
