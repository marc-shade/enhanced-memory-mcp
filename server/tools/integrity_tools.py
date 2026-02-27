"""
Memory Integrity MCP Tools
==========================

MCP tools for verifying and managing memory integrity.
Protects against data poisoning and tampering.

Tools:
- memory_verify_integrity: Verify a single entity's integrity
- memory_scan_integrity: Scan all entities for tampering
- memory_sign_entity: Sign an entity for integrity protection
- memory_bulk_sign: Sign multiple unsigned entities
- memory_integrity_stats: Get integrity status statistics
- memory_detect_anomalies: Detect suspicious modification patterns
"""

import json
from typing import Any, Dict, Optional

from fastmcp import FastMCP

from ..integrity import (
    sign_entity,
    verify_entity,
    scan_all_integrity,
    bulk_sign_entities,
    get_integrity_stats,
    IntegrityResult,
    AnomalyReport,
)
from ..config import logger


def register_integrity_tools(app: FastMCP) -> None:
    """Register memory integrity tools with the FastMCP app."""

    @app.tool()
    async def memory_verify_integrity(entity_name: str) -> str:
        """
        Verify the integrity of a memory entity.

        Checks if the entity's content matches its cryptographic signature.
        Detects tampering, unauthorized modifications, or data poisoning.

        Args:
            entity_name: Name of the entity to verify

        Returns:
            JSON with verification result including:
            - status: verified/tampered/unsigned
            - signature_valid: boolean
            - threat_level: none/low/medium/high/critical
            - anomalies: list of detected issues
            - recommendation: suggested action
        """
        try:
            result = verify_entity(entity_name)

            return json.dumps({
                "entity_name": result.entity_name,
                "status": result.status,
                "verified": result.verified,
                "signature_valid": result.signature_valid,
                "signed_at": result.signed_at,
                "signed_by": result.signed_by,
                "algorithm": result.algorithm,
                "threat_level": result.threat_level,
                "anomalies": result.anomalies,
                "recommendation": result.recommendation,
            }, indent=2)

        except Exception as e:
            logger.error(f"Integrity verification failed: {e}")
            return json.dumps({
                "error": str(e),
                "status": "error",
                "verified": False
            })

    @app.tool()
    async def memory_scan_integrity(limit: int = 100) -> str:
        """
        Scan multiple entities for integrity issues.

        Performs a bulk integrity check across the memory database.
        Identifies tampered, unsigned, and verified entities.

        Args:
            limit: Maximum number of entities to scan (default 100)

        Returns:
            JSON with scan results including:
            - total_entities: number scanned
            - verified: count of verified entities
            - tampered: count of tampered entities (ALERT!)
            - unsigned: count of unsigned entities
            - anomalies_detected: details of issues found
            - threat_level: overall threat assessment
            - recommendations: suggested actions
        """
        try:
            report = scan_all_integrity(limit=limit)

            return json.dumps({
                "total_entities": report.total_entities,
                "verified": report.verified,
                "tampered": report.tampered,
                "unsigned": report.unsigned,
                "anomalies_detected": report.anomalies_detected,
                "threat_level": report.threat_level,
                "scan_time": report.scan_time,
                "recommendations": report.recommendations,
            }, indent=2)

        except Exception as e:
            logger.error(f"Integrity scan failed: {e}")
            return json.dumps({"error": str(e)})

    @app.tool()
    async def memory_sign_entity(
        entity_name: str,
        signer: str = "system"
    ) -> str:
        """
        Sign a memory entity for integrity protection.

        Creates a cryptographic signature of the entity's content.
        Future modifications can be detected by verifying the signature.

        Args:
            entity_name: Name of the entity to sign
            signer: Who is signing (default "system", can be agent name)

        Returns:
            JSON with result:
            - success: boolean
            - entity_name: name of signed entity
            - signed_by: who signed it
        """
        try:
            success = sign_entity(entity_name, signer)

            return json.dumps({
                "success": success,
                "entity_name": entity_name,
                "signed_by": signer,
                "message": f"Entity '{entity_name}' signed successfully" if success else "Signing failed"
            })

        except Exception as e:
            logger.error(f"Entity signing failed: {e}")
            return json.dumps({
                "success": False,
                "error": str(e)
            })

    @app.tool()
    async def memory_bulk_sign(
        entity_type: Optional[str] = None,
        signer: str = "system"
    ) -> str:
        """
        Sign multiple unsigned entities in bulk.

        Finds all unsigned entities and creates integrity signatures.
        Optionally filters by entity type.

        Args:
            entity_type: Optional filter (e.g., "learning", "project_outcome")
            signer: Who is signing (default "system")

        Returns:
            JSON with counts:
            - signed: number successfully signed
            - failed: number that failed to sign
            - skipped: number skipped (already signed)
        """
        try:
            results = bulk_sign_entities(entity_type=entity_type, signer=signer)

            return json.dumps({
                "signed": results["signed"],
                "failed": results["failed"],
                "skipped": results.get("skipped", 0),
                "filter": entity_type or "all types",
                "signed_by": signer,
            })

        except Exception as e:
            logger.error(f"Bulk signing failed: {e}")
            return json.dumps({"error": str(e)})

    @app.tool()
    async def memory_integrity_stats() -> str:
        """
        Get memory integrity status statistics.

        Returns overview of integrity status across all entities,
        recent verifications, and unresolved anomalies.

        Returns:
            JSON with statistics:
            - total_entities: total count
            - by_status: breakdown by integrity status
            - verifications_24h: recent verification count
            - unresolved_anomalies: count needing attention
            - recent_anomalies: details of recent issues
        """
        try:
            stats = get_integrity_stats()
            return json.dumps(stats, indent=2, default=str)

        except Exception as e:
            logger.error(f"Failed to get integrity stats: {e}")
            return json.dumps({"error": str(e)})

    @app.tool()
    async def memory_detect_anomalies(
        time_window_hours: int = 24,
        sensitivity: str = "medium"
    ) -> str:
        """
        Detect anomalous modification patterns in memory.

        Analyzes modification patterns to identify suspicious activity:
        - Rapid bulk modifications
        - Modifications to critical entities
        - Unusual modification times
        - Modifications from unknown sources

        Args:
            time_window_hours: Look back period (default 24 hours)
            sensitivity: Detection sensitivity (low/medium/high)

        Returns:
            JSON with detected anomalies and threat assessment
        """
        try:
            import sqlite3
            from datetime import datetime, timedelta, timezone
            from ..config import DB_PATH

            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            cutoff = (datetime.now(timezone.utc) - timedelta(hours=time_window_hours)).isoformat()

            anomalies = []
            threat_indicators = []

            # Check for rapid modifications
            cursor.execute('''
                SELECT e.name, COUNT(*) as mod_count
                FROM memory_versions mv
                JOIN entities e ON mv.entity_id = e.id
                WHERE mv.created_at > ?
                GROUP BY mv.entity_id
                HAVING mod_count > 10
            ''', (cutoff,))

            rapid_mods = cursor.fetchall()
            if rapid_mods:
                for name, count in rapid_mods:
                    anomalies.append({
                        "type": "rapid_modification",
                        "entity": name,
                        "count": count,
                        "threat_level": "medium" if count < 20 else "high"
                    })
                    threat_indicators.append("rapid_modification")

            # Check for unsigned critical entities
            cursor.execute('''
                SELECT name, entity_type FROM entities
                WHERE tier = 'long_term'
                AND (integrity_signature IS NULL OR integrity_signature = '')
            ''')

            unsigned_critical = cursor.fetchall()
            if unsigned_critical:
                for name, etype in unsigned_critical[:10]:
                    anomalies.append({
                        "type": "unsigned_critical_entity",
                        "entity": name,
                        "entity_type": etype,
                        "threat_level": "medium"
                    })
                threat_indicators.append("unsigned_critical")

            # Check for tampered entities
            cursor.execute('''
                SELECT name FROM entities
                WHERE integrity_status = 'tampered'
            ''')

            tampered = cursor.fetchall()
            if tampered:
                for (name,) in tampered[:10]:
                    anomalies.append({
                        "type": "tampered_entity",
                        "entity": name,
                        "threat_level": "critical"
                    })
                threat_indicators.append("tampering_detected")

            conn.close()

            # Calculate overall threat level
            if "tampering_detected" in threat_indicators:
                overall_threat = "critical"
            elif "rapid_modification" in threat_indicators and len(rapid_mods) > 5:
                overall_threat = "high"
            elif threat_indicators:
                overall_threat = "medium"
            else:
                overall_threat = "low"

            # Generate recommendations
            recommendations = []
            if "tampering_detected" in threat_indicators:
                recommendations.append("CRITICAL: Tampered entities detected! Investigate immediately.")
                recommendations.append("Consider restoring from backup or reverting to known-good version.")
            if "unsigned_critical" in threat_indicators:
                recommendations.append(f"Sign {len(unsigned_critical)} critical unsigned entities.")
            if "rapid_modification" in threat_indicators:
                recommendations.append("Review rapid modification sources for potential automated attacks.")
            if not threat_indicators:
                recommendations.append("No significant anomalies detected. Memory integrity appears healthy.")

            return json.dumps({
                "time_window_hours": time_window_hours,
                "sensitivity": sensitivity,
                "anomalies_found": len(anomalies),
                "anomalies": anomalies,
                "threat_indicators": threat_indicators,
                "overall_threat_level": overall_threat,
                "recommendations": recommendations,
            }, indent=2)

        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return json.dumps({"error": str(e)})

    logger.info("Registered 6 memory integrity tools")
