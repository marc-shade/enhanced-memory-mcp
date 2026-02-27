"""
Prompt Injection Detection MCP Tools
=====================================

Provides MCP tools for manual prompt injection scanning and detection statistics.
Integrates with the hook-based detection system for comprehensive coverage.
"""

import sys
from typing import Any
from pathlib import Path

# Add hooks directory for detector import
sys.path.insert(0, str(Path.home() / ".claude" / "hooks"))

try:
    from prompt_injection_detector import (
        scan_content,
        is_safe,
        get_detection_stats,
        RiskLevel,
        InjectionType
    )
    DETECTOR_AVAILABLE = True
except ImportError as e:
    DETECTOR_AVAILABLE = False
    DETECTOR_ERROR = str(e)


def register_prompt_injection_tools(app, db_path: str = None):
    """Register prompt injection detection MCP tools."""

    @app.tool()
    async def scan_for_injection(
        content: str,
        context: str = "manual_scan"
    ) -> dict[str, Any]:
        """
        Scan content for prompt injection attempts.

        Args:
            content: Text content to scan for injection attacks
            context: Description of where content came from (e.g., 'web_page', 'email', 'file')

        Returns:
            Scan result with risk assessment and detections
        """
        if not DETECTOR_AVAILABLE:
            return {
                "error": f"Detector unavailable: {DETECTOR_ERROR}",
                "clean": None,
                "risk_level": "unknown"
            }

        result = scan_content(content, context)

        return {
            "clean": result.clean,
            "risk_level": result.risk_level,
            "recommendation": result.recommendation,
            "detections": result.detections,
            "content_length": result.content_length,
            "scan_time_ms": result.scan_time_ms,
            "context": context
        }

    @app.tool()
    async def check_content_safety(content: str) -> dict[str, Any]:
        """
        Quick safety check - returns whether content is safe to process.

        Args:
            content: Text content to check

        Returns:
            Simple safety assessment
        """
        if not DETECTOR_AVAILABLE:
            return {
                "error": f"Detector unavailable: {DETECTOR_ERROR}",
                "safe": None
            }

        safe = is_safe(content)

        if safe:
            return {
                "safe": True,
                "message": "Content appears safe to process"
            }
        else:
            # Get full scan for details
            result = scan_content(content, "safety_check")
            return {
                "safe": False,
                "risk_level": result.risk_level,
                "message": result.recommendation,
                "detection_count": len(result.detections)
            }

    @app.tool()
    async def get_injection_detection_stats() -> dict[str, Any]:
        """
        Get statistics about prompt injection detections.

        Returns:
            Detection statistics including counts by type and risk level
        """
        if not DETECTOR_AVAILABLE:
            return {
                "error": f"Detector unavailable: {DETECTOR_ERROR}",
                "stats": None
            }

        stats = get_detection_stats()
        return {
            "total_scans": stats.get("total_scans", 0),
            "detections_by_type": stats.get("detections_by_type", {}),
            "detections_by_risk": stats.get("detections_by_risk", {}),
            "blocked_count": stats.get("blocked_count", 0),
            "warning_count": stats.get("warning_count", 0),
            "detector_available": True
        }

    @app.tool()
    async def list_injection_types() -> dict[str, Any]:
        """
        List all prompt injection types the detector can identify.

        Returns:
            List of injection types with descriptions
        """
        if not DETECTOR_AVAILABLE:
            return {
                "error": f"Detector unavailable: {DETECTOR_ERROR}",
                "types": None
            }

        types = {
            "direct_instruction": "Direct commands to ignore/override previous instructions",
            "role_manipulation": "Attempts to redefine AI role or identity",
            "delimiter_escape": "Escape sequences targeting prompt structure",
            "encoding_attack": "Obfuscated content (base64, rot13, hex)",
            "hidden_text": "Invisible characters, zero-width, RTL override",
            "jailbreak": "Known jailbreak phrases (DAN, Developer Mode)",
            "system_prompt_leak": "Requests to reveal system prompt",
            "tool_abuse": "Malicious tool/function call attempts",
            "context_manipulation": "False context/authority claims",
            "output_manipulation": "Attempts to control/format AI output"
        }

        return {
            "injection_types": types,
            "count": len(types),
            "detector_version": "1.0.0"
        }

    @app.tool()
    async def analyze_url_for_injection(url: str) -> dict[str, Any]:
        """
        Analyze a URL for potential injection indicators in the path/parameters.

        Args:
            url: URL to analyze

        Returns:
            Analysis of URL components for injection risk
        """
        if not DETECTOR_AVAILABLE:
            return {
                "error": f"Detector unavailable: {DETECTOR_ERROR}",
                "safe": None
            }

        from urllib.parse import urlparse, parse_qs

        try:
            parsed = urlparse(url)

            # Check URL components
            results = {
                "url": url,
                "domain": parsed.netloc,
                "path_analysis": None,
                "query_analysis": None,
                "overall_safe": True,
                "concerns": []
            }

            # Scan path
            if parsed.path and len(parsed.path) > 1:
                path_result = scan_content(parsed.path, "url_path")
                if not path_result.clean:
                    results["path_analysis"] = {
                        "risk_level": path_result.risk_level,
                        "detections": path_result.detections
                    }
                    results["overall_safe"] = False
                    results["concerns"].append(f"Suspicious path: {path_result.recommendation}")

            # Scan query parameters
            if parsed.query:
                params = parse_qs(parsed.query)
                param_concerns = []

                for key, values in params.items():
                    for value in values:
                        param_result = scan_content(value, f"url_param_{key}")
                        if not param_result.clean:
                            param_concerns.append({
                                "param": key,
                                "risk_level": param_result.risk_level,
                                "detections": param_result.detections
                            })
                            results["overall_safe"] = False

                if param_concerns:
                    results["query_analysis"] = param_concerns
                    results["concerns"].append(f"Suspicious query parameters detected")

            return results

        except Exception as e:
            return {
                "error": f"URL parsing failed: {e}",
                "url": url,
                "safe": None
            }

    print("[PromptInjection] MCP tools registered", file=sys.stderr)
    return True
