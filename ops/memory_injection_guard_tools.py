#!/usr/bin/env python3
"""
MCP tool surface for the memory-side injection guard.

Registers `memory_injection_check` so an agent can pre-check content before
writing it to shared memory. The scan itself lives in
`ops.memory_injection_guard` (stdlib-only, no harness-hook dependency); this
module is only the thin `@app.tool()` wrapper.

Signature follows the repo's feature-module convention:
    register_memory_injection_guard_tools(app, db_path)
(db_path is accepted for uniformity with the other *_tools modules; the guard
is content-only and does not touch the database.)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from ops.memory_injection_guard import (
    guard_shared_write,
    is_allowlisted,
    scan_for_injection,
)

logger = logging.getLogger("memory_injection_guard_tools")


def register_memory_injection_guard_tools(app, db_path: Optional[str] = None):
    """Register the memory_injection_check MCP tool."""

    @app.tool()
    async def memory_injection_check(
        content: str,
        allowlist_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Pre-check content for prompt-injection patterns before writing it to
        shared memory.

        Runs the memory-side injection scanner (mirrors the harness hook's
        detection logic, implemented in-process so the server needs no hook
        dependency) and reports the verdict plus allowlist status. A CRITICAL
        verdict with no allowlist entry means a `guard_shared_write` gate would
        block the write (fail closed).

        Args:
            content: The text to scan (e.g. a memory observation or episode).
            allowlist_path: Optional override path to an injection_overrides.json
                allowlist (defaults to ~/.claude/security/injection_overrides.json).

        Returns:
            Dict with verdict, matched_patterns, reason, allowlisted, allowed
        """
        path = Path(allowlist_path) if allowlist_path else None
        scan = scan_for_injection(content)
        gate = guard_shared_write(content, path)
        return {
            "verdict": scan["verdict"],
            "matched_patterns": scan["matched_patterns"],
            "reason": scan["reason"],
            "allowlisted": is_allowlisted(content, path),
            "allowed": gate["allowed"],
            "hint": (
                "A write to shared memory is blocked (allowed=False) only when "
                "verdict=critical AND the content hash is not allowlisted."
            ),
        }

    logger.info("Registered 1 memory injection guard MCP tool")
    return True
