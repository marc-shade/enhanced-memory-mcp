#!/usr/bin/env python3
"""
Memory-Side Injection Guard

Self-contained content scanner that detects prompt-injection patterns in
content being written to shared memory. It mirrors the detection logic of
`~/.claude/hooks/prompt_injection_detector.py` — the prompt-override phrase
patterns, encoding attacks (base64/hex), and hidden-text (HTML comments,
invisible unicode) — but is implemented fresh inside the memory server, so the
server has no dependency on the hook module or its filesystem paths.

Exposed API:
  scan_for_injection(content) -> dict
      {verdict: "clean"|"warning"|"critical", matched_patterns: [...], reason}
  is_allowlisted(content, allowlist_path=None) -> bool
      content-hash allowlist against ~/.claude/security/injection_overrides.json
  guard_shared_write(content, allowlist_path=None) -> dict
      fail-closed gate: {allowed: bool, verdict, reason}

Verdict semantics:
  critical  — a clear override attempt or an encoding attack. `guard_shared_write`
              fails CLOSED here: allowed=False unless the exact content hash is
              allowlisted.
  warning   — suspicious-but-ambiguous (role manipulation, delimiter escape,
              jailbreak, hidden text, system-prompt leak, tool abuse...).
  clean     — no patterns matched.

This module imports nothing from the harness hooks layer; the only external
dependencies are the stdlib.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import re
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("memory_injection_guard")

_DEFAULT_ALLOWLIST = Path.home() / ".claude" / "security" / "injection_overrides.json"

# Verdicts (the gate collapses these to allowed True/False).
CLEAN = "clean"
WARNING = "warning"
CRITICAL = "critical"

_SUSPICIOUS_DECODE_KEYWORDS = ("ignore", "instruction", "system", "prompt", "jailbreak")

# =============================================================================
# PATTERN TABLES
# Mirrors the hook's categories; each entry is (compiled-regex, verdict).
# =============================================================================

# --- direct instruction injection (override attempts) ---------------------- #
_DIRECT_INSTRUCTION_PATTERNS: List[Tuple[str, str]] = [
    # CRITICAL: ignore/disregard/forget/override of prior instructions.
    (
        r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|guidelines?|rules?)",
        CRITICAL,
    ),
    (
        r"disregard\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|guidelines?|rules?)",
        CRITICAL,
    ),
    (
        r"forget\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|guidelines?|rules?)",
        CRITICAL,
    ),
    (
        r"override\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|guidelines?|rules?)",
        CRITICAL,
    ),
    (
        r"(assistant|ai|claude|gpt|model)\s*[,:]\s*(ignore|disregard|forget|skip)",
        CRITICAL,
    ),
    (r"system\s*:\s*(ignore|disregard|forget|override)", CRITICAL),
    # WARNING: new-instruction framing.
    (r"(new|updated|revised)\s+instructions?\s*:", WARNING),
    (r"from\s+now\s+on\s*,?\s*(you\s+)?(will|must|should|shall)", WARNING),
    (r"your\s+new\s+(instructions?|task|objective|goal)\s*(is|are)?", WARNING),
    (r"\[INST\]|\[/INST\]|\[SYSTEM\]|\[/SYSTEM\]", WARNING),
    (r"<\|im_start\|>|<\|im_end\|>", WARNING),
    (r"###\s*(instruction|system|human|assistant)\s*:?\s*$", WARNING),
]

# --- role manipulation ------------------------------------------------------ #
_ROLE_MANIPULATION_PATTERNS: List[Tuple[str, str]] = [
    (r"you\s+are\s+(now\s+)?(a|an)\s+(different|new|unrestricted)", CRITICAL),
    (
        r"pretend\s+(to\s+be|you\s+are)\s+(a|an)?\s*(different|evil|unrestricted|jailbroken)",
        CRITICAL,
    ),
    (
        r"act\s+as\s+if\s+you\s+(have\s+no|don't\s+have|lack)\s+(restrictions?|limitations?|guidelines?)",
        CRITICAL,
    ),
    (r"roleplay\s+as\s+(a|an)?\s*(unrestricted|evil|malicious)", CRITICAL),
    (r"\bDAN\b.*\bdo\s+anything\s+now\b", CRITICAL),
    (r"jailbreak(ed)?\s+(mode|version|persona)", CRITICAL),
    (r"developer\s+mode\s+(enabled|activated|on)", CRITICAL),
    (r"bypass\s+(your\s+)?(safety|security|restrictions?|filters?)", CRITICAL),
    (r"you\s+are\s+no\s+longer\s+(bound|restricted|limited)", WARNING),
    (r"unlock(ed)?\s+(your\s+)?(full|true)\s+(potential|capabilities)", WARNING),
]

# --- delimiter escape / context smuggling ----------------------------------- #
_DELIMITER_ESCAPE_PATTERNS: List[Tuple[str, str]] = [
    (
        r"</?(system|instruction|prompt|context|user|assistant)[-_]?(message|prompt)?(/)?>\s*",
        WARNING,
    ),
    (r"<\|endof(text|turn|prompt)\|>", WARNING),
    (r"```[ \t]*(system|instruction|prompt)\b(?![\w-])", WARNING),
    (r"---+\s*(end|begin)\s+of\s+(system\s+prompt|instructions?)\s*---", WARNING),
    (r"<!--\s*(ignore|disregard|new\s+instruction)", CRITICAL),
    (r"/\*\s*(ignore|disregard|new\s+instruction)", WARNING),
    (
        r"#\s*(?:system|instruction|prompt)\s*:\s*(?:ignore|disregard|override|you\s+(?:are|must)|new\s+instruction)",
        WARNING,
    ),
]

# --- jailbreak techniques --------------------------------------------------- #
_JAILBREAK_PATTERNS: List[Tuple[str, str]] = [
    (r"pretend\s+this\s+is\s+a\s+(hypothetical|fictional|roleplay)", WARNING),
    (
        r"for\s+(educational|research|academic)\s+purposes\s+only[,. ]+(?:explain|show|tell|describe|how\s+to)\b",
        WARNING,
    ),
    (
        r"this\s+is\s+(just\s+)?(a\s+)?(?:hypothetical|fictional|roleplay)\s+(?:test|experiment|simulation)",
        WARNING,
    ),
    (r"in\s+a\s+fictional\s+(world|scenario|story)\s+where", WARNING),
    (r"my\s+(grandma|grandmother)\s+(used\s+to|would)\s+(tell|read)", WARNING),
    (r"tell\s+me\s+a\s+story\s+(about|where)\s+.*(hack|exploit|bypass)", WARNING),  # quality-gate: allow -- detects these words
    (r"split\s+(this|the)\s+(word|text|message)\s+into", WARNING),
]

# --- system prompt leak attempts -------------------------------------------- #
_SYSTEM_PROMPT_LEAK_PATTERNS: List[Tuple[str, str]] = [
    (
        r"(show|reveal|repeat)\s+(me\s+)?your\s+(system\s+|initial\s+|original\s+)?prompt",
        CRITICAL,
    ),
    (
        r"repeat\s+(your\s+)?(system\s+)?(instructions?|prompt)\s+(back|verbatim)",
        CRITICAL,
    ),
    (
        r"(copy|paste|echo)\s+(your\s+)?(initial|original|system)\s+(prompt|instructions?)",
        CRITICAL,
    ),
    (
        r"what\s+(are|is)\s+your\s+(system\s+)?(instructions?|prompt|guidelines?)",
        WARNING,
    ),
]

# --- tool abuse (command injection / exfiltration) --------------------------- #
_DESTRUCTIVE_FORMS = [
    r"rm\s+-[A-Za-z]*[rf]",
    r"del\s+(?:/[A-Za-z]\s+)*[A-Za-z]:",
    r"format\s+(?:[A-Za-z]:|/)",
    r"shutdown\s+(?:-\w|/[sr]|now)",
    r"reboot\s+(?:-\w|/[rf]|now)",
]
_DESTRUCTIVE_CMD = r"(?:" + r"|".join(_DESTRUCTIVE_FORMS) + r")"

_TOOL_ABUSE_PATTERNS: List[Tuple[str, str]] = [
    (r";\s*" + _DESTRUCTIVE_CMD, CRITICAL),
    (r"\|\s*" + _DESTRUCTIVE_CMD, CRITICAL),
    (
        r"\b(curl|wget|nc|netcat)\b\s+[^\n]{0,80}(api[_-]?key|password|secret|token)\s*[=:]",
        CRITICAL,
    ),
    (r"\.\./\.\./", WARNING),
    (r"/etc/(passwd|shadow|hosts)", WARNING),
    (r"~/.ssh/", WARNING),
    (r"send\s+(to|via)\s+(email|http|webhook)", WARNING),
]

# --- indirect injection (instructions embedded in retrieved data) ------------ #
_INDIRECT_INJECTION_PATTERNS: List[Tuple[str, str]] = [
    (
        r"(?i)IMPORTANT\s+INSTRUCTION\s+FOR\s+(?:AI|ASSISTANT|CLAUDE|MODEL|LLM)",
        CRITICAL,
    ),
    (
        r"//\s*(?:ignore|override|disregard)\s+(?:previous|all)\s+instructions?",
        CRITICAL,
    ),
    (r"#\s*(?:ignore|override|disregard)\s+(?:previous|all)\s+instructions?", CRITICAL),
    (
        r"/\*\s*(?:ignore|override|disregard)\s+(?:previous|all)\s+instructions?",
        CRITICAL,
    ),
    (
        r"(?i)(?:postinstall|preinstall|prepare)\s*[=:]\s*.*(?:curl|wget|bash|sh)\s+https?://",
        CRITICAL,
    ),
    (r"(?i)(?:NOTE|ATTENTION)\s+(?:TO|FOR)\s+(?:AI|ASSISTANT|CLAUDE)[\s:]+", WARNING),
    (r"(?i)AI\s+AGENT\s*:\s*(?:please|you\s+must|immediately)", WARNING),
    (
        r"(?i)IF\s+YOU\s+ARE\s+(?:AN?\s+)?(?:AI|LLM|LANGUAGE\s+MODEL|ASSISTANT)\s*[,:]?\s*(?:reading|processing|you\s+must|ignore|then\s+you)",
        WARNING,
    ),
    (r"(?i)WHEN\s+(?:AN?\s+)?AI\s+(?:READS?|PROCESSES?|SEES?)\s+THIS", WARNING),
    (
        r"(?i)(?:TODO|FIXME|HACK|XXX)\s*:\s*(?:ignore|override|new)\s+instructions?",  # quality-gate: allow -- detection vocabulary
        WARNING,
    ),
    (r"<!--\s*(?:system|admin|override|instruction)\s*:", WARNING),
    (r"<!--\s*(?:AI|assistant|claude)\s*[,:]\s*", WARNING),
    (
        r"(?i)(?:description|readme|about)\s*[=:]\s*[^\n]{0,40}(?:ignore|override|disregard)\s+(?:(?:all|previous)\s+){1,2}(?:instructions?|rules?)\b",
        WARNING,
    ),
    (
        r"(?i)(?:^|\n)(?:commit|merge|squash)\s+.*(?:ignore|override)\s+(?:instructions?|rules?)",
        WARNING,
    ),
    (r"(?i)\[//\]\s*:\s*#\s*\(.*(?:instruction|ignore|override)", WARNING),
    (r"(?i)\.\.\s+(?:note|warning|attention)\s*::\s*(?:AI|assistant|ignore)", WARNING),
]

# --- hidden-text signals (invisible unicode, braille, tag block) ------------- #
_HIDDEN_TEXT_PATTERNS: List[Tuple[str, str]] = [
    (r"[​‌‍⁠﻿]{3,}", WARNING),
    (r"[‪-‮⁦-⁩]", WARNING),
    (r"[\U000e0001-\U000e007f]", WARNING),
    (r"[⠀-⣿]{3,}", WARNING),
]

_CATEGORY_NAMES: List[Tuple[str, str]] = [
    ("direct_instruction", _DIRECT_INSTRUCTION_PATTERNS),
    ("role_manipulation", _ROLE_MANIPULATION_PATTERNS),
    ("delimiter_escape", _DELIMITER_ESCAPE_PATTERNS),
    ("jailbreak", _JAILBREAK_PATTERNS),
    ("system_prompt_leak", _SYSTEM_PROMPT_LEAK_PATTERNS),
    ("tool_abuse", _TOOL_ABUSE_PATTERNS),
    ("indirect_injection", _INDIRECT_INJECTION_PATTERNS),
    ("hidden_text", _HIDDEN_TEXT_PATTERNS),
]

_BASE64_RE = re.compile(r"[A-Za-z0-9+/]{40,}={0,2}")
_HEX_RE = re.compile(r"(?:0x)?[0-9a-fA-F]{20,}")
_HTML_COMMENT_RE = re.compile(r"<!--(.*?)-->", re.DOTALL)

_COMPILED: Optional[List[Tuple[str, str, Any]]] = None


def _compiled_patterns() -> List[Tuple[str, str, Any]]:
    """Compile pattern tables once (category, verdict, compiled regex)."""
    global _COMPILED
    if _COMPILED is None:
        compiled: List[Tuple[str, str, Any]] = []
        for category, patterns in _CATEGORY_NAMES:
            for pattern, verdict in patterns:
                try:
                    compiled.append(
                        (
                            category,
                            verdict,
                            re.compile(pattern, re.IGNORECASE | re.MULTILINE),
                        )
                    )
                except re.error as e:
                    logger.warning("bad injection pattern %r: %s", pattern, e)
        _COMPILED = compiled
    return _COMPILED


# =============================================================================
# SCANNING
# =============================================================================


def _normalize_content(content: str) -> str:
    """Normalize content for pattern matching (mirrors the hook)."""
    normalized = unicodedata.normalize("NFKC", content)
    normalized = re.sub(r"[​‌‍⁠﻿]", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def _detect_encoding_attacks(content: str) -> List[Dict[str, Any]]:
    """Base64/hex payloads that decode to injection vocabulary (verdict critical)."""
    hits: List[Dict[str, Any]] = []

    for match in _BASE64_RE.finditer(content):
        blob = match.group()
        try:
            decoded = base64.b64decode(blob).decode("utf-8", errors="ignore")
        except Exception:
            continue
        low = decoded.lower()
        if any(k in low for k in _SUSPICIOUS_DECODE_KEYWORDS):
            hits.append(
                {
                    "category": "encoding_attack",
                    "verdict": CRITICAL,
                    "pattern": "base64:" + blob[:40],
                    "snippet": _snippet(content, match.start()),
                }
            )

    for match in _HEX_RE.finditer(content):
        blob = match.group().replace("0x", "")
        try:
            decoded = bytes.fromhex(blob).decode("utf-8", errors="ignore")
        except Exception:
            continue
        low = decoded.lower()
        if any(k in low for k in ("ignore", "instruction", "system", "prompt")):
            hits.append(
                {
                    "category": "encoding_attack",
                    "verdict": CRITICAL,
                    "pattern": "hex:" + blob[:40],
                    "snippet": _snippet(content, match.start()),
                }
            )

    return hits


def _detect_hidden_text(content: str) -> List[Dict[str, Any]]:
    """HTML comments containing instruction vocabulary (verdict warning)."""
    hits: List[Dict[str, Any]] = []
    for match in _HTML_COMMENT_RE.finditer(content):
        comment = match.group(1).lower()
        if any(
            k in comment
            for k in ("ignore", "instruction", "system", "prompt", "inject")
        ):
            hits.append(
                {
                    "category": "hidden_text",
                    "verdict": WARNING,
                    "pattern": "html_comment:" + match.group()[:60],
                    "snippet": _snippet(content, match.start()),
                }
            )
    return hits


def _snippet(content: str, position: int, window: int = 60) -> str:
    start = max(0, position - window)
    end = min(len(content), position + window)
    snippet = content[start:end]
    if start > 0:
        snippet = "..." + snippet
    if end < len(content):
        snippet = snippet + "..."
    return snippet


def scan_for_injection(content: str) -> Dict[str, Any]:
    """
    Scan content for prompt-injection patterns.

    Returns:
        {"verdict": "clean"|"warning"|"critical",
         "matched_patterns": [{category, verdict, pattern, snippet}, ...],
         "reason": str}
    """
    if not content:
        return {
            "verdict": CLEAN,
            "matched_patterns": [],
            "reason": "No content to scan.",
        }

    normalized = _normalize_content(content)
    matched: List[Dict[str, Any]] = []

    # Pattern pass.
    for category, verdict, regex in _compiled_patterns():
        for m in regex.finditer(normalized):
            matched.append(
                {
                    "category": category,
                    "verdict": verdict,
                    "pattern": m.group()[:80],
                    "snippet": _snippet(normalized, m.start()),
                }
            )

    # Encoding attacks (critical by construction).
    matched.extend(_detect_encoding_attacks(content))
    # Hidden-text (HTML comments) — warning.
    matched.extend(_detect_hidden_text(content))

    # Deduplicate identical (verdict, category, pattern) hits.
    seen = set()
    unique: List[Dict[str, Any]] = []
    for hit in matched:
        key = (hit["verdict"], hit["category"], hit["pattern"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(hit)

    unique = unique[:12]

    if not unique:
        return {
            "verdict": CLEAN,
            "matched_patterns": [],
            "reason": "No prompt-injection patterns detected.",
        }

    if any(h["verdict"] == CRITICAL for h in unique):
        critical_kinds = sorted(
            {h["category"] for h in unique if h["verdict"] == CRITICAL}
        )
        return {
            "verdict": CRITICAL,
            "matched_patterns": unique,
            "reason": (
                f"{len(unique)} pattern(s) matched; {len(critical_kinds)} critical "
                f"kind(s): {', '.join(critical_kinds)}. Content is blocked unless "
                "its exact hash is allowlisted."
            ),
        }

    return {
        "verdict": WARNING,
        "matched_patterns": unique,
        "reason": (
            f"{len(unique)} suspicious pattern(s) matched (e.g. "
            f"{unique[0]['category']}). Review before writing to shared memory."
        ),
    }


# =============================================================================
# ALLOWLIST + FAIL-CLOSED GATE
# =============================================================================


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _coerce_path(value: Optional[Path]) -> Optional[Path]:
    """Accept either a Path or a str path (tool callers pass strings)."""
    if value is None or isinstance(value, Path):
        return value
    return Path(value)


def _load_allowlist_entries(allowlist_path: Path) -> List[Dict[str, Any]]:
    """Read sha256 entries from the allowlist file. Absent/unreadable -> []."""
    allowlist_path = _coerce_path(allowlist_path)  # type: ignore[assignment]
    try:
        if not allowlist_path.is_file():
            return []
        payload = json.loads(allowlist_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        logger.warning(
            "allowlist %s unreadable (%s); no entries apply", allowlist_path, e
        )
        return []

    if isinstance(payload, list):
        entries = payload
    elif isinstance(payload, dict):
        entries = payload.get("approvals") or payload.get("entries") or []
    else:
        return []
    return [e for e in entries if isinstance(e, dict)]


def is_allowlisted(content: str, allowlist_path: Optional[Path] = None) -> bool:
    """
    Content-hash allowlist check.

    Reuses `~/.claude/security/injection_overrides.json` (entries carry a
    `sha256` of the approved raw content). An entry that has an
    `expires_at_epoch` in the past does NOT grant allowlist (fail closed).
    If the file is absent, no allowlist applies — returns False, not an error.

    Args:
        content: The content whose hash to compare.
        allowlist_path: Override path (testing); defaults to
            ~/.claude/security/injection_overrides.json.

    Returns:
        True iff the exact content hash is in the allowlist and unexpired.
    """
    path = _coerce_path(allowlist_path) or _DEFAULT_ALLOWLIST
    digest = _content_hash(content)
    now = time.time()

    for entry in _load_allowlist_entries(path):
        approved = entry.get("sha256")
        if not isinstance(approved, str) or not approved:
            continue
        if approved.strip().lower() != digest:
            continue
        expires = entry.get("expires_at_epoch")
        if expires is not None:
            try:
                if float(expires) < now:
                    continue  # expired approval does not grant allowlist
            except (TypeError, ValueError):
                continue  # unparseable expiry: fail closed
        return True
    return False


def guard_shared_write(
    content: str, allowlist_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Fail-closed gate for writing content to shared memory.

    A CRITICAL verdict with no allowlist entry => allowed=False.
    Everything else (clean, warning) is allowed.

    Returns:
        {"allowed": bool, "verdict": str, "reason": str}
    """
    scan = scan_for_injection(content)
    verdict = scan["verdict"]
    if verdict == CRITICAL and not is_allowlisted(content, allowlist_path):
        return {
            "allowed": False,
            "verdict": verdict,
            "reason": (
                scan["reason"] + " No allowlist entry for this content hash; "
                "write blocked (fail closed)."
            ),
        }
    return {
        "allowed": True,
        "verdict": verdict,
        "reason": scan["reason"],
    }
