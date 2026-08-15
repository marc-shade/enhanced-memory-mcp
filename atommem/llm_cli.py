"""Headless-CLI LLM helper for the AtomMem upgrades.

Project rule (rules/intent-engineering.md, NON-NEGOTIABLE): AI calls in any
agentic-system path go through headless CLIs (claude --print, codex exec,
gemini), NEVER a provider SDK. This module is the single LLM entry point for
the atommem package. It deliberately does not import anthropic/openai/gemini.

Design contract:
  * Fail SOFT. Every dependent feature (atomic-fact extraction, conflict
    detection, profile-update decisions) must degrade gracefully when no CLI is
    installed (e.g. a headless cron host). call_json returns
    {"_unavailable": True} or {"_error": ...} instead of raising, so callers
    fall back to deterministic behaviour.
  * Redact obvious secrets before transmission (secret firewall), mirroring
    intelligent-agents/maker_headless_providers.py.
  * Resolve binaries via PATH first, then known install locations — never
    pin a single node version path inline.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# ---- binary resolution ---------------------------------------------------- #
_KNOWN_DIRS = [
    os.path.expanduser("~/.nvm/versions/node/v24.7.0/bin"),
    os.path.expanduser("~/.bun/bin"),
    # npx-style installs live here (codex, claude). Without this, the launchd
    # MCP daemon (minimal PATH) can only resolve ornith + ollama and the
    # provider fallback chain loses its cloud legs (observed 2026-08-05: the
    # daemon's provider list was ["ornith", "ollama"] while an interactive
    # shell resolved ["ornith", "claude", "codex", "ollama"]).
    os.path.expanduser("~/.local/bin"),
    "/opt/homebrew/bin",
    "/usr/local/bin",
]


def _resolve(binary: str) -> Optional[str]:
    found = shutil.which(binary)
    if found:
        return found
    for d in _KNOWN_DIRS:
        cand = os.path.join(d, binary)
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return None


# ---- secret firewall ------------------------------------------------------ #
_SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),  # OpenAI / Anthropic style
    re.compile(r"(?i)\b(api[_-]?key|secret|token|password)\b\s*[:=]\s*\S+"),
    re.compile(r"AKIA[0-9A-Z]{16}"),  # AWS access key id
    re.compile(r"ghp_[A-Za-z0-9]{36}"),  # GitHub PAT
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),  # Slack
]


def redact_secrets(text: str) -> tuple[str, int]:
    redactions = 0
    out = text
    for pat in _SECRET_PATTERNS:
        out, n = pat.subn("[REDACTED]", out)
        redactions += n
    return out, redactions


# ---- providers ------------------------------------------------------------ #
@dataclass
class Provider:
    name: str
    binary: str
    args: List[str]
    timeout: int = 60


def _default_providers() -> List[Provider]:
    # Order = preference. Each entry is a CLI looked up on PATH; a provider that
    # is not installed exits non-zero, _run returns None, and the next provider
    # is tried. All of them are optional -- with none installed the callers fall
    # back to their non-LLM path.
    # Set MEMORY_LLM_CLI to prepend your own command (e.g. a local model runner).
    providers: List[Provider] = []
    custom = os.environ.get("MEMORY_LLM_CLI")
    if custom:
        providers.append(Provider("custom", custom, [], timeout=120))
    providers += [
        Provider("claude", "claude", ["--print", "--model", "haiku", "--"], timeout=60),
        Provider("codex", "codex", ["exec", "-m", "gpt-5.5", "--"], timeout=90),
        Provider("gemini", "gemini", [], timeout=60),
        Provider("ollama", "ollama", ["run", "gpt-oss:20b-cloud"], timeout=90),
    ]
    return providers


def _extract_json(output: str) -> Optional[Any]:
    """Parse JSON from possibly-noisy CLI stdout. Supports objects and arrays.

    Handles markdown fences and reasoning-model preambles (e.g. ollama
    gpt-oss emits "Thinking..." + prose before the JSON). Without preamble
    stripping, a brace or bracket inside the prose wins the balanced-block scan
    and the real JSON is never found (observed 2026-08-05: ollama output parsed
    False while ornith/codex parsed True).
    """
    if not output:
        return None
    s = output.strip()
    # Strip markdown fences.
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", s, re.DOTALL)
    if fence:
        s = fence.group(1).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # Reasoning-model preambles never start a line with an opening brace or
    # bracket; jump to the first line that does. Prose like "Here: {...}"
    # is handled by the balanced-block scan below as a last resort.
    lines = s.splitlines()
    for i, ln in enumerate(lines):
        if ln.lstrip()[:1] in ("{", "["):
            s = "\n".join(lines[i:]).strip()
            break
    # Find the first balanced {...} or [...] block.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = s.find(opener)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(s)):
            if s[i] == opener:
                depth += 1
            elif s[i] == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(s[start : i + 1])
                    except json.JSONDecodeError:
                        break
    return None


class HeadlessLLM:
    """Single-shot headless-CLI LLM wrapper with provider fallback."""

    def __init__(self, providers: Optional[List[Provider]] = None):
        self._providers = providers or _default_providers()
        self._resolved: Dict[str, Optional[str]] = {}

    def _path_for(self, p: Provider) -> Optional[str]:
        if p.binary not in self._resolved:
            self._resolved[p.binary] = _resolve(p.binary)
        return self._resolved[p.binary]

    def available(self) -> bool:
        return any(self._path_for(p) for p in self._providers)

    def available_providers(self) -> List[str]:
        return [p.name for p in self._providers if self._path_for(p)]

    def _run(self, p: Provider, prompt: str) -> Optional[str]:
        path = self._path_for(p)
        if not path:
            return None
        cmd = [path] + p.args + [prompt]
        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=p.timeout,
                stdin=subprocess.DEVNULL,  # codex/others hang waiting on stdin otherwise
            )
        except subprocess.TimeoutExpired:
            return None
        except Exception:
            return None
        out = (res.stdout or "").strip()
        if not out and res.stderr:
            # Some CLIs emit the answer on stderr; only use if it looks like content.
            err = res.stderr.strip()
            if err and "error" not in err.lower()[:40]:
                out = err
        return out or None

    def call_text(
        self, system: str, user: str, prefer: Optional[str] = None
    ) -> Dict[str, Any]:
        """Return {"text": ...} or {"_unavailable": True}/{"_error": ...}."""
        prompt, n = redact_secrets(f"{system.strip()}\n\n{user.strip()}")
        if n:
            # Visible, not silent — secret firewall fired.
            import sys

            sys.stderr.write(
                f"[atommem.llm_cli] redacted {n} secret(s) before transmission\n"
            )
        providers = self._ordered(prefer)
        if not any(self._path_for(p) for p in providers):
            return {"_unavailable": True}
        for p in providers:
            out = self._run(p, prompt)
            if out:
                return {"text": out, "_provider": p.name}
        return {"_error": "all providers failed"}

    def call_json(
        self, system: str, user: str, prefer: Optional[str] = None
    ) -> Dict[str, Any]:
        """Return parsed JSON dict/list under {"data": ...}, or a soft error.

        Adds a strict "Output ONLY valid JSON" instruction to the system prompt.
        """
        sys_json = (
            system.strip()
            + "\n\nIMPORTANT: Output ONLY valid JSON. No prose, no markdown fences."
        )
        res = self.call_text(sys_json, user, prefer=prefer)
        if "_unavailable" in res or "_error" in res:
            return res
        data = _extract_json(res["text"])
        if data is None:
            return {
                "_error": "unparseable_json",
                "_raw": res["text"][:500],
                "_provider": res.get("_provider"),
            }
        return {"data": data, "_provider": res.get("_provider")}

    def _ordered(self, prefer: Optional[str]) -> List[Provider]:
        if not prefer:
            return self._providers
        pref = [p for p in self._providers if p.name == prefer]
        rest = [p for p in self._providers if p.name != prefer]
        return pref + rest


# Module-level singleton for cheap reuse.
_DEFAULT: Optional[HeadlessLLM] = None


def get_llm() -> HeadlessLLM:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = HeadlessLLM()
    return _DEFAULT


if __name__ == "__main__":
    llm = get_llm()
    print("resolved providers:", llm.available_providers())
    print("available:", llm.available())
    if llm.available():
        r = llm.call_json(
            "You convert a sentence into JSON.",
            'Return {"echo": "<the word HELLO in lowercase>"} for input "say hello".',
        )
        print("json call result:", r)
