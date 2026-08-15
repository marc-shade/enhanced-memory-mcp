"""Delta 1: Atomic-fact extraction / normalization.

Ports AtomMem's Atomic Fact Extractor (data/SFT_training_data.json instruction +
prompts/fact_metadata_extraction_prompt.txt) onto our headless-CLI convention.

Transforms raw dialogue / observation text into objective, self-contained,
coreference-resolved, time-anchored third-person facts, each with structured
metadata (people, keywords, time, profile flag).

The AtomMem paper fine-tunes a Qwen3-14B extractor for this; we use an
instruction-tuned headless model (claude --print) which is adequate for online
use. Falls back to a deterministic single-fact passthrough when no CLI is
available, so the write path never blocks on the LLM.

Output unit (one "structured atomic fact"):
    {
      "fact": "<standalone third-person statement>",
      "people": ["Emma", ...],
      "keywords": ["psychology", "exam", ...],   # <= 5, singular noun-ish
      "time": ["<event_time YYYY-MM-DD|''>", "<interaction_time>"],
      "needs_profile": <bool>,                    # stable attribute about a person?
    }
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .keywords import extract_keywords
from .llm_cli import HeadlessLLM, get_llm

# Ported verbatim from AtomMem SFT instruction (data/SFT_training_data.json).
_EXTRACT_SYSTEM = """Extract high-value factual information from the given dialogue and rewrite it as objective third-person facts.

Requirements:
- Ignore greetings, pleasantries, acknowledgements, and other low-value content.
- Use dialogue context to infer implicit references, causality, or speaker intent when necessary.
- Resolve all pronouns and vague references into explicit entities.
- Infer specific time information when possible (anchor relative times like "yesterday" against the session time).
- Each fact must be complete and standalone (understandable with no other context).
- Output the result as a JSON array of strings."""

# Ported from prompts/fact_metadata_extraction_prompt.txt (condensed).
_METADATA_SYSTEM = """Extract metadata for a fact already extracted from conversation.

Extract:
1. people: all complete person names mentioned in the fact.
2. keywords: up to 5 retrieval keywords (most important topics/entities/fields),
   noun form, singular when possible. Never use a speaker's name, "user",
   "assistant", or a time as a keyword.
3. time: two fields [event_time, interaction_time].
   - event_time: when the event/action occurred (YYYY-MM-DD, or year-only, or "").
     Anchor relative references ("yesterday", "last summer") against session_time.
   - interaction_time: always the session_time.
4. needs_profile_extraction: true only if the fact states a STABLE long-term
   attribute about a person (gender, occupation, hometown, hobby, relationship,
   personality, interest). False for one-time events, actions, or emotions.

Output JSON:
{"people": [...], "keywords": [...], "time": ["event_or_empty", "interaction"], "needs_profile_extraction": true/false}"""


class AtomicFactExtractor:
    def __init__(self, llm: Optional[HeadlessLLM] = None):
        self.llm = llm or get_llm()
        # Populated when the LLM path fails (unavailable / unparseable output),
        # so callers can distinguish "extraction genuinely failed" from "the
        # model was available and deliberately extracted nothing".
        self.last_extract_error: Optional[Dict[str, Any]] = None

    # ---- raw text -> list of standalone fact strings ---------------------- #
    def extract_facts(
        self,
        dialogue: str,
        session_time: str = "",
        speaker: str = "",
        context: str = "",
    ) -> List[str]:
        user = self._build_extract_user(dialogue, session_time, speaker, context)
        res = self.llm.call_json(_EXTRACT_SYSTEM, user)
        data = res.get("data")
        if isinstance(data, list):
            self.last_extract_error = None
            return [str(x).strip() for x in data if str(x).strip()]
        if isinstance(data, dict):
            # Some models wrap the array, e.g. {"facts": [...]}.
            for v in data.values():
                if isinstance(v, list):
                    self.last_extract_error = None
                    return [str(x).strip() for x in v if str(x).strip()]
        # LLM unavailable or unparseable; record why so the caller can report
        # degradation honestly instead of a silent passthrough.
        if "_unavailable" in res:
            self.last_extract_error = {"error": "llm_unavailable"}
        elif "_error" in res:
            self.last_extract_error = {k: v for k, v in res.items() if k != "data"}
        else:
            self.last_extract_error = {"error": "no_facts_in_llm_output"}
        return []

    @staticmethod
    def _build_extract_user(
        dialogue: str, session_time: str, speaker: str, context: str
    ) -> str:
        parts = []
        if session_time:
            parts.append(f"Session time: {session_time}")
        if context:
            parts.append(f"Previous context: {context}")
        head = f"[{speaker}]: " if speaker else ""
        parts.append(f"Current dialogue:\n{head}{dialogue}")
        parts.append(
            "Extract all factual information and output a JSON array of strings."
        )
        return "\n\n".join(parts)

    # ---- fact -> metadata ------------------------------------------------- #
    def extract_metadata(
        self, fact: str, session_time: str = "", context: str = ""
    ) -> Dict[str, Any]:
        user_parts = [f"Fact: {fact}"]
        if session_time:
            user_parts.append(f"session_time: {session_time}")
        if context:
            user_parts.append(f"Context (previous turn): {context}")
        res = self.llm.call_json(_METADATA_SYSTEM, "\n".join(user_parts))
        data = res.get("data")
        if isinstance(data, dict):
            return self._coerce_metadata(data, fact, session_time)
        return self._fallback_metadata(fact, session_time)

    @staticmethod
    def _coerce_metadata(
        data: Dict[str, Any], fact: str, session_time: str
    ) -> Dict[str, Any]:
        people = [str(p).strip() for p in (data.get("people") or []) if str(p).strip()]
        keywords = [
            str(k).strip().lower()
            for k in (data.get("keywords") or [])
            if str(k).strip()
        ]
        if not keywords:
            keywords = extract_keywords(fact)
        time = data.get("time")
        if not (isinstance(time, list) and len(time) == 2):
            time = ["", session_time]
        return {
            "people": people,
            "keywords": keywords[:5],
            "time": [str(time[0] or ""), str(time[1] or session_time)],
            "needs_profile": bool(data.get("needs_profile_extraction", False)),
        }

    @staticmethod
    def _fallback_metadata(fact: str, session_time: str) -> Dict[str, Any]:
        return {
            "people": [],
            "keywords": extract_keywords(fact),
            "time": ["", session_time],
            "needs_profile": False,
        }

    # ---- combined: raw text -> structured atomic facts -------------------- #
    def extract_structured(
        self,
        dialogue: str,
        session_time: str = "",
        speaker: str = "",
        context: str = "",
        with_metadata: bool = True,
    ) -> List[Dict[str, Any]]:
        facts = self.extract_facts(dialogue, session_time, speaker, context)
        if not facts:
            if self.last_extract_error:
                # LLM path failed (unavailable / unparseable): keep the raw text
                # as a single fact so nothing is silently dropped.
                cleaned = " ".join(dialogue.split()).strip()
                if not cleaned:
                    return []
                return [
                    {
                        "fact": cleaned,
                        **self._fallback_metadata(cleaned, session_time),
                        "_extracted": False,
                    }
                ]
            # LLM was available and deliberately extracted nothing (empty array).
            # Do NOT invent a passthrough "fact" for that case.
            return []
        out: List[Dict[str, Any]] = []
        for f in facts:
            if with_metadata:
                meta = self.extract_metadata(f, session_time, context)
            else:
                meta = self._fallback_metadata(f, session_time)
            out.append({"fact": f, **meta, "_extracted": True})
        return out


if __name__ == "__main__":
    ex = AtomicFactExtractor()
    print("LLM available:", ex.llm.available(), ex.llm.available_providers())
    facts = ex.extract_structured(
        dialogue="Hey Zoe, guess what! I got an A on my first psychology exam last Friday! "
        "Also I'm a sophomore studying psych and I just got a mini cactus for my dorm.",
        session_time="2023-05-17",
        speaker="Emma",
        with_metadata=True,
    )
    import json

    print(json.dumps(facts, indent=2, ensure_ascii=False))
