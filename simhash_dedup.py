"""Near-duplicate observation detection via 64-bit word-token simhash.

Why this exists. create_entities skips EXACT-content duplicate observations
(issue #8), but a re-worded re-import -- the same seed file lightly edited --
still multiplies rows and skews FTS relevance toward whatever was re-imported.
LLM-based similarity would be slow and non-deterministic on a write path with
a 300ms-class budget; simhash is O(tokens), stdlib-only, and gives the same
answer every run (the Zep/Graphiti observation that traditional IR beats LLM
calls for extraction-side hygiene, applied here).

Why REPORT and not DROP. A correction is a near-duplicate: "ai-lab has 62Gi
RAM" -> "ai-lab has 125Gi RAM" measured hamming distance 15 on this very
store's history, and silently dropping the second would have preserved a
wrong fact forever (that exact under-read persisted for six weeks once; see
the ai-lab node card). The standing memory rule is stale > deletion: never
silently drop facts. So the default policy reports near-duplicates in the
response and inserts anyway; `skip` exists as an OPT-IN for import pipelines
that know they are re-importing, set via ENHANCED_MEMORY_NEAR_DUP_POLICY.

Threshold. Measured 2026-08-15 on three populations (word-token simhash,
64-bit, blake2b feature hashing):

    re-worded same fact      distance  0..14   (n=4, incl. a pure reorder at 0)
    corrections              distance 15..25   (n=3)
    unrelated observations   distance 27..31   (n=4)

DEFAULT_MAX_DISTANCE = 16 covers every measured reword with margin 2 and sits
11 under the nearest unrelated pair. The 62Gi-style correction at 15 lands
inside the flag range, which is the desired behavior under report-don't-drop:
the caller is told "this looks like a rewrite of an existing row" and the row
is stored regardless. Char-4-gram shingles were measured too and REJECTED:
their reword band (5..21) overlaps their correction band (13..19).

Small n; bands are indicative, not proof. Re-run the calibration before
tightening the threshold.

Gaps / not covered: very short observations (1-2 tokens) produce noisy
fingerprints -- pairs like "short note"/"tiny memo" measured 27, safely
apart, but no exhaustive short-text sweep was done. Cross-entity
near-duplicates are out of scope here (per-entity check only); the
consolidation layer is the right home for a store-wide sweep.
"""

import hashlib
import os
import re
from typing import List, Optional, Tuple

DEFAULT_MAX_DISTANCE = 16
POLICY_ENV = "ENHANCED_MEMORY_NEAR_DUP_POLICY"
_VALID_POLICIES = ("report", "skip")


def near_dup_policy() -> str:
    """Current policy: 'report' (default; flag and insert) or 'skip' (drop
    near-duplicates -- opt-in for deliberate re-imports). An unknown value
    falls back to 'report', the safe side: worst case is noise in a response
    field, never a silently dropped fact."""
    policy = os.environ.get(POLICY_ENV, "report").strip().lower()
    return policy if policy in _VALID_POLICIES else "report"


def simhash64(text: str) -> int:
    """64-bit simhash over lowercased word tokens.

    Word tokens, not char shingles, per the measured band separation above.
    Bag-of-words means a pure reorder hashes identically (distance 0) --
    correct for this purpose: a reordered sentence IS the same observation.
    """
    tokens = re.findall(r"\w+", text.lower())
    if not tokens:
        return 0
    v = [0] * 64
    for tok in tokens:
        h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "big")
        for bit in range(64):
            v[bit] += 1 if (h >> bit) & 1 else -1
    out = 0
    for bit in range(64):
        if v[bit] > 0:
            out |= 1 << bit
    return out


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def find_near_duplicate(
    new_text: str,
    existing_texts: List[str],
    max_distance: int = DEFAULT_MAX_DISTANCE,
) -> Optional[Tuple[str, int]]:
    """The closest existing text within max_distance, or None.

    Exact matches are the caller's job (cheap SQL equality, runs first);
    this only answers the near-miss question. Returns (existing_text,
    distance) for the closest hit so the response can show WHICH row the new
    observation resembles -- a bare count would tell the operator something
    happened but not what to review.
    """
    new_hash = simhash64(new_text)
    best = None
    for text in existing_texts:
        if text == new_text:
            continue  # exact dup, already handled upstream
        d = hamming(new_hash, simhash64(text))
        if d <= max_distance and (best is None or d < best[1]):
            best = (text, d)
    return best
