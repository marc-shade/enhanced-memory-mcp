"""Delta 3: Residual-delta verification write path.

Ports AtomMem's fact-verification pipeline (src/fact_storage.py): before a new
fact is stored, decide whether it is a duplicate, a conflict, or genuinely novel
-- and when novel-but-overlapping, store only the residual content not already
entailed by existing memory (paper: (c'_new, U) <- LLM(c_new || C_ret)).

Decision flow (mirrors fact_storage.process_new_fact):
  1. Dedup gate     : cosine(new.emb, existing.emb) > DUPLICATE_THRESHOLD -> IGNORE
  2. Conflict gate  : hybrid_sim = 0.7*cos + 0.3*jaccard > CONFLICT_SIM_FLOOR,
                      take top-5, ask LLM for a logical conflict -> CONFLICT_RESOLVED
  3. Residual       : otherwise CREATE; optionally trim to residual novel content
                      against the most-similar existing facts.

People gate: candidates with non-empty, disjoint people sets are skipped (a fact
about Emma cannot duplicate/conflict a fact about Zoe).

Embeddings are injected via embed_fn so the logic is testable offline and the
caller controls the embedding backend. LLM is optional; absent it, the path
degrades to embedding+keyword dedup with no conflict resolution and no residual
trimming (store-as-is), never blocking the write.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from .idf_keyword_graph import cosine_similarity, hybrid_similarity, normalize_people
from .llm_cli import HeadlessLLM

EmbedFn = Callable[[str], Sequence[float]]


@dataclass
class VerifyConfig:
    duplicate_threshold: float = 0.95  # AtomMem DUPLICATE_THRESHOLD
    conflict_sim_floor: float = 0.60  # AtomMem conflict candidate gate
    conflict_top_k: int = 5
    alpha: float = 0.7  # hybrid: embedding weight
    beta: float = 0.3  # hybrid: keyword weight
    enable_residual: bool = True


_CONFLICT_SYSTEM = """You detect logical conflicts between facts.
A conflict exists when two facts make contradictory claims about the same thing
(e.g. different ages, different locations, opposite preferences for the same object).
Output JSON: {"has_conflict": true/false, "conflict_fact_id": "<id>"|null, "reason": "<short>"}"""

_RESIDUAL_SYSTEM = """You compare a NEW fact against EXISTING related facts and return only the
information in the new fact that is NOT already entailed by the existing facts.
If the new fact is fully entailed (adds nothing), set is_novel=false.
Keep the residual a standalone third-person statement.
Output JSON: {"is_novel": true/false, "residual": "<novel content or empty>"}"""


def _people_compatible(a: Sequence[str], b: Sequence[str]) -> bool:
    """Reject only when both people sets are non-empty and disjoint."""
    sa, sb = set(normalize_people(a)), set(normalize_people(b))
    return (not sa) or (not sb) or bool(sa & sb)


def _ensure_embedding(fact: Dict[str, Any], embed_fn: EmbedFn) -> Sequence[float]:
    emb = fact.get("embedding")
    if emb:
        return emb
    text = fact.get("fact") or fact.get("content") or ""
    emb = embed_fn(text) if text else []
    fact["embedding"] = emb
    return emb


class FactVerifier:
    def __init__(
        self,
        embed_fn: EmbedFn,
        llm: Optional[HeadlessLLM] = None,
        config: Optional[VerifyConfig] = None,
    ):
        self.embed_fn = embed_fn
        self.llm = llm
        self.config = config or VerifyConfig()

    def verify(
        self, new_fact: Dict[str, Any], existing_facts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Return a decision dict:
        {"action": "IGNORE"|"CONFLICT_RESOLVED"|"CREATE",
         "reason": str,
         "conflict_fact": <existing fact>|None,    # for CONFLICT_RESOLVED
         "store_text": str,                         # residual or original (CREATE)
         "is_residual": bool}
        """
        cfg = self.config
        new_emb = _ensure_embedding(new_fact, self.embed_fn)
        new_kw = new_fact.get("keywords", [])
        new_people = new_fact.get("people", [])
        new_text = new_fact.get("fact") or new_fact.get("content") or ""

        # candidate filter by people gate
        candidates = [
            f
            for f in existing_facts
            if _people_compatible(new_people, f.get("people", []))
        ]

        # ---- Step 1: dedup gate -------------------------------------------- #
        for f in candidates:
            sim = cosine_similarity(new_emb, _ensure_embedding(f, self.embed_fn))
            if sim > cfg.duplicate_threshold:
                return {
                    "action": "IGNORE",
                    "reason": f"duplicate (cosine {sim:.3f} > {cfg.duplicate_threshold})",
                    "conflict_fact": None,
                    "store_text": new_text,
                    "is_residual": False,
                }

        # ---- Step 2: conflict gate ----------------------------------------- #
        scored = []
        for f in candidates:
            hs = hybrid_similarity(
                new_emb,
                _ensure_embedding(f, self.embed_fn),
                new_kw,
                f.get("keywords", []),
                cfg.alpha,
                cfg.beta,
            )
            if hs > cfg.conflict_sim_floor:
                scored.append((hs, f))
        scored.sort(key=lambda kv: kv[0], reverse=True)
        top = [f for _s, f in scored[: cfg.conflict_top_k]]

        if top and self.llm is not None and self.llm.available():
            conflict = self._llm_conflict(new_text, top)
            if conflict is not None:
                return {
                    "action": "CONFLICT_RESOLVED",
                    "reason": "logical conflict detected; existing fact superseded",
                    "conflict_fact": conflict,
                    "store_text": new_text,
                    "is_residual": False,
                }

        # ---- Step 3: residual / create ------------------------------------- #
        store_text = new_text
        is_residual = False
        if (
            cfg.enable_residual
            and top
            and self.llm is not None
            and self.llm.available()
        ):
            residual = self._llm_residual(
                new_text, [f.get("fact") or f.get("content") or "" for f in top]
            )
            if residual is not None:
                if not residual.get("is_novel", True):
                    return {
                        "action": "IGNORE",
                        "reason": "fully entailed by existing facts (no residual novelty)",
                        "conflict_fact": None,
                        "store_text": new_text,
                        "is_residual": False,
                    }
                r = (residual.get("residual") or "").strip()
                if r and r.lower() != new_text.strip().lower():
                    store_text = r
                    is_residual = True

        return {
            "action": "CREATE",
            "reason": "novel fact" + (" (residual-trimmed)" if is_residual else ""),
            "conflict_fact": None,
            "store_text": store_text,
            "is_residual": is_residual,
        }

    # ---- LLM helpers ------------------------------------------------------- #
    def _llm_conflict(
        self, new_text: str, candidates: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        assert self.llm is not None  # guarded by caller (self.llm.available())
        listing = "\n".join(
            f"- {f.get('id', f.get('fact_id', i))}: {f.get('fact') or f.get('content') or ''}"
            for i, f in enumerate(candidates)
        )
        user = f"New fact: {new_text}\n\nCandidate facts:\n{listing}\n\nIs there a logical conflict?"
        res = self.llm.call_json(_CONFLICT_SYSTEM, user)
        data = res.get("data")
        if not isinstance(data, dict) or not data.get("has_conflict"):
            return None
        cid = data.get("conflict_fact_id")
        for i, f in enumerate(candidates):
            fid = str(f.get("id", f.get("fact_id", i)))
            if str(cid) == fid:
                return f
        return candidates[0] if candidates else None

    def _llm_residual(
        self, new_text: str, existing_texts: List[str]
    ) -> Optional[Dict[str, Any]]:
        assert self.llm is not None  # guarded by caller (self.llm.available())
        listing = "\n".join(f"- {t}" for t in existing_texts if t)
        user = f"NEW fact: {new_text}\n\nEXISTING related facts:\n{listing}\n\nReturn the residual novel content."
        res = self.llm.call_json(_RESIDUAL_SYSTEM, user)
        data = res.get("data")
        return data if isinstance(data, dict) else None


# --------------------------------------------------------------------------- #
# Self-test (deterministic embeddings + scripted LLM)                          #
# --------------------------------------------------------------------------- #
def _selftest() -> int:
    failures = 0

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal failures
        if not cond:
            failures += 1
        print(
            f"  [{'PASS' if cond else 'FAIL'}] {name}{('' if cond else ' :: ' + detail)}"
        )

    # Deterministic embedder: map known texts to fixed unit vectors.
    vecs = {
        "emma likes coffee": [1.0, 0.0, 0.0],
        "emma loves coffee": [0.999, 0.044, 0.0],  # near-duplicate of likes coffee
        "emma dislikes coffee": [
            0.9,
            0.43,
            0.0,
        ],  # high sim, opposite meaning -> conflict
        "emma lives in boston": [0.0, 1.0, 0.0],  # unrelated
        "zoe likes coffee": [1.0, 0.0, 0.0],  # same vec but different person
    }

    def embed(text: str):
        return vecs.get(text.strip().lower(), [0.0, 0.0, 1.0])

    existing = [
        {
            "id": "F1",
            "fact": "Emma likes coffee",
            "keywords": ["coffee"],
            "people": ["Emma"],
        },
        {
            "id": "F2",
            "fact": "Emma lives in Boston",
            "keywords": ["boston"],
            "people": ["Emma"],
        },
    ]

    print("Delta 3 — fact verification self-test")

    # 1. No LLM: near-duplicate -> IGNORE via cosine > 0.95
    v = FactVerifier(embed, llm=None)
    d = v.verify(
        {"fact": "Emma loves coffee", "keywords": ["coffee"], "people": ["Emma"]},
        existing,
    )
    check("near-duplicate -> IGNORE (no LLM)", d["action"] == "IGNORE", d["reason"])

    # 2. No LLM: unrelated novel -> CREATE
    d = v.verify(
        {"fact": "Emma plays violin", "keywords": ["violin"], "people": ["Emma"]},
        existing,
    )
    check("novel fact -> CREATE (no LLM)", d["action"] == "CREATE", d["reason"])

    # 3. Person gate: identical vector but different person -> not a duplicate
    d = v.verify(
        {"fact": "Zoe likes coffee", "keywords": ["coffee"], "people": ["Zoe"]},
        existing,
    )
    check(
        "different-person same-topic -> CREATE (person gate)",
        d["action"] == "CREATE",
        d["reason"],
    )

    # 4. Scripted LLM conflict: 'dislikes coffee' vs 'likes coffee' -> CONFLICT_RESOLVED
    class ScriptLLM:
        def __init__(self, mode):
            self.mode = mode

        def available(self):
            return True

        def call_json(self, system, user, prefer=None):
            if "logical conflict" in user:
                return {
                    "data": {
                        "has_conflict": True,
                        "conflict_fact_id": "F1",
                        "reason": "opposite preference",
                    }
                }
            if "residual" in user.lower():
                if self.mode == "entailed":
                    return {"data": {"is_novel": False, "residual": ""}}
                return {
                    "data": {
                        "is_novel": True,
                        "residual": "Emma started drinking decaf.",
                    }
                }
            return {"data": {}}

    v2 = FactVerifier(embed, llm=ScriptLLM("novel"))
    d = v2.verify(
        {"fact": "Emma dislikes coffee", "keywords": ["coffee"], "people": ["Emma"]},
        existing,
    )
    check(
        "conflict via LLM -> CONFLICT_RESOLVED",
        d["action"] == "CONFLICT_RESOLVED",
        d["reason"],
    )
    check(
        "conflict identifies F1",
        d.get("conflict_fact", {}).get("id") == "F1",
        str(d.get("conflict_fact")),
    )

    # 5. Residual trimming: overlapping-but-novel -> CREATE with residual text
    #    Use a fact that is similar enough to trigger the conflict gate but the
    #    scripted LLM says no conflict, then returns residual novelty.
    class ResidualLLM(ScriptLLM):
        def call_json(self, system, user, prefer=None):
            if "logical conflict" in user:
                return {"data": {"has_conflict": False, "conflict_fact_id": None}}
            if "residual" in user.lower():
                return {
                    "data": {
                        "is_novel": True,
                        "residual": "Emma started drinking decaf.",
                    }
                }
            return {"data": {}}

    v3 = FactVerifier(embed, llm=ResidualLLM("novel"))
    d = v3.verify(
        {"fact": "Emma dislikes coffee", "keywords": ["coffee"], "people": ["Emma"]},
        existing,
    )
    check(
        "residual-trim -> CREATE residual text",
        d["action"] == "CREATE" and d["is_residual"],
        d["reason"],
    )
    check(
        "residual text applied",
        d["store_text"] == "Emma started drinking decaf.",
        d["store_text"],
    )

    # 6. Fully entailed -> IGNORE
    class EntailedLLM(ScriptLLM):
        def call_json(self, system, user, prefer=None):
            if "logical conflict" in user:
                return {"data": {"has_conflict": False}}
            if "residual" in user.lower():
                return {"data": {"is_novel": False, "residual": ""}}
            return {"data": {}}

    v4 = FactVerifier(embed, llm=EntailedLLM("entailed"))
    d = v4.verify(
        {"fact": "Emma dislikes coffee", "keywords": ["coffee"], "people": ["Emma"]},
        existing,
    )
    check("fully-entailed -> IGNORE", d["action"] == "IGNORE", d["reason"])

    print(
        f"\n{'ALL PASS' if failures == 0 else str(failures) + ' FAILURE(S)'} — Delta 3"
    )
    return failures


if __name__ == "__main__":
    import sys

    sys.exit(_selftest())
