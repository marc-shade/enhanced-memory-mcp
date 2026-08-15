"""Delta 2: IDF-weighted keyword-overlap graph + Personalized PageRank.

Faithful port of AtomMem's entity-edge channel
(atommem_core/multichannel_graph.py, paper Eq. 3) generalized to operate on our
entity/fact items. An "item" is any dict carrying:

    {"id": <str>, "keywords": [<str>, ...], "people": [<str>, ...]}

People are optional; when both items name disjoint, non-empty people sets the
edge is gated to zero (prevents cross-subject keyword bleed). Keywords drive the
graph; supply them from atomic-fact metadata (Delta 1) or the deterministic
extractor in keywords.py.

Edge weight between facts i and j (IDF-weighted cosine over shared keywords):

    w(k)        = idf(k) * boost(k) * penalty(k)
    idf(k)      = log((N + 1) / (df(k) + 1))
    boost(k)    = query_keyword_boost if k in query else 1.0
    penalty(k)  = frequency penalty that down-weights very common keywords
    edge(i, j)  = Σ_{k in shared} w(k) / sqrt(Σ_{k in i} w(k) * Σ_{k in j} w(k) + ε)

Retrieval = seed scores from query-keyword match, expand a bounded local graph,
run RWR/Personalized PageRank, return top-k non-seed facts by activation.

Pure standard library. Deterministic. No LLM, no I/O.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

# Roles that must never act as "people" for the person-gate.
_ROLE_PEOPLE = {"user", "assistant", "system", "me", "you"}


# --------------------------------------------------------------------------- #
# Generic similarity helpers (reused by Delta 3/4)                            #
# --------------------------------------------------------------------------- #
def jaccard_similarity(a: Sequence[str], b: Sequence[str]) -> float:
    """Jaccard overlap of two keyword sets (normalized, case-insensitive)."""
    set_a = {normalize_keyword(x) for x in a if normalize_keyword(x)}
    set_b = {normalize_keyword(x) for x in b if normalize_keyword(x)}
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union else 0.0


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two dense vectors. 0.0 on empty/mismatched input."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def hybrid_similarity(
    emb_a: Sequence[float],
    emb_b: Sequence[float],
    kw_a: Sequence[str],
    kw_b: Sequence[str],
    alpha: float = 0.7,
    beta: float = 0.3,
) -> float:
    """AtomMem hybrid metric S_h = alpha*cosine + beta*jaccard (paper Eq. 2).

    Defaults alpha=0.7, beta=0.3 match AtomMem config.
    """
    return alpha * cosine_similarity(emb_a, emb_b) + beta * jaccard_similarity(
        kw_a, kw_b
    )


def normalize_keyword(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.strip().lower().split())


def normalize_people(values: Iterable[Any]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for value in values or []:
        if not isinstance(value, str):
            continue
        item = " ".join(value.strip().lower().split())
        if not item or item in _ROLE_PEOPLE or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #
@dataclass
class IDFGraphConfig:
    # PPR / RWR
    restart_prob: float = 0.34
    max_iter: int = 20
    tol: float = 1e-6
    # Hub suppression, MemGraphRAG (arXiv 2606.00610) eq. 7: divide a seed's
    # initial weight by log(degree + 1). A node wired to most of the local
    # graph spreads importance over everything it touches and drags recall
    # toward whatever is merely well-connected; the penalty keeps propagation
    # anchored on specific nodes. Worth 2.18 points in the paper's ablation
    # (67.22 -> 69.40 on HotpotQA), the second-largest component after conflict
    # resolution.
    #
    # OFF by default: this changes ranking, and it has not been measured on our
    # store yet. Turn it on behind a benchmark, not on a hunch.
    hub_suppression: bool = False
    # Local graph bounds
    max_seed_facts: int = 10
    max_neighbors_per_fact: int = 30
    max_local_nodes: int = 180
    max_hops: int = 2
    query_keyword_hit_top_k: int = 80
    # Keyword weighting
    query_keyword_boost: float = 2.5
    query_penalty_floor: float = 0.45
    query_penalty_tau: float = 0.05
    query_penalty_gamma: float = 0.7
    non_query_penalty_tau: float = 0.10
    non_query_penalty_gamma: float = 1.0
    edge_epsilon: float = 1e-8
    # Seed blend: existing(vector) score vs query-keyword match score
    seed_existing_weight: float = 0.6
    seed_query_weight: float = 0.4
    graph_recall_top_k: int = 10


# --------------------------------------------------------------------------- #
# Graph                                                                        #
# --------------------------------------------------------------------------- #
class IDFKeywordGraph:
    """IDF-weighted keyword graph over a set of items with Personalized PageRank.

    Parameters
    ----------
    items : list of dicts, each with "id", "keywords", optionally "people".
    config : IDFGraphConfig, optional.
    """

    def __init__(
        self, items: List[Dict[str, Any]], config: Optional[IDFGraphConfig] = None
    ):
        self.config = config or IDFGraphConfig()
        self.lookup: Dict[str, Dict[str, Any]] = {}
        self.fact_keywords: Dict[str, List[str]] = {}
        self.fact_people: Dict[str, List[str]] = {}
        self.keyword_to_facts: Dict[str, List[str]] = defaultdict(list)
        self.keyword_df: Dict[str, int] = {}
        self._weight_cache: Dict[Tuple[str, bool], float] = {}
        self._build(items)

    def _build(self, items: List[Dict[str, Any]]) -> None:
        for item in items:
            fid = item.get("id")
            if fid is None:
                continue
            fid = str(fid)
            self.lookup[fid] = item
            seen: Set[str] = set()
            kws: List[str] = []
            for kw in item.get("keywords") or []:
                nk = normalize_keyword(kw)
                if nk and nk not in seen:
                    seen.add(nk)
                    kws.append(nk)
                    self.keyword_to_facts[nk].append(fid)
            self.fact_keywords[fid] = kws
            self.fact_people[fid] = normalize_people(item.get("people", []))
        self.num_facts = len(self.lookup)
        self.keyword_df = {k: len(set(v)) for k, v in self.keyword_to_facts.items()}

    # ---- IDF / weighting --------------------------------------------------- #
    def idf(self, keyword: str) -> float:
        return math.log((self.num_facts + 1) / (self.keyword_df.get(keyword, 0) + 1))

    def df_ratio(self, keyword: str) -> float:
        return self.keyword_df.get(keyword, 0) / max(self.num_facts, 1)

    def _frequency_penalty(self, keyword: str, is_query: bool) -> float:
        cfg = self.config
        ratio = self.df_ratio(keyword)
        if is_query:
            base = (
                cfg.query_penalty_tau / max(ratio, cfg.query_penalty_tau)
            ) ** cfg.query_penalty_gamma
            return max(cfg.query_penalty_floor, base)
        return (
            cfg.non_query_penalty_tau / max(ratio, cfg.non_query_penalty_tau)
        ) ** cfg.non_query_penalty_gamma

    def keyword_weight(self, keyword: str, query_keywords: Set[str]) -> float:
        is_query = keyword in query_keywords
        key = (keyword, is_query)
        if key in self._weight_cache:
            return self._weight_cache[key]
        idf = self.idf(keyword)
        if idf <= 0:
            self._weight_cache[key] = 0.0
            return 0.0
        boost = self.config.query_keyword_boost if is_query else 1.0
        weight = idf * boost * self._frequency_penalty(keyword, is_query)
        self._weight_cache[key] = weight
        return weight

    # ---- edges ------------------------------------------------------------- #
    def _passes_person_gate(self, left: str, right: str) -> bool:
        lp = set(self.fact_people.get(left, []))
        rp = set(self.fact_people.get(right, []))
        if lp and rp and not (lp & rp):
            return False
        return True

    def _keyword_neighbor_ids(self, fid: str) -> Set[str]:
        out: Set[str] = set()
        for kw in self.fact_keywords.get(fid, []):
            for nb in self.keyword_to_facts.get(kw, []):
                if nb != fid and self._passes_person_gate(fid, nb):
                    out.add(nb)
        return out

    def edge_weight(
        self, left: str, right: str, query_keywords: Optional[Set[str]] = None
    ) -> float:
        """IDF-weighted cosine over shared keywords (paper Eq. 3)."""
        q = query_keywords or set()
        if not self._passes_person_gate(left, right):
            return 0.0
        lk = set(self.fact_keywords.get(left, []))
        rk = set(self.fact_keywords.get(right, []))
        shared = lk & rk
        if not shared:
            return 0.0
        shared_w = sum(self.keyword_weight(k, q) for k in shared)
        left_w = sum(self.keyword_weight(k, q) for k in lk)
        right_w = sum(self.keyword_weight(k, q) for k in rk)
        denom = math.sqrt(left_w * right_w + self.config.edge_epsilon)
        return shared_w / denom if denom > 0 else 0.0

    # ---- query / seed scoring --------------------------------------------- #
    def _query_match_score(self, fid: str, query_keywords: Set[str]) -> float:
        fk = set(self.fact_keywords.get(fid, []))
        return sum(self.keyword_weight(k, query_keywords) for k in fk & query_keywords)

    def _build_seed_scores(
        self,
        query_keywords: Set[str],
        query_people: Set[str],
        prior_scores: Optional[Dict[str, float]],
    ) -> Dict[str, float]:
        cfg = self.config
        existing: Dict[str, float] = {}
        for fid, sc in (prior_scores or {}).items():
            fid = str(fid)
            if fid in self.lookup:
                existing[fid] = max(existing.get(fid, 0.0), float(sc or 0.0))

        query_match: Dict[str, float] = {}
        for fid in self.lookup:
            if query_people:
                fp = set(self.fact_people.get(fid, []))
                if fp and not (query_people & fp):
                    continue
            sc = self._query_match_score(fid, query_keywords)
            if sc > 0:
                query_match[fid] = sc
        query_match = dict(
            sorted(query_match.items(), key=lambda kv: (-kv[1], kv[0]))[
                : cfg.query_keyword_hit_top_k
            ]
        )

        existing_max = max(existing.values(), default=0.0) or 1.0
        query_max = max(query_match.values(), default=0.0) or 1.0
        raw: Dict[str, float] = {}
        for fid in set(existing) | set(query_match):
            en = existing.get(fid, 0.0) / existing_max
            qn = query_match.get(fid, 0.0) / query_max
            s = cfg.seed_existing_weight * en + cfg.seed_query_weight * qn
            if s > 0:
                raw[fid] = s
        total = sum(raw.values())
        return {f: s / total for f, s in raw.items()} if total > 0 else {}

    def _build_local_nodes(
        self, seed_scores: Dict[str, float], query_keywords: Set[str]
    ) -> Set[str]:
        cfg = self.config
        local: Set[str] = set()
        ranked = [
            f for f, _ in sorted(seed_scores.items(), key=lambda kv: (-kv[1], kv[0]))
        ]
        for f in ranked:
            local.add(f)
            if len(local) >= cfg.max_local_nodes:
                return local
        frontier = set(local)
        for _hop in range(cfg.max_hops):
            nxt: Set[str] = set()
            for f in sorted(frontier):
                scored: List[Tuple[str, float]] = []
                for nb in self._keyword_neighbor_ids(f):
                    w = self.edge_weight(f, nb, query_keywords)
                    if w > 0:
                        scored.append((nb, w))
                scored.sort(key=lambda kv: (-kv[1], kv[0]))
                for nb, _w in scored[: cfg.max_neighbors_per_fact]:
                    if nb not in local:
                        local.add(nb)
                        nxt.add(nb)
                        if len(local) >= cfg.max_local_nodes:
                            return local
            if not nxt:
                break
            frontier = nxt
        return local

    def _adjacency(
        self, local: Set[str], query_keywords: Set[str]
    ) -> Dict[str, List[Tuple[str, float]]]:
        cfg = self.config
        adj: Dict[str, List[Tuple[str, float]]] = {f: [] for f in local}
        for f in sorted(local):
            for nb in sorted(self._keyword_neighbor_ids(f)):
                if nb not in local:
                    continue
                w = self.edge_weight(f, nb, query_keywords)
                if w > 0:
                    adj[f].append((nb, w))
            adj[f].sort(key=lambda kv: (-kv[1], kv[0]))
            adj[f] = adj[f][: cfg.max_neighbors_per_fact]
        return adj

    def _run_ppr(
        self,
        local: Set[str],
        adj: Dict[str, List[Tuple[str, float]]],
        seed: Dict[str, float],
    ) -> Dict[str, float]:
        cfg = self.config
        local_seed = {f: s for f, s in seed.items() if f in local and s > 0}
        if cfg.hub_suppression:
            # eq. 7. Degree 0 would divide by log(1) = 0; a node with no
            # neighbours is not a hub, so it keeps its weight untouched.
            local_seed = {
                f: (s / math.log1p(len(adj.get(f, ()))) if adj.get(f) else s)
                for f, s in local_seed.items()
            }
        total = sum(local_seed.values())
        if total <= 0:
            return {}
        seed_n = {f: s / total for f, s in local_seed.items()}
        scores = {f: seed_n.get(f, 0.0) for f in local}
        for _ in range(cfg.max_iter):
            nxt = {f: cfg.restart_prob * seed_n.get(f, 0.0) for f in local}
            sink = 0.0
            for src in sorted(local):
                s = scores.get(src, 0.0)
                neighbors = adj.get(src, [])
                edge_total = sum(w for _t, w in neighbors)
                if edge_total <= 0:
                    sink += s
                    continue
                for tgt, w in neighbors:
                    nxt[tgt] = (
                        nxt.get(tgt, 0.0) + (1 - cfg.restart_prob) * s * w / edge_total
                    )
            if sink:
                for f, sv in seed_n.items():
                    nxt[f] = nxt.get(f, 0.0) + (1 - cfg.restart_prob) * sink * sv
            delta = sum(abs(nxt.get(f, 0.0) - scores.get(f, 0.0)) for f in local)
            scores = nxt
            if delta < cfg.tol:
                break
        return scores

    # ---- public API -------------------------------------------------------- #
    def neighbors(
        self, fid: str, query_keywords: Optional[Sequence[str]] = None, top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Top-k IDF-weighted keyword neighbors of a single fact."""
        fid = str(fid)
        q = {
            normalize_keyword(k) for k in (query_keywords or []) if normalize_keyword(k)
        }
        scored: List[Tuple[str, float]] = []
        for nb in self._keyword_neighbor_ids(fid):
            w = self.edge_weight(fid, nb, q)
            if w > 0:
                scored.append((nb, w))
        scored.sort(key=lambda kv: (-kv[1], kv[0]))
        out = []
        for nb, w in scored[:top_k]:
            item = dict(self.lookup[nb])
            item["edge_weight"] = round(w, 6)
            out.append(item)
        return out

    def graph_recall(
        self,
        query_keywords: Sequence[str],
        query_people: Optional[Sequence[str]] = None,
        prior_scores: Optional[Dict[str, float]] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Personalized-PageRank recall seeded by query-keyword match (+ optional
        prior vector scores). Returns ranked non-seed items with activation score.
        """
        if self.num_facts == 0:
            return []
        cfg = self.config
        top_k = top_k or cfg.graph_recall_top_k
        q = list(
            dict.fromkeys(
                normalize_keyword(k) for k in query_keywords if normalize_keyword(k)
            )
        )
        q_set = set(q)
        qp = set(normalize_people(query_people or []))

        seed_scores = self._build_seed_scores(q_set, qp, prior_scores)
        if not seed_scores:
            return []
        seed_ids = set(seed_scores.keys())
        local = self._build_local_nodes(seed_scores, q_set)
        if not local:
            return []
        adj = self._adjacency(local, q_set)
        ppr = self._run_ppr(local, adj, seed_scores)
        ranked = sorted(ppr.items(), key=lambda kv: (-kv[1], kv[0]))

        out: List[Dict[str, Any]] = []
        for fid, score in ranked:
            if fid in seed_ids:
                continue
            base = self.lookup.get(fid)
            if not base:
                continue
            item = dict(base)
            item["graph_score"] = round(float(score), 8)
            out.append(item)
            if len(out) >= top_k:
                break
        return out


# --------------------------------------------------------------------------- #
# Self-test                                                                     #
# --------------------------------------------------------------------------- #
def _selftest() -> int:
    failures = 0

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal failures
        status = "PASS" if cond else "FAIL"
        if not cond:
            failures += 1
        print(f"  [{status}] {name}{(' :: ' + detail) if detail and not cond else ''}")

    # IDF: a keyword appearing in every fact has idf ~ log((N+1)/(N+1)) = 0.
    items = [
        {"id": "F1", "keywords": ["psychology", "exam", "grade"], "people": ["emma"]},
        {"id": "F2", "keywords": ["psychology", "study"], "people": ["emma"]},
        {"id": "F3", "keywords": ["psychology", "cactus", "dorm"], "people": ["emma"]},
        {"id": "F4", "keywords": ["plants", "cactus", "garden"], "people": ["zoe"]},
        {"id": "F5", "keywords": ["plants", "indoor", "collection"], "people": ["zoe"]},
    ]
    g = IDFKeywordGraph(items)

    print("Delta 2 — IDF keyword graph self-test")
    # 'psychology' is in 3/5 facts -> low idf; 'exam' in 1/5 -> high idf.
    idf_psych = g.idf("psychology")
    idf_exam = g.idf("exam")
    check(
        "rare keyword has higher IDF than common keyword",
        idf_exam > idf_psych,
        f"exam={idf_exam:.4f} psych={idf_psych:.4f}",
    )

    # Frequency penalty: common non-query keyword penalized below rare one's weight.
    w_psych = g.keyword_weight("psychology", set())
    w_exam = g.keyword_weight("exam", set())
    check(
        "common keyword down-weighted vs rare",
        w_exam > w_psych,
        f"w_exam={w_exam:.4f} w_psych={w_psych:.4f}",
    )

    # Edge: F1-F2 share 'psychology' only (common); F4-F5 share 'plants' (common).
    # F1-F3 share 'psychology'; F3-F4 share 'cactus' (rarer). Person gate: F3(emma)
    # vs F4(zoe) disjoint people -> edge must be 0.
    e_f3_f4 = g.edge_weight("F3", "F4")
    check(
        "person gate zeroes cross-subject edge (F3 emma / F4 zoe)",
        e_f3_f4 == 0.0,
        f"edge={e_f3_f4}",
    )

    # Same-subject shared-keyword edge is positive.
    e_f1_f2 = g.edge_weight("F1", "F2")
    check(
        "same-subject shared-keyword edge > 0 (F1/F2)", e_f1_f2 > 0.0, f"edge={e_f1_f2}"
    )

    # Self-similar edge approaches 1 for identical keyword sets.
    g2 = IDFKeywordGraph(
        [
            {"id": "A", "keywords": ["alpha", "beta"]},
            {"id": "B", "keywords": ["alpha", "beta"]},
            {"id": "C", "keywords": ["gamma"]},
        ]
    )
    e_ab = g2.edge_weight("A", "B")
    check(
        "identical keyword sets -> edge ~ 1.0",
        abs(e_ab - 1.0) < 1e-6,
        f"edge={e_ab:.6f}",
    )

    # graph_recall: query 'cactus' should surface emma's cactus facts, not seed itself.
    recall = g.graph_recall(query_keywords=["cactus", "dorm"], top_k=3)
    recall_ids = [r["id"] for r in recall]
    check(
        "graph_recall returns ranked non-seed items",
        len(recall) > 0,
        f"ids={recall_ids}",
    )
    check(
        "graph_recall scores are descending",
        all(
            recall[i]["graph_score"] >= recall[i + 1]["graph_score"]
            for i in range(len(recall) - 1)
        ),
        f"scores={[r['graph_score'] for r in recall]}",
    )

    # hybrid_similarity sanity
    hs = hybrid_similarity([1.0, 0.0], [1.0, 0.0], ["a", "b"], ["a", "c"], 0.7, 0.3)
    # cosine=1.0, jaccard=1/3 -> 0.7 + 0.1 = 0.8
    check("hybrid_similarity matches 0.7*cos+0.3*jac", abs(hs - 0.8) < 1e-9, f"hs={hs}")

    # jaccard empty safety
    check("jaccard empty -> 0.0", jaccard_similarity([], ["a"]) == 0.0)

    print(
        f"\n{'ALL PASS' if failures == 0 else str(failures) + ' FAILURE(S)'} — Delta 2"
    )
    return failures


if __name__ == "__main__":
    import sys

    sys.exit(_selftest())
