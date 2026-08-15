"""Delta 4: Versioned temporal profiles with valid-time intervals.

Ports AtomMem's temporal profile version chain
(atommem_core/temporal_profile_version_chain.py): per-subject stable attributes
that evolve over time, where each change pushes the prior state into a history
list carrying [valid_from, valid_to). Point-in-time queries return the version
valid at a given date, so "where did Caroline live in 2020?" returns the 2020
state, not today's.

This is genuinely new for our memory system: we have Git-like entity versioning
(full snapshots, memory_revert), but no per-subject attribute timeline with
valid-time intervals and time-aware selection.

Storage: new SQLite table `temporal_profiles` (created on demand; isolated from
the core schema). Profile row:
    profile_id, subject, content, keywords(json), valid_from, evidence(json),
    history(json: [{version_id, content, valid_from, valid_to, evidence}]),
    updated_at

The update decision (confirm / update_current / update_history / new) is
deterministic by default (text identity + recency) and can be upgraded with an
injected LLM decision function. The version-chain mechanics are pure and tested
without any LLM.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from datetime import date
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .idf_keyword_graph import jaccard_similarity, normalize_keyword


# --------------------------------------------------------------------------- #
# Date helpers (ported)                                                        #
# --------------------------------------------------------------------------- #
def normalize_date_value(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    value = value.strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return value
    if re.fullmatch(r"\d{4}-\d{2}", value):
        return value
    if re.fullmatch(r"\d{4}", value):
        return value
    return ""


def date_start(value: Any) -> Optional[date]:
    value = normalize_date_value(value)
    try:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            y, m, d = (int(x) for x in value.split("-"))
            return date(y, m, d)
        if re.fullmatch(r"\d{4}-\d{2}", value):
            y, m = (int(x) for x in value.split("-"))
            return date(y, m, 1)
        if re.fullmatch(r"\d{4}", value):
            return date(int(value), 1, 1)
    except ValueError:
        return None
    return None


def compare_dates(a: Any, b: Any) -> Optional[int]:
    da, db = date_start(a), date_start(b)
    if da is None or db is None:
        return None
    return -1 if da < db else (1 if da > db else 0)


def time_in_interval(query_time: Any, valid_from: Any, valid_to: Any) -> bool:
    qt, vf, vt = date_start(query_time), date_start(valid_from), date_start(valid_to)
    if qt is None or vf is None:
        return False
    if qt < vf:
        return False
    if vt is not None and qt >= vt:
        return False
    return True


def _unique(items: Sequence[Any]) -> List[Any]:
    out, seen = [], set()
    for it in items or []:
        if it is None:
            continue
        k = str(it)
        if k not in seen:
            seen.add(k)
            out.append(it)
    return out


# --------------------------------------------------------------------------- #
# Decision types                                                               #
# --------------------------------------------------------------------------- #
# A decision function takes (candidate, [matched_profiles]) and returns one of:
#   {"action": "confirm"|"update_current"|"update_history"|"new",
#    "profile_id": <id or None>, "updated_content": <str or "">}
DecisionFn = Callable[[Dict[str, Any], List[Dict[str, Any]]], Dict[str, Any]]


def _norm_text(text: Any) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(text).lower())).strip()


def deterministic_decision(
    candidate: Dict[str, Any], matches: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Default decision: no LLM.

    - identical normalized text to top match -> confirm (or update_history if the
      candidate is older than the matched current state).
    - high keyword overlap (>=0.6) but different text -> update_current.
    - otherwise -> new.
    """
    if not matches:
        return {"action": "new", "profile_id": None, "updated_content": ""}
    top = matches[0]
    if _norm_text(candidate.get("content")) == _norm_text(top.get("content")):
        cvf, tvf = candidate.get("valid_from", ""), top.get("valid_from", "")
        if cvf and tvf and compare_dates(cvf, tvf) == -1:
            return {
                "action": "update_history",
                "profile_id": top.get("profile_id"),
                "updated_content": candidate.get("content", ""),
            }
        return {
            "action": "confirm",
            "profile_id": top.get("profile_id"),
            "updated_content": "",
        }
    sim = jaccard_similarity(candidate.get("keywords", []), top.get("keywords", []))
    if sim >= 0.6:
        return {
            "action": "update_current",
            "profile_id": top.get("profile_id"),
            "updated_content": candidate.get("content", ""),
        }
    return {"action": "new", "profile_id": None, "updated_content": ""}


# --------------------------------------------------------------------------- #
# Version-chain mechanics (pure, operate on one profile dict)                  #
# --------------------------------------------------------------------------- #
def _renumber_history(profile: Dict[str, Any]) -> None:
    for idx, h in enumerate(profile.get("history", []) or [], 1):
        h["version_id"] = f"{profile.get('profile_id')}_v{idx}"


def apply_update_current(
    profile: Dict[str, Any], candidate: Dict[str, Any], updated_content: str
) -> None:
    """Promote the candidate to the current state; archive the prior current into
    history with valid_to = candidate's effective time."""
    current_vf = normalize_date_value(profile.get("valid_from", ""))
    effective = normalize_date_value(candidate.get("valid_from", "")) or current_vf
    if current_vf and effective and compare_dates(effective, current_vf) == -1:
        # Candidate is older than current -> it belongs in history instead.
        apply_update_history(profile, candidate, updated_content)
        return
    old_content = (profile.get("content") or "").strip()
    if (
        old_content
        and effective
        and (not current_vf or compare_dates(current_vf, effective) != 0)
    ):
        history = profile.setdefault("history", [])
        history.append(
            {
                "version_id": f"{profile.get('profile_id')}_v{len(history) + 1}",
                "content": old_content,
                "valid_from": current_vf,
                "valid_to": effective,
                "evidence": list(profile.get("evidence", []) or []),
            }
        )
    profile["content"] = (updated_content or candidate.get("content", "")).strip()
    profile["valid_from"] = effective or current_vf
    profile["keywords"] = _unique(
        list(profile.get("keywords", [])) + list(candidate.get("keywords", []))
    )
    profile["evidence"] = _unique(list(candidate.get("evidence", [])))


def apply_update_history(
    profile: Dict[str, Any], candidate: Dict[str, Any], updated_content: str
) -> None:
    """Insert a past-only state into history without disturbing the current state."""
    effective = normalize_date_value(candidate.get("valid_from", ""))
    evidence = _unique(candidate.get("evidence", []))
    content = (updated_content or candidate.get("content", "")).strip()
    if not effective:
        profile["evidence"] = _unique(list(profile.get("evidence", [])) + evidence)
        return
    # valid_to = the next-later valid_from among current + history.
    later = []
    cvf = profile.get("valid_from", "")
    if cvf and compare_dates(effective, cvf) == -1:
        later.append(cvf)
    for h in profile.get("history", []) or []:
        vf = h.get("valid_from", "")
        if vf and compare_dates(effective, vf) == -1:
            later.append(vf)
    later.sort(key=lambda t: date_start(t) or date.max)
    valid_to = later[0] if later else cvf
    history = profile.setdefault("history", [])
    history.append(
        {
            "version_id": f"{profile.get('profile_id')}_v{len(history) + 1}",
            "content": content,
            "valid_from": effective,
            "valid_to": valid_to,
            "evidence": evidence,
        }
    )
    history.sort(key=lambda h: date_start(h.get("valid_from")) or date.max)
    _renumber_history(profile)


def apply_confirm(profile: Dict[str, Any], candidate: Dict[str, Any]) -> None:
    profile["evidence"] = _unique(
        list(profile.get("evidence", [])) + list(candidate.get("evidence", []))
    )


def current_view(profile: Dict[str, Any]) -> Dict[str, Any]:
    out = {k: v for k, v in profile.items() if k != "history"}
    return out


def view_at_time(profile: Dict[str, Any], query_time: Any) -> Optional[Dict[str, Any]]:
    """Return the profile view valid at query_time (current or a history version).
    Returns None only when query_time precedes all known states."""
    cvf = profile.get("valid_from", "")
    cmp = compare_dates(query_time, cvf)
    if cmp is None or cmp >= 0:
        return current_view(profile)
    for h in profile.get("history", []) or []:
        if time_in_interval(query_time, h.get("valid_from"), h.get("valid_to")):
            v = current_view(profile)
            v["content"] = h.get("content", "")
            v["valid_from"] = h.get("valid_from", "")
            v["valid_to"] = h.get("valid_to", "")
            v["evidence"] = h.get("evidence", [])
            v["profile_version_id"] = h.get("version_id", "")
            return v
    return None


# --------------------------------------------------------------------------- #
# SQLite-backed store                                                          #
# --------------------------------------------------------------------------- #
DEFAULT_DB = os.path.expanduser("~/.claude/enhanced_memories/memory.db")


class TemporalProfileStore:
    def __init__(
        self, db_path: str = DEFAULT_DB, decision_fn: Optional[DecisionFn] = None
    ):
        self.db_path = db_path
        self.decision_fn = decision_fn or deterministic_decision
        self._init_table()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_table(self) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS temporal_profiles (
                    profile_id TEXT PRIMARY KEY,
                    subject TEXT NOT NULL,
                    content TEXT NOT NULL,
                    keywords TEXT DEFAULT '[]',
                    valid_from TEXT DEFAULT '',
                    evidence TEXT DEFAULT '[]',
                    history TEXT DEFAULT '[]',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_temporal_profiles_subject ON temporal_profiles(subject)"
            )

    # ---- (de)serialization ------------------------------------------------ #
    @staticmethod
    def _row_to_profile(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "profile_id": row["profile_id"],
            "subject": row["subject"],
            "content": row["content"],
            "keywords": json.loads(row["keywords"] or "[]"),
            "valid_from": row["valid_from"] or "",
            "evidence": json.loads(row["evidence"] or "[]"),
            "history": json.loads(row["history"] or "[]"),
            "updated_at": row["updated_at"],
        }

    def _save(self, conn: sqlite3.Connection, profile: Dict[str, Any]) -> None:
        conn.execute(
            """
            INSERT INTO temporal_profiles
                (profile_id, subject, content, keywords, valid_from, evidence, history, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(profile_id) DO UPDATE SET
                content=excluded.content, keywords=excluded.keywords,
                valid_from=excluded.valid_from, evidence=excluded.evidence,
                history=excluded.history, updated_at=CURRENT_TIMESTAMP
            """,
            (
                profile["profile_id"],
                profile["subject"],
                profile["content"],
                json.dumps(profile.get("keywords", [])),
                profile.get("valid_from", ""),
                json.dumps(profile.get("evidence", [])),
                json.dumps(profile.get("history", [])),
            ),
        )

    def _next_profile_id(self, conn: sqlite3.Connection) -> str:
        cur = conn.execute("SELECT COUNT(*) AS n FROM temporal_profiles")
        return f"P{cur.fetchone()['n'] + 1}"

    # ---- public API ------------------------------------------------------- #
    def get_profiles_for_subject(self, subject: str) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM temporal_profiles WHERE subject = ? COLLATE NOCASE",
                (subject,),
            ).fetchall()
        return [self._row_to_profile(r) for r in rows]

    def _rank_matches(
        self, candidate: Dict[str, Any], profiles: List[Dict[str, Any]], top_k: int
    ) -> List[Dict[str, Any]]:
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for p in profiles:
            sim = jaccard_similarity(
                candidate.get("keywords", []), p.get("keywords", [])
            )
            scored.append((sim, p))
        scored.sort(key=lambda kv: kv[0], reverse=True)
        return [p for _s, p in scored[:top_k]]

    def upsert(
        self,
        subject: str,
        content: str,
        valid_from: str = "",
        keywords: Optional[Sequence[str]] = None,
        evidence: Optional[Sequence[str]] = None,
        top_k: int = 8,
    ) -> Dict[str, Any]:
        """Insert/merge an attribute observation for a subject. Returns
        {"action": ..., "profile_id": ...}."""
        candidate = {
            "subject": subject,
            "content": content.strip(),
            "valid_from": normalize_date_value(valid_from),
            "keywords": [
                normalize_keyword(k) for k in (keywords or []) if normalize_keyword(k)
            ],
            "evidence": list(evidence or []),
        }
        if not candidate["content"]:
            return {"action": "skip", "profile_id": None, "reason": "empty content"}

        with self._conn() as conn:
            existing = [
                self._row_to_profile(r)
                for r in conn.execute(
                    "SELECT * FROM temporal_profiles WHERE subject = ? COLLATE NOCASE",
                    (subject,),
                ).fetchall()
            ]
            matches = self._rank_matches(candidate, existing, top_k)
            decision = self.decision_fn(candidate, matches)
            action = decision.get("action", "new")

            if action == "new" or not decision.get("profile_id"):
                pid = self._next_profile_id(conn)
                profile = {
                    "profile_id": pid,
                    "subject": subject,
                    "content": candidate["content"],
                    "keywords": candidate["keywords"],
                    "valid_from": candidate["valid_from"],
                    "evidence": _unique(candidate["evidence"]),
                    "history": [],
                }
                self._save(conn, profile)
                return {"action": "new", "profile_id": pid}

            target = next(
                (p for p in existing if p.get("profile_id") == decision["profile_id"]),
                None,
            )
            if target is None:
                pid = self._next_profile_id(conn)
                profile = {
                    "profile_id": pid,
                    "subject": subject,
                    "content": candidate["content"],
                    "keywords": candidate["keywords"],
                    "valid_from": candidate["valid_from"],
                    "evidence": _unique(candidate["evidence"]),
                    "history": [],
                }
                self._save(conn, profile)
                return {"action": "new", "profile_id": pid}

            if action == "confirm":
                apply_confirm(target, candidate)
            elif action == "update_current":
                apply_update_current(
                    target, candidate, decision.get("updated_content", "")
                )
            elif action == "update_history":
                apply_update_history(
                    target, candidate, decision.get("updated_content", "")
                )
            else:
                apply_confirm(target, candidate)
            self._save(conn, target)
            return {"action": action, "profile_id": target["profile_id"]}

    def query(
        self,
        subject: Optional[str] = None,
        query_time: str = "",
        keywords: Optional[Sequence[str]] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Return point-in-time profile views. If query_time is set, historical
        versions are selected by valid-time interval."""
        with self._conn() as conn:
            if subject:
                rows = conn.execute(
                    "SELECT * FROM temporal_profiles WHERE subject = ? COLLATE NOCASE",
                    (subject,),
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM temporal_profiles").fetchall()
        profiles = [self._row_to_profile(r) for r in rows]

        if keywords:
            kw = [normalize_keyword(k) for k in keywords if normalize_keyword(k)]
            profiles.sort(
                key=lambda p: jaccard_similarity(kw, p.get("keywords", [])),
                reverse=True,
            )

        out: List[Dict[str, Any]] = []
        for p in profiles[:top_k] if not query_time else profiles:
            if query_time:
                v = view_at_time(p, query_time)
                if v is not None:
                    out.append(v)
            else:
                out.append(current_view(p))
        return out[:top_k]


# --------------------------------------------------------------------------- #
# Self-test (pure version-chain logic + in-memory store, no LLM)               #
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

    print("Delta 4 — temporal profile self-test")

    # Pure mechanics: living-location timeline.
    profile = {
        "profile_id": "P1",
        "subject": "Caroline",
        "content": "Caroline lives in Boston.",
        "keywords": ["location", "boston"],
        "valid_from": "2019",
        "evidence": ["F1"],
        "history": [],
    }
    # Move to Seattle in 2021 -> update_current archives Boston as 2019..2021.
    apply_update_current(
        profile,
        {
            "content": "Caroline lives in Seattle.",
            "valid_from": "2021",
            "keywords": ["location", "seattle"],
            "evidence": ["F8"],
        },
        "Caroline lives in Seattle.",
    )
    check(
        "current state updated to Seattle",
        profile["content"] == "Caroline lives in Seattle.",
    )
    check(
        "Boston archived to history",
        len(profile["history"]) == 1 and "Boston" in profile["history"][0]["content"],
    )
    check(
        "history interval 2019..2021",
        profile["history"][0]["valid_from"] == "2019"
        and profile["history"][0]["valid_to"] == "2021",
        str(profile["history"][0]),
    )

    # Point-in-time queries.
    v2020 = view_at_time(profile, "2020")
    v2022 = view_at_time(profile, "2022")
    check(
        "2020 query -> Boston (historical)",
        v2020 is not None and "Boston" in v2020["content"],
        str(v2020),
    )
    check(
        "2022 query -> Seattle (current)",
        v2022 is not None and "Seattle" in v2022["content"],
        str(v2022),
    )

    # SQLite store round-trip with deterministic decisions.
    import tempfile

    tmp = os.path.join(
        tempfile.gettempdir(), f"atommem_tp_test_{int(time.time() * 1000)}.db"
    )
    try:
        store = TemporalProfileStore(db_path=tmp)
        r1 = store.upsert(
            "Dana",
            "Dana works as a nurse.",
            valid_from="2018",
            keywords=["occupation", "nurse"],
            evidence=["E1"],
        )
        check("store: first upsert -> new", r1["action"] == "new", str(r1))
        # Same content again -> confirm (evidence merged, no new profile).
        r2 = store.upsert(
            "Dana",
            "Dana works as a nurse.",
            valid_from="2018",
            keywords=["occupation", "nurse"],
            evidence=["E2"],
        )
        check("store: identical content -> confirm", r2["action"] == "confirm", str(r2))
        # Career change, same domain keywords -> update_current.
        r3 = store.upsert(
            "Dana",
            "Dana works as a doctor.",
            valid_from="2022",
            keywords=["occupation", "doctor", "nurse"],
            evidence=["E9"],
        )
        check(
            "store: career change -> update_current",
            r3["action"] == "update_current",
            str(r3),
        )
        # Point-in-time: 2019 -> nurse, now -> doctor.
        q2019 = store.query(subject="Dana", query_time="2019")
        qnow = store.query(subject="Dana")
        check(
            "store: 2019 -> nurse",
            bool(q2019) and "nurse" in q2019[0]["content"],
            str(q2019),
        )
        check(
            "store: current -> doctor",
            bool(qnow) and "doctor" in qnow[0]["content"],
            str(qnow),
        )
        # Exactly one profile row for Dana (no duplication).
        check(
            "store: single Dana profile",
            len(store.get_profiles_for_subject("Dana")) == 1,
        )
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

    print(
        f"\n{'ALL PASS' if failures == 0 else str(failures) + ' FAILURE(S)'} — Delta 4"
    )
    return failures


if __name__ == "__main__":
    import sys

    sys.exit(_selftest())
