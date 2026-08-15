#!/usr/bin/env python3
"""
Profile Drift Curation (AtomMem Delta-4 ops-hardening)

The `temporal_profiles` table (created on demand by atommem.temporal_profile)
is populated only through `upsert_temporal_profile` — and nothing calls it, so
the table sits at 0 rows while the live `entities` table carries ~11.5k
entities / ~72k observations, including bi-temporal facts
(valid_until / superseded_by). This module:

  1. `_backfill_profiles(conn)` — populates temporal_profiles from the live
     entities table. Candidates are entities that carry a
     valid_until / superseded_by (bi-temporal facts) or whose entity_type is
     fact-like (`fact`, `fact/...`, `session_episode/...`, `audit_probe`,
     `auto_memory/...`). For each observation of each candidate it upserts a
     profile row via the AtomMem version-chain mechanics
     (deterministic_decision + apply_update_current / apply_update_history /
     apply_confirm from atommem.temporal_profile), with
     valid_from = entity.created_at and valid_to = entity.valid_until.
     Idempotent: a per-observation evidence marker (`entity:<id>:<obs_idx>`)
     and a content-hash keyword marker skip already-backfilled rows.
  2. `curate_profiles(subject=None, limit=200)` — runs the backfill, then
     surfaces CONTRADICTIONS: pairs of facts about the same subject+attribute
     with overlapping valid ranges whose differing span is a value, not a
     paraphrase (lexical diff via difflib).
  3. `profile_summary(subject=None)` — the current snapshot per subject
     (newest valid fact per attribute = per profile row).

Storage is direct sqlite3 against db_path; schema creation is idempotent
(CREATE TABLE IF NOT EXISTS), mirroring the other *_tools feature modules.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import logging
import pickle  # tolerant read of legacy zlib+pickle entity rows (first-party db)
import re
import sqlite3
import zlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

from atommem.idf_keyword_graph import jaccard_similarity
from atommem.keywords import extract_keywords
from atommem.temporal_profile import (
    apply_confirm,
    apply_update_current,
    apply_update_history,
    date_start,
    deterministic_decision,
    normalize_date_value,
)

logger = logging.getLogger("curate_profiles")

# Candidate entity_type prefixes (fact-like). Bi-temporal entities
# (valid_until / superseded_by set) are candidates regardless of entity_type.
_FACT_LIKE_PREFIXES = (
    "fact",
    "fact/",
    "session_episode/",
    "audit_probe",
    "auto_memory/",
)

# Same-attribute threshold for contradiction detection: two facts are the same
# attribute when their keyword sets overlap at least this much.
_SAME_ATTRIBUTE_JACCARD = 0.3

# Value-substitution threshold: matching-block coverage of the longer text.
# Above this the two facts share a common template with a differing span
# (a value change), below it they are paraphrases.
_VALUE_SUBSTITUTION_COVERAGE = 0.5

_PROFILE_COLS = (
    "profile_id",
    "subject",
    "content",
    "keywords",
    "valid_from",
    "evidence",
    "history",
    "updated_at",
)


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Create the temporal_profiles table if absent (idempotent; mirrors
    atommem.temporal_profile.TemporalProfileStore._init_table)."""
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
        "CREATE INDEX IF NOT EXISTS idx_temporal_profiles_subject "
        "ON temporal_profiles(subject)"
    )


def _uniq(items: Sequence[Any]) -> List[Any]:
    out: List[Any] = []
    seen = set()
    for it in items or []:
        if it is None:
            continue
        k = str(it)
        if k not in seen:
            seen.add(k)
            out.append(it)
    return out


def _date_only(ts: Optional[str]) -> str:
    """Truncate a SQL timestamp / date string to a profile-valid date value
    (YYYY-MM-DD / YYYY-MM / YYYY). Empty input -> ''."""
    if not ts:
        return ""
    value = str(ts).strip()
    # SQLite CURRENT_TIMESTAMP -> 'YYYY-MM-DD HH:MM:SS'; take the date part.
    if re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(\.\d+)?", value):
        value = value[:10]
    return normalize_date_value(value)


def _entity_columns(conn: sqlite3.Connection) -> set:
    return {r[1] for r in conn.execute("PRAGMA table_info(entities)").fetchall()}


def _decompress_entity_data(data: Optional[bytes]) -> Optional[Dict[str, Any]]:
    """Tolerant reader for entities.compressed_data — zlib+json, zlib+pickle,
    gzip, or plain text. Returns a dict (or None on empty/undecodable).

    SECURITY NOTE on the pickle branch below: `pickle.loads` is only reached on
    first-party rows in this server's OWN local memory.db (the same trust model
    as memory_db_service._decompress_data — our own write path produces the
    rows). The primary path in _entity_observations uses the plain-text
    `observations` table and never touches compressed_data, so the pickle
    fallback is cold except for legacy rows written before that table existed.
    """
    if not data:
        return None
    try:
        decompressed = zlib.decompress(data)
    except zlib.error:
        try:
            decompressed = zlib.decompress(data, zlib.MAX_WBITS | 32)
        except zlib.error:
            try:
                import gzip

                decompressed = gzip.decompress(data)
            except Exception:
                decompressed = data  # treat raw bytes as payload
    except TypeError:
        return None
    try:
        return json.loads(decompressed.decode("utf-8", errors="replace"))
    except (ValueError, UnicodeDecodeError):
        # JSON failed; fall through to the pickle attempt (first-party rows only).
        pass
    try:
        obj = pickle.loads(decompressed)
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, list):
            return {"observations": [str(x) for x in obj]}
        if isinstance(obj, str):
            return {"observations": [obj]}
        return None
    except Exception:
        # Pickle failed; fall through to the plain-text attempt.
        pass
    try:
        text = decompressed.decode("utf-8", errors="replace")
        if text.strip():
            return {"observations": [text]}
    except Exception:
        # Plain-text attempt failed; the payload is not a shape we can read.
        pass
    return None


def _entity_observations(conn: sqlite3.Connection, entity_id: int) -> List[str]:
    """Observation text for an entity: the observations table first, then a
    compressed_data fallback (the observations table is the live write path;
    the fallback covers rows written before the table existed)."""
    try:
        rows = conn.execute(
            "SELECT content FROM observations WHERE entity_id = ? ORDER BY id",
            (entity_id,),
        ).fetchall()
        texts = [r[0] for r in rows if r[0] and str(r[0]).strip()]
        if texts:
            return texts
    except sqlite3.OperationalError:
        pass  # observations table absent in some legacy DBs

    row = conn.execute(
        "SELECT compressed_data FROM entities WHERE id = ?", (entity_id,)
    ).fetchone()
    if row is None:
        return []
    data = _decompress_entity_data(row[0])
    obs = (data or {}).get("observations") or []
    return [str(o) for o in obs if str(o).strip()]


def _derive_subject(name: str, entity_type: str, data: Optional[Dict[str, Any]]) -> str:
    """Subject name for a candidate entity: a `people`-style key in the entity
    payload wins; otherwise the entity name with fact-ish prefixes stripped."""
    if isinstance(data, dict):
        people = data.get("people")
        if isinstance(people, list):
            named = [p for p in people if isinstance(p, str) and p.strip()]
            if named:
                return named[0].strip()
        if isinstance(data.get("person"), str) and data["person"].strip():
            return data["person"].strip()
    n = (name or "").strip()
    for prefix in ("fact:", "test:", "auto_memory/", "session_episode/"):
        if n.startswith(prefix):
            n = n[len(prefix) :]
            break
    n = re.sub(r"_v\d+$", "", n)  # drop a trailing _v<N> index suffix
    n = n.strip()
    return n or (name or "unknown")


def _is_candidate(row: Tuple[Any, ...], cols: set) -> bool:
    """Candidates: bi-temporal facts (valid_until / superseded_by set) or
    fact-like entity_type. Defensive when the entities schema lacks the
    bi-temporal columns."""
    et_idx = _COL_INDEX.get("entity_type")
    entity_type = (row[et_idx] if et_idx is not None else "") or ""
    entity_type = entity_type.lower()
    if any(entity_type.startswith(p) for p in _FACT_LIKE_PREFIXES):
        return True
    idx_vu = _COL_INDEX.get("valid_until")
    idx_sb = _COL_INDEX.get("superseded_by")
    if idx_vu is not None and idx_sb is not None:
        if row[idx_vu] is not None or row[idx_sb] is not None:
            return True
    return False


# Column index map for the entities SELECT in _backfill_profiles; populated per
# call based on PRAGMA (the entities table has drifted across environments).
_COL_INDEX: Dict[str, int] = {}


def _row_to_profile(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "profile_id": row[0],
        "subject": row[1],
        "content": row[2],
        "keywords": json.loads(row[3] or "[]"),
        "valid_from": row[4] or "",
        "evidence": json.loads(row[5] or "[]"),
        "history": json.loads(row[6] or "[]"),
        "updated_at": row[7],
    }


def _load_profiles_for_subject(
    conn: sqlite3.Connection, subject: str
) -> List[Dict[str, Any]]:
    rows = conn.execute(
        f"""
        SELECT {", ".join(_PROFILE_COLS)}
        FROM temporal_profiles WHERE subject = ? COLLATE NOCASE
        """,
        (subject,),
    ).fetchall()
    return [_row_to_profile(r) for r in rows]


def _evidence_markers(profile: Dict[str, Any]) -> set:
    """All evidence markers across a profile's current state and history."""
    markers = set(profile.get("evidence") or [])
    for h in profile.get("history") or []:
        markers.update(h.get("evidence") or [])
    return markers


def _upsert_profile(
    conn: sqlite3.Connection,
    subject: str,
    content: str,
    valid_from: str,
    keywords: List[str],
    evidence: List[str],
) -> Dict[str, Any]:
    """Mirror of TemporalProfileStore.upsert using the pure version-chain
    mechanics, operating on the caller's connection."""
    content = content.strip()
    if not content:
        return {"action": "skip", "reason": "empty content"}

    existing = _load_profiles_for_subject(conn, subject)
    scored = sorted(
        ((jaccard_similarity(keywords, p.get("keywords", [])), p) for p in existing),
        key=lambda kv: kv[0],
        reverse=True,
    )
    matches = [p for _s, p in scored[:8]]

    candidate = {
        "subject": subject,
        "content": content,
        "valid_from": normalize_date_value(valid_from),
        "keywords": keywords,
        "evidence": evidence,
    }
    decision = deterministic_decision(candidate, matches)
    action = decision.get("action", "new")

    if action == "new" or not decision.get("profile_id"):
        count = conn.execute("SELECT COUNT(*) FROM temporal_profiles").fetchone()[0]
        profile = {
            "profile_id": f"P{count + 1}",
            "subject": subject,
            "content": content,
            "keywords": keywords,
            "valid_from": candidate["valid_from"],
            "evidence": _uniq(evidence),
            "history": [],
        }
        conn.execute(
            f"""
            INSERT INTO temporal_profiles
                ({", ".join(_PROFILE_COLS)})
            VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (
                profile["profile_id"],
                profile["subject"],
                profile["content"],
                json.dumps(profile["keywords"]),
                profile["valid_from"],
                json.dumps(profile["evidence"]),
                json.dumps(profile["history"]),
            ),
        )
        return {"action": "new", "profile_id": profile["profile_id"]}

    target = next(
        (p for p in existing if p.get("profile_id") == decision["profile_id"]),
        None,
    )
    if target is None:
        return {"action": "skip", "reason": "decision profile missing"}
    if action == "confirm":
        apply_confirm(target, candidate)
    elif action == "update_current":
        apply_update_current(target, candidate, decision.get("updated_content", ""))
    elif action == "update_history":
        apply_update_history(target, candidate, decision.get("updated_content", ""))
    else:
        apply_confirm(target, candidate)
    conn.execute(
        f"""
        INSERT INTO temporal_profiles
            ({", ".join(_PROFILE_COLS)})
        VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(profile_id) DO UPDATE SET
            content=excluded.content, keywords=excluded.keywords,
            valid_from=excluded.valid_from, evidence=excluded.evidence,
            history=excluded.history, updated_at=CURRENT_TIMESTAMP
        """,
        (
            target["profile_id"],
            target["subject"],
            target["content"],
            json.dumps(target.get("keywords", [])),
            target.get("valid_from", ""),
            json.dumps(target.get("evidence", [])),
            json.dumps(target.get("history", [])),
        ),
    )
    return {"action": action, "profile_id": target["profile_id"]}


def _renumber_history(profile: Dict[str, Any]) -> None:
    """Re-number history version_ids after an insert (mirrors the AtomMem
    version-chain bookkeeping)."""
    for idx, h in enumerate(profile.get("history", []) or [], 1):
        h["version_id"] = f"{profile.get('profile_id')}_v{idx}"


def _save_profile(conn: sqlite3.Connection, profile: Dict[str, Any]) -> None:
    conn.execute(
        f"""
        INSERT INTO temporal_profiles
            ({", ".join(_PROFILE_COLS)})
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


def _insert_history_entry(
    conn: sqlite3.Connection,
    profile: Dict[str, Any],
    content: str,
    valid_from: str,
    valid_to: str,
    evidence: List[str],
) -> None:
    """Insert a past-only state into a profile's history with its EXPLICIT
    [valid_from, valid_to) interval (the version-chain's apply_update_history
    infers valid_to from the next state; a bi-temporal fact's valid_until IS
    the interval end, so we record it directly)."""
    history = profile.setdefault("history", [])
    history.append(
        {
            "version_id": f"{profile.get('profile_id')}_v{len(history) + 1}",
            "content": content.strip(),
            "valid_from": normalize_date_value(valid_from),
            "valid_to": normalize_date_value(valid_to),
            "evidence": _uniq(evidence),
        }
    )
    history.sort(key=lambda h: date_start(h.get("valid_from")) or date.max)
    _renumber_history(profile)
    _save_profile(conn, profile)


def _match_profile(
    profiles: List[Dict[str, Any]], keywords: List[str]
) -> Tuple[Optional[Dict[str, Any]], float]:
    """Best profile for a fact by keyword overlap (same-attribute match)."""
    best, best_score = None, -1.0
    for p in profiles:
        score = jaccard_similarity(keywords, p.get("keywords", []))
        if score > best_score:
            best, best_score = p, score
    return best, best_score


def _backfill_profiles(conn: sqlite3.Connection) -> Dict[str, Any]:
    """
    Populate temporal_profiles from the live entities table.

    Candidates: entities with valid_until / superseded_by set (bi-temporal
    facts) or fact-like entity_type (fact, session_episode/..., audit_probe,
    auto_memory/...). For each observation of each candidate, upsert a profile
    row with valid_from = created_at and valid_to = valid_until.

    Two-phase, because a superseded fact is a PAST state:
      Phase 1 — CURRENT facts (no valid_until) become profile current states
                via the AtomMem deterministic_decision chain.
      Phase 2 — SUPERSEDED facts (valid_until set) are inserted into the
                matching profile's history with their explicit
                [created_at, valid_until) interval, honoring the spec's
                "valid_to = valid_until".

    Idempotent: skips observations whose evidence marker (entity:<id>:<idx>) or
    content-hash marker (hash:<sha16>) already exists in a profile.

    Returns:
        {"candidates", "observations_processed", "profiles_created",
         "profiles_updated", "history_inserts", "skipped", "subjects",
         "errors"}
    """
    _ensure_schema(conn)
    cols = _entity_columns(conn)
    need = ["id", "name", "entity_type", "created_at", "compressed_data"]
    present = [c for c in need if c in cols]
    if "valid_until" in cols:
        present.append("valid_until")
    if "superseded_by" in cols:
        present.append("superseded_by")
    # Column index map for the SELECT; must be built AFTER the optional
    # bi-temporal columns are appended (a stale idx silently read NULL for
    # valid_until/superseded_by, so every fact looked "current").
    idx = {c: present.index(c) for c in present}

    # Cache the optional bi-temporal column positions for _is_candidate.
    global _COL_INDEX
    _COL_INDEX = dict(idx)

    select = f"SELECT {', '.join(present)} FROM entities ORDER BY id"
    rows = conn.execute(select).fetchall()

    stats = {
        "candidates": 0,
        "observations_processed": 0,
        "profiles_created": 0,
        "profiles_updated": 0,
        "history_inserts": 0,
        "skipped": 0,
        "subjects": 0,
        "errors": 0,
    }
    subjects_seen = set()

    # Decode every candidate once, recording the per-observation facts.
    # Each fact: {entity_id, subject, obs, valid_from, valid_to, kw, evidence,
    #             superseded_by (int|None)}.
    facts: List[Dict[str, Any]] = []
    entity_subjects: Dict[int, str] = {}
    for row in rows:
        try:
            if not _is_candidate(row, cols):
                continue
            stats["candidates"] += 1
            entity_id = row[idx["id"]]
            name = row[idx.get("name", -1)] if "name" in idx else ""
            entity_type = (
                row[idx.get("entity_type", -1)] if "entity_type" in idx else ""
            )
            created_at = row[idx.get("created_at", -1)] if "created_at" in idx else ""
            data = None
            if "compressed_data" in idx:
                data = _decompress_entity_data(row[idx["compressed_data"]])
            subject = _derive_subject(name, entity_type, data)
            entity_subjects[entity_id] = subject
            subjects_seen.add(subject)
            valid_from = _date_only(created_at)
            valid_to = (
                _date_only(row[idx.get("valid_until", -1)])
                if "valid_until" in idx and idx.get("valid_until", -1) >= 0
                else ""
            )
            superseded_by = (
                row[idx.get("superseded_by", -1)]
                if "superseded_by" in idx and idx.get("superseded_by", -1) >= 0
                else None
            )
            obs_list = _entity_observations(conn, entity_id)
            if not obs_list:
                continue
            for oi, obs in enumerate(obs_list):
                obs = str(obs).strip()
                if not obs:
                    continue
                evidence_marker = f"entity:{entity_id}:{oi}"
                content_hash = hashlib.sha256(obs.encode("utf-8")).hexdigest()[:16]
                hash_marker = f"hash:{content_hash}"
                # Keywords stay PURE semantic tokens: mixing a hash marker into
                # keywords dragged the AtomMem jaccard below the update_current
                # threshold (0.6), so superseded pairs were split into separate
                # profiles instead of archived. The content-hash idempotency
                # marker lives in `evidence`, where it cannot skew decisions.
                kw = extract_keywords(obs, max_keywords=5)
                facts.append(
                    {
                        "entity_id": entity_id,
                        "subject": subject,
                        "obs": obs,
                        "valid_from": valid_from,
                        "valid_to": valid_to,
                        "kw": kw,
                        "evidence": [evidence_marker, hash_marker],
                        "superseded_by": superseded_by,
                    }
                )
        except Exception as e:  # one bad entity must not abort the backfill
            logger.warning("backfill entity %s failed: %s", row, e)
            stats["errors"] += 1

    def _already_backfilled(subject: str, evidence: List[str]) -> bool:
        markers = set(evidence)
        existing = _load_profiles_for_subject(conn, subject)
        return any(markers & _evidence_markers(p) for p in existing)

    # ---- Phase 1: current facts (no valid_until) become current states ---- #
    for fact in facts:
        if fact["valid_to"]:
            continue
        if _already_backfilled(fact["subject"], fact["evidence"]):
            stats["skipped"] += 1
            continue
        stats["observations_processed"] += 1
        res = _upsert_profile(
            conn,
            fact["subject"],
            fact["obs"],
            fact["valid_from"],
            fact["kw"],
            fact["evidence"],
        )
        if res.get("action") == "new":
            stats["profiles_created"] += 1
        elif res.get("action") in ("update_current", "update_history", "confirm"):
            stats["profiles_updated"] += 1

    # ---- Phase 2: superseded facts land in history with valid_to ---------- #
    for fact in facts:
        if not fact["valid_to"]:
            continue
        if _already_backfilled(fact["subject"], fact["evidence"]):
            stats["skipped"] += 1
            continue
        stats["observations_processed"] += 1
        # Prefer the successor's profile (superseded_by points at the entity
        # that replaced this fact); fall back to the fact's own subject.
        target_subject = entity_subjects.get(fact["superseded_by"]) or fact["subject"]
        candidates = _load_profiles_for_subject(conn, target_subject)
        target, score = _match_profile(candidates, fact["kw"])
        if target is not None and score >= _SAME_ATTRIBUTE_JACCARD:
            _insert_history_entry(
                conn,
                target,
                fact["obs"],
                fact["valid_from"],
                fact["valid_to"],
                fact["evidence"],
            )
            stats["history_inserts"] += 1
        else:
            # No same-attribute successor profile: the superseded fact has no
            # live state to attach to — fall back to a current row so the fact
            # is not dropped (it will be surfaced as a contradiction/stale).
            res = _upsert_profile(
                conn,
                fact["subject"],
                fact["obs"],
                fact["valid_from"],
                fact["kw"],
                fact["evidence"],
            )
            if res.get("action") == "new":
                stats["profiles_created"] += 1
            elif res.get("action") in (
                "update_current",
                "update_history",
                "confirm",
            ):
                stats["profiles_updated"] += 1
        conn.commit()

    conn.commit()
    stats["subjects"] = len(subjects_seen)
    return stats


# --------------------------------------------------------------------------- #
# Contradiction detection                                                      #
# --------------------------------------------------------------------------- #


def _collect_facts(
    conn: sqlite3.Connection, subject: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Flatten profile rows (current + history) into fact records, each with a
    valid-time interval and keywords derived from content."""
    if subject:
        rows = conn.execute(
            f"SELECT {', '.join(_PROFILE_COLS)} FROM temporal_profiles "
            "WHERE subject = ? COLLATE NOCASE",
            (subject,),
        ).fetchall()
    else:
        rows = conn.execute(
            f"SELECT {', '.join(_PROFILE_COLS)} FROM temporal_profiles"
        ).fetchall()

    facts: List[Dict[str, Any]] = []
    for r in rows:
        prof = _row_to_profile(r)
        subj = prof["subject"]
        facts.append(
            {
                "subject": subj,
                "content": prof.get("content", ""),
                "valid_from": prof.get("valid_from", ""),
                "valid_to": None,
                "keywords": extract_keywords(prof.get("content", ""), max_keywords=5),
                "source": prof.get("profile_id"),
            }
        )
        for h in prof.get("history") or []:
            hc = h.get("content", "")
            facts.append(
                {
                    "subject": subj,
                    "content": hc,
                    "valid_from": h.get("valid_from", ""),
                    "valid_to": h.get("valid_to", ""),
                    "keywords": extract_keywords(hc, max_keywords=5),
                    "source": h.get("version_id", prof.get("profile_id")),
                }
            )
    return facts


def _ranges_overlap(vf_a: str, vt_a: str, vf_b: str, vt_b: str) -> bool:
    """True when two [valid_from, valid_to) intervals overlap. None valid_to is
    open-ended (the state is still current)."""
    sa = date_start(vf_a)
    sb = date_start(vf_b)
    if sa is None or sb is None:
        return False
    ea = date_start(vt_a)
    eb = date_start(vt_b)
    a_before_b_end = eb is None or sa < eb
    b_before_a_end = ea is None or sb < ea
    return a_before_b_end and b_before_a_end


def _value_substitution(a: str, b: str) -> bool:
    """True when the two facts share a common template and differ only in a
    middle span (a value change), not a paraphrase."""
    if not a or not b or a == b:
        return False
    norm_a = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", a.lower())).strip()
    norm_b = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", b.lower())).strip()
    if norm_a == norm_b:
        return False  # punctuation/whitespace-only difference
    sm = difflib.SequenceMatcher(None, a, b)
    matching = sum(block.size for block in sm.get_matching_blocks())
    coverage = matching / max(len(a), len(b), 1)
    return coverage >= _VALUE_SUBSTITUTION_COVERAGE


def _find_contradictions(
    conn: sqlite3.Connection,
    subject: Optional[str] = None,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    """Pairs of facts for the same subject+attribute with overlapping valid
    ranges whose differing span is a value change (not a paraphrase)."""
    facts = _collect_facts(conn, subject)
    by_subject: Dict[str, List[Dict[str, Any]]] = {}
    for f in facts:
        by_subject.setdefault(f["subject"], []).append(f)

    contradictions: List[Dict[str, Any]] = []
    for subj, subj_facts in by_subject.items():
        n = len(subj_facts)
        for i in range(n):
            if len(contradictions) >= limit:
                break
            for j in range(i + 1, n):
                a, b = subj_facts[i], subj_facts[j]
                if a["content"] == b["content"]:
                    continue
                kw_a, kw_b = a["keywords"], b["keywords"]
                if jaccard_similarity(kw_a, kw_b) < _SAME_ATTRIBUTE_JACCARD:
                    continue  # different attribute, not a contradiction
                if not _ranges_overlap(
                    a["valid_from"], a["valid_to"], b["valid_from"], b["valid_to"]
                ):
                    continue  # disjoint valid ranges — sequential states
                if not _value_substitution(a["content"], b["content"]):
                    continue  # paraphrase, not a value flip
                contradictions.append(
                    {
                        "subject": subj,
                        "fact_a": {
                            "content": a["content"],
                            "valid_from": a["valid_from"],
                            "valid_to": a["valid_to"],
                            "source": a["source"],
                        },
                        "fact_b": {
                            "content": b["content"],
                            "valid_from": b["valid_from"],
                            "valid_to": b["valid_to"],
                            "source": b["source"],
                        },
                        "valid_ranges": {
                            "a": f"{a['valid_from'] or '?'} → {a['valid_to'] or 'now'}",
                            "b": f"{b['valid_from'] or '?'} → {b['valid_to'] or 'now'}",
                        },
                        "recommendation": (
                            f"Same subject '{subj}', same attribute, overlapping validity "
                            f"(A: {a['valid_from'] or '?'}→{a['valid_to'] or 'now'}, "
                            f"B: {b['valid_from'] or '?'}→{b['valid_to'] or 'now'}). The "
                            "differing span looks like a value change, not a paraphrase. "
                            "Confirm which state is authoritative; if one is superseded, "
                            "set its valid_until (bi-temporal) or archive it via history "
                            "to resolve the overlap."
                        ),
                    }
                )
                if len(contradictions) >= limit:
                    break
        if len(contradictions) >= limit:
            break
    return contradictions


# --------------------------------------------------------------------------- #
# Tool registration                                                            #
# --------------------------------------------------------------------------- #


def register_curate_profiles_tools(app, db_path: str):
    """Register the curate_profiles and profile_summary MCP tools."""

    def _connect() -> sqlite3.Connection:
        conn = sqlite3.connect(db_path, timeout=30)
        conn.execute("PRAGMA busy_timeout = 30000")
        _ensure_schema(conn)
        return conn

    @app.tool()
    async def curate_profiles(
        subject: Optional[str] = None,
        limit: int = 200,
    ) -> Dict[str, Any]:
        """
        Curate profile drift: backfill temporal_profiles from the live entities
        table, then surface contradictions (same subject+attribute, overlapping
        valid ranges, value-change not paraphrase).

        Args:
            subject: Restrict curation to one subject (case-insensitive).
            limit: Max contradictions to return (default 200).

        Returns:
            Dict with profiles_count, backfill stats, and contradictions
            (each with subject, fact_a, fact_b, valid_ranges, recommendation).
        """
        conn = _connect()
        try:
            backfill = _backfill_profiles(conn)
            contradictions = _find_contradictions(conn, subject, limit)
            profiles_count = conn.execute(
                "SELECT COUNT(*) FROM temporal_profiles"
            ).fetchone()[0]
        finally:
            conn.close()
        return {
            "profiles_count": profiles_count,
            "backfill": backfill,
            "contradictions": contradictions,
            "contradiction_count": len(contradictions),
        }

    @app.tool()
    async def profile_summary(
        subject: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Current snapshot per subject: the newest valid fact per attribute
        (one row per profile; archived states are folded into `history_count`).

        Args:
            subject: Restrict to one subject (case-insensitive).

        Returns:
            Dict with profiles: [{profile_id, subject, content, valid_from,
            keywords, history_count}] and count.
        """
        conn = _connect()
        try:
            if subject:
                rows = conn.execute(
                    f"SELECT {', '.join(_PROFILE_COLS)} FROM temporal_profiles "
                    "WHERE subject = ? COLLATE NOCASE",
                    (subject,),
                ).fetchall()
            else:
                rows = conn.execute(
                    f"SELECT {', '.join(_PROFILE_COLS)} FROM temporal_profiles "
                    "ORDER BY subject"
                ).fetchall()
            profiles = []
            for r in rows:
                prof = _row_to_profile(r)
                profiles.append(
                    {
                        "profile_id": prof["profile_id"],
                        "subject": prof["subject"],
                        "content": prof["content"],
                        "valid_from": prof["valid_from"],
                        "keywords": prof["keywords"],
                        "history_count": len(prof.get("history") or []),
                    }
                )
        finally:
            conn.close()
        return {"profiles": profiles, "count": len(profiles)}

    logger.info("Registered 2 curate-profiles MCP tools")
    return True
