#!/usr/bin/env python3
"""Schema-health report for the memory store's entity types.

Applies MemGraphRAG's (arXiv 2606.00610) eq. 4 frequency criterion,

    M_ont^stable = { s in M_ont : Freq(s) >= tau }

to our own entity_type vocabulary. Their pilot found that a large share of
LLM-extracted relation types are one-off inventions rather than real types, and
that dropping the low-frequency tail *raised* accuracy. The same pattern is
present here: as of 2026-07-19, 636 distinct entity types across 10,059
entities, of which 405 (64%) are used exactly once.

This reports; it does not delete. The paper filters *facts* that fail schema
alignment during graph construction, which is safe because the graph is being
rebuilt. Our entities are the durable record -- a singleton type means the
label is inconsistent, not that the memory is junk. So the output is a
consolidation worklist: for each singleton type, the stable types it most
plausibly belongs to, by token overlap.

Read-only. Opens the database in immutable mode and never writes.

Usage:
    python3 schema_health.py [--db PATH] [--tau N] [--limit N] [--json]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "libs"))

from memgraph.conflicts import normalize  # noqa: E402
from memgraph.ontology import OntologyLayer  # noqa: E402
from memory_paths import get_db_path

DEFAULT_DB = get_db_path()


def load_type_counts(db_path: Path) -> Counter[str]:
    """Read entity_type frequencies without taking a write lock."""
    uri = f"file:{db_path}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        rows = conn.execute(
            "SELECT entity_type, COUNT(*) FROM entities "
            "WHERE entity_type IS NOT NULL AND entity_type != '' "
            "GROUP BY entity_type"
        ).fetchall()
    return Counter({name: count for name, count in rows})


def _tokens(type_name: str) -> set[str]:
    # Types are written as slug/snake/path forms (auto_memory/project,
    # reasoning_lesson/success), so split on the separators rather than
    # treating the whole label as opaque.
    return {
        t for t in normalize(type_name.replace("/", " ").replace("_", " ")).split() if t
    }


def suggest_merges(
    singletons: list[str], stable: list[str], max_suggestions: int = 3
) -> dict[str, list[str]]:
    """For each singleton type, the stable types sharing the most tokens.

    Token overlap, not embeddings: this runs read-only against a live database
    and should not require a model to be up. Types with no overlap get an empty
    list rather than a forced low-confidence guess.
    """
    stable_tokens = [(s, _tokens(s)) for s in stable]
    out: dict[str, list[str]] = {}
    for singleton in singletons:
        tokens = _tokens(singleton)
        if not tokens:
            out[singleton] = []
            continue
        scored = []
        for name, other in stable_tokens:
            shared = tokens & other
            if shared:
                # Jaccard, so a short exact-ish match beats an incidental
                # overlap with a long compound type.
                scored.append((len(shared) / len(tokens | other), name))
        scored.sort(reverse=True)
        out[singleton] = [name for _, name in scored[:max_suggestions]]
    return out


def build_report(db_path: Path, tau: int, limit: int) -> dict:
    counts = load_type_counts(db_path)
    if not counts:
        return {"error": f"no entity types found in {db_path}"}

    ontology = OntologyLayer(tau=tau)
    for type_name, count in counts.items():
        # One schema per type: the relation dimension is not modelled here
        # because relation_type has only 12 distinct values and is not the
        # source of the sprawl.
        ontology.observe(type_name, "instance_of", "entity", n=count)

    stable = [s.head_type for s in ontology.stable_schemas()]
    candidates = [s.head_type for s in ontology.candidate_schemas()]
    singletons = sorted(t for t, c in counts.items() if c == 1)
    total_entities = sum(counts.values())

    # Types that are distinct strings but identical after casefolding and
    # punctuation stripping are pure duplicates -- the cheapest merges
    # available, and the reason distinct_types exceeds the normalized
    # vocabulary size.
    normalized: dict[str, list[str]] = {}
    for type_name in counts:
        normalized.setdefault(normalize(type_name), []).append(type_name)
    collisions = {k: sorted(v) for k, v in normalized.items() if len(v) > 1}

    return {
        "database": str(db_path),
        "tau": tau,
        "totals": {
            "entities": total_entities,
            "distinct_types": len(counts),
            "stable_types": len(stable),
            "below_tau_types": len(candidates),
            "singleton_types": len(singletons),
            "singleton_share": round(len(singletons) / len(counts), 4),
            "entities_under_singleton_types": len(singletons),
            "normalized_vocabulary": len(normalized),
            "case_punctuation_duplicates": len(collisions),
        },
        "duplicate_types": collisions,
        "top_types": counts.most_common(10),
        "consolidation_worklist": {
            k: v for k, v in list(suggest_merges(singletons, stable).items())[:limit]
        },
    }


def render(report: dict) -> str:
    if "error" in report:
        return f"ERROR: {report['error']}"
    t = report["totals"]
    lines = [
        f"Schema health for {report['database']}",
        f"  tau = {report['tau']} (a type is stable once it labels this many entities)",
        "",
        f"  entities            {t['entities']:>6}",
        f"  distinct types      {t['distinct_types']:>6}",
        f"  stable types        {t['stable_types']:>6}",
        f"  below tau           {t['below_tau_types']:>6}",
        f"  singletons          {t['singleton_types']:>6}"
        f"  ({t['singleton_share']:.1%} of the vocabulary)",
        "",
        "  most-used types:",
    ]
    lines += [f"    {name:<40} {count:>6}" for name, count in report["top_types"]]

    duplicates = report.get("duplicate_types", {})
    if duplicates:
        lines += [
            "",
            f"  types differing only by case or punctuation "
            f"({t['case_punctuation_duplicates']}) -- safe to merge:",
        ]
        lines += [
            f"    {' == '.join(variants)}"
            for variants in list(duplicates.values())[:10]
        ]

    worklist = report["consolidation_worklist"]
    if worklist:
        lines += [
            "",
            "  consolidation candidates (singleton -> plausible stable type):",
        ]
        for singleton, suggestions in worklist.items():
            target = ", ".join(suggestions) if suggestions else "(no token overlap)"
            lines.append(f"    {singleton:<40} -> {target}")
    lines += [
        "",
        "  Report only. Nothing was modified; merging types is a separate,",
        "  explicitly-invoked operation.",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument(
        "--tau",
        type=int,
        default=2,
        help="entity count at which a type is considered stable (eq. 4)",
    )
    ap.add_argument(
        "--limit", type=int, default=25, help="consolidation entries to show"
    )
    ap.add_argument("--json", action="store_true", help="emit raw JSON")
    args = ap.parse_args()

    if not args.db.exists():
        print(f"FATAL: no database at {args.db}", file=sys.stderr)
        return 2

    report = build_report(args.db, args.tau, args.limit)
    print(json.dumps(report, indent=2) if args.json else render(report))
    return 1 if "error" in report else 0


if __name__ == "__main__":
    raise SystemExit(main())
