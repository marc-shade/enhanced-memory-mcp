#!/usr/bin/env python3
"""Propose bridging edges to reconnect the fragmented entity graph.

MemGraphRAG (arXiv 2606.00610) sec. 4.2.3 reconnects disconnected subgraphs with
two kinds of edge: type-based (entities sharing a stable ontology type) and
similarity-based (entities whose embeddings are near-duplicates). See
libs/memgraph/bridging.py.

Why this exists: as of 2026-07-19 the live store holds 968 relations across
10,059 entities, with only 539 entities (5.4%) in any relation at all. Graph
retrieval -- personalized PageRank included -- has nothing to traverse. The
missing piece is graph CONSTRUCTION, not a better ranker.

Read-only. Opens SQLite immutable and only scrolls Qdrant. Proposals go to
stdout or a JSON file for review; nothing is written back, because a wrong
bridge merges two distinct things and every later multi-hop answer inherits the
error.

Usage:
    python3 graph_bridge_report.py --limit 30
    python3 graph_bridge_report.py --json proposals.json --similarity 0.90
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "libs"))

from memgraph import bridging  # noqa: E402
from memgraph.ontology import OntologyLayer  # noqa: E402

DEFAULT_DB = Path.home() / ".claude" / "enhanced_memories" / "memory.db"
QDRANT = "http://127.0.0.1:6333"
# The live collection. Bare `enhanced_memory` and `enhanced_memory__bge_m3` are
# legacy and hold 364 stale points each.
COLLECTION = "enhanced_memory__embeddinggemma"


def load_graph(db_path: Path):
    """Entity types and existing relations, without taking a write lock."""
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        entities = {
            str(row[0]): (row[1] or "", row[2] or "")
            for row in conn.execute("SELECT id, name, entity_type FROM entities")
        }
        relations = [
            (str(a), str(b))
            for a, b in conn.execute(
                "SELECT from_entity_id, to_entity_id FROM relations "
                "WHERE from_entity_id IS NOT NULL AND to_entity_id IS NOT NULL"
            )
        ]
    return entities, relations


def fetch_embeddings(collection: str, limit: int):
    """Scroll vectors out of Qdrant, keyed by entity name.

    Qdrant point ids are not entity ids here, so points are matched back by
    the `name` payload field. Points without a name are skipped rather than
    guessed at.
    """
    names, vectors, offset = [], [], None
    while len(names) < limit:
        body = {
            "limit": min(256, limit - len(names)),
            "with_payload": True,
            "with_vector": True,
        }
        if offset is not None:
            body["offset"] = offset
        request = urllib.request.Request(
            f"{QDRANT}/collections/{collection}/points/scroll",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = json.loads(response.read())["result"]
        except (urllib.error.URLError, KeyError, json.JSONDecodeError) as e:
            print(
                f"WARNING: Qdrant unavailable ({e}); skipping similarity pass",
                file=sys.stderr,
            )
            return [], np.zeros((0, 0), dtype=np.float32)

        for point in payload.get("points", []):
            name = (point.get("payload") or {}).get("name")
            vector = point.get("vector")
            if name and isinstance(vector, list):
                names.append(name)
                vectors.append(vector)
        offset = payload.get("next_page_offset")
        if offset is None:
            break

    if not vectors:
        return [], np.zeros((0, 0), dtype=np.float32)
    return names, np.asarray(vectors, dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--collection", default=COLLECTION)
    ap.add_argument("--tau", type=int, default=2, help="schema stability threshold")
    ap.add_argument(
        "--max-fanout",
        type=int,
        default=bridging.DEFAULT_MAX_TYPE_FANOUT,
        help="skip types labelling more entities than this (too generic to bridge)",
    )
    ap.add_argument(
        "--similarity",
        type=float,
        default=bridging.DEFAULT_SIMILARITY_THRESHOLD,
        help="cosine threshold for a similarity bridge",
    )
    ap.add_argument("--max-vectors", type=int, default=4000)
    ap.add_argument("--limit", type=int, default=25, help="proposals to print")
    ap.add_argument("--json", type=Path, help="write all proposals here")
    args = ap.parse_args()

    if not args.db.exists():
        print(f"FATAL: no database at {args.db}", file=sys.stderr)
        return 2

    entities, relations = load_graph(args.db)
    if not entities:
        print("FATAL: no entities", file=sys.stderr)
        return 2

    before = bridging.connectivity(list(entities), relations)
    print(f"Graph connectivity BEFORE bridging ({args.db}):")
    print(f"  entities            {before['entities']:>6}")
    print(f"  relations           {before['edges']:>6}")
    print(
        f"  connected entities  {before['connected_entities']:>6}"
        f"  ({before['connected_fraction']:.1%})"
    )
    print(f"  edges per entity    {before['edges_per_entity']:>6.3f}")

    # Stable types only: bridging on a type seen once would wire together
    # whatever that one-off label happened to cover.
    type_counts = Counter(t for _, t in entities.values() if t)
    ontology = OntologyLayer(tau=args.tau)
    for type_name, count in type_counts.items():
        ontology.observe(type_name, "instance_of", "entity", n=count)
    stable = {s.head_type for s in ontology.stable_schemas()}

    entity_types = {eid: t for eid, (_, t) in entities.items() if t}
    type_props = bridging.type_bridges(
        entity_types, stable_types=stable, max_fanout=args.max_fanout
    )
    print(f"\n  stable types {len(stable)}, type-based proposals {len(type_props)}")

    names, vectors = fetch_embeddings(args.collection, args.max_vectors)
    sim_props = []
    if len(names) > 1:
        by_name = {name: eid for eid, (name, _) in entities.items() if name}
        keep = [(i, by_name[n]) for i, n in enumerate(names) if n in by_name]
        if len(keep) > 1:
            idx = [i for i, _ in keep]
            sim_props = bridging.similarity_bridges(
                [eid for _, eid in keep], vectors[idx], threshold=args.similarity
            )
        print(
            f"  vectors {len(names)} ({len(keep)} matched to entities), "
            f"similarity proposals {len(sim_props)}"
        )
    else:
        print("  vectors 0 -- similarity pass skipped")

    proposals = bridging.merge_proposals(type_props, sim_props)
    projected = bridging.connectivity(
        list(entities), relations + [p.pair() for p in proposals]
    )
    print(f"\nProjected connectivity IF all {len(proposals)} proposals were accepted:")
    print(
        f"  connected entities  {projected['connected_entities']:>6}"
        f"  ({projected['connected_fraction']:.1%})"
    )
    print(f"  edges per entity    {projected['edges_per_entity']:>6.3f}")

    if proposals:
        name_of = {eid: name for eid, (name, _) in entities.items()}

        def show(items, heading):
            if not items:
                return
            print(f"\n{heading}")
            for bridge in items[: args.limit]:
                a, b = bridge.pair()
                print(
                    f"  [{bridge.score:.4f}] {name_of.get(a, a)[:34]:<36}"
                    f" <-> {name_of.get(b, b)[:34]}"
                )

        similarity_only = [p for p in proposals if "similarity" in p.kind]
        type_only = [p for p in proposals if p.kind == "type"]

        # Dated series embed at ~0.99 and are NOT duplicates; separating them
        # means a reviewer cannot destroy a time series by acting on the wrong
        # row of one undifferentiated list.
        mergeable, series = bridging.partition_by_mergeability(similarity_only, name_of)

        show(
            mergeable,
            f"REVIEW QUEUE -- similarity bridges, dated series removed "
            f"({len(mergeable)}).",
        )
        if mergeable:
            print(
                "\n  This is a review queue, NOT a merge list. The filter removes one\n"
                "  specific high-volume false-positive class (identical name shell,\n"
                "  different embedded date). It does NOT catch same-date entries that\n"
                "  differ only by suffix -- e.g. '...2026-02-15-Training-completed-\n"
                "  macpro51_grpo' vs '..._wm_cuda' are distinct runs and will appear\n"
                "  here. Every row still needs eyes before anything is merged."
            )
        show(
            series,
            f"RELATED ONLY -- dated series ({len(series)}). Same name, different "
            "day. Safe as related-to edges; NEVER merge these.",
        )

        show(
            type_only,
            f"LOW PRECISION -- type-only bridges ({len(type_only)}). Sharing a "
            "type is weak evidence; do not accept in bulk without review.",
        )

    if args.json:
        args.json.write_text(
            json.dumps(
                [
                    {
                        "source": p.source,
                        "target": p.target,
                        "kind": p.kind,
                        "score": p.score,
                        "evidence": p.evidence,
                    }
                    for p in proposals
                ],
                indent=2,
            )
        )
        print(f"\nwrote {len(proposals)} proposals -> {args.json}")

    print("\nProposals only. Nothing was written to the graph; accepting a bridge")
    print("is a separate, explicitly-invoked operation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
