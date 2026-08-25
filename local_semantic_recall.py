#!/usr/bin/env python3
"""Semantic recall for enhanced-memory — model-agnostic backfill + search.

Embeds via the FEDORA ollama node using ANY local embedding model you've pulled
there (nomic-embed-text, embeddinggemma, snowflake-arctic-embed2, qwen3-embedding,
granite-embedding, bge-m3, ...). The SAME model embeds both documents (backfill)
and queries (search), so retrieval is consistent.

Each model has its own dimension (nomic 768, granite-30m 384, arctic2 1024,
qwen3 up to 4096), and a Qdrant collection is fixed-dimension — so each model
gets its OWN collection, dimension auto-detected from a probe embedding. nomic
keeps the legacy `enhanced_memory` collection for backward-compat; every other
model uses `enhanced_memory__<model>`. Switching models never clobbers another.

Config (env, both overridable by --model / --ollama-url):
  MEMORY_EMBED_MODEL   default "nomic-embed-text"
  MEMORY_OLLAMA_URL    default "http://127.0.0.1:11434"  (mac-studio local; fedora ollama masked 2026-07-02, cluster gateway = ai-lab.local:11435)

Usage (mcp venv python):
  python local_semantic_recall.py list-models                 # embed models on the endpoint
  python local_semantic_recall.py backfill [--auto-only] [--model M]
  python local_semantic_recall.py search "how to compress context" -k 5 [--model M]
  python local_semantic_recall.py count [--model M]
"""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
import sys

import httpx
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from memory_paths import get_db_path

# Env seams (ADDITIVE — default to prod, so unset env == prior behavior). A scratch/benchmark
# instance can redirect the backfill source DB (ENHANCED_MEMORY_DB_PATH, the same var the
# server uses), the Qdrant endpoint (MEMORY_QDRANT_URL), and the collection (see
# collection_for) so it can build/query an isolated per-question index without touching the
# prod enhanced_memory* collections.
DB = str(get_db_path())
QDRANT = os.environ.get("MEMORY_QDRANT_URL", "http://localhost:6333")
DEFAULT_MODEL = os.environ.get("MEMORY_EMBED_MODEL", "embeddinggemma")
# Retrieval-quality signal (Phase G, 2026-08-05): a top cosine below this flags a
# weak/quiet miss. Calibrated for embeddinggemma 768d where real matches run
# >=0.55 and nonsense queries sit ~0.46.
LOW_CONFIDENCE_THRESHOLD = float(os.environ.get("MEMORY_LOW_CONF_THRESHOLD", "0.50"))
# Default changed fedora -> localhost 2026-07-02 (Phase 0 spine repair): recall
# and write-path indexing must not depend on a remote node being up (fedora was
# verifiably unreachable during the 2026-07-01 audit). mac-studio's local ollama
# serves the same embeddinggemma model. Cluster nodes override via env.
OLLAMA_URL = os.environ.get("MEMORY_OLLAMA_URL", "http://127.0.0.1:11434")
BATCH = 64

# Substrings that mark an ollama model as embedding-capable (for list-models).
_EMBED_HINTS = (
    "embed",
    "bge",
    "nomic",
    "minilm",
    "mpnet",
    "granite-embed",
    "embeddinggemma",
    "arctic",
    "qwen3-embedding",
)


def collection_for(model: str) -> str:
    """nomic keeps the legacy collection; others get a per-model collection.

    MEMORY_QDRANT_COLLECTION overrides everything — a scratch/benchmark instance pins its own
    isolated per-question collection here so it never reads or writes the prod enhanced_memory*.
    """
    override = os.environ.get("MEMORY_QDRANT_COLLECTION")
    if override:
        return override
    if model == "nomic-embed-text":
        return "enhanced_memory"
    safe = re.sub(r"[^a-zA-Z0-9]+", "_", model).strip("_").lower()
    return f"enhanced_memory__{safe}"


def embed(texts: list[str], model: str) -> list[list[float]]:
    out = []
    with httpx.Client(timeout=120) as c:
        for t in texts:
            r = c.post(
                f"{OLLAMA_URL}/api/embeddings", json={"model": model, "prompt": t}
            )
            r.raise_for_status()
            out.append(r.json()["embedding"])
    return out


def ensure_collection(client: QdrantClient, name: str, dim: int) -> None:
    """Create the collection at the model's dimension if it doesn't exist."""
    try:
        info = client.get_collection(name)
        existing = info.config.params.vectors.size  # type: ignore[union-attr]
        if existing != dim:
            raise SystemExit(
                f"Collection {name} exists at dim {existing} but model emits {dim}. "
                f"Delete it first (client.delete_collection) or use a different model."
            )
    except Exception as e:
        if isinstance(e, SystemExit):
            raise
        client.create_collection(
            name, vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
        )
        print(f"  created collection {name} (dim={dim}, cosine)")


def _rows(auto_only: bool):
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    # Mirror the canonical liveness filter every SQL search path applies
    # (memory_db_service adds archived_at/superseded_by; the write-path
    # indexer excludes the archive/quarantine tiers, and supersede EVICTS
    # the Qdrant point). Without this, a backfill resurrects every hidden
    # memory into the vector path that injects into prompts - it did,
    # 2026-08-24: 2,048 points (filtered corpus) became 12,017 (whole
    # store, incl. 8,414 archive-tier + 1,553 quarantine-tier entities).
    q = "SELECT id, name, entity_type FROM entities WHERE " + LIVE_PREDICATE
    if auto_only:
        q += " AND entity_type LIKE 'auto_memory/%'"
    ents = conn.execute(q).fetchall()
    out = []
    for e in ents:
        obs = conn.execute(
            "SELECT content FROM observations WHERE entity_id=? LIMIT 8", (e["id"],)
        ).fetchall()
        # Template contextual prefixes measurably HURT vague-query retrieval
        # (2026-08-24 A/B, 30 targets + 400 distractors, embeddinggemma:
        # top-1 0.53 with vs 0.60 without, MRR 0.644 vs 0.715; stripping
        # improved 9/30 entities, hurt 2). Filter matches the template's
        # rigid skeleton, BOTH halves required, so LLM-generated
        # "[Context: ...]" prefixes - which measured best - stay embedded.
        contents = [
            o["content"]
            for o in obs
            if not (
                o["content"].startswith("[Context: This is a ")
                and "' with information about" in o["content"]
            )
        ]
        if obs and not contents:
            continue  # nothing but boilerplate: no document worth embedding
        text = (e["name"] + ": " + " ".join(contents)).strip()
        if len(text) > 8:
            out.append((e["id"], e["name"], e["entity_type"], text[:4000]))
    conn.close()
    return out


def backfill(model: str, auto_only: bool) -> int:
    coll = collection_for(model)
    rows = _rows(auto_only)
    client = QdrantClient(url=QDRANT)
    dim = len(embed(["dimension probe"], model)[0])
    ensure_collection(client, coll, dim)
    print(f"backfilling {len(rows)} entities via {model} ({dim}d) -> {coll} ...")
    done = 0
    for i in range(0, len(rows), BATCH):
        chunk = rows[i : i + BATCH]
        vecs = embed([c[3] for c in chunk], model)
        pts = [
            PointStruct(id=c[0], vector=v, payload={"name": c[1], "entity_type": c[2]})
            for c, v in zip(chunk, vecs)
        ]
        client.upsert(coll, points=pts)
        done += len(pts)
        print(f"  {done}/{len(rows)}")
    print(f"done. {coll} points_count={client.count(coll).count}")
    return 0


LIVE_PREDICATE = (
    "COALESCE(tier,'') NOT IN ('archive','quarantine')"
    " AND archived_at IS NULL AND superseded_by IS NULL"
)


def _drop_dead(hits):
    """Keep only hits whose entity is still live in sqlite.

    The payload is a cache written at index time; archiving does not evict
    the point (only supersede does), so a reader that trusts the payload
    resurrects archived memories. Measured 2026-08-25: 42 freshly archived
    ARC-era entities, and this CLI ranked one of them first while the
    SQL-gated MCP tool and per-prompt hook were clean. Same predicate as
    _rows, so the backfill and the reader agree on what "live" means.
    """
    ids = [int(h.id) for h in hits]
    if not ids:
        return []
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    marks = ",".join("?" * len(ids))
    live = {
        r[0]
        for r in conn.execute(
            f"SELECT id FROM entities WHERE id IN ({marks}) AND {LIVE_PREDICATE}", ids
        )
    }
    conn.close()
    return [h for h in hits if int(h.id) in live]


def search(model: str, query: str, k: int) -> int:
    coll = collection_for(model)
    qv = embed([query], model)[0]
    client = QdrantClient(url=QDRANT)
    # Over-fetch, then gate on the database: dead points may still be in the
    # collection and must never reach the reader as a result.
    hits = _drop_dead(client.query_points(coll, query=qv, limit=k * 3).points)[:k]
    print(f"query: {query!r}  (model={model}, coll={coll})")
    for h in hits:
        pl = h.payload or {}
        print(f"  {h.score:.3f}  [{pl.get('entity_type')}] {pl.get('name')}")
    return 0


def count(model: str) -> int:
    coll = collection_for(model)
    try:
        print(f"{coll} points_count:", QdrantClient(url=QDRANT).count(coll).count)
    except Exception as e:
        print(f"{coll}: not created yet ({type(e).__name__})")
    return 0


def list_models() -> int:
    try:
        with httpx.Client(timeout=10) as c:
            tags = c.get(f"{OLLAMA_URL}/api/tags").json().get("models", [])
    except Exception as e:
        print(f"  fedora ollama unreachable at {OLLAMA_URL}: {e}")
        return 1
    names = [m["name"] for m in tags]
    embed_models = [n for n in names if any(h in n.lower() for h in _EMBED_HINTS)]
    print(f"embed-capable models on {OLLAMA_URL}:")
    for n in embed_models:
        print(f"  {n}  (collection: {collection_for(n.split(':')[0])})")
    print(f"(pull more on the host serving {OLLAMA_URL}: ollama pull <model>)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL, help="ollama embedding model")
    ap.add_argument("--ollama-url", default=None, help="override MEMORY_OLLAMA_URL")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backfill")
    b.add_argument("--auto-only", action="store_true")
    s = sub.add_parser("search")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=5)
    sub.add_parser("count")
    sub.add_parser("list-models")
    args = ap.parse_args()
    if args.ollama_url:
        global OLLAMA_URL
        OLLAMA_URL = args.ollama_url
    if args.cmd == "backfill":
        return backfill(args.model, args.auto_only)
    if args.cmd == "search":
        return search(args.model, args.query, args.k)
    if args.cmd == "count":
        return count(args.model)
    return list_models()


if __name__ == "__main__":
    sys.exit(main())
