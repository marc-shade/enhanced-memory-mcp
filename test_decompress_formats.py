#!/usr/bin/env python3
"""Regression test for decompress_data format-tolerance (server.py).

The entities.compressed_data column historically holds 3+ encodings: zlib+pickle
(canonical), zlib+plain-text (episodic/insight), and gzip (service_event). The
original decompress_data assumed only zlib+pickle and raised on ~76% of entities,
breaking detect_memory_conflicts and any tool reading compressed_data.

This test asserts decompress_data reads all three synthetic formats AND that it
reads ~100% of the live DB (not just the 24% pickle subset).

Run: .venv/bin/python test_decompress_formats.py
"""

import gzip
import json
import os
import pickle
import sqlite3
import sys
import zlib

sys.path.insert(0, "/Volumes/SSDRAID0/agentic-system/mcp-servers/enhanced-memory-mcp")
from server import decompress_data


def test_synthetic_formats():
    # zlib + pickle (canonical)
    assert decompress_data(zlib.compress(pickle.dumps({"a": 1}))) == {"a": 1}
    # zlib + json
    assert decompress_data(zlib.compress(json.dumps({"b": 2}).encode())) == {"b": 2}
    # zlib + plain text -> wrapped as {"observations": [text]}
    out = decompress_data(zlib.compress(b"Type: pattern\nDescription: x"))
    assert isinstance(out, dict) and "observations" in out
    # gzip (service_event style)
    g = decompress_data(gzip.compress(b"some service event text"))
    assert "service event" in str(g)
    print("PASS: synthetic formats (zlib+pickle, zlib+json, zlib+text, gzip)")


def test_live_db_readability():
    db = os.path.expanduser("~/.claude/enhanced_memories/memory.db")
    rows = (
        sqlite3.connect(db)
        .execute(
            "SELECT compressed_data FROM entities WHERE compressed_data IS NOT NULL"
        )
        .fetchall()
    )
    ok = 0
    for (blob,) in rows:
        try:
            decompress_data(blob)
            ok += 1
        except Exception:
            pass
    pct = 100 * ok // max(len(rows), 1)
    assert pct >= 99, f"only {pct}% readable ({ok}/{len(rows)}) — format gap remains"
    print(f"PASS: live DB readability {ok}/{len(rows)} ({pct}%)")


if __name__ == "__main__":
    test_synthetic_formats()
    test_live_db_readability()
    print("ALL PASS")
