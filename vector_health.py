"""Honest-degradation guard for the vector / semantic tool surface.

The enhanced-memory MCP exposes vector tools (`semantic_recall`,
`semantic_cache_*`) that depend on optional, heavy, environment-specific
dependencies: `qdrant_client`, `sentence_transformers`, a reachable Qdrant
server, and a loadable local embedder. When the interpreter running the server
lacks one of these, the *honest* behaviour is to return a STRUCTURED
"unavailable"/"degraded" dict that names the missing piece and how to fix it --
never a raw `ModuleNotFoundError`, a bare traceback, or a silent empty result
dressed up as success.

This module provides:

- `check_vector_semantic_health()` -- structured booleans + reasons for every
  dependency, computed against the ACTUAL interpreter the server runs under.
- `degraded_payload()` -- the canonical structured-degradation dict shape.
- `GuardedToolApp` -- a thin proxy around the FastMCP `app` whose `.tool()`
  decorator wraps each registered tool with a cheap pre-check. If the pre-check
  reports the backend unavailable, the tool returns the structured dict instead
  of executing (and instead of leaking a dependency error). The proxy preserves
  the wrapped function's signature/docstring via `functools.wraps`, so FastMCP
  schema generation is unaffected (verified against fastmcp 2.14.1).

Nothing here installs anything. The guard is the deliverable; the install is a
separate, optional remediation that the payload tells the operator how to run.
"""

from __future__ import annotations

import functools
import importlib.util
import json
import os
import sys
import time
import urllib.request
from typing import Any, Callable, Dict, Optional


# --------------------------------------------------------------------------- #
# Cheap, cached dependency probes
# --------------------------------------------------------------------------- #
# Whether a module is importable does not change within a process lifetime, so
# we cache it. Qdrant reachability CAN change, so it gets a short TTL.

_import_cache: Dict[str, bool] = {}
_qdrant_cache: Dict[str, Any] = {"ts": 0.0, "reachable": None, "reason": ""}
_QDRANT_TTL_SECONDS = 30.0


def _module_importable(name: str) -> bool:
    """True if `name` can be imported, without actually importing it."""
    if name in _import_cache:
        return _import_cache[name]
    try:
        ok = importlib.util.find_spec(name) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        ok = False
    _import_cache[name] = ok
    return ok


def _qdrant_url() -> str:
    host = os.environ.get("QDRANT_HOST", "127.0.0.1")
    port = os.environ.get("QDRANT_PORT", "6333")
    # Allow a fully-qualified override.
    url = os.environ.get("QDRANT_URL")
    if url:
        return url.rstrip("/")
    return f"http://{host}:{port}"


def _qdrant_reachable(timeout: float = 1.5, use_cache: bool = True) -> Dict[str, Any]:
    """Probe the Qdrant readiness endpoint. Returns {reachable, reason, url}.

    Result is cached for `_QDRANT_TTL_SECONDS` to keep per-call latency low on
    healthy systems.
    """
    now = time.time()
    if (
        use_cache
        and _qdrant_cache["reachable"] is not None
        and (now - _qdrant_cache["ts"]) < _QDRANT_TTL_SECONDS
    ):
        return {
            "reachable": _qdrant_cache["reachable"],
            "reason": _qdrant_cache["reason"],
            "url": _qdrant_url(),
        }

    url = _qdrant_url()
    reachable = False
    reason = ""
    try:
        with urllib.request.urlopen(f"{url}/readyz", timeout=timeout) as resp:
            reachable = 200 <= resp.status < 300
            reason = (
                "ok" if reachable else f"Qdrant /readyz returned HTTP {resp.status}"
            )
    except Exception as e:  # noqa: BLE001 - any failure means "not reachable"
        reachable = False
        reason = f"{type(e).__name__}: {e}"

    _qdrant_cache.update({"ts": now, "reachable": reachable, "reason": reason})
    return {"reachable": reachable, "reason": reason, "url": url}


# --------------------------------------------------------------------------- #
# Structured payload builders
# --------------------------------------------------------------------------- #


def _remediation_for(missing: Optional[list]) -> str:
    interp = sys.executable
    if not missing:
        return f"Backend reachable from interpreter {interp}."
    pkgs = " ".join(sorted(set(missing)))
    return (
        f"pip install {pkgs} into the MCP interpreter: '{interp} -m pip install {pkgs}'"
    )


def degraded_payload(
    reason: str,
    status: str = "unavailable",
    missing: Optional[list] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Canonical structured-degradation dict.

    status: "unavailable" (backend fundamentally missing) or
            "degraded" (partial: e.g. deps present but Qdrant unreachable).
    """
    payload: Dict[str, Any] = {
        "status": status,
        "reason": reason,
        "remediation": _remediation_for(missing),
        "interpreter": sys.executable,
    }
    if missing:
        payload["missing"] = sorted(set(missing))
    if extra:
        payload.update(extra)
    return payload


# --------------------------------------------------------------------------- #
# Full health report
# --------------------------------------------------------------------------- #


def check_vector_semantic_health() -> Dict[str, Any]:
    """Structured health of every vector/semantic dependency, computed against
    the interpreter actually running this process.

    Returns a dict with per-dependency {available, reason} blocks plus an
    overall `status` ("available" | "degraded" | "unavailable").
    """
    qc_ok = _module_importable("qdrant_client")
    st_ok = _module_importable("sentence_transformers")

    # Embedder availability is the *real* test for sentence-transformers: it
    # imports + loads the model. Keep it best-effort and never raise.
    embedder_ok = False
    embedder_reason = "not checked"
    if st_ok:
        try:
            from atommem import embedder  # local, lazy, fail-soft

            embedder_ok = bool(embedder.available())
            embedder_reason = (
                f"all-MiniLM-L6-v2 loaded (dim={embedder.dimension()})"
                if embedder_ok
                else "sentence_transformers present but model failed to load"
            )
        except Exception as e:  # noqa: BLE001
            embedder_ok = False
            embedder_reason = f"{type(e).__name__}: {e}"
    else:
        embedder_reason = "sentence_transformers not importable"

    qdrant = (
        _qdrant_reachable(use_cache=False)
        if qc_ok
        else {
            "reachable": False,
            "reason": "qdrant_client not importable",
            "url": _qdrant_url(),
        }
    )

    checks = {
        "qdrant_client_importable": {
            "available": qc_ok,
            "reason": "ok" if qc_ok else "module not installed in this interpreter",
        },
        "qdrant_reachable": {
            "available": bool(qdrant["reachable"]),
            "reason": qdrant["reason"],
            "url": qdrant["url"],
        },
        "sentence_transformers_importable": {
            "available": st_ok,
            "reason": "ok" if st_ok else "module not installed in this interpreter",
        },
        "embedder_available": {
            "available": embedder_ok,
            "reason": embedder_reason,
        },
    }

    # Vector recall needs qdrant_client AND a reachable server.
    vector_ok = qc_ok and bool(qdrant["reachable"])
    # Semantic cache needs sentence_transformers (its model is local).
    cache_ok = st_ok

    if vector_ok and cache_ok:
        status = "available"
    elif not qc_ok and not st_ok:
        status = "unavailable"
    else:
        status = "degraded"

    missing = []
    if not qc_ok:
        missing.append("qdrant-client")
    if not st_ok:
        missing.append("sentence-transformers")

    return {
        "status": status,
        "interpreter": sys.executable,
        "python_version": sys.version.split()[0],
        "checks": checks,
        "vector_recall_ready": vector_ok,
        "semantic_cache_ready": cache_ok,
        "missing_packages": sorted(set(missing)),
        "remediation": _remediation_for([m for m in missing] or None),
    }


# --------------------------------------------------------------------------- #
# Pre-check callables (returned None == proceed; dict == short-circuit degraded)
# --------------------------------------------------------------------------- #


def _vector_precheck() -> Optional[Dict[str, Any]]:
    """Guard for tools that need qdrant_client + a reachable Qdrant."""
    if not _module_importable("qdrant_client"):
        return degraded_payload(
            "qdrant_client is not installed in the running interpreter; "
            "vector recall is unavailable.",
            status="unavailable",
            missing=["qdrant-client"],
        )
    probe = _qdrant_reachable()
    if not probe["reachable"]:
        return degraded_payload(
            f"qdrant_client present but Qdrant is not reachable at "
            f"{probe['url']} ({probe['reason']}).",
            status="degraded",
            extra={"qdrant_url": probe["url"]},
        )
    return None


def _semantic_cache_precheck() -> Optional[Dict[str, Any]]:
    """Guard for tools that need sentence_transformers (cache embedding model)."""
    if not _module_importable("sentence_transformers"):
        return degraded_payload(
            "sentence_transformers is not installed in the running interpreter; "
            "the semantic cache embedding model cannot load.",
            status="unavailable",
            missing=["sentence-transformers"],
        )
    return None


# --------------------------------------------------------------------------- #
# GuardedToolApp proxy
# --------------------------------------------------------------------------- #


class GuardedToolApp:
    """Proxy around a FastMCP `app` that wraps each registered tool with a
    cheap availability pre-check.

    Usage:
        register_semantic_vector_tools(GuardedToolApp(app, _vector_precheck))

    The wrapped tool:
      - runs `precheck()`; if it returns a dict, returns that dict (the
        structured-degradation payload) WITHOUT executing the real tool;
      - otherwise delegates to the real tool, catching any unexpected
        ModuleNotFoundError/ImportError and converting it to a structured dict
        rather than letting a raw traceback reach the caller.

    Signature/docstring are preserved via functools.wraps so FastMCP schema
    generation sees the original function (verified against fastmcp 2.14.1).
    Any attribute other than `.tool` is forwarded to the wrapped app.
    """

    def __init__(self, app: Any, precheck: Callable[[], Optional[Dict[str, Any]]]):
        self._app = app
        self._precheck = precheck

    @staticmethod
    def _conform(payload: Dict[str, Any], fn: Callable) -> Any:
        """Return the degradation payload in the shape the tool promises.

        functools.wraps keeps `fn`'s return annotation, so FastMCP validates the
        wrapper's output against the ORIGINAL signature. A tool annotated
        `-> str` that receives a dict fails output validation, and the caller
        gets an opaque ToolError instead of the remediation text. Serializing
        here keeps the guidance readable.
        """
        annotation = getattr(fn, "__annotations__", {}).get("return")
        if annotation is str or annotation == "str":
            return json.dumps(payload)
        return payload

    def tool(self, *t_args, **t_kwargs):
        real_decorator = self._app.tool(*t_args, **t_kwargs)
        precheck = self._precheck
        conform = self._conform

        def decorator(fn):
            import inspect

            if inspect.iscoroutinefunction(fn):

                @functools.wraps(fn)
                async def wrapped(*args, **kwargs):
                    blocked = precheck()
                    if blocked is not None:
                        return conform(blocked, fn)
                    try:
                        return await fn(*args, **kwargs)
                    except (ModuleNotFoundError, ImportError) as e:
                        return conform(
                            degraded_payload(
                                f"vector/semantic dependency missing at call time: "
                                f"{type(e).__name__}: {e}",
                                status="unavailable",
                            ),
                            fn,
                        )

            else:

                @functools.wraps(fn)
                def wrapped(*args, **kwargs):
                    blocked = precheck()
                    if blocked is not None:
                        return conform(blocked, fn)
                    try:
                        return fn(*args, **kwargs)
                    except (ModuleNotFoundError, ImportError) as e:
                        return conform(
                            degraded_payload(
                                f"vector/semantic dependency missing at call time: "
                                f"{type(e).__name__}: {e}",
                                status="unavailable",
                            ),
                            fn,
                        )

            return real_decorator(wrapped)

        return decorator

    def __getattr__(self, name: str) -> Any:
        # Forward everything except `.tool` (handled above) to the real app.
        return getattr(self._app, name)


# --------------------------------------------------------------------------- #
# Health tool registration
# --------------------------------------------------------------------------- #


def check_recall_path() -> Dict[str, Any]:
    """Probe the ACTUAL recall path (Phase 0 spine repair, 2026-07-02).

    The dependency checks above validate atommem's MiniLM (used by the
    semantic cache) — but `semantic_recall` embeds via ollama and queries the
    collection_for(DEFAULT_MODEL) Qdrant collection. The 2026-07-01 audit found
    this split-brain: health said "available" while recall coverage was 3.7%.
    This block tests the real path: embed a probe with the real model, query
    the real collection, and report indexing coverage from the sweeper's
    bookkeeping. A canary round-trip (upsert + query + delete of a reserved
    point) proves write->search works end to end.
    """
    report: Dict[str, Any] = {}
    try:
        from local_semantic_recall import DEFAULT_MODEL, QDRANT, collection_for, embed

        report["model"] = DEFAULT_MODEL
        report["collection"] = collection_for(DEFAULT_MODEL)
        t0 = time.time()
        probe_vec = embed(["recall path health probe"], DEFAULT_MODEL)[0]
        report["embed_ok"] = True
        report["embed_dim"] = len(probe_vec)
        report["embed_latency_ms"] = round((time.time() - t0) * 1000)
    except Exception as e:  # noqa: BLE001
        report["embed_ok"] = False
        report["reason"] = f"{type(e).__name__}: {e}"
        return report

    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import PointStruct

        client = QdrantClient(url=QDRANT)
        coll = report["collection"]
        report["qdrant_points"] = client.count(coll).count

        # canary round-trip on a reserved id far above any entity id
        canary_id = 999_999_999
        client.upsert(
            coll,
            points=[
                PointStruct(
                    id=canary_id,
                    vector=probe_vec,
                    payload={"name": "__health_canary__", "entity_type": "health"},
                )
            ],
        )
        hits = client.query_points(coll, query=probe_vec, limit=1).points
        report["canary_roundtrip_ok"] = bool(hits) and hits[0].id == canary_id
        client.delete(coll, points_selector=[canary_id])
    except Exception as e:  # noqa: BLE001
        report["canary_roundtrip_ok"] = False
        report["reason"] = f"{type(e).__name__}: {e}"
        return report

    try:
        from vector_write_indexer import coverage, pending_ids, sweeper_alive

        report["coverage"] = coverage()
        report["pending_index"] = len(pending_ids(limit=10_000))
        # Report the MECHANISM, not just the symptom. A backlog is ambiguous on
        # its own: writes outpacing a healthy sweeper looks identical to a dead
        # sweeper, and only the second is a fault. Without this, diagnosing it
        # means reading the source to find that the drainer is an in-process
        # thread rather than a scheduled job.
        report["sweeper_alive"] = sweeper_alive()
    except Exception as e:  # noqa: BLE001
        report["coverage_error"] = f"{type(e).__name__}: {e}"

    report["ok"] = bool(
        report.get("embed_ok")
        and report.get("canary_roundtrip_ok")
        # A pending backlog with no sweeper never drains on its own.
        and not (report.get("pending_index") and report.get("sweeper_alive") is False)
    )
    return report


def register_vector_health_tools(app: Any) -> None:
    """Register the `vector_semantic_health` diagnostic MCP tool."""

    @app.tool()
    async def vector_semantic_health() -> Dict[str, Any]:
        """Report honest availability of the vector/semantic backend.

        Two layers: (1) dependency checks against the interpreter actually
        running this MCP server (qdrant_client, Qdrant reachability,
        sentence_transformers, cache embedder); (2) the REAL recall path —
        embeds a probe with the same ollama model semantic_recall uses,
        runs a canary upsert/query/delete round-trip against the same
        collection, and reports write-path indexing coverage. Overall
        status is degraded unless the recall path round-trip passes.
        """
        health = check_vector_semantic_health()
        recall = check_recall_path()
        health["recall_path"] = recall
        if not recall.get("ok") and health.get("status") == "available":
            health["status"] = "degraded"
            health["degraded_reason"] = (
                "dependency checks pass but the live recall path failed: "
                + recall.get("reason", "see recall_path block")
            )
        return health
