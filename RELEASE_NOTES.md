# Release Notes — standalone v1

This branch packages enhanced-memory-mcp as a repository you can clone and
install on a machine that is not the one it grew up on.

The starting point was two diverged trees: a working local deployment and the
published `main`. This release takes the **local deployment** as the source of
truth, because that is the line that is verifiably running, and removes the
parts of it that only make sense on the original host.

Every number below was measured on this branch. Commands are included so you can
re-run them rather than trust them. What was *not* verified is listed at the end.

---

## Measured tool surface

| Dependency set | Tools registered |
|---|---|
| `requirements.txt` only | **188** |
| plus `requirements-optional.txt` | **206** |

Front door (eagerly loaded, flagged `anthropic/alwaysLoad`): `search_nodes`,
`semantic_recall`, `create_entities`, `get_memory_status`, `execute_code`.
The remaining tools are registered but deferred.

Source comments in `server.py` previously described the surface as "~142 tools".
That figure was the historical decision point for the front-door design, not a
current count; it is now labelled as such. **Re-measure rather than quoting any
of these numbers** — the total moves with what is importable.

---

## Honesty fixes

### 1. A dead daemon no longer looks like an empty memory system

Previously, when the memory-db socket was unreachable, the front-door tools
returned an error field *alongside* zero-filled data:

```json
{"error": "...", "entities": {"total": 0}, "compression": {"ratio": "N/A"}}
```

A caller rendering `entities.total` showed a healthy, empty memory system. The
failure and the empty state serialized to nearly the same thing.

Failures now carry `status`, `error`, and a `daemon` field naming the socket,
and **no data-shaped keys at all** — no counts, no empty result lists, no zeroed
totals. Measured with a socket path that does not exist:

```
$ MEMORY_DB_SOCKET_PATH=/tmp/does-not-exist.sock <server> get_memory_status
{
  "status": "failed",
  "error": "Memory-DB service error: [Errno 2] No such file or directory",
  "daemon": "UNREACHABLE (/tmp/does-not-exist.sock)"
}
```

`search_nodes` adds `query`, `create_entities` adds `requested` (the number of
entities submitted — a request echo, not a claim about what happened). The same
treatment was applied to `semantic_recall` (whose payload names the failing
backend instead of the socket) and to the daemon's own status error path in
`memory_db_service.py`.

This is a real but bounded improvement: a caller that writes
`result.get("results", [])` still gets an empty list. What changed is that a
caller reading the response as data now fails or sees `status: failed`, instead
of quietly rendering zeros.

### 2. An entire tool group was silently skipped at every startup

`agi_tools.py` passed `outputSchema=` to `@app.tool()`. FastMCP's parameter is
`output_schema`, so registration raised and the AGI Memory Phase 1 group was
skipped on **every** boot, logging a warning nobody read:

```
AGI Memory Phase 1 integration skipped: FastMCP.tool() got an unexpected
keyword argument 'outputSchema'. Did you mean 'output_schema'?
```

Fixed. The group registers and the surface goes from 172 to 188 tools — the 16
tools defined in that file: `get_agent_identity`, `update_agent_skills`,
`add_agent_belief`, `update_agent_personality`, `set_agent_preference`,
`start_session`, `end_session`, `get_session_context`, `get_recent_sessions`,
`get_session_chain`, `record_action_outcome`, `get_similar_actions`,
`get_action_success_rate`, `get_learnings_for_action`, `should_retry_action`,
`get_action_statistics`.

### 3. The database split-brain is now announced at startup

Setting `ENHANCED_MEMORY_DB_PATH` on the server alone does not move the daemon.
The server would create and report an empty database at the new path while every
front-door tool kept serving the daemon's database. Both halves looked healthy,
and the reasonable conclusion was that the memories had been lost.

At startup the server now asks the daemon which database it serves (the daemon
already reported `database_path`; no new operation was needed) and prints a
banner when they disagree:

```
!!  DATABASE SPLIT-BRAIN: server and daemon disagree          !!
  server resolved : .../dbB/memory.db
  daemon serves   : .../dbA/memory.db   <-- the tools read and write HERE
  socket          : /tmp/em-release-test.sock
```

When the paths match it logs `DB-path agreement OK`, and when the daemon is
unreachable it says the check did not run rather than implying agreement. All
three branches were exercised.

The comparison resolves symlinks on both sides, which matters more than it
sounds: a common setup keeps `~/.claude/enhanced_memories` as a symlink to
storage elsewhere, so the daemon and the server can name the same database with
two different strings. A string comparison would have fired this banner on
every boot of a correctly-configured system, and a warning that cries wolf every
time is worse than no warning. Verified both ways — the same database reached
via a symlink reports agreement, and a genuinely different directory still
raises the banner.

### 4. A fresh install stored nothing and reported success

**This one only bit new installations, which is exactly who this release is
for.** It was invisible on any database that had already been migrated.

`memory_db_service.init_database()` created the `entities` table without the
`modality` and `raw_data_pointer` columns, but `create_entities` INSERTs nine
columns including both. On a database the daemon created itself, every write
therefore failed with `table entities has no column named modality`.

Two things then combined to hide it:

1. `create_entities` initialised its result to `{"success": True, ...}` and
   never revised it. The per-entity `except` incremented `failed` and left
   `success` alone, so the response was `{"success": true, "created": 0,
   "failed": 1}` — and `server.py` branches on `response.get("success")`, so the
   MCP caller was told the write worked. Measured before the fix, on a fresh
   database: `success: True, created: 0, failed: 1`, with 0 rows in the table.
2. The columns were only added by `migrations/005_omnimem_mau_fields.sql`, which
   ran from `server.py` at startup. So whether a new user's memories were stored
   at all depended on whether a server had ever started against that file —
   and daemon-first is both the documented and the container start order.

Fixed in three places:

- The `CREATE TABLE` now declares both columns.
- `init_database()` also runs an additive `PRAGMA table_info` / `ALTER TABLE`
  guard, so databases created before this change converge on the same schema no
  matter which process starts first. Verified against a database built with the
  old schema: both columns were added, the pre-existing row survived, and writes
  then succeeded.
- `success` is now computed after the loop as `failed == 0`, with an `error`
  summary and a per-entity `errors` list. A partial batch reports what did land
  (`stored`) alongside the failure, because that is a measured fact rather than
  a zero-fill.

Measured after the fix, through the MCP stdio path, on a database no server had
ever touched: create then search round-tripped, and a deliberately invalid
entity returned

```json
{"status": "failed", "error": "1 of 1 entities failed to store",
 "daemon": "ERROR (...)", "requested": 1, "failed": 1,
 "errors": [{"name": null, "error": "NOT NULL constraint failed: entities.name"}]}
```

**The production tree carries the same latent code.** It is masked there because
those databases were migrated long ago. Backporting this to the running system
is a separate follow-up, not part of this branch.

### 5. The code-execution memory API returned coroutines

`api/memory.py` is what agent code calls inside the `execute_code` sandbox. All
three of its memory functions called the **async** `MemoryClient` methods and
returned the coroutine object. RestrictedPython has no event loop and cannot
await, so every memory call from inside the sandbox failed with
`'coroutine' object has no attribute 'get'` and a
`coroutine ... was never awaited` warning. The entire memory API of the
code-execution feature was non-functional.

Fixed to call the `_sync` variants, which already existed and were unused.

`search_nodes` needed a second fix in the same place: it treated the daemon's
response as a bare list and filtered it directly, so even after awaiting
correctly, `entity_type` or `min_confidence` filtering would have iterated the
envelope's KEYS and raised. It now unwraps `results` and raises on a failed
search rather than filtering an error dict.

Measured after the fix, against a live daemon:
`create_entities` -> `created: 1`; `search_nodes` -> a `list` containing the
written entity; `get_status` -> a `dict`; no coroutines returned.

### 6. The code-execution API ignored the configured socket

`api/memory.py` constructed `MemoryClient()` with no argument, taking the
built-in `/tmp/memory-db.sock` regardless of `MEMORY_DB_SOCKET_PATH`. Only
`server.py` passed the environment value through, so the sandbox always talked
to the default socket even when the daemon was configured elsewhere.

Fixed in `MemoryClient.__init__` rather than at each call site, so every
consumer honours the environment. Resolution order is now explicit argument,
then `MEMORY_DB_SOCKET_PATH`, then the default — verified all three, including
that an unset environment still yields the original literal default.

### 7. update_entity wrote to the operator's real database

`api/memory.py`'s `update_entity` — reachable from agent code inside
`execute_code` — carried `~/.claude/enhanced_memories/memory.db` inline and opened
SQLite **directly**, bypassing the daemon socket entirely. It therefore ignored
`ENHANCED_MEMORY_DB_PATH`, `MEMORY_DB_PATH`, `ENHANCED_MEMORY_DIR` and
`MEMORY_DIR`, and because it does not use the socket, configuring the socket
did not contain it either. A second instance, a test run, or a container with a
mounted database would have written to the operator's real store.

The resolution order now lives in one place, `memory_paths.py`, imported by both
`server.py` and `api/memory.py` so the direct-SQLite path cannot drift from what
the server uses. Verified: with `ENHANCED_MEMORY_DB_PATH` set, an
`update_entity` call wrote to the configured database and the default database
had zero matching rows.

### 8. A degradation message that never reached the caller

`GuardedToolApp` (`vector_health.py`) returns a structured "this backend is
unavailable, here is the pip command" payload. Because `functools.wraps`
preserves the wrapped tool's `-> str` annotation, FastMCP validated that dict
against a string schema and raised, so `semantic_recall` answered with an opaque
`ToolError` instead of the remediation text. The guard now serializes its
payload to match the declared return type.

### 9. Importing the server created the operator's memory directory

`server.py` ran `MEMORY_DIR.mkdir(parents=True, exist_ok=True)` at **module
scope**, so merely `import server` reached into the configured memory location
and created it. `exist_ok=True` made it silent on a machine where the directory
already existed — and on one where it did not, an import would have created it.
Anything that imported the module for introspection, tooling or testing had this
side effect before a single function was called.

Directory creation moved into `init_database()`, which the `__main__` block
calls before anything touches the database. Verified both directions: `import
server` now performs zero `mkdir` calls, and booting the server against a
nested path that does not exist still creates it.

Found by the test suite author while auditing why their isolation guards could
be bypassed — a connection-level guard is structurally blind to `mkdir`, so this
route was invisible to it.

### 10. Tool discovery advertised five MCP servers that do not exist

`sandbox/tool_discovery.py` held a fixed in-source `MCP_TOOL_REGISTRY` naming
`voice-mode`, `arduino-surface`, `agent-runtime`, `safla-enhanced` and
`sequential-thinking` alongside `enhanced-memory`. `init_tool_registry()` wrote
those to `tool_registry/`, and `list_servers()` returned the directory names
with no availability check. Agent code inside `execute_code` was therefore told
it could call tools from five servers a standalone install does not have.

The `enhanced-memory` entry was wrong too, in the opposite direction: it
declared 6 tools while the server registers 188.

The registry now defaults to **empty** — this module cannot know what an
installation runs, so it claims nothing. Servers are declared by pointing
`MEMORY_TOOL_REGISTRY_FILE` at a JSON file of the same shape, or by writing the
`tool_registry/` tree directly. The six committed directories were removed;
deleting them alone would not have worked, because `init_tool_registry()`
regenerated them from the dict.

`tool_registry/` is now in `.gitignore`. It is per-install state written at
runtime, and committing it is precisely how this repository came to make claims
on behalf of every future clone — `list_servers()` reports whatever directory
names it finds there. Verified that a regenerated tree containing a real
declaration is ignored rather than picked up by `git add`.

Measured on a clean tree with no declaration:

```
list_servers()                            -> []
list_tools("voice-mode")                  -> []
get_tool_schema("voice-mode", "converse") -> None
```

and with `MEMORY_TOOL_REGISTRY_FILE` set to a one-server file,
`list_servers() -> ['my-real-server']` with its tool and schema retrievable. A
malformed or missing declaration file logs the specific reason and declares
nothing, rather than emptying itself silently.

### Smaller items

- `semantic_vector_tools.py` had `except Exception: pass` around retrieval
  telemetry. It now logs the failure.
- Three scope-validation errors in `semantic_recall` returned `count: 0,
  results: []` next to their error. Same treatment as above.
- `.gitignore` ended with a truncated pattern (`${AGENTIC_SYSTEM_PATH:-*`) that
  matched nothing, which is why directories literally named `$HOME` and `~`
  — carrying a stray `feedback.db` — had been committed. Fixed and the junk
  directories are excluded.

---

## Portability

Machine-specific absolute paths and LAN addresses were replaced with
environment variables and repository-relative defaults in runtime code:

- `server.py` — `_get_storage_base()` probed four fixed volume paths; now
  reads `AGENTIC_SYSTEM_PATH` or resolves relative to the repository.
- `graphrag_tools.py`, `visual_memory_tools.py` — same treatment.
- `surprise_consolidation.py` — metrics file now defaults beside the memory
  database (`SURPRISE_METRICS_FILE` overrides).
- `nmf_config.yaml` — sqlite/files/log paths now under
  `~/.claude/enhanced_memories/`; the embedding `base_url` is localhost rather
  than a LAN IP.
- `neural_memory_fabric.py` — the redirect that rescued those paths was guarded
  by `if not Path("/mnt").exists()`. `/mnt` exists on most Linux systems even
  when the configured directory under it does not, so the redirect would not
  have fired for anyone else and the sqlite backend would have failed to open.
  It now tests the configured directory itself.
- `start_server.sh` — venv path is `MEMORY_VENV_PATH` or beside the script, and
  it fails with a clear message instead of sourcing a missing activate script.
- `atommem/llm_cli.py` — dropped a provider pointing at an absolute path on the
  original host. Remaining providers are PATH lookups; `MEMORY_LLM_CLI` adds
  your own.
- `local_semantic_recall.py` — a help string told you to ssh to a specific LAN
  host to pull models.

Remaining matches are confined to test files (see Gaps).

---

## Removed from the release

### Present in main's history, not in the verified release line

These 87 files exist on `main` but not in the working deployment. They are not
part of the line this release verifies, so they are removed here. Nothing in the
shipped code imports them; the server boots and round-trips without them. They
remain in git history on `main`.

**Feature modules (29):** `activation_field_tools.py`, `adversarial_learning.py`,
`adversarial_validation_test.py`, `agi/activation_field.py`,
`agi/procedural_evolution.py`, `agi/routing_learning.py`,
`anti_hallucination.py`, `audit_entities.py`, `causal_inference.py`,
`continuous_learning.py`, `entropy_scoring.py`, `fact_integration.py`,
`fact_validator.py`, `filesystem_tools.py`, `holdout_test_framework.py`,
`http_api_server.py`, `manifold_working_memory.py`,
`manifold_working_memory_tools.py`, `procedural_evolution_tools.py`,
`reasoning_bank.py`, `routing_learning_tools.py`, `safla_remote_integration.py`,
`start_http_server.py`, `strange_loops.py`, `tool_usage_logger.py`,
`trajectory_compression.py`, `triple_signal_search.py`,
`triple_signal_tools.py`, `unified_search_api.py`

**Their tests (18):** `test_activation_field.py`, `test_anti_hallucination.py`,
`test_causal_inference.py`, `test_continuous_learning.py`,
`test_entropy_integration.py`, `test_fact_integration.py`,
`test_graphrag_integration.py`, `test_letta_integration.py`,
`test_manifold_working_memory.py`, `test_memory_routing.py`,
`test_performance_comparison.py`, `test_procedural_evolution.py`,
`test_reasoning_bank.py`, `test_routing_learning.py`,
`test_semantic_file_search.py`, `test_strange_loops.py`,
`test_trajectory_compression.py`, `test_triple_signal_search.py`

**Packages (38 files):** `nmf/`, `router/`, `server/`

Note on `server/`: a package directory named `server/` sitting beside
`server.py` shadows the module on import, so shipping both is a trap
independent of whether the package is wanted. In the working deployment these
three directories contained only stale bytecode — the sources had already been
removed.

**Result files (2):** `adversarial_test_results.json`, `holdout_test_results.json`

### Legacy server variants

`server_fastmcp.py`, `server_fixed.py`, `server_otel.py`, `server_simple.py`,
`server_code_exec.py`, `server_git_enhanced.py`, `server_resources.py`,
`server_legacy_20251108_090108.py`, `server_original_backup.py`,
`server.py.backup`, `server.py.backup-20251019-090456`,
`server.py.preMemFix-20260601.bak`.

Also `server_wrapper.py`, which was not on the original exclusion list: it does
`from server import main`, and `server.py` defines no `main`, so it raises
ImportError if used. Nothing referenced it.

Four test files that existed only to exercise those variants went with them:
`test_fastmcp_server.py` and `test_resources.py`, `simple_mcp_test.py`,
`validate_mcp_resources.py`. Each imports directly from `server_fastmcp` or
`server_resources`, so with those modules gone they tested nothing that ships.

`memory_db_service_v2.py` looks like another variant and is NOT one: `server.py`
and `memory_db_service.py` both import it. It stays.

### Development scratch

`debug_server.py`, `debug_comm_detailed.py`, `debug_comm_fixed.py`,
`debug_claude_desktop_integration.py`, `direct_test.py`, `echo_test.py`,
`echo_simple.py`, `simple_test.py`, `simple_comm_test.py` — probe scripts
carrying paths from the original machine inline, referenced only by each other.

### Host-specific operations

`agi_cluster_bridge.py`, `ops/apply_quarantine.sh`, `ops/scheduled_review.sh`,
`ops/com.phoenix.memory-compounding-review.plist` — launchd jobs and one-shot
maintenance tied to the original deployment. The importable parts of `ops/`
(`retrieval_log.py`, `memory_injection_guard*.py`, `memory_compounding_report.py`,
`memory_quarantine.py`) are runtime dependencies and were kept.

`install.sh` was removed: it printed instructions and installed nothing.

Databases, logs, virtualenvs, bytecode and the accidental `$HOME` / `~`
directories were excluded.

---

## Python version

Verified on **Python 3.11.11**. The original deployment runs 3.14; 3.11 is the
floor this release was tested against, and the full optional stack — including
torch 2.13.0 behind sentence-transformers — resolved on it without pins beyond
those in the requirements files.

`mcp` is pinned to `1.25.0` deliberately. `fastmcp` will resolve `mcp` 2.x,
which removes the low-level decorators and the `_call_tool` hook `server.py`
patches for large results. Raising that pin means re-testing that patch.

---

## Test suite

Two gates ship, and they answer different questions.

```bash
python comprehensive_test.py    # does the installed server work on this machine?
python -m pytest tests/         # do the units behave as their contracts say?
```

Measured on this branch, 2026-08-14:

| Gate | Result |
|---|---|
| `python comprehensive_test.py` | **exit 0**, 0 failed (106 assertions in the default sandbox on this machine) |
| `python -m pytest tests/ -q` | **138 passed, 0 failed, 0 skipped** |

**Judge `comprehensive_test.py` by its exit code, not by its assertion count.**
The count is mode-dependent by design: a default run makes four checks that a
run against a configured deployment cannot make, because they describe a sandbox
that run did not generate. Measured on the same machine, same commit:

| Mode | Assertions | Exit |
|---|---|---|
| isolated sandbox (default) | 106 | 0 |
| operator-directed (env overrides set) | 102 | 0 |

Both are correct. The run prints its mode and names the inapplicable checks
before the summary, so the two numbers are never silently comparable. A count
difference between two green runs was in fact briefly mistaken for an
environment-dependent tool surface during this work; it was the mode.

### comprehensive_test.py was rewritten, not repaired

The previous version carried a docstring declaring itself OBSOLETE and
instructing readers not to use its exit code. It measured 42 passed / 17 failed
because it asserted a tool surface that no longer exists (`read_graph`,
`create_relations`), response keys that were renamed (`success`,
`entities_created` for what is now `created` / `failed` / `results`), and
`entity_type` where search results carry `entityType`.

It also was not isolated. Line 646 read
`Path.home()/".claude"/"enhanced_memories"/"memory.db"` directly, so its
database-integrity test inspected the operator's live memory store and printed
its 11,950 entities during an otherwise "isolated" run.

The replacement:

- **Builds its own sandbox.** A temporary database and socket, created fresh
  per run. It never reads `~/.claude/enhanced_memories/memory.db` and never
  binds `/tmp/memory-db.sock`. The isolation is asserted rather than intended:
  the run prints its entity count, and a clean run reads in single digits.
- **Honours `ENHANCED_MEMORY_DB_PATH`, `MEMORY_DB_PATH`, `ENHANCED_MEMORY_DIR`,
  `MEMORY_DIR` and `MEMORY_DB_SOCKET_PATH`** when they are set, so you can point
  it at a real deployment. Because it writes entities, it refuses a database
  that already holds any, and refuses the default production path outright.
  `ENHANCED_MEMORY_TEST_FORCE=1` overrides the refusal.
- **Keeps socket paths under the AF_UNIX 104-byte limit** and asserts it, rather
  than letting `bind()` fail with a truncated path.
- **Asserts the current contracts**, read out of `server.py` and
  `memory_db_service.py` rather than remembered.

### A regression test for the fresh-install schema bug

The daemon's `CREATE TABLE entities` omitted the `modality` and
`raw_data_pointer` columns that its own `INSERT` writes. `server.py` runs a
migration at startup that adds them, so the bug was invisible in any database a
server had ever opened. Daemon-first on a fresh database is the ordering that
exposes it, and it is the documented start order.

TEST 2 covers it, and the test was verified by falsification rather than by
assumption:

| Daemon code | Result |
|---|---|
| fixed (shipping) | 106/106, exit 0 |
| schema fix reverted, both layers | **5 assertions fail, exit 1**, `table entities has no column named modality` |
| `success` flag fix reverted alone | **7 assertions fail, exit 1** |

The middle row needed both layers reverted. Reverting only the `CREATE TABLE`
columns still passes, because the fix also carries an `ALTER TABLE` self-heal
that repairs an older database. A regression test written against one layer
would have been a test that cannot fail.

### Tests deleted, and what their absence means

Twelve tests were removed. None was skipped: a skipped test is a deleted test
that still prints a line.

- **6 in `test_provenance.py`** covering `_detect_citation_cycle`,
  `_calculate_source_quality_penalty`, and a `ValueError` on a citation-gaming
  attempt. **None of those methods exists on `ProvenanceManager`**, here or in
  the deployment this release was cut from. They were never implemented.
  Read that as a capability statement: **L-Scores are descriptive, not
  adversarial.** A caller that cites in a loop, or cites a low-quality source,
  is not penalised for it.
- **3 in `test_server.py`** (`TestToolUsageLogging`) importing
  `_set_tool_usage_callback` and `_log_tool_usage`. Neither symbol, nor the
  `tool_usage_logger.py` module behind them, exists in this tree or in
  production.
- **3 in `test_server.py`** (`TestEntropyFallback`) referencing
  `server.ENTROPY_SCORING_AVAILABLE` and `server.combine_scores`. Neither
  exists. These were doubly inert: each body was wrapped in
  `if not server.ENTROPY_SCORING_AVAILABLE:`, so even had the attribute
  existed and been true, the test would have asserted nothing and passed.

### Tests that were passing while asserting nothing

Three were fixed rather than deleted, because the behaviour they claimed to
cover does ship and does work:

- **`TestStoragePath`** had two tests that mocked `platform.system()`, opened a
  `with` block, and ended. No assertion, no possible failure, and they described
  a platform-detection branch `_get_storage_base` no longer has. Replaced with
  three tests of what it actually does: the `AGENTIC_SYSTEM_PATH` override, `~`
  and `$VAR` expansion, and the repo-relative fallback.
- **`test_timeout`** ran `import time; time.sleep(10)` inside a sandbox that
  blocks `__import__`. It failed in under a millisecond with
  `ImportError: __import__ not found`, and the assertion
  `"time" in result.error.lower()` accepted that string. Timeout enforcement
  was never exercised; the test would have passed with the feature removed.
  It now runs a busy loop and asserts both the error and the elapsed time.
  Measured: interrupted at 2.01s against a 2s limit.
- **`test_token_savings`** built confidence scores as `i/10` over
  `range(100)`, producing values up to 9.9, so a 0.8 threshold selected 92 of
  100 items while the test asserted 20. The fixture now uses `i/100`, a real
  0-to-1 confidence range, where the threshold selects exactly the 20 intended.

### The unit suite was reading the operator's live memory store

`tests/code_exec/test_integration.py::test_api_access` called `get_status()`
through the shipped code-execution API. `MemoryClient()` built with no argument
falls back to `/tmp/memory-db.sock`, so on any machine with a memory-db daemon
running, that test connected to it. Measured while investigating: it reached a
database of **11,952 entities** at `~/.claude/enhanced_memories/memory.db` and
reported itself as a passing unit test.

`tests/conftest.py` now points every test at a socket path that does not exist,
by default and without opt-in, so an accidental memory call fails with
`FileNotFoundError` instead of silently reaching production. Tests that need a
real round-trip take the `memory_daemon` / `live_memory` fixtures, which start a
daemon against a throwaway database and assert against *that* path:

```python
assert result.result["database_path"] == str(live_memory)
assert result.result["entity_total"] == 0
```

Verified by probe: a test calling the memory API without the fixture is blocked
with `FileNotFoundError: [Errno 2] No such file or directory`, not with a
result.

**The socket guard is not enough on its own**, because around a dozen modules
resolve `~/.claude/enhanced_memories/` themselves and open SQLite directly,
bypassing the socket entirely. `api.memory.update_entity` did exactly that until
2026-08-14, and it is a *write* path. A second, independent guard therefore
wraps `sqlite3.connect` for the whole session and raises on any path under the
real memory directory:

```
RuntimeError: Test opened the real memory database:
  .../enhanced_memories/memory.db
```

Both guards were verified by probe rather than by inspection. The SQLite guard
fires on the plain path form, on the `file:...?mode=ro` URI form, and on sibling
files in the same directory (`nmf.db`), and it resolves symlinks first — which
matters here, because `~/.claude/enhanced_memories` is itself a symlink on the
machine this was measured on, and a guard comparing path *strings* would have
matched nothing while reporting a clean run.

`tests/test_isolation_guards.py` keeps the guards honest: removing the
interception from conftest makes exactly its five production-database tests
fail. It also asserts statically that no module binds `from sqlite3 import
connect`, which would hold its own reference and bypass the guard silently.
Measured 2026-08-14: no module in the tree does.

**A third escape route, which neither guard could see: `mkdir`.** `import server`
runs `MEMORY_DIR.mkdir(parents=True, exist_ok=True)` at module scope. With no
override that resolves to `~/.claude/enhanced_memories`, so importing the module
under test touched the operator's real memory directory — and on a machine where
that directory does not exist yet, a test run would have created it. No
connection is opened, so a connection-level guard is structurally blind to it.

Fixed at the source rather than only guarded: `tests/conftest.py` redirects
`ENHANCED_MEMORY_DIR` / `ENHANCED_MEMORY_DB_PATH` (and the legacy aliases) to a
throwaway directory *at conftest import time*, before any test module is
imported, because the offending call runs at module scope rather than inside a
test. A `mkdir`/`makedirs` guard backs it up. Both layers are independently
falsified: removing the guard fails exactly its two tests, removing the redirect
fails exactly the two that assert `import server` stays out of the real store.

**A measurement that did NOT work, recorded because the method is tempting.**
An earlier draft of this section claimed the real memory directory was
"unchanged — identical entry count and mtime fingerprint before and after
`pytest tests/`". That comparison is worthless on a machine with a live daemon:
a control of four idle windows of the same length showed the directory mtime
changing in 3 of 4, the same rate as four windows containing a full test run.
The daemons write continuously, so the fingerprint measures their activity, not
the suite's. The single "unchanged" reading that produced the original claim was
a quiet window, not evidence. The guards are the evidence; the fingerprint is
not.

`update_entity` now has a test asserting the row lands in the *configured*
database. Simulating the pre-fix behaviour, where that path was baked in, makes
the test fail with the guard's RuntimeError — so the coverage is known to be
capable of failing.

### tests/conftest.py

The suite marks async tests `@pytest.mark.asyncio`, but `pytest-asyncio` is not
a dependency, and without it pytest does not fail those tests, it errors them
with "async def functions are not natively supported" -- 18 failures that read
like broken code and are actually a missing plugin. `tests/conftest.py` runs
coroutine test functions directly, so the suite needs no plugin; if
`pytest-asyncio` is installed it takes precedence.

### Machine-specific paths

Zero machine-specific paths remain under `tests/` or in
`comprehensive_test.py`:

```
$ grep -rn -e "/Users/marc" -e "/Volumes/SSDRAID0" -e "/mnt/" tests/ comprehensive_test.py
(no matches)
```

The absolute paths that remain are synthetic fixture values that are never
touched on disk (`/opt/some/checkout`, `/custom/path.sock`), plus two
`tempfile.mkdtemp(dir="/tmp")` calls, which are deliberate: a longer temporary
directory would push the Unix socket past the AF_UNIX limit.

## Documentation corrections

`CLAUDE.md` described `comprehensive_test.py` as obsolete and told readers not
to rely on it. That was true of the previous version and is no longer true: it
was rewritten for this release against the current MCP contract and is now the
post-install gate. The note now says so, and says to judge it by **exit code**
rather than a pass count.

The reason no figure is quoted: the suite runs in two modes whose totals are not
comparable. With no `ENHANCED_MEMORY_*` / `MEMORY_DB_*` variables set it
generates its own sandbox and runs every check; with them set it runs
operator-directed and skips the checks that describe a sandbox it did not
create. Measured on one machine at one commit: **106 isolated, 102
operator-directed, both exit 0**, a delta of exactly 4.

Worth recording how that was established, because the first explanation was
wrong. Two runs disagreed (97 vs 102) and the difference was attributed to which
optional backends were importable — plausible, and false. It was the run mode.
Had it been backends, coverage would have been silently varying with the
environment, which is a far worse property; the guess was not only incorrect but
would have masked a real defect had one existed. The suite now prints its mode,
names the checks it skipped and why, and states that the count is not comparable
between modes.

Stale `76/76` counts for that suite were removed from
`ENHANCED_MEMORY_SYSTEM_GUIDE.md` (both copies), `MEMVID_CLEANUP_COMPLETE.md`
(both copies) and replaced with the command.

The same figure still appears in `TEST_RESULTS_SUMMARY.md` and
`MEMORY_SYSTEM_VERIFICATION_COMPLETE.md` (each present at the top level and
under `docs/implementation-reports/`). Those are **dated historical reports**,
not statements about this release, and they were deliberately left intact:
rewriting a record of what was measured on a past date would falsify it. Read
everything under `docs/implementation-reports/` and every `*_COMPLETE.md` as a
snapshot of its own moment, not as current status.

## Gaps / not covered

- **Only Python 3.11 was tested.** 3.12 and 3.13 are unknown. The original 3.14
  deployment is unaffected by this branch but was not re-tested against it.
- **Only macOS (arm64) was tested.** No Linux or container run was performed
  here; the from-scratch container install is a separate piece of work.
- **The test suite is not green and was not repaired.** `tests/` currently
  reports 30 failed / 93 passed. Every failure traces to the removed
  main-only modules or to APIs that diverged (for example
  `_set_tool_usage_callback`, which lived in the removed `tool_usage_logger.py`).
  Per-file: `test_memory_client.py` 18 failed / 15 passed, `test_provenance.py`
  6 / 44, `test_server.py` 6 / 34. These three files came from `main` rather
  than the working deployment and were kept, rather than deleted with the other
  84, because the 93 passing tests cover code that survives. Repairing them is
  tracked separately.
- **Test files still contain machine-specific paths.** `test_source_attribution.py`,
  `test_decompression.py`, `test_new_tools_direct.py` name a home directory literally;
  `test_server.py`, `test_minimal.py`, `test_main_server_comm.py`,
  `test_always_include.py`, `test_decompress_formats.py`,
  `test_surprise_consolidation.py`, `test_phase2_features.py` and
  `tests/test_server.py` name volume paths literally. Runtime code is clean; these
  were left for the test repair pass.
- **56 `except: pass` blocks remain** across the tree. One was fixed where it
  sat in a front-door path. The rest were not audited.
- **The governance/ACL tool group cannot activate here.** It imports
  `visibility`, which lives outside this repository, so it logs
  `Governance integration skipped: No module named 'visibility'` on every boot.
  The module was kept because it degrades loudly and correctly, but in a
  standalone install it is inert.
- **Optional backends were verified as importable, not as functional.** Qdrant
  and ollama were never contacted: no Qdrant server was running and no
  embedding model was pulled. What was proven is that the tool groups register
  when the packages are present and degrade with a specific reason when absent.
- **The Anthropic enrichment path was never exercised with a key.** It was
  confirmed to report `using_llm: false` and create entities without it.
- **Concurrency, load and crash recovery were not tested.** The round-trip was a
  single client writing one entity and reading it back.
- **Other modules still carry the default database path inline.**
  `memory_paths.py` fixed the one reachable from `execute_code`, but a grep
  finds the same literal `~/.claude/enhanced_memories/...` in roughly a dozen
  other modules
  (`graph_traversal.py`, `schema_health.py`, `fast_batch_sync.py`,
  `contextual_enrichment_migration.py`, `letta_memory_blocks.py`,
  `mirror_mind_enhancements.py`, several migration scripts and others). Most are
  standalone utilities rather than server code paths, and none was changed or
  tested here. If you run one of those scripts, check which database it opens
  before trusting that your overrides apply.
- **Only the `entities` table was audited for the schema drift described above.**
  `create_entities` was the write path that failed and the one that was fixed and
  re-tested. The other tables created by `init_database()` were not compared
  against their INSERT sites, so a similar mismatch elsewhere would not have been
  caught by this work.
- **`neural_memory_fabric.py` carries three `TODO` markers** (file-system storage
  in one code path, graph traversal and LLM re-ranking in another). They came
  from the source deployment and describe genuinely unimplemented functionality.
  They are left in place deliberately: removing the labels would make the code
  read as more finished than it is.
- **Tool discovery declares nothing by default, so it can under-report.** The
  registry (fix 10 above) starts empty. An installation that runs other MCP
  servers and does not declare them will have `list_servers()` return `[]` even
  though those servers are reachable. That is the deliberate direction of the
  error: silence is recoverable, a confident list of servers that do not exist
  is not. Full reachability detection — asking each declared server whether it
  is actually answering — was not built.
- **The test suite proves contracts, not correctness at scale.** Both gates run
  a single client against a fresh database. `comprehensive_test.py` calls five
  of the 206 registered tools; a tool being listed is not evidence that it
  works. Concurrency is untested on both gates, which matters because the socket
  daemon exists specifically to serve concurrent clients.
- **`pytest` is in no requirements file.** `pytest tests/` needs pytest
  installed separately; a from-scratch install per the README cannot run the
  unit suite. `comprehensive_test.py` has no such dependency and runs on the
  install as documented.
- **Only two of the fixes shipped tonight were verified by falsification.** The
  fresh-database schema fix and the `success`-on-partial-failure fix were each
  reverted in isolation and shown to break the gate. The other honesty fixes in
  this document were confirmed by observing the fixed behaviour only, not by
  re-breaking the code to watch a test catch it.
- **The deleted provenance tests were not replaced with tests of the
  implemented behaviour.** Citation cycles and source-quality penalties are
  unimplemented, and nothing now asserts what `create_entity_with_provenance`
  does when handed a cyclic source chain. It does not raise; beyond that, the
  behaviour is unexamined.
- **`comprehensive_test.py` was measured on macOS only.** It uses the stricter
  104-byte AF_UNIX limit and `/tmp` for its sandbox, which should hold on Linux,
  but no Linux run was made.
- **The isolation guards were added after the first green run.** The "121
  passed" reported earlier in this work included a test reading a live
  production database. Both guards now block that and both were verified by
  probe, but they cover the **memory** store only: no equivalent guard exists
  for tests that might reach Qdrant, ollama, or the Anthropic API, and none was
  audited for.
- **The SQLite guard only fires on code paths a test actually executes.** It
  proves no test in this suite *did* open the real database on any run since it
  was installed. It does not prove no test *could*: a module can be imported
  without its database-opening function ever being called, and
  `neural_memory_fabric` is in fact imported transitively by `server`.
- **There is no independent confirmation that the guards are sufficient.** The
  obvious external check — watching the real memory directory for changes during
  a run — was tried and is invalid on any machine running a daemon, for the
  reason given above. Everything resting on "the suite does not touch the real
  store" rests on the guards themselves, and they intercept exactly three
  things: `sqlite3.connect`, `os.mkdir`/`os.makedirs`, and the socket path. A
  write by any other route — `open()` in write mode, a file copy, a subprocess,
  an ORM binding its own driver — would not be caught. The `mkdir` route was
  found only because a reviewer pointed out that a connection-level guard
  cannot see directory creation; there has been no systematic enumeration of
  what else it cannot see.
- **Around a dozen modules still resolve `~/.claude/enhanced_memories/`
  themselves** (`graph_traversal.py`, `schema_health.py`, `fast_batch_sync.py`,
  `contextual_enrichment_migration.py`, `letta_memory_blocks.py`,
  `mirror_mind_enhancements.py`, several migration scripts). Only the one
  reachable from `execute_code` was fixed. The rest are mostly standalone
  utilities, they were sampled rather than audited, and no test covers them.

## Verifying commit signatures

Release commits are SSH-signed. To verify:

```bash
git config gpg.ssh.allowedSignersFile .allowed_signers
git log --show-signature -1
```
