# memory ops — compounding instrumentation

Tooling to answer one question and then fix it: **is the memory getting smarter
about the user across sessions, or just accumulating?** Built 2026-06-21 after a
direct audit of the live store.

## What the audit found (verified against the live DB, read-only)

| Finding | Number |
|---|---|
| Entities with no recorded retrieval | 9,393 / 9,524 (98.6%) |
| Largest type `platonic_insight` (Jan-2026 bulk import), ever read | 6,301, **0** |
| About-you layer (`auto_memory/*`), never retrieved | ~85% |
| Decay applied but eviction wired | 9,523 decayed, **0 evicted** |
| Consolidation jobs vs promotions produced | 231 jobs → 31 promotions |
| Episodic → semantic consolidation | 64 episodes, **0** consolidated (dead since Jan) |
| Retrieval telemetry | none (`retrieval_contexts` frozen at 5 seed rows since 2025-12-02) |
| MD notes not linked from the loaded `MEMORY.md` index | 178 / 349 (51%) |

**Verdict: accumulating, not compounding.** The only artifact that genuinely
compounds session-to-session is `MEMORY.md`, because it is the one thing
force-loaded into every session. The 34MB DB is largely write-only. The root
cause is that there is no trustworthy retrieval signal, so promotion-by-use and
decay-by-disuse are both unimplementable.

## The three tools

### `memory_compounding_report.py` (read-only, safe anytime)
Repeatable diagnostic. Opens the DB `mode=ro` so it is safe alongside the live
daemon. Reports store state, decay actuation, consolidation output, the live-layer
orphan gap, and the **true compounding metric** (cross-session resurfacing rate)
— which reports `UNAVAILABLE` until `retrieval_log` has data rather than
fabricating a number.

```bash
python3 memory_compounding_report.py          # human-readable
python3 memory_compounding_report.py --json    # machine-readable
```

### `retrieval_log.py` (additive table; the precondition for everything)
Append-only `retrieval_log` table — the missing retrieval signal. The table and
its self-test are live (the mechanism is proven end-to-end). The one change that
turns it from infrastructure into a live signal — a single `log_retrieval(...)`
call in the MCP server search path — is staged in the proposal, not applied here,
because it modifies the running production server.

```bash
python3 retrieval_log.py --init --selftest --status
```

### `memory_quarantine.py` (dry-run safe; apply gated)
Moves never-retrieved bulk-import entities out of the default search path so
recall over the about-you layer stops being diluted. Default mode is read-only
dry-run. Apply is additive (`archived_at` column, default NULL), fully reversible
(`--restore`), and **refuses to run while the daemon holds the DB open**.

The default target is an explicit allowlist (`platonic_insight` only, 6,301 rows).
`--broad` would archive all cold non-protected types (8,979 rows) but that trusts
the access_count proxy we already proved unreliable — a Goodhart trap — so it must
wait for `retrieval_log` to corroborate disuse.

```bash
python3 memory_quarantine.py                    # safe dry-run (allowlist)
python3 memory_quarantine.py --broad            # show the over-aggressive sweep
python3 memory_quarantine.py --apply --confirm  # gated; daemon must be quiesced
python3 memory_quarantine.py --restore --confirm
```

## Safety model

- Reports and dry-runs use `file:...?mode=ro` — they cannot write.
- The about-you layer (`auto_memory/*`) and pinned anchors are never archive
  candidates, even when cold. They are what we want to resurface, not hide.
- Apply is additive + reversible + daemon-gated. No row is ever deleted; child
  rows (observations, forgetting_curves) are left intact.
- Nothing here modifies an L0/L1 governance surface. The pieces that do
  (server search-path wiring, `memory_promotion.py`, the `MEMORY.md` ritual,
  the consolidation daemon) are written up in
  `docs/plans/proposals/memory-compounding/` to go through the gate.

## Env

- `ENHANCED_MEMORY_DB_PATH` — override DB path (default `~/.claude/enhanced_memories/memory.db`)
- `MEMORY_MD_DIR` — override MD-corpus dir (default `~/.claude/projects/-Users-marc/memory`)
