# Enhanced Memory MCP Server

[![MCP](https://img.shields.io/badge/MCP-Compatible-blue)](https://modelcontextprotocol.io)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-green)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Tools](https://img.shields.io/badge/MCP_Tools-186_core-informational)]()

Persistent, searchable memory for AI agents, over the
[Model Context Protocol](https://modelcontextprotocol.io/). Entities and their
observations live in a compressed SQLite database with checksums and version
history; a tiered store and a multi-strategy retrieval pipeline sit on top; and
the whole thing is exposed to your client as MCP tools.

How many tools depends on what you installed, and the difference is not a bug:
a tool whose backend is missing is not registered at all. A core install
(`requirements.txt`) registers 186; adding the optional backends
(`requirements-optional.txt`) brings it to 204. If you counted 186 after a
plain `pip install -r requirements.txt`, nothing is broken.

Both figures were measured on Python 3.11.11 through `tools/list` over stdio,
with `AGENTIC_SYSTEM_PATH` unset. That last condition is not pedantry. If that
variable points at the separate system described under
[GraphRAG](#graphrag-is-optional-and-external), seven more tools register and
you get 193 and 211 instead. Earlier drafts of this file said 188 and 206
because they were measured on machines that had it exported, and two of us
reproduced the same wrong number without noticing we shared the cause. Unset it
before you re-measure.

Everything core runs locally with no API keys and no network. The optional
vector stack (Qdrant plus ollama) upgrades recall from keyword matching to
meaning-based, and its absence degrades gracefully rather than breaking.

## The one thing to know first

**This is two processes, not one.** Nearly every support question about this
project comes from running only half of it.

```
   your MCP client  (Claude Code, Claude Desktop, an SDK, curl)
            |
            |   stdio JSON-RPC, one server process per client session
            v
   +-------------------------------------------------------+
   |  MCP server            server.py                       |
   |  start with            setup/bin/mcp-server.sh          |
   +-------------------------------------------------------+
            |
            |   JSON over a Unix socket: $MEMORY_DB_SOCKET_PATH
            |   (default /tmp/memory-db.sock)
            v
   +-------------------------------------------------------+
   |  memory-db daemon      memory_db_service.py            |
   |  start with            setup/bin/memory-db-daemon.sh    |
   |  REQUIRED. Owns the database file exclusively so that   |
   |  several clients can share it without corrupting it.    |
   +-------------------------------------------------------+
            |
            v
     memory.db   (SQLite, default ~/.claude/enhanced_memories/)


   optional, off to the side:
     Qdrant  http://localhost:6333    vector index for semantic recall
     ollama  http://127.0.0.1:11434   local embeddings that feed that index
```

The daemon is not optional and it is not started for you by the MCP server.
Without it, the server still starts, still answers, and returns objects like
these:

```json
{"query": "anything", "count": 0, "results": [],
 "error": "Memory-DB service error: [Errno 2] No such file or directory"}

{"error": "Memory-DB service error: ...", "entities": {"total": 0},
 "compression": {"ratio": "N/A"}}
```

Well formed, parseable, and empty. An agent reading that concludes your memory
is empty rather than deaf. `./healthcheck.sh` exists to tell the two apart.

## Prerequisites

- Python 3.11 or newer. On some macOS machines a bare `python3` is
  still 3.9, so the installer looks for versioned names first.
- git, and disk for the virtualenv. Measured on macOS arm64 with Python 3.11:
  **83 MB** for a core install, **964 MB** with the optional backends, since
  those pull sentence-transformers and torch. On Linux x86_64 the core figure
  is **131 MB** (measured in a python:3.11-slim container) — the wheels differ
  by platform, so expect the number to move with yours. The checkout itself
  is 5 MB.
- Optional: podman or docker, if you want the container path or a local Qdrant.
- Optional: [ollama](https://ollama.com), for local embeddings.

No sudo is required at any point. Nothing is installed system wide.

## Already running an enhanced-memory system?

Read this before step 2 below if this machine might already have one: an older
checkout, a second clone, a service you installed months ago. By default every
install wants the same two things — the socket `/tmp/memory-db.sock` and the
database `~/.claude/enhanced_memories/memory.db` — and they cannot be shared.

Check first:

```bash
lsof /tmp/memory-db.sock        # macOS or Linux
ss -xl | grep memory-db.sock    # Linux
pgrep -af memory_db_service.py
```

Anything listed means an install is live. Starting a second daemon on an
occupied socket is refused: it exits nonzero and prints the socket path and the
database the answering daemon is using, rather than taking the socket over.
That is a guard, not coexistence — the second daemon does not run at all.

To run two installs side by side, give this one its own everything in `.env`:

```ini
ENHANCED_MEMORY_DIR=/home/you/.enhanced-memory-second
MEMORY_DB_SOCKET_PATH=/tmp/memory-db-second.sock
# Only if you want the Neural Memory Fabric somewhere else again; by default it
# follows ENHANCED_MEMORY_DIR:
# NMF_SQLITE_PATH=/home/you/.enhanced-memory-second/nmf.db
# NMF_FILES_ROOT=/home/you/.enhanced-memory-second/nmf_files
```

`ENHANCED_MEMORY_DIR` is the one that gets forgotten. Two daemons on two
sockets sharing one `memory.db` is not coexistence: it is two exclusive owners
of one file, which is exactly what the daemon exists to prevent.

## Quick start

```bash
git clone <this-repo> enhanced-memory-mcp
cd enhanced-memory-mcp

# 1. venv, dependencies, .env, database directory. Idempotent, re-runnable.
setup/setup.sh

# 2. start the daemon (foreground). Leave it running, or install it as a
#    background service: setup/service/install-services.sh
setup/bin/memory-db-daemon.sh &

# 3. prove the install works before you trust it
./healthcheck.sh
```

A healthy run ends with `Required checks passed.` and exit code 0. Anything else
is a real problem: see [Troubleshooting](#troubleshooting).

Settings live in `.env`, which step 1 creates from `.env.example` only when
`.env` is absent. **Editing that file is how a setting persists**; re-running
`setup/setup.sh` never overwrites it.

Then register the server with your MCP client. In `~/.claude.json`:

```json
{
  "mcpServers": {
    "enhanced-memory": {
      "command": "/absolute/path/to/enhanced-memory-mcp/setup/bin/mcp-server.sh"
    }
  }
}
```

Point the client at the **launcher**, not at `python server.py`. The launcher
applies this checkout's `.env`, which is what guarantees the MCP server and the
daemon resolve the same database file. A client that execs python directly
inherits only whatever environment that client happened to have, and the two
processes drift apart silently. See
[the split brain trap](#the-server-and-the-daemon-disagree-about-the-database).

### The alternative: one shared HTTP server

stdio spawns one server process per client session, which is what desktop
clients expect. If you would rather run a single shared server over HTTP, use
the SSE transport:

```bash
MCP_TRANSPORT=sse setup/bin/mcp-server.sh     # or setup/bin/mcp-server-sse.sh
```

```json
{
  "mcpServers": {
    "enhanced-memory": { "type": "sse", "url": "http://127.0.0.1:9106/sse" }
  }
}
```

There is no authentication on that port. Keep `MCP_HOST` at `127.0.0.1`.

## Configuration

Configuration is environment variables. `setup/setup.sh` writes a `.env` from
[`.env.example`](.env.example), which documents every setting inline. **Editing
`.env` is the persistent mechanism**: the copy happens only when `.env` does not
exist, so your edits survive every re-run of the installer (and, for the same
reason, a new release's defaults do not arrive on their own — compare the two
files after an upgrade). A variable already set in your environment beats the
file for that one invocation:

```bash
MEMORY_DB_SOCKET_PATH=/tmp/other.sock ./healthcheck.sh
```

| Variable | Default | Purpose |
|---|---|---|
| `ENHANCED_MEMORY_DIR` | `~/.claude/enhanced_memories` | Directory holding `memory.db`. |
| `ENHANCED_MEMORY_DB_PATH` | (unset) | Full path to the database file. Overrides the directory setting. |
| `MEMORY_DB_SOCKET_PATH` | `/tmp/memory-db.sock` | Unix socket between the two processes. Keep it short, see the AF_UNIX note below. Give a second install on the same machine its own. |
| `NMF_SQLITE_PATH` | `$ENHANCED_MEMORY_DIR/nmf.db` | Optional. The Neural Memory Fabric database. It follows `ENHANCED_MEMORY_DIR` on its own; set this only to put it somewhere else. |
| `NMF_FILES_ROOT` | `$ENHANCED_MEMORY_DIR/nmf_files` | Optional. The NMF file store, same rule. |
| `MCP_TRANSPORT` | `stdio` | `stdio`, `sse`, or `streamable-http`. |
| `MCP_HOST` | `127.0.0.1` | HTTP transports only. Do not expose this to a network. |
| `MCP_PORT` | `9106` | HTTP transports only. |
| `ENHANCED_MEMORY_SURFACE` | `frontdoor` | `frontdoor` registers every tool and marks five as always-loaded (`search_nodes`, `semantic_recall`, `create_entities`, `get_memory_status`, `execute_code`), leaving the rest for the client's tool search; `consolidated` exposes 7 and hides the rest behind one dispatcher; `full` registers everything and marks nothing. |
| `MEMORY_PROFILE` | `full` | `minimal` skips the optional integrations and starts faster. |
| `MEMORY_QDRANT_URL` | `http://localhost:6333` | Optional vector store. |
| `MEMORY_OLLAMA_URL` | `http://127.0.0.1:11434` | Optional embedding provider. |
| `MEMORY_EMBED_MODEL` | `embeddinggemma` | Embedding model to pull and use. |
| `MEMORY_LOW_CONF_THRESHOLD` | `0.50` | Score under which a result is flagged low confidence. |
| `MEMORY_TOOL_REGISTRY_FILE` | (unset) | JSON file declaring which *other* MCP servers code inside `execute_code` may call. Unset means none are declared, which is the honest default for a package that cannot know what your machine runs. |
| `MEMORY_LOG_STDERR` | `1` | Send `WARNING` and above to stderr as well as the log file, so skipped tool groups are visible. Set `0` if your MCP client treats stderr as errors. |
| `AGENTIC_SYSTEM_PATH` | (unset) | Only enables GraphRAG, whose implementation is not shipped here. Setting it takes the tool count from 186 to 193, or 204 to 211 with the optional backends. |
| `EXPECTED_TOOL_COUNT` | (unset) | Pins the tool count `./healthcheck.sh` requires. |

`ENHANCED_MEMORY_SURFACE` and `MEMORY_PROFILE` both change how many tools
`tools/list` returns, and so does which optional dependencies are installed:
tools whose backend is missing are not registered. A core-only install and an
install with the optional extras report different counts from the same code. An
expected tool count is only meaningful next to all three.

## Optional services, and what you lose without them

Neither is required. Both are worth having.

| | Present | Absent |
|---|---|---|
| **Qdrant** | Search ranks by meaning: a query about "permission gating" can surface an entity that never uses those words. | Search still works and still returns results, but ranking falls back to lexical matching. Nothing errors, which is why it is easy not to notice. |
| **ollama** | Generates the embeddings Qdrant indexes. | Qdrant has nothing to index, so recall stays lexical even with Qdrant running. |

Provision either or both:

```bash
setup/setup.sh --with-qdrant     # container on 127.0.0.1:6333, named volume
setup/setup.sh --with-ollama     # verifies ollama, pulls the embedding model
```

`./healthcheck.sh` reports both as OPTIONAL and never fails the gate on their
absence. Pass `--require-optional` if you want the stricter contract.

Already running a Qdrant? Point `MEMORY_QDRANT_URL` at it and skip
`--with-qdrant` entirely; nothing here needs to own the instance. The port
conflict discussed under the container profile below is specific to that
profile, which publishes its own container on 6333 and cannot bind a port
something else already holds. A host install only makes outbound requests.

### GraphRAG is optional and external

The GraphRAG tools (`graph_enhanced_search`, `get_entity_neighbors`) do not ship
here. `graphrag_tools.py` loads its implementation from
`$AGENTIC_SYSTEM_PATH/scripts/graph-rag.py`, a file that belongs to a separate
system and is not part of this package. `AGENTIC_SYSTEM_PATH` defaults to the
grandparent of the checkout, so on a standalone install that path does not
exist.

Nothing breaks. Registration is wrapped, the server logs
`GraphRAG integration skipped: ...` and starts without those tools. If you do
have that system, point `AGENTIC_SYSTEM_PATH` at its root and they register.
Note that the skip message goes to the log file, not your terminal, so absent
tools look like tools that were never there.

## Running in a container

The delivery path for shared environments. Podman first, docker compatible.

```bash
podman-compose up --build                              # core only
WITH_OPTIONAL=1 podman-compose --profile qdrant up     # with a USABLE vector store
```

**`WITH_OPTIONAL=1` is load-bearing on the qdrant profile.** The default image
installs `requirements.txt` only, which does not include `qdrant-client` — so
`--profile qdrant` without it gives you a healthy, reachable, entirely unused
Qdrant: the healthcheck reports the service reachable (true) while the server
logs "qdrant-client not installed - vector search disabled" and every search
stays lexical. A green signal beside an inert capability is exactly the failure
mode this project exists to kill, so it is named here rather than left for you
to find. `WITH_OPTIONAL=1` builds the image with `requirements-optional.txt`
and the vector path actually engages. (Measured: a core image alongside the
qdrant profile answered `/readyz` with "all shards are ready" and used it for
nothing.)

Use the hyphen. On Fedora 44, `podman compose` (a space) hands off to an
external provider, `/usr/libexec/docker/cli-plugins/docker-compose`, which needs
a Docker-compatible API socket. With `podman.socket` inactive, which is the
default, `podman compose up` fails:

```
failed to connect to the docker API at unix:///run/user/1000/podman/podman.sock:
  connect: no such file or directory
```

`systemctl --user start podman.socket` fixes that, or just use `podman-compose`
(1.6.0 here), which drives podman directly and needs no socket. Measured on
Fedora 44 with podman 5.8.4: `podman compose up` failed as above, `podman-compose
up -d` brought the stack up and the container reported `healthy`.

The image runs both processes under `container-entrypoint.sh`, which starts the
daemon, waits for the socket to answer, and only then starts the MCP server on
the SSE transport. If either process exits, the container exits, because a live
MCP server next to a dead daemon is exactly the state that returns well formed
zeros forever.

Notes that will save you time:

- **`podman build` discards the `HEALTHCHECK`.** Podman defaults to the OCI
  image format, which has no field for it. It does warn, once, at build time:

  ```
  HEALTHCHECK is not supported for OCI image format and will be ignored.
  Must use `docker` format
  ```

  Miss that line in the build output and nothing mentions it again: the image
  carries no healthcheck and `podman ps` shows no health state, ever. Measured
  on podman 5.8.4, Fedora 44: the OCI image's `.HealthCheck` inspects as `nil`,
  and rebuilding with `podman build --format docker` gives
  `[CMD /app/setup/lib/container-health.sh]`.

  Three ways out, all verified: build with `--format docker`; use compose, whose
  service-level healthcheck is defined in `compose.yaml` and applies regardless
  of image format (a compose-managed container reports `healthy` from the same
  image that inspects as `nil`); or check on demand with
  `podman exec <name> /app/healthcheck.sh --skip-mcp`.
- The MCP port is published on the **host loopback only**
  (`127.0.0.1:9106:9106`). Inside the container the server binds `0.0.0.0`,
  which is correct there and wrong on a workstation.
- Qdrant's host ports are `${QDRANT_PORT:-6333}` and
  `${QDRANT_ADMIN_PORT:-6334}`. Set them in `.env` if you already run Qdrant on
  6333, which is otherwise a bind conflict that stops the profile from starting.
- **The image is a core install, so the qdrant profile does nothing on its own.**
  `podman-compose --profile qdrant up` gives you a Qdrant that starts, passes its
  healthcheck and answers on its port, while the server has no `qdrant-client`
  to talk to it with. Everything looks green and nothing is indexed. Build with
  the optional stack to actually use it:
  ```bash
  podman build --build-arg WITH_OPTIONAL=1 -t enhanced-memory:local -f Containerfile .
  # or, through compose:
  WITH_OPTIONAL=1 podman-compose up --build
  ```
  `./healthcheck.sh` distinguishes the two cases: it reports Qdrant as reachable
  *and* usable only when the client library is importable, and warns when the
  service is up but nothing can use it.
- The database lives in the named volume `enhanced-memory-data`. Without a
  volume, your memory dies with the container.
- ollama runs on your host, and a container cannot reach it at `127.0.0.1`.
  Uncomment `MEMORY_OLLAMA_URL` in `compose.yaml`
  (`host.containers.internal` for podman, `host.docker.internal` for docker).
- Verify a running container the same way you verify a host install. Use the
  absolute path: not every engine resolves a relative one against `WORKDIR`.
  ```bash
  podman exec enhanced-memory /app/healthcheck.sh --skip-mcp
  ```
- Your local `.env` is not configuration for the container. The image ships an
  empty one on purpose, and everything real comes from the runtime environment
  in `compose.yaml`. `.containerignore` and `.dockerignore` exclude the file,
  but not every engine honours them (Apple's `container build` did not, verified
  2026-08-14), so the Containerfile also empties it in a discarded build stage
  and then fails the build if a populated one survives.

## Running as a background service

```bash
setup/service/install-services.sh              # daemon only
setup/service/install-services.sh --with-sse   # and a shared SSE server
setup/service/uninstall-services.sh
```

launchd user agents on macOS (`~/Library/LaunchAgents`), systemd user units on
Linux (`~/.config/systemd/user`). No root, no system units. Every path is
rendered from this checkout's location, so two checkouts can coexist if you give
them different `--label-prefix` values, different `MEMORY_DB_SOCKET_PATH`
values, **and different `ENHANCED_MEMORY_DIR` values**. All three, not the first
two: separate sockets alone leave both daemons opening the same `memory.db`,
and each of them is meant to own that file exclusively.

The installer waits for the socket and fails loudly with a log tail if the
service does not come up. Logs land in `~/Library/Logs/enhanced-memory` or
`${XDG_STATE_HOME:-~/.local/state}/enhanced-memory/log`, deliberately not in the
checkout: launchd cannot create a log file on an external volume at spawn time
and the job dies with exit 78 before your code ever runs.

On Linux, user units stop at logout unless you enable lingering:

```bash
loginctl enable-linger $USER
```

## Verify your install

Two gates, in this order.

```bash
./healthcheck.sh                 # the post-install gate
python3 comprehensive_test.py    # the functional suite (needs the daemon running)
```

A third, developer-facing suite lives under `tests/` and needs
`pip install -r dev-requirements.txt` first — pytest deliberately ships in no
runtime requirements file, and the two gates above run on the stdlib alone.

Judge `comprehensive_test.py` by its exit code, not by a pass count. The number
of checks depends on the mode it selects: with no `ENHANCED_MEMORY_*` or
`MEMORY_DB_*` variables set it builds its own sandbox and runs everything, and
with them set it runs against your deployment and skips the checks that describe
a sandbox it did not create. Measured on one machine, one commit: **106**
isolated and **102** operator-directed, both exit 0. The run prints its own mode
and names what it skipped.

Installing the optional backends changes that count by **zero**, measured both
ways. An earlier revision of this file said the backends were the cause. They
are not, and the same wrong guess was attached to the `pytest` skip count before
anyone tested it; see the test-suite section of `RELEASE_NOTES.md` for what
actually moves that one.

`./healthcheck.sh` is built so that it can fail. It writes a probe entity
through the daemon socket, searches it back, and deletes it. It treats an
`error` or `daemon` key in any response as failure regardless of the rest of the
payload, and it compares the database path the daemon reports against the one
your environment resolves. It checks:

1. venv, interpreter version, `.env`, socket path length, sources present
2. daemon round trip (status, database agreement, write, read back, cleanup)
   and a schema check: every literal `INSERT` in the two files that own this
   database is compared against the live table definitions, because a column
   the schema lacks fails every write while the daemon reports the failure per
   row rather than raising
3. MCP handshake over stdio, tool count, and that nothing polluted stdout
4. Qdrant and ollama, marked OPTIONAL, never fatal

Useful flags: `--skip-mcp` for a fast daemon-only check, `--expect-tools N` to
pin the count, `--require-optional` to demand the vector stack.

### Where the logs are

`/tmp/enhanced-memory-mcp.log`, always, for every install on the host.

The MCP server clears every logging handler at startup and sends everything to
that one rotating file (50 MB, two backups), because on the stdio transport
anything on stdout corrupts the protocol. Routine `INFO` lives only there, and
the path is fixed, so two checkouts on one machine interleave into the same
file with timestamps and pids as your only separator.

`WARNING` and above additionally go to **stderr**, unless you set
`MEMORY_LOG_STDERR=0`. That is deliberate: every `... integration skipped:
<reason>` line is a feature that did not load, and routing those only to a file
under `/tmp` meant nobody ever read them. If your MCP client treats any stderr
output as an error, set the variable to 0 and read the file instead.

`./healthcheck.sh` reports these too, as a `WARN mcp-startup` line listing the
distinct warnings, so a missing feature shows up in the gate rather than only in
a log. Measured on this branch: a core install produces 11 of them (numpy,
qdrant-client, sentence-transformers, redis, neo4j and so on), a full install
3. None of them fail the gate. They are the inventory of what your install does
not have, which is worth reading once and then ignoring.

### Verifying the signatures on this release

Commits are signed with SSH. Git will not verify them until you tell it which
keys to trust, and that config does not travel with a clone:

```bash
git config gpg.ssh.allowedSignersFile .allowed_signers
git log --show-signature -1
```

Without the first line, `git log --format=%G?` reports `N` for every commit,
which means "cannot verify", not "unsigned". The signatures are present either
way: `git cat-file commit HEAD` shows the `gpgsig` block.

## Troubleshooting

### Every tool returns zeros, or an `error` field

The daemon is not running. This is the common case by a wide margin.

```json
{"count": 0, "results": [], "error": "Memory-DB service error: ..."}
```

```bash
setup/bin/memory-db-daemon.sh          # foreground, watch it
./healthcheck.sh --skip-mcp            # confirm the round trip
```

### The server and the daemon disagree about the database

Symptom: writes appear to succeed but searches never find them, or
`get_memory_status` reports a count that does not match what you stored. The two
processes resolved different files, and neither one errors.

`./healthcheck.sh` detects this directly:

```
FAIL db-agreement  SPLIT BRAIN: daemon holds /path/A/memory.db,
                   this environment resolves /path/B/memory.db
```

Cause: something started one process with a different `ENHANCED_MEMORY_DIR`,
`ENHANCED_MEMORY_DB_PATH`, or `HOME` than the other. Usually an MCP client
configured to exec `python server.py` directly, bypassing the launcher that
applies `.env`. Fix the client config to use `setup/bin/mcp-server.sh`, then
restart both processes.

### Content queries return zero while name queries work

Since `e9ca30c` this cannot happen silently: when search cannot see
observation content, the response says so —

```json
{"count": 0, "results": [], "degraded": "name-only (observations_fts missing)"}
```

`degraded` means the database predates the full-text index and no daemon has
initialized against it since upgrading. Restart the daemon: `init_database()`
now creates the index and backfills every existing row. The other value,
`name-only (FTS query error)`, is per-query and means the query text broke
FTS syntax after sanitization; the name/type match still ran.

### Re-importing a seed appends duplicate observations

Fixed in `e9ca30c`: `create_entities` skips observations whose exact content
already exists for that entity and reports the skips as
`observations_deduped` in its response, so repeated seed imports are
idempotent. Genuinely new observations still append. Duplicates created by
re-imports from before the fix are not deleted for you — issue #8 has the
one-time cleanup SQL.

### `OSError` when the daemon starts, with no useful message

The socket path is too long. AF_UNIX caps the path string at 104 bytes on macOS
and 108 on Linux, and `bind()` fails with an error that mentions neither the cap
nor the path. Deep checkouts hit this the moment the socket is placed inside
them.

Keep `MEMORY_DB_SOCKET_PATH` short and outside the checkout, for example
`/tmp/em-myproject.sock`. `setup/setup.sh` measures it and refuses to continue
if it is too long.

### macOS: the service installs but the daemon never starts

If the log shows `Operation not permitted` on the launcher path, the checkout is
somewhere launchd is not allowed to execute from. Verified 2026-08-14: a
checkout on an external volume under `/Volumes` installs and loads fine, then
every spawn fails with EPERM, because launchd runs without the disk access your
terminal has.

Move the checkout to your home directory or another local path and reinstall,
or grant Full Disk Access to launchd if the location is not negotiable. The
installer surfaces this rather than hiding it: it waits for the socket, fails
after 30 seconds, and prints the tail of the error log.

### `ConnectionRefusedError` while the socket file exists

A killed daemon left the file behind. Start the daemon again and it removes the
file itself, logging `removed stale socket <path>`; the launcher does the same
before it execs. Do not delete a socket file by hand as a habit — a file that is
still being served looks exactly like a stale one, and removing it cuts off
every client of the daemon that owns it.

### `REFUSING TO START: another daemon is already serving ...`

Working as intended: something else is answering on that socket path. The
message names the socket and, when the other daemon replies to a status
request, the database it holds. Either stop that daemon, or give this one its
own `MEMORY_DB_SOCKET_PATH` **and** `ENHANCED_MEMORY_DIR` — see
[Already running an enhanced-memory system?](#already-running-an-enhanced-memory-system).

### The MCP client fails at the handshake with a JSON parse error

Something printed to stdout, which belongs exclusively to the JSON-RPC stream on
the stdio transport. `./healthcheck.sh` check 3 reports this as
`FAIL mcp-stdout` along with the offending line.

### `python3` is 3.9

Common on macOS. Install a supported interpreter
(`brew install python@3.11`) and re-run `setup/setup.sh`, which prefers
versioned names. To force one: `setup/setup.sh --python /path/to/python3.11`.

## Gaps and known issues

Written to be re-checked, not trusted.

- The healthcheck does not exercise the SSE transport, does not call individual
  tools (it lists them), does not test concurrent access from several clients,
  and does not measure recall quality with or without the vector stack.
- Performance figures quoted in older revisions of this README have not been
  reproduced here and were removed rather than repeated. Nothing in this file
  claims a throughput, a latency or a compression ratio.
- The container path is verified with podman 5.8.4 on Fedora 44, linux/amd64:
  built, run, the full healthcheck green inside it, the supervision test
  producing `Exited (1)` with the entrypoint naming which half died, and
  `podman-compose` bringing the stack up healthy. It was also built and run
  under Apple's `container` and under Docker on macOS/arm64 during development.
  Not covered: any distribution other than Fedora 44, and rootful podman (all of
  the above was rootless).
- The service units are installed and started by the installer, which waits for
  the socket and fails loudly if it does not appear. Survival across a real
  reboot or logout has not been tested.
- The tool count varies with the surface, the profile and which optional
  dependencies are installed. Treat any single number as specific to one
  machine's configuration.

## License

MIT
