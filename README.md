# Enhanced Memory MCP Server

[![MCP](https://img.shields.io/badge/MCP-Compatible-blue)](https://modelcontextprotocol.io)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-green)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Tools](https://img.shields.io/badge/MCP_Tools-188_core-informational)]()

Persistent, searchable memory for AI agents, over the
[Model Context Protocol](https://modelcontextprotocol.io/). Entities and their
observations live in a compressed SQLite database with checksums and version
history; a tiered store and a multi-strategy retrieval pipeline sit on top; and
the whole thing is exposed to your client as MCP tools.

How many tools depends on what you installed, and the difference is not a bug:
a tool whose backend is missing is not registered at all. A core install
(`requirements.txt`) registers 188; adding the optional backends
(`requirements-optional.txt`) brings it to 206. If you counted 188 after a
plain `pip install -r requirements.txt`, nothing is broken.

Both figures were measured on Python 3.11.11 through `tools/list` over stdio,
in one virtualenv, before and after installing the optional set.

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
- git, and about 2 GB of disk for the virtual environment.
- Optional: podman or docker, if you want the container path or a local Qdrant.
- Optional: [ollama](https://ollama.com), for local embeddings.

No sudo is required at any point. Nothing is installed system wide.

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
[`.env.example`](.env.example), which documents every setting inline. A variable
already set in your environment beats the file, so a one-off override works as
you would expect:

```bash
MEMORY_DB_SOCKET_PATH=/tmp/other.sock ./healthcheck.sh
```

| Variable | Default | Purpose |
|---|---|---|
| `ENHANCED_MEMORY_DIR` | `~/.claude/enhanced_memories` | Directory holding `memory.db`. |
| `ENHANCED_MEMORY_DB_PATH` | (unset) | Full path to the database file. Overrides the directory setting. |
| `MEMORY_DB_SOCKET_PATH` | `/tmp/memory-db.sock` | Unix socket between the two processes. Keep it short, see the AF_UNIX note below. |
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

## Running in a container

The delivery path for shared environments. Podman first, docker compatible.

```bash
podman compose up --build                 # core only
podman compose --profile qdrant up        # with the vector store
```

The image runs both processes under `container-entrypoint.sh`, which starts the
daemon, waits for the socket to answer, and only then starts the MCP server on
the SSE transport. If either process exits, the container exits, because a live
MCP server next to a dead daemon is exactly the state that returns well formed
zeros forever.

Notes that will save you time:

- The MCP port is published on the **host loopback only**
  (`127.0.0.1:9106:9106`). Inside the container the server binds `0.0.0.0`,
  which is correct there and wrong on a workstation.
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
them different `--label-prefix` values and different sockets.

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

Judge `comprehensive_test.py` by its exit code, not by a pass count. The number
of checks it runs depends on which optional backends are importable, so any
figure printed in a README would be wrong on some machines.

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

A killed daemon left the file behind. The launcher removes a stale socket on
start; if you are starting the daemon some other way, delete the file first.

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
- The container path was built and run with Docker and with Apple's `container`
  on macOS/arm64. It has **not** been exercised with podman, and not on
  linux/amd64. The Containerfile uses no engine-specific syntax, but that is an
  expectation, not a measurement.
- The service units are installed and started by the installer, which waits for
  the socket and fails loudly if it does not appear. Survival across a real
  reboot or logout has not been tested.
- The tool count varies with the surface, the profile and which optional
  dependencies are installed. Treat any single number as specific to one
  machine's configuration.

## License

MIT
