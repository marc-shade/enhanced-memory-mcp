# Enhanced Memory MCP, containerized.
#
# Podman first, Docker compatible: this file uses no runtime-specific syntax, so
#   podman build -t enhanced-memory:local -f Containerfile .
#   docker build  -t enhanced-memory:local -f Containerfile .
# both work, as does `container build` on macOS.
#
# The image runs BOTH processes under container-entrypoint.sh, which refuses to
# serve a live MCP server next to a dead database daemon. See that file for why.
#
# Transport is SSE/HTTP on 9106, because a container has no stdio peer. Desktop
# MCP clients want stdio and should run the checkout directly (see README).

ARG PYTHON_VERSION=3.11

# --- stage 1: keep the developer's .env out of the image --------------------
#
# .containerignore / .dockerignore exclude .env, .git and .venv, and podman and
# docker honour them. Apple's `container build` (buildkit shim 0.7.0) did NOT,
# verified 2026-08-14: a real .env was copied straight into the image, and an
# .env is exactly where an ANTHROPIC_API_KEY lives.
#
# So an ignore file is a convenience, not a guarantee. Emptying the file in a
# later layer of the final image would not help either, since the content would
# still sit in the earlier layer for anyone who pulls the image. Doing it in a
# stage that gets discarded does help: only what survives here is copied below.
#
# Emptied rather than deleted, and scoped to this one path: the image ships an
# empty .env, which configures nothing, and every real setting arrives through
# the runtime environment (see compose.yaml).
FROM docker.io/library/python:${PYTHON_VERSION}-slim AS source
WORKDIR /src
COPY . /src
RUN : > /src/.env && : > /src/.env.local

# --- stage 2: runtime -------------------------------------------------------
FROM docker.io/library/python:${PYTHON_VERSION}-slim

# tini is not used: container-entrypoint.sh is a bash supervisor that reaps its
# own children and forwards signals. Keep the image to what is actually needed.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    EMM_REPO=/app \
    MEMORY_DB_SOCKET_PATH=/tmp/memory-db.sock \
    ENHANCED_MEMORY_DIR=/data/enhanced_memories \
    MCP_TRANSPORT=sse \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=9106

WORKDIR /app

# Requirements come straight from the context so this layer's cache key is the
# requirements files alone: editing a source file must not trigger a reinstall.
# They carry no secrets.
COPY requirements*.txt /app/
ARG WITH_OPTIONAL=0
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt \
    && if [ "$WITH_OPTIONAL" = "1" ] && [ -f requirements-optional.txt ]; then \
           python -m pip install -r requirements-optional.txt; \
       fi

COPY --from=source /src /app

# A check that can fail. If an engine ever gets a populated .env past the stage
# above, the build stops here instead of shipping someone's keys.
RUN test ! -s /app/.env || { \
        echo "FATAL: a non-empty .env reached the image; refusing to build" >&2; \
        exit 1; \
    }

# Run as a non-root user. The uid is fixed so a bind-mounted host directory has a
# predictable owner; rootless podman maps it into your user namespace.
RUN useradd --create-home --uid 10001 memory \
    && mkdir -p /data/enhanced_memories \
    && chown -R memory:memory /data /app \
    && chmod 700 /data/enhanced_memories \
    && chmod +x /app/container-entrypoint.sh /app/healthcheck.sh /app/setup/setup.sh \
                /app/setup/bin/*.sh /app/setup/service/*.sh /app/setup/lib/*.sh
USER memory

# Named volume or bind mount goes here. Without one, memory does not survive the
# container.
VOLUME ["/data/enhanced_memories"]

EXPOSE 9106

# Liveness, not correctness: connects to the daemon socket, asks for status, and
# checks that the MCP port is accepting. It deliberately does NOT write a probe
# entity every interval. The full write/read/delete round trip is
# ./healthcheck.sh, run once after install:
#   podman exec <container> /app/healthcheck.sh --skip-mcp
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD ["/app/setup/lib/container-health.sh"]

ENTRYPOINT ["/app/container-entrypoint.sh"]
