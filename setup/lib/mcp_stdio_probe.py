#!/usr/bin/env python3
"""Speak MCP over stdio to the server and count the tools it advertises.

Runs the same handshake a real client runs (initialize, notifications/initialized,
tools/list, following nextCursor to the end) against ``setup/bin/mcp-server.sh``,
so it exercises the launcher, the venv and the .env at once rather than the
python module in isolation.

Two things it deliberately checks beyond the count:

* stdout purity. The MCP stdio transport is JSON-RPC on stdout. One stray print
  from any imported module corrupts the stream and every client fails at the
  handshake with an opaque parse error. Non-JSON lines are reported.
* what the count depends on. The number of tools is a function of
  ENHANCED_MEMORY_SURFACE (consolidated exposes 7, frontdoor and full expose the
  whole set) and MEMORY_PROFILE (minimal skips the optional integrations). An
  expected count is only meaningful next to those two values, so both are printed.

Output: RESULT <PASS|FAIL|WARN> <check-id> <message> on stdout, diagnostics on
stderr. Exit 1 on failure.

Gaps / not covered: lists tools, does not call them. A tool that is registered
but broken counts here as present.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional

PROTOCOL_VERSION = "2025-06-18"

_failed = False


def result(status: str, check: str, message: str) -> None:
    global _failed
    if status == "FAIL":
        _failed = True
    print(f"RESULT {status} {check} {message}", flush=True)


class Server:
    """A spawned MCP server speaking newline-delimited JSON-RPC on stdio."""

    def __init__(self, command: List[str], timeout: float):
        self.timeout = timeout
        self.stderr_file = tempfile.NamedTemporaryFile(
            prefix="mcp-stdio-probe-", suffix=".log", delete=False
        )
        # Pin the transport for our own child. This probe speaks stdio, so it
        # must not inherit MCP_TRANSPORT=sse from a .env or from the container
        # image, where the server would instead try to bind an HTTP port that
        # the already-running server holds and exit 3 before saying anything.
        child_env = dict(os.environ, MCP_TRANSPORT="stdio")
        self.proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr_file,
            text=True,
            bufsize=1,
            env=child_env,
        )
        self.next_id = 0
        self.stdout_pollution: List[str] = []

    def send(
        self, method: str, params: Optional[Dict[str, Any]] = None
    ) -> Optional[int]:
        """Send a request (returns its id) or a notification (returns None)."""
        message: Dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        request_id = None
        if not method.startswith("notifications/"):
            self.next_id += 1
            request_id = self.next_id
            message["id"] = request_id
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(message) + "\n")
        self.proc.stdin.flush()
        return request_id

    def read_response(self, request_id: int) -> Dict[str, Any]:
        """Read until the response with this id arrives, the process dies, or we
        run out of patience."""
        deadline = time.time() + self.timeout
        assert self.proc.stdout is not None
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if line == "":
                raise RuntimeError(
                    f"server exited (code {self.proc.poll()}) before answering"
                )
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                # Not fatal for us, fatal for a real client. Record and continue.
                self.stdout_pollution.append(line[:200])
                continue
            if message.get("id") == request_id:
                return message
        raise TimeoutError(
            f"no response to request {request_id} within {self.timeout}s"
        )

    def stderr_tail(self, lines: int = 15) -> str:
        self.stderr_file.flush()
        try:
            with open(self.stderr_file.name, "r", errors="replace") as handle:
                return "".join(handle.readlines()[-lines:])
        except OSError:
            return "(stderr unavailable)"

    def close(self) -> None:
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
            self.proc.terminate()
            self.proc.wait(timeout=10)
        except Exception:  # noqa: BLE001 - teardown must not mask the real result
            self.proc.kill()
        finally:
            self.stderr_file.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--command",
        required=True,
        help="command to launch the server, for example setup/bin/mcp-server.sh",
    )
    parser.add_argument(
        "--expect",
        type=int,
        default=None,
        help="exact tool count to require. Omit to report the count and warn "
        "that it is unpinned.",
    )
    parser.add_argument(
        "--min",
        type=int,
        default=1,
        help="minimum tool count accepted when --expect is not given",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    surface = os.environ.get("ENHANCED_MEMORY_SURFACE", "frontdoor (default)")
    profile = os.environ.get("MEMORY_PROFILE", "full (default)")

    try:
        server = Server([args.command], args.timeout)
    except OSError as exc:
        result("FAIL", "mcp-spawn", f"could not launch {args.command}: {exc}")
        return 1

    try:
        request_id = server.send(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "enhanced-memory-healthcheck", "version": "1"},
            },
        )
        response = server.read_response(request_id)
        if "error" in response:
            result("FAIL", "mcp-handshake", f"initialize failed: {response['error']}")
            print(server.stderr_tail(), file=sys.stderr)
            return 1
        server_info = response.get("result", {}).get("serverInfo", {})
        result(
            "PASS",
            "mcp-handshake",
            f"initialize ok (server {server_info.get('name', '?')} "
            f"{server_info.get('version', '?')}, protocol "
            f"{response.get('result', {}).get('protocolVersion', '?')})",
        )

        server.send("notifications/initialized")

        tools: List[Dict[str, Any]] = []
        cursor: Optional[str] = None
        while True:
            params: Dict[str, Any] = {"cursor": cursor} if cursor else {}
            request_id = server.send("tools/list", params)
            response = server.read_response(request_id)
            if "error" in response:
                result("FAIL", "mcp-tools", f"tools/list failed: {response['error']}")
                print(server.stderr_tail(), file=sys.stderr)
                return 1
            page = response.get("result", {})
            tools.extend(page.get("tools", []))
            cursor = page.get("nextCursor")
            if not cursor:
                break

        count = len(tools)
        detail = f"{count} tools (surface={surface}, profile={profile})"
        if args.expect is not None:
            if count == args.expect:
                result("PASS", "mcp-tools", detail)
            else:
                result(
                    "FAIL",
                    "mcp-tools",
                    f"expected {args.expect} tools, got {detail}. Either the "
                    "install is incomplete or EXPECTED_TOOL_COUNT is stale for "
                    "this surface/profile.",
                )
        elif count >= args.min:
            result("PASS", "mcp-tools", detail)
            result(
                "WARN",
                "mcp-tools",
                "EXPECTED_TOOL_COUNT is unset, so this check only proves the "
                f"count is >= {args.min}. Set it in .env to catch a partial "
                "registration.",
            )
        else:
            result(
                "FAIL",
                "mcp-tools",
                f"{detail}, below the minimum of {args.min}",
            )

        if server.stdout_pollution:
            result(
                "FAIL",
                "mcp-stdout",
                f"{len(server.stdout_pollution)} non-JSON line(s) on stdout, which "
                f"breaks the stdio transport for real clients. First: "
                f"{server.stdout_pollution[0]!r}",
            )
        else:
            result("PASS", "mcp-stdout", "stdout carried JSON-RPC only")

        if args.expect is None and count:
            print(
                f"tool names: {', '.join(sorted(t.get('name', '?') for t in tools))}",
                file=sys.stderr,
            )

    except (RuntimeError, TimeoutError) as exc:
        result("FAIL", "mcp-handshake", str(exc))
        print(server.stderr_tail(), file=sys.stderr)
        return 1
    finally:
        server.close()

    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
