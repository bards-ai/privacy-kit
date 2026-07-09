"""Detect (and, for Cursor, restart) running AI clients after a config change.

The ``setup <tool> --apply/--remove`` edits only affect *new* sessions: Claude
Code reads ``settings.json`` per session, Codex reads ``config.toml`` on start,
and Cursor loads ``hooks.json`` only when the editor starts. This module finds
the client processes still running on the old configuration so the CLI can
warn about them — and, for Cursor, terminate and relaunch the editor once the
user confirms.

Detection uses a single ``ps`` snapshot filtered in Python rather than
``pgrep``: flag semantics differ between Linux and macOS, ``pgrep -f``
substring matching happily matches its own caller, and the full argv is needed
anyway to tell Cursor's Electron main process apart from its ``--type=…``
helpers. All functions are pure/return data; the CLI owns every message.
"""

from __future__ import annotations

import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

RESTART_TOOLS = ("claude-code", "codex", "cursor")


@dataclass(frozen=True)
class ClientProcess:
    """One running client process, with enough context to kill and relaunch it."""

    pid: int
    exe: str  # basename of the executable (from comm or argv[0])
    args: str  # full command line, for display and relaunch fallback


def list_processes() -> list[tuple[int, str, str]]:
    """Snapshot of ``(pid, comm-basename, args)`` for every visible process.

    Returns ``[]`` on Windows or on any ``ps`` failure — callers then see
    nothing running, so the restart feature degrades to a silent no-op rather
    than breaking setup.
    """
    if sys.platform.startswith("win"):
        return []
    try:
        result = subprocess.run(
            ["ps", "-ww", "-e", "-o", "pid=,comm=,args="],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []
    procs: list[tuple[int, str, str]] = []
    for line in result.stdout.splitlines():
        parts = line.split(None, 2)
        if len(parts) != 3:
            continue
        pid_text, comm, args = parts
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        # macOS prints the full path for comm, Linux the (truncated) name.
        procs.append((pid, os.path.basename(comm), args))
    return procs


def _argv(args: str) -> list[str]:
    try:
        return shlex.split(args)
    except ValueError:
        return args.split()


def _basename(token: str) -> str:
    return os.path.basename(token)


def _excluded(pid: int, args: str) -> bool:
    # Never match ourselves, the invoking make/shell, or anything privacy-kit
    # runs (the gateway under `uv run privacy-kit serve`, this very command).
    return pid in (os.getpid(), os.getppid()) or "privacy-kit" in args


def detect_claude_code(procs: list[tuple[int, str, str]]) -> list[ClientProcess]:
    """Running Claude Code sessions (native binary or ``node …/claude``)."""
    found: list[ClientProcess] = []
    for pid, comm, args in procs:
        if _excluded(pid, args):
            continue
        argv = _argv(args)
        argv0 = _basename(argv[0]) if argv else ""
        if (
            comm == "claude"
            or argv0 == "claude"
            or (argv0 in ("node", "nodejs") and len(argv) > 1 and _basename(argv[1]) == "claude")
        ):
            found.append(ClientProcess(pid=pid, exe="claude", args=args))
    return found


def detect_codex(procs: list[tuple[int, str, str]]) -> list[ClientProcess]:
    """Running Codex sessions (a native binary named ``codex``)."""
    found: list[ClientProcess] = []
    for pid, comm, args in procs:
        if _excluded(pid, args):
            continue
        argv = _argv(args)
        argv0 = _basename(argv[0]) if argv else ""
        if comm == "codex" or argv0 == "codex":
            found.append(ClientProcess(pid=pid, exe="codex", args=args))
    return found


def detect_cursor_main(procs: list[tuple[int, str, str]]) -> list[ClientProcess]:
    """Cursor's Electron *main* process only.

    Helpers (renderers, zygotes, ``--type=utility`` re-execs) all carry a
    ``--type=`` flag; killing the main process tears the whole tree down.
    """
    found: list[ClientProcess] = []
    for pid, _comm, args in procs:
        if _excluded(pid, args) or "--type=" in args:
            continue
        argv = _argv(args)
        argv0 = _basename(argv[0]).lower() if argv else ""
        if argv0 == "cursor":
            found.append(ClientProcess(pid=pid, exe="cursor", args=args))
    return found


def detect(tool: str, procs: list[tuple[int, str, str]]) -> list[ClientProcess]:
    """Dispatch to the right detector for a ``RESTART_TOOLS`` name."""
    if tool == "claude-code":
        return detect_claude_code(procs)
    if tool == "codex":
        return detect_codex(procs)
    if tool == "cursor":
        return detect_cursor_main(procs)
    raise ValueError(f"unknown tool {tool!r}")


def terminate(pids: list[int], grace: float = 8.0) -> list[int]:
    """SIGTERM the pids, escalate to SIGKILL after ``grace`` seconds.

    SIGTERM first so Electron can save window state; SIGKILL is only the
    escape hatch. Returns the pids that are still alive afterwards
    (normally empty; includes pids we lack permission to signal).
    """
    alive: list[int] = []
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
            alive.append(pid)
        except ProcessLookupError:
            continue  # already gone
        except PermissionError:
            alive.append(pid)
    alive = _wait_gone(alive, grace)
    for pid in alive:
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            continue
    return _wait_gone(alive, 2.0)


def _wait_gone(pids: list[int], timeout: float) -> list[int]:
    deadline = time.monotonic() + timeout
    remaining = list(pids)
    while remaining and time.monotonic() < deadline:
        time.sleep(0.2)
        still: list[int] = []
        for pid in remaining:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                continue
            except PermissionError:
                still.append(pid)
            else:
                still.append(pid)
        remaining = still
    return remaining


def cursor_relaunch_argv(killed: list[ClientProcess]) -> list[str] | None:
    """Best command to start Cursor again, or ``None`` if we can't find one."""
    if sys.platform == "darwin":
        return ["open", "-a", "Cursor"]
    launcher = shutil.which("cursor")
    if launcher:
        return [launcher]
    for proc in killed:
        argv = _argv(proc.args)
        if argv and os.path.isabs(argv[0]) and os.path.exists(argv[0]):
            return [argv[0]]
    return None


def relaunch_detached(argv: list[str]) -> None:
    """Start ``argv`` in its own session so it survives make/the CLI exiting."""
    subprocess.Popen(
        argv,
        start_new_session=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=Path.home(),
    )


def stdin_is_interactive() -> bool:
    """True when we can genuinely prompt the user (both ends are a tty)."""
    return sys.stdin.isatty() and sys.stdout.isatty()
