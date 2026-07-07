"""Detection/relaunch logic for `privacy-kit restart-clients` — no real processes."""

from __future__ import annotations

import subprocess
from typing import Any

import pytest

from privacy_kit.gateway import clients

# Synthetic process table modeled on a real Linux snapshot: Cursor's Electron
# tree, a native Claude Code binary, a node-launched claude, a codex binary,
# and lookalikes that must NOT match.
PROCS = [
    (5968, "cursor", "/usr/share/cursor/cursor"),
    (6010, "cursor", "/usr/share/cursor/cursor --type=zygote"),
    (6055, "cursor", "/proc/self/exe --type=utility --utility-sub-type=node.mojom.NodeService"),
    (6002, "chrome-sandbox", "/usr/share/cursor/chrome-sandbox /usr/share/cursor/cursor"),
    (6003, "chrome_crashpad", "/usr/share/cursor/chrome_crashpad_handler --database=/tmp/x"),
    (7001, "claude", "/home/u/.cursor/extensions/anthropic.claude/native-binary/claude"),
    (7002, "node", "node /usr/local/bin/claude --continue"),
    (7100, "codex", "codex"),
    (8000, "bash", "bash -c 'cat /home/u/.claude/shell-snapshots/snap.sh'"),
    (8001, "vim", "vim notes-about-cursor.md"),
    (8002, "uv", "uv run privacy-kit restart-clients cursor"),
    (8003, "python3", "/usr/bin/python3 -m privacy_kit.gateway serve"),
]


def test_detect_cursor_matches_only_the_main_process() -> None:
    found = clients.detect_cursor_main(PROCS)
    assert [p.pid for p in found] == [5968]


def test_detect_claude_code_matches_binary_and_node_launch() -> None:
    found = clients.detect_claude_code(PROCS)
    assert sorted(p.pid for p in found) == [7001, 7002]


def test_detect_claude_code_ignores_argv_mentions_of_dot_claude() -> None:
    found = clients.detect_claude_code([(8000, "bash", "bash -c 'cat /home/u/.claude/x.sh'")])
    assert found == []


def test_detect_codex_matches_binary_only() -> None:
    found = clients.detect_codex(PROCS)
    assert [p.pid for p in found] == [7100]


def test_detect_excludes_privacy_kit_processes() -> None:
    own = [(8002, "uv", "uv run privacy-kit restart-clients cursor")]
    assert clients.detect_cursor_main(own) == []


def test_detect_excludes_own_pid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("privacy_kit.gateway.clients.os.getpid", lambda: 5968)
    assert clients.detect_cursor_main(PROCS) == []


def test_detect_dispatch_rejects_unknown_tool() -> None:
    with pytest.raises(ValueError):
        clients.detect("vim", PROCS)


def test_list_processes_parses_ps_output(monkeypatch: pytest.MonkeyPatch) -> None:
    blob = (
        "    1 systemd         /sbin/init\n"
        " 5968 cursor          /usr/share/cursor/cursor\n"
        "malformed\n"
        " oops bash            bash\n"
    )

    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=args, returncode=0, stdout=blob, stderr="")

    monkeypatch.setattr("privacy_kit.gateway.clients.subprocess.run", fake_run)
    procs = clients.list_processes()
    assert (1, "systemd", "/sbin/init") in procs
    assert (5968, "cursor", "/usr/share/cursor/cursor") in procs
    assert len(procs) == 2


def test_list_processes_returns_empty_on_ps_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise OSError("no ps")

    monkeypatch.setattr("privacy_kit.gateway.clients.subprocess.run", boom)
    assert clients.list_processes() == []


def test_cursor_relaunch_prefers_open_on_macos(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("privacy_kit.gateway.clients.sys.platform", "darwin")
    assert clients.cursor_relaunch_argv([]) == ["open", "-a", "Cursor"]


def test_cursor_relaunch_uses_cli_launcher(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("privacy_kit.gateway.clients.sys.platform", "linux")
    monkeypatch.setattr("privacy_kit.gateway.clients.shutil.which", lambda name: "/usr/bin/cursor")
    assert clients.cursor_relaunch_argv([]) == ["/usr/bin/cursor"]


def test_cursor_relaunch_falls_back_to_killed_argv0(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("privacy_kit.gateway.clients.sys.platform", "linux")
    monkeypatch.setattr("privacy_kit.gateway.clients.shutil.which", lambda name: None)
    monkeypatch.setattr(
        "privacy_kit.gateway.clients.os.path.exists", lambda path: path == "/opt/cursor/cursor"
    )
    killed = [clients.ClientProcess(pid=1, exe="cursor", args="/opt/cursor/cursor --foo")]
    assert clients.cursor_relaunch_argv(killed) == ["/opt/cursor/cursor"]


def test_cursor_relaunch_none_when_nothing_found(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("privacy_kit.gateway.clients.sys.platform", "linux")
    monkeypatch.setattr("privacy_kit.gateway.clients.shutil.which", lambda name: None)
    monkeypatch.setattr("privacy_kit.gateway.clients.os.path.exists", lambda path: False)
    killed = [clients.ClientProcess(pid=1, exe="cursor", args="cursor")]
    assert clients.cursor_relaunch_argv(killed) is None
