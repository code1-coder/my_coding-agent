import json
import shutil
import subprocess

import pytest

from coding_agent.agents.subagent import AgentPool
from coding_agent.security.permission import Verdict, WorkspaceChecker
from coding_agent.tools import (
    ApplyPatchTool, BashTool, CodeOutlineTool, GitDiffTool, GitStatusTool, ReadFileTool, SearchCodeTool,
)
from coding_agent.tools.registry import ToolRegistry


def test_bash_preserves_exit_code_and_output(tmp_path):
    result = BashTool(tmp_path).execute(
        'python -c "import sys; print(\'stdout\'); print(\'stderr\', file=sys.stderr); sys.exit(7)"'
    )
    assert result.error == "Error: Command exited with code 7"
    assert result.content == "stdout\nstderr"
    assert result.execution["exit_code"] == 7
    assert result.execution["stdout"].strip() == "stdout"
    assert result.execution["stderr"].strip() == "stderr"
    assert result.execution["timed_out"] is False
    assert result.execution["truncated"] is False
    assert result.to_dict()["is_error"] is True


def test_bash_success_has_zero_exit_code(tmp_path):
    result = BashTool(tmp_path).execute('python -c "print(42)"')
    assert result.error is None
    assert result.content == "42"
    assert result.execution["exit_code"] == 0
    assert result.execution["duration_ms"] >= 0
    assert result.execution["cancelled"] is False
    assert '"exit_code": 0' in result.to_dict()["content"]


def test_bash_reports_output_truncation_and_nonzero_exit(tmp_path):
    result = BashTool(tmp_path).execute(
        'python -c "import sys; sys.stdout.write(\'x\' * 50001); sys.exit(7)"'
    )
    assert result.error == "Error: Command exited with code 7"
    assert result.execution["exit_code"] == 7
    assert result.execution["stdout"] == "x" * 50000
    assert result.execution["stderr"] == ""
    assert result.execution["truncated"] is True
    assert result.execution["failure_stage"] == "nonzero_exit"
    assert result.to_dict()["is_error"] is True


def test_bash_reports_timeout_and_startup_failure(tmp_path):
    timed_out = BashTool(tmp_path, timeout_seconds=0.01).execute(
        'python -c "import time; time.sleep(0.2)"'
    )
    assert timed_out.error == "Error: Timeout (0.01s)"
    assert timed_out.execution["exit_code"] is None
    assert timed_out.execution["timed_out"] is True
    assert timed_out.execution["cancelled"] is False
    assert timed_out.execution["failure_stage"] == "timeout"
    assert timed_out.to_dict()["is_error"] is True

    startup_failed = BashTool(tmp_path / "missing-workdir").execute("echo never")
    assert startup_failed.error
    assert startup_failed.execution["exit_code"] is None
    assert startup_failed.execution["timed_out"] is False
    assert startup_failed.execution["failure_stage"] == "startup"
    assert startup_failed.to_dict()["is_error"] is True


def test_bash_rejects_empty_command_with_structured_failure(tmp_path):
    result = BashTool(tmp_path).execute("   ")
    assert result.error == "Error: command must be a non-empty string"
    assert result.execution["failure_stage"] == "validation"
    assert result.execution["exit_code"] is None


def test_read_file_segments_and_legacy_output(tmp_path):
    (tmp_path / "source.py").write_text("first\nsecond\nthird\nfourth\n", encoding="utf-8")
    tool = ReadFileTool(tmp_path)
    assert tool.execute("source.py") == "first\nsecond\nthird\nfourth"
    assert tool.execute("source.py", start_line=2, end_line=3).startswith("2: second\n3: third")
    assert "next start_line=4" in tool.execute("source.py", start_line=2, end_line=3)
    assert tool.execute("source.py", start_line=3, limit=1, show_line_numbers=False).startswith("third\n")
    for arguments in ({"start_line": 0}, {"limit": -1}, {"start_line": 8},
                      {"start_line": 3, "end_line": 2}):
        assert tool.execute("source.py", **arguments).startswith("Error:")
    assert tool.execute("../outside").startswith("Error:")


@pytest.mark.skipif(shutil.which("rg") is None, reason="requires ripgrep")
def test_search_code_literal_regex_context_and_limits(tmp_path):
    (tmp_path / "main.py").write_text("before\ncall(x)\nCALL(x)\ncallZx\nafter\n", encoding="utf-8")
    (tmp_path / "other.txt").write_text("call(x)", encoding="utf-8")
    (tmp_path / ".hidden.py").write_text("call(x)", encoding="utf-8")
    tool = SearchCodeTool(tmp_path)
    result = json.loads(tool.execute("call(x)", file_glob="*.py", context_lines=1))
    matches = [row for row in result["results"] if row["is_match"]]
    assert [(row["path"], row["line"]) for row in matches] == [("main.py", 2)]
    assert any(not row["is_match"] for row in result["results"])
    result = json.loads(tool.execute("call(x)", case_sensitive=False, file_glob="*.py", max_results=1))
    assert result["match_count"] == 1 and result["truncated"]
    result = json.loads(tool.execute("call.x", regex=True, file_glob="*.py"))
    assert result["match_count"] == 2
    assert json.loads(tool.execute("nothing"))["results"] == []
    with pytest.raises(RuntimeError, match="ripgrep failed"):
        tool.execute("[", regex=True)
    with pytest.raises(ValueError, match="工作区"):
        tool.execute("anything", path="..")


@pytest.mark.skipif(shutil.which("rg") is None or shutil.which("git") is None, reason="requires rg and Git")
def test_search_respects_gitignore(tmp_path):
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / ".gitignore").write_text("ignored.py\n", encoding="utf-8")
    (tmp_path / "ignored.py").write_text("secret_match", encoding="utf-8")
    (tmp_path / "visible.py").write_text("secret_match", encoding="utf-8")
    rows = json.loads(SearchCodeTool(tmp_path).execute("secret_match", file_glob="*.py"))["results"]
    assert [row["path"] for row in rows] == ["visible.py"]


def test_tools_reject_symlink_targets(tmp_path):
    target = tmp_path / "target.py"
    target.write_text("value = 1\n", encoding="utf-8")
    link = tmp_path / "link.py"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation unavailable")
    with pytest.raises(ValueError, match="symlink"):
        SearchCodeTool(tmp_path).execute("value", path="link.py")
    with pytest.raises(ValueError, match="symlink"):
        ApplyPatchTool(tmp_path).execute("*** Begin Patch\n*** Delete File: link.py\n*** End Patch")
    assert target.read_text() == "value = 1\n"


def test_search_missing_dependency_is_explicit(tmp_path, monkeypatch):
    monkeypatch.setattr("coding_agent.tools.code.shutil.which", lambda name: None)
    with pytest.raises(RuntimeError, match="ripgrep"):
        SearchCodeTool(tmp_path).execute("example")


def test_python_outline_nested_async_and_no_execution(tmp_path):
    (tmp_path / "source.py").write_text(
        "import os\nraise RuntimeError('must not execute')\nclass Example:\n"
        "    @staticmethod\n    async def run(value: int = 2) -> str:\n"
        "        def nested():\n            return value\n        return str(nested())\n", encoding="utf-8")
    result = json.loads(CodeOutlineTool(tmp_path).execute("source.py"))
    symbols = {entry.get("name"): entry for entry in result["symbols"]}
    method = symbols["Example.run"]
    assert method["signature"] == "async def run(value: int=2) -> str"
    assert method["line"] == 4 and method["end_line"] == 8
    assert method["decorators"] == ["staticmethod"]
    assert "Example.run.nested" in symbols
    assert json.loads(CodeOutlineTool(tmp_path).execute("source.py", max_symbols=1))["truncated"]
    (tmp_path / "bad.py").write_text("def broken(", encoding="utf-8")
    with pytest.raises(SyntaxError):
        CodeOutlineTool(tmp_path).execute("bad.py")


def test_patch_add_update_move_delete_and_crlf(tmp_path):
    tool = ApplyPatchTool(tmp_path)
    tool.execute("*** Begin Patch\n*** Add File: src/new.py\n+first\n+second\n*** End Patch")
    assert (tmp_path / "src/new.py").read_bytes() == b"first\nsecond\n"
    (tmp_path / "existing.py").write_bytes(b"first\r\nsecond\r\n")
    tool.execute("*** Begin Patch\n*** Update File: existing.py\n*** Move to: moved.py\n@@\n first\n-second\n+changed\n*** Delete File: src/new.py\n*** End Patch")
    assert (tmp_path / "moved.py").read_bytes() == b"first\r\nchanged\r\n"
    assert not (tmp_path / "existing.py").exists()
    assert not (tmp_path / "src/new.py").exists()


def test_patch_preflights_all_files_and_rejects_ambiguity(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("same\nsame\n", encoding="utf-8")
    tool = ApplyPatchTool(tmp_path)
    with pytest.raises(ValueError, match="ambiguous"):
        tool.execute("*** Begin Patch\n*** Add File: new.py\n+new\n*** Update File: source.py\n@@\n-same\n+other\n*** End Patch")
    assert not (tmp_path / "new.py").exists()
    assert source.read_text() == "same\nsame\n"
    tool.execute("*** Begin Patch\n*** Update File: source.py\n@@\n-same\n+last\n*** End of File\n*** End Patch")
    assert source.read_text() == "same\nlast\n"


def test_patch_anchor_multiple_hunks_and_bom(tmp_path):
    source = tmp_path / "source.py"
    source.write_bytes(b"\xef\xbb\xbfdef first():\r\n    return 1\r\ndef second():\r\n    return 1\r\n")
    ApplyPatchTool(tmp_path).execute(
        "*** Begin Patch\n*** Update File: source.py\n@@ def first():\n"
        "-    return 1\n+    return 2\n def second():\n@@\n"
        "-    return 1\n+    return 3\n*** End of File\n*** End Patch")
    assert source.read_bytes() == b"\xef\xbb\xbfdef first():\r\n    return 2\r\ndef second():\r\n    return 3\r\n"


@pytest.mark.parametrize("patch", [
    "*** Begin Patch\n*** Update File: file\n@@\n+value\n*** End Patch",
    "*** Begin Patch\n*** Add File: file\ninvalid\n*** End Patch",
    "*** Begin Patch\n*** Delete File: file\n+unexpected\n*** End Patch",
    "*** Begin Patch\n*** Update File: file\n*** Move to: .git/config\n@@\n-old\n+new\n*** End Patch",
])
def test_patch_rejects_invalid_format_without_writes(tmp_path, patch):
    source = tmp_path / "file"
    source.write_text("old", encoding="utf-8")
    with pytest.raises(ValueError):
        ApplyPatchTool(tmp_path).execute(patch)
    assert source.read_text() == "old"


@pytest.mark.parametrize("path", ["../outside.py", "/absolute.py", "C:/outside.py", ".git/config",
                                  "nested/../outside.py", "file:stream"])
def test_patch_rejects_unsafe_paths(tmp_path, path):
    with pytest.raises(ValueError):
        ApplyPatchTool(tmp_path).execute(f"*** Begin Patch\n*** Add File: {path}\n+x\n*** End Patch")


def test_patch_refuses_overwrite_and_rolls_back_write_failure(tmp_path, monkeypatch):
    import coding_agent.tools.patch as patch_module

    (tmp_path / "existing").write_bytes(b"original")
    tool = ApplyPatchTool(tmp_path)
    with pytest.raises(ValueError, match="already exists"):
        tool.execute("*** Begin Patch\n*** Add File: existing\n+overwrite\n*** End Patch")
    write = patch_module._atomic_write

    def fail_second(path, content, mode):
        if path.name == "second":
            raise OSError("simulated disk failure")
        return write(path, content, mode)

    monkeypatch.setattr(patch_module, "_atomic_write", fail_second)
    with pytest.raises(RuntimeError, match="rolled back"):
        tool.execute("*** Begin Patch\n*** Update File: existing\n@@\n-original\n+changed\n*** Add File: second\n+new\n*** End Patch")
    assert (tmp_path / "existing").read_bytes() == b"original"
    assert not (tmp_path / "second").exists()


@pytest.mark.skipif(shutil.which("git") is None, reason="requires Git")
def test_git_views_staged_unstaged_and_literal_paths(tmp_path):
    def git(*arguments):
        return subprocess.run(["git", *arguments], cwd=tmp_path, check=True, capture_output=True)

    git("init")
    (tmp_path / "file.txt").write_text("staged\n", encoding="utf-8")
    git("add", "file.txt")
    (tmp_path / "file.txt").write_text("working\n", encoding="utf-8")
    (tmp_path / "untracked.txt").write_text("new", encoding="utf-8")
    assert "AM file.txt" in GitStatusTool(tmp_path).execute()
    assert "?? untracked.txt" in GitStatusTool(tmp_path).execute()
    tool = GitDiffTool(tmp_path)
    assert "+working" in tool.execute()
    assert "+staged" in tool.execute(staged=True)
    assert "file.txt" in tool.execute(stat=True)
    assert tool.execute(path="*.txt") == "(no changes)"
    assert "+working" in tool.execute(path="file.txt")


@pytest.mark.asyncio
async def test_registry_errors_permissions_and_subagent_tools(tmp_path):
    registry = ToolRegistry()
    registry.register_tool(ApplyPatchTool(tmp_path))
    result = await registry.dispatch("apply_patch", {"patch": "invalid"}, tool_use_id="patch-1")
    assert result.error and result.to_dict()["is_error"]
    checker = WorkspaceChecker(tmp_path)
    for name in ("search_code", "code_outline", "git_diff"):
        verdict, _ = await checker.check(name, {"path": "../outside"})
        assert verdict == Verdict.NEEDS_APPROVAL
    verdict, _ = await checker.check("apply_patch", {"patch":
        "*** Begin Patch\n*** Add File: ../outside\n+x\n*** End Patch"})
    assert verdict == Verdict.DENY
    pool = AgentPool(workdir=tmp_path, size=2)
    explorer, worker = pool._specs
    assert {"search_code", "code_outline", "git_status", "git_diff"} <= explorer.tool_map.keys()
    assert "apply_patch" not in explorer.tool_map
    assert "apply_patch" in worker.tool_map
