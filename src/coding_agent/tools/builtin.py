"""
内置工具 —— 命令执行、文件读写、文本编辑和文件搜索。

* ``BashTool``    — 运行 shell 命令
* ``ReadFileTool`` — 读取文件内容
* ``WriteFileTool`` — 将内容写入文件
* ``EditFileTool`` — 在文件中做精确字符串替换
* ``GlobTool``    — 按 glob 模式搜索文件
"""

from __future__ import annotations

import glob as _glob
import subprocess
import time
from pathlib import Path

from coding_agent.tools.base import Tool, ToolResult


# ═══════════════════════════════════════════════════════════════
#  辅助函数
# ═══════════════════════════════════════════════════════════════

def _safe_path(workdir: Path, rel: str) -> Path:
    """将 *rel* 相对路径基于 *workdir* 解析，并验证路径未越界。"""
    path = (workdir / rel).resolve()
    if not path.is_relative_to(workdir.resolve()):
        raise ValueError(f"路径逃逸工作区: {rel}")
    return path


# ═══════════════════════════════════════════════════════════════
#  1. Bash 工具
# ═══════════════════════════════════════════════════════════════

class BashTool(Tool):
    name = "bash"
    description = "运行 shell 命令，返回退出码、stdout、stderr、耗时和截断状态；非零退出码表示失败。输出共保留最多 50000 字符。"
    input_schema = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "要运行的 shell 命令。"},
        },
        "required": ["command"],
    }

    def __init__(self, workdir: Path | str | None = None) -> None:
        self._workdir = Path(workdir) if workdir else Path.cwd()

    def execute(self, command: str) -> ToolResult:
        """在工作目录执行命令，保留退出状态和独立输出，最多返回 50000 字符。

        非零退出码、危险命令、超时和启动失败均返回结构化失败结果。
        """
        started = time.perf_counter()
        # 阻止明显危险的命令
        dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
        if any(d in command for d in dangerous):
            return self._result(started, error="Error: Dangerous command blocked")

        try:
            completed = subprocess.run(
                command,
                shell=True,
                cwd=str(self._workdir),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=120,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else exc.stdout or ""
            stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr or ""
            return self._result(started, stdout, stderr, error="Error: Timeout (120s)", timed_out=True)
        except (FileNotFoundError, OSError) as exc:
            return self._result(started, error=f"Error: {exc}")

        error = f"Error: Command exited with code {completed.returncode}" if completed.returncode != 0 else None
        return self._result(started, completed.stdout, completed.stderr, exit_code=completed.returncode, error=error)

    def _result(self, started: float, stdout: str = "", stderr: str = "", *,
                exit_code: int | None = None, error: str | None = None, timed_out: bool = False) -> ToolResult:
        """限制输出大小，同时保留执行状态；未正常退出时退出码为空。"""
        truncated = len(stdout) + len(stderr) > 50000
        stdout = stdout[:50000]
        stderr = stderr[:max(0, 50000 - len(stdout))]
        return ToolResult(tool_use_id="", content=(stdout + stderr).strip() or "(no output)", error=error,
                          execution={"exit_code": exit_code, "stdout": stdout, "stderr": stderr,
                                     "duration_ms": round((time.perf_counter() - started) * 1000, 3),
                                     "timed_out": timed_out, "cancelled": False, "truncated": truncated})


# ═══════════════════════════════════════════════════════════════
#  2. 读取文件
# ═══════════════════════════════════════════════════════════════

class ReadFileTool(Tool):
    name = "read_file"
    description = "读取 UTF-8 文件；start_line/end_line 为从 1 开始的闭区间。分段读取默认展示行号，limit 可限制行数。"
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "文件的相对路径。"},
            "limit": {"type": "integer", "description": "最大返回行数（可选）。"},
            "start_line": {"type": "integer", "minimum": 1},
            "end_line": {"type": "integer", "minimum": 1},
            "show_line_numbers": {"type": "boolean"},
        },
        "required": ["path"],
    }

    def __init__(self, workdir: Path | str | None = None) -> None:
        self._workdir = Path(workdir) if workdir else Path.cwd()

    def execute(self, path: str, limit: int | None = None, start_line: int | None = None,
                end_line: int | None = None, show_line_numbers: bool | None = None) -> str:
        """读取工作区内的 UTF-8 文件，按 limit 截取行数并注明省略的行数。

        路径越界或读取失败时返回错误文本；不修改原文件。
        """
        try:
            for name, value in (("limit", limit), ("start_line", start_line), ("end_line", end_line)):
                if value is not None and (type(value) is not int or value < 1):
                    raise ValueError(f"{name} must be a positive integer")
            first = start_line or 1
            if end_line is not None and end_line < first:
                raise ValueError("end_line must be >= start_line")
            lines = _safe_path(self._workdir, path).read_text(encoding="utf-8", errors="replace").splitlines()
            if first > len(lines) and first != 1:
                raise ValueError(f"start_line exceeds file length ({len(lines)} lines)")
            last = min(len(lines), end_line if end_line is not None else len(lines))
            if limit is not None:
                last = min(last, first + limit - 1)
            numbered = show_line_numbers if show_line_numbers is not None else (
                start_line is not None or end_line is not None)
            selected = [f"{index + 1}: {lines[index]}" if numbered else lines[index]
                        for index in range(first - 1, last)]
            if last < len(lines):
                selected.append(f"... ({len(lines) - last} more lines; next start_line={last + 1})")
            return "\n".join(selected)
        except Exception as exc:
            return f"Error: {exc}"


# ═══════════════════════════════════════════════════════════════
#  3. 写入文件
# ═══════════════════════════════════════════════════════════════

class WriteFileTool(Tool):
    name = "write_file"
    description = "将内容写入文件，自动创建父目录。"
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "文件的相对路径。"},
            "content": {"type": "string", "description": "要写入的完整文件内容。"},
        },
        "required": ["path", "content"],
    }

    def __init__(self, workdir: Path | str | None = None) -> None:
        self._workdir = Path(workdir) if workdir else Path.cwd()

    def execute(self, path: str, content: str) -> str:
        """在工作区内写入完整的 UTF-8 文本，必要时创建父目录。

        已存在的文件会被覆盖；路径越界或写入失败时返回错误文本。
        """
        try:
            file_path = _safe_path(self._workdir, path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")
            return f"Wrote {len(content)} bytes to {path}"
        except Exception as exc:
            return f"Error: {exc}"


# ═══════════════════════════════════════════════════════════════
#  4. 编辑文件
# ═══════════════════════════════════════════════════════════════

class EditFileTool(Tool):
    name = "edit_file"
    description = "在文件中将首次出现的 *old_text* 替换为 *new_text*。"
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "文件的相对路径。"},
            "old_text": {"type": "string", "description": "要替换的精确文本。"},
            "new_text": {"type": "string", "description": "替换后的文本。"},
        },
        "required": ["path", "old_text", "new_text"],
    }

    def __init__(self, workdir: Path | str | None = None) -> None:
        self._workdir = Path(workdir) if workdir else Path.cwd()

    def execute(self, path: str, old_text: str, new_text: str) -> str:
        """将文件中首次出现的 old_text 精确替换为 new_text 并写回磁盘。

        未找到原文本、路径越界或读写失败时返回错误文本。
        """
        try:
            file_path = _safe_path(self._workdir, path)
            text = file_path.read_text(encoding="utf-8")
            if old_text not in text:
                return f"Error: text not found in {path}"
            file_path.write_text(text.replace(old_text, new_text, 1), encoding="utf-8")
            return f"Edited {path}"
        except Exception as exc:
            return f"Error: {exc}"


# ═══════════════════════════════════════════════════════════════
#  5. Glob 文件搜索
# ═══════════════════════════════════════════════════════════════

class GlobTool(Tool):
    name = "glob"
    description = "在工作区中按 glob 模式查找匹配的文件。"
    input_schema = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Glob 模式，例如 '**/*.py'。"},
        },
        "required": ["pattern"],
    }

    def __init__(self, workdir: Path | str | None = None) -> None:
        self._workdir = Path(workdir) if workdir else Path.cwd()

    def execute(self, pattern: str) -> str:
        """按 pattern 搜索工作区文件，过滤越界结果并返回以换行分隔的路径。

        没有匹配项时返回提示文本，搜索失败时返回错误文本。
        """
        try:
            results: list[str] = []
            for match in _glob.glob(pattern, root_dir=str(self._workdir)):
                resolved = (self._workdir / match).resolve()
                if resolved.is_relative_to(self._workdir):
                    results.append(match)
            return "\n".join(results) if results else "(no matches)"
        except Exception as exc:
            return f"Error: {exc}"
