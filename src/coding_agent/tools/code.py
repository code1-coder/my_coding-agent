"""工作区代码搜索、Python 符号提纲和只读 Git 查询。"""

from __future__ import annotations

import ast
import json
import shutil
import subprocess
import tempfile
import tokenize
from pathlib import Path

from coding_agent.tools.base import Tool
from coding_agent.tools.builtin import _safe_path


def _run_command(command: list[str], workdir: Path, max_bytes: int = 50000) -> tuple[int, str, str, bool]:
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        result = subprocess.run(command, cwd=workdir, stdout=stdout, stderr=stderr,
                                stdin=subprocess.DEVNULL, timeout=30)
        stdout.seek(0)
        output = stdout.read(max_bytes + 1)
        stderr.seek(0)
        error = stderr.read(4000).decode("utf-8", errors="replace")
    return result.returncode, output[:max_bytes].decode("utf-8", errors="replace"), error, len(output) > max_bytes


def _integer(value: int, name: str, minimum: int, maximum: int) -> None:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")


class WorkspaceTool(Tool):
    def __init__(self, workdir: Path | str | None = None) -> None:
        self._workdir = Path(workdir or Path.cwd()).resolve()


class SearchCodeTool(WorkspaceTool):
    name = "search_code"
    description = ("用 ripgrep 搜索工作区，返回 JSON：路径、1-based 行号、匹配和上下文。"
                   "默认按字面量搜索，遵循 .gitignore，跳过隐藏、二进制文件和符号链接；regex=true 启用正则。")
    input_schema = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "path": {"type": "string", "default": "."},
            "file_glob": {"type": "string", "description": "例如 *.py 或 *.{ts,tsx}。"},
            "regex": {"type": "boolean", "default": False},
            "case_sensitive": {"type": "boolean", "default": True},
            "context_lines": {"type": "integer", "minimum": 0, "maximum": 10},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 500},
        },
        "required": ["pattern"],
    }

    def execute(self, pattern: str, path: str = ".", file_glob: str | None = None,
                regex: bool = False, case_sensitive: bool = True, context_lines: int = 0,
                max_results: int = 100) -> str:
        _integer(context_lines, "context_lines", 0, 10)
        _integer(max_results, "max_results", 1, 500)
        if not isinstance(pattern, str) or not pattern or "\n" in pattern:
            raise ValueError("pattern must be a non-empty single-line string")
        target = _safe_path(self._workdir, path)
        if not target.exists():
            raise FileNotFoundError(path)
        requested = self._workdir / path
        if any(candidate.is_symlink() for candidate in (requested, *requested.parents)
               if candidate != self._workdir):
            raise ValueError("Searching through symlinks is not supported")
        executable = shutil.which("rg")
        if executable is None:
            raise RuntimeError("search_code requires ripgrep (rg) on PATH")
        command = [executable, "--no-config", "--json", "--color=never", "--sort=path",
                   "--max-filesize=2M", f"--max-count={max_results + 1}",
                   f"--context={context_lines}", "--glob=!.git/**"]
        if not regex:
            command.append("--fixed-strings")
        if not case_sensitive:
            command.append("--ignore-case")
        if file_glob:
            command.extend(["--type-add", f"agentfilter:{file_glob}", "--type", "agentfilter"])
        command.append("--glob=!.*")
        command.extend(["--", pattern, str(target.relative_to(self._workdir))])
        returncode, output, error, truncated = _run_command(command, self._workdir, 2000000)
        if returncode not in (0, 1):
            raise RuntimeError(f"ripgrep failed ({returncode}): {error.strip()}")
        rows = []
        matches = 0
        for line in output.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                truncated = True
                break
            if event["type"] not in ("match", "context"):
                continue
            data = event["data"]
            if event["type"] == "match":
                if matches == max_results:
                    truncated = True
                    break
                matches += 1
            source_path = data["path"].get("text")
            source_text = data["lines"].get("text")
            if source_path is None or source_text is None:
                continue
            if not _safe_path(self._workdir, source_path).is_file():
                continue
            rows.append({"path": Path(source_path).as_posix(), "line": data["line_number"],
                         "text": source_text.rstrip("\r\n")[:2000],
                         "line_truncated": len(source_text.rstrip("\r\n")) > 2000,
                         "is_match": event["type"] == "match"})
            if len(rows) >= 1000:
                truncated = True
                break
        return json.dumps({"results": rows, "match_count": matches, "truncated": truncated,
                           "limits": "Files <= 2 MiB; lines <= 2000 chars; <= 1000 rows. Narrow path or pattern if truncated."},
                          ensure_ascii=False)


class CodeOutlineTool(WorkspaceTool):
    name = "code_outline"
    description = "静态解析 Python 文件，返回导入、类、函数/方法签名、装饰器、限定名及行范围；不执行代码。"
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "max_symbols": {"type": "integer", "minimum": 1, "maximum": 1000},
        },
        "required": ["path"],
    }

    def execute(self, path: str, max_symbols: int = 200) -> str:
        _integer(max_symbols, "max_symbols", 1, 1000)
        target = _safe_path(self._workdir, path)
        if target.suffix not in (".py", ".pyi"):
            raise ValueError("code_outline supports .py and .pyi files")
        if target.stat().st_size > 2000000:
            raise ValueError("Python file exceeds 2 MB; read a segment instead")
        with tokenize.open(target) as source:
            tree = ast.parse(source.read(), filename=path)
        symbols = []

        def visit(node: ast.AST, parents: tuple[str, ...] = ()) -> None:
            if len(symbols) > max_symbols:
                return
            scope = parents
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if isinstance(node, ast.ClassDef):
                    signature = f"class {node.name}(" + ", ".join(
                        [ast.unparse(base) for base in node.bases] +
                        [ast.unparse(keyword) for keyword in node.keywords]) + ")"
                    kind = "class"
                else:
                    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
                    signature = f"{prefix} {node.name}({ast.unparse(node.args)})"
                    if node.returns:
                        signature += f" -> {ast.unparse(node.returns)}"
                    kind = "function"
                scope = (*parents, node.name)
                symbols.append({"kind": kind, "name": ".".join(scope), "signature": signature[:2000],
                                "line": min([node.lineno] + [item.lineno for item in node.decorator_list]),
                                "end_line": node.end_lineno,
                                "decorators": [ast.unparse(item)[:300] for item in node.decorator_list]})
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                symbols.append({"kind": "import", "signature": ast.unparse(node)[:2000],
                                "line": node.lineno, "end_line": node.end_lineno})
            for child in ast.iter_child_nodes(node):
                visit(child, scope)

        visit(tree)
        return json.dumps({"path": target.relative_to(self._workdir).as_posix(),
                           "symbols": symbols[:max_symbols], "truncated": len(symbols) > max_symbols},
                          ensure_ascii=False)


class GitStatusTool(WorkspaceTool):
    name = "git_status"
    description = "只读查看工作区 Git 状态，返回分支及 porcelain v1 状态；含未跟踪文件，不修改索引。"
    input_schema = {"type": "object", "properties": {}}

    def _git(self, arguments: list[str]) -> str:
        executable = shutil.which("git")
        if executable is None:
            raise RuntimeError("Git is not installed or not on PATH")
        command = [executable, "--no-optional-locks", "--no-pager",
                   "-c", "core.quotePath=false", "-c", "core.fsmonitor=false", *arguments]
        returncode, output, error, truncated = _run_command(command, self._workdir)
        if returncode:
            raise RuntimeError(f"Git failed ({returncode}): {error.strip()}")
        if truncated:
            output += "\n[truncated at 50000 bytes; narrow path or use stat=true]"
        return output.rstrip() or "(no changes)"

    def execute(self) -> str:
        return self._git(["status", "--porcelain=v1", "--branch", "--untracked-files=normal", "--", "."])


class GitDiffTool(GitStatusTool):
    name = "git_diff"
    description = "只读查看 Git 差异：默认未暂存，staged=true 查看暂存区；支持路径、stat、上下文行数，不包含未跟踪文件内容。"
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "staged": {"type": "boolean", "default": False},
            "stat": {"type": "boolean", "default": False},
            "context_lines": {"type": "integer", "minimum": 0, "maximum": 20},
        },
    }

    def execute(self, path: str | None = None, staged: bool = False, stat: bool = False,
                context_lines: int = 3) -> str:
        _integer(context_lines, "context_lines", 0, 20)
        arguments = ["diff", "--no-ext-diff", "--no-textconv", "--no-color", f"--unified={context_lines}"]
        if staged:
            arguments.append("--cached")
        if stat:
            arguments.append("--stat")
        selected = _safe_path(self._workdir, path or ".").relative_to(self._workdir).as_posix()
        arguments.extend(["--", f":(literal){selected}"])
        return self._git(arguments)
