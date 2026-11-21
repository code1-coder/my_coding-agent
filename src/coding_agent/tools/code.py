"""工作区代码搜索、Python 符号提纲和只读 Git 查询。"""

from __future__ import annotations

import ast
import asyncio
import fnmatch
import json
import os
import shutil
import stat
import subprocess
import tempfile
import threading
import time
import tokenize
from pathlib import Path

from coding_agent.security.permission import approved_search_path, resolve_search_path
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


class _SearchRun:
    """每次搜索独立持有预算和取消信号，不能在并发调用间共享。"""

    def __init__(self) -> None:
        self.stopped = threading.Event()
        self.deadline = time.monotonic() + 30
        self.workdir = Path.cwd()
        self.target = self.workdir
        self.target_is_directory = True
        self.external = False

    def check(self) -> None:
        if self.stopped.is_set():
            raise asyncio.CancelledError()
        if time.monotonic() >= self.deadline:
            raise TimeoutError("search_code exceeded its 30-second total budget")

    def command(self, arguments: list[str], workdir: Path, *, data: bytes = b"",
                max_bytes: int = 2000000) -> tuple[int, bytes, str, bool]:
        # 仅启动受控的 rg/Git；轮询预算并回收进程，不把取消等待当成停止执行。
        self.check()
        with tempfile.TemporaryFile() as stdin, tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            stdin.write(data)
            stdin.seek(0)
            process = subprocess.Popen(arguments, cwd=workdir, stdin=stdin, stdout=stdout, stderr=stderr)
            clipped = False
            try:
                while process.poll() is None:
                    self.check()
                    if os.fstat(stdout.fileno()).st_size > max_bytes:
                        clipped = True
                        break
                    if os.fstat(stderr.fileno()).st_size > 2000000:
                        raise RuntimeError("Search subprocess error output exceeded its limit")
                    try:
                        process.wait(timeout=0.05)
                    except subprocess.TimeoutExpired:
                        continue
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait()
            self.check()
            stdout.seek(0)
            output = stdout.read(max_bytes + 1)
            stderr.seek(0)
            error = stderr.read(4000).decode("utf-8", errors="replace")
        return (0 if clipped else process.returncode), output[:max_bytes], error, clipped or len(output) > max_bytes


class SearchCodeTool(WorkspaceTool):
    name = "search_code"
    description = ("搜索工作区或经用户单次批准的外部文件/目录，返回 JSON：路径、1-based 行号、匹配和上下文。优先 ripgrep，"
                   "缺失时用 Git 筛选文件并用 Python 字面量搜索；Git 也缺失时须指定文件。"
                   "跳过隐藏、二进制文件和符号链接；regex 和复杂 glob 需要 rg。备用限制见 warnings。")
    _MAX_FILES = 5000
    _MAX_READ_BYTES = 20 * 1024 * 1024
    _MAX_OUTPUT_BYTES = 2000000
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

    def __init__(self, workdir: Path | str | None = None) -> None:
        super().__init__(workdir)
        self._capability_lock = threading.Lock()
        self._capability_key: tuple[str, str] | None = None
        self._capability_time = 0.0
        self._executables: dict[str, str | None] = {}
        self._capabilities()

    def _capabilities(self) -> dict[str, str | None]:
        key = (os.environ.get("PATH", os.defpath), os.environ.get("PATHEXT", ""))
        with self._capability_lock:
            if key != self._capability_key or time.monotonic() - self._capability_time >= 30:
                # 将相对 PATH 项按子进程 cwd 解析，避免查找环境与执行环境不一致。
                directories = [self._workdir / item.strip(chr(34)) for item in key[0].split(os.pathsep)]
                extensions = [""] + (key[1].split(os.pathsep) if os.name == "nt" else [])
                self._executables = {}
                for name in ("rg", "git"):
                    self._executables[name] = next((found for directory in directories
                                                    for extension in extensions
                                                    if (found := shutil.which(str(directory / (name + extension))))), None)
                self._capability_key, self._capability_time = key, time.monotonic()
            return dict(self._executables)

    def _checked_path(self, run: _SearchRun, path: str) -> Path:
        target = resolve_search_path(run.workdir, path)
        if (not target.is_relative_to(run.workdir)
                or (target != run.target and not (run.target_is_directory and target.is_relative_to(run.target)))):
            raise ValueError("Path outside the requested search scope")
        return target

    def execute(self, pattern: str, path: str = ".", file_glob: str | None = None,
                regex: bool = False, case_sensitive: bool = True, context_lines: int = 0,
                max_results: int = 100) -> str:
        return self._execute(_SearchRun(), pattern, path, file_glob, regex, case_sensitive,
                             context_lines, max_results)

    async def aexecute(self, **kwargs) -> str:
        run = _SearchRun()
        worker = asyncio.create_task(asyncio.to_thread(self._execute, run, **kwargs))
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError:
            run.stopped.set()
            finishing = asyncio.gather(worker, return_exceptions=True)
            # 重复取消也必须等待本次扫描和受控子进程释放资源。
            while not finishing.done():
                try:
                    await asyncio.shield(finishing)
                except asyncio.CancelledError:
                    continue
            raise

    def _execute(self, run: _SearchRun, pattern: str, path: str = ".", file_glob: str | None = None,
                 regex: bool = False, case_sensitive: bool = True, context_lines: int = 0,
                 max_results: int = 100) -> str:
        _integer(context_lines, "context_lines", 0, 10)
        _integer(max_results, "max_results", 1, 500)
        if not isinstance(pattern, str) or not pattern or "\n" in pattern or "\r" in pattern:
            raise ValueError("pattern must be a non-empty single-line string")
        target = approved_search_path(self._workdir, path)
        if not target.exists():
            raise FileNotFoundError(path)
        # 搜索根与结果路径属于单次调用；不能修改共享工具的工作区或依赖查找环境。
        run.target = target
        run.target_is_directory = target.is_dir()
        run.external = not target.is_relative_to(self._workdir)
        run.workdir = (target if run.target_is_directory else target.parent) if run.external else self._workdir
        self._checked_path(run, str(target))
        executable = self._capabilities()["rg"]
        if executable is not None:
            return self._ripgrep(run, executable, target, pattern, file_glob, regex,
                                 case_sensitive, context_lines, max_results)
        return self._python_search(run, target, pattern, file_glob, regex, case_sensitive,
                                   context_lines, max_results)

    def _ripgrep(self, run: _SearchRun, executable: str, target: Path, pattern: str,
                 file_glob: str | None, regex: bool, case_sensitive: bool,
                 context_lines: int, max_results: int) -> str:
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
        command.extend(["--", pattern, str(target.relative_to(run.workdir))])
        try:
            returncode, raw, error, truncated = run.command(command, run.workdir)
        except FileNotFoundError:
            # 仅恢复启动时消失的依赖，不将结果文件消失或解析错误误判成缺少 rg。
            if not run.workdir.exists():
                raise
            with self._capability_lock:
                self._executables["rg"] = None
            return self._python_search(run, target, pattern, file_glob, regex, case_sensitive,
                                       context_lines, max_results)
        output = raw.decode("utf-8", errors="replace")
        if returncode not in (0, 1):
            raise RuntimeError(f"ripgrep failed ({returncode}): {error.strip()}")
        rows = []
        matches = 0
        for line in output.splitlines():
            run.check()
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
            checked = self._checked_path(run, source_path)
            if not checked.is_file():
                continue
            rows.append({"path": (checked if run.external else checked.relative_to(self._workdir)).as_posix(),
                         "line": data["line_number"],
                         "text": source_text.rstrip("\r\n")[:2000],
                         "line_truncated": len(source_text.rstrip("\r\n")) > 2000,
                         "is_match": event["type"] == "match"})
            if len(rows) >= 1000:
                truncated = True
                break
        return json.dumps({"backend": "rg", "results": rows, "match_count": matches, "truncated": truncated,
                           "limits": "Files <= 2 MiB; lines <= 2000 chars; <= 1000 rows. Narrow path or pattern if truncated."},
                          ensure_ascii=False)


    def _git(self, run: _SearchRun, executable: str, arguments: list[str], *, data: bytes = b""):
        command = [executable, "--no-optional-locks", "--no-pager", "-c", "core.fsmonitor=false",
                   "-c", "core.quotePath=false", *arguments]
        return run.command(command, run.workdir, data=data)

    def _preflight(self, run: _SearchRun, target: Path, executable: str) -> list[Path]:
        """只检查目录元数据；不能忽视 rg 专属忽略文件后直接扩大读取范围。"""
        scope = target if target.is_dir() else target.parent
        for parent in (scope, *scope.parents):
            run.check()
            for name in (".ignore", ".rgignore"):
                if (parent / name).exists():
                    raise RuntimeError(f"ripgrep unavailable: fallback cannot apply {name}; install rg")
        if not target.is_dir():
            return []
        excluded = []
        entries = 0
        pending = [target]
        while pending:
            run.check()
            current = pending.pop()
            if current != run.workdir and ((current / ".git").exists()):
                excluded.append(current)
                continue
            directories = []
            with os.scandir(current) as children:
                for child in children:
                    run.check()
                    entries += 1
                    if entries > self._MAX_FILES:
                        raise RuntimeError("ripgrep unavailable: fallback metadata limit reached; narrow path")
                    if child.name in (".ignore", ".rgignore"):
                        raise RuntimeError(f"ripgrep unavailable: fallback cannot apply {child.name}; install rg")
                    if child.name.startswith("."):
                        continue
                    info = child.stat(follow_symlinks=False)
                    if stat.S_ISDIR(info.st_mode) and not getattr(info, "st_file_attributes", 0) & 0x400:
                        directories.append(Path(child.path))
            if directories:
                # 预检也按 Git 规则裁剪，避免扫描已忽略的依赖目录耗尽元数据预算。
                data = b"\0".join(os.fsencode(item.relative_to(run.workdir)) for item in directories) + b"\0"
                code, output, error, clipped = self._git(run, executable,
                    ["check-ignore", "--no-index", "-z", "--stdin"], data=data)
                if code not in (0, 1) or clipped:
                    raise RuntimeError(f"Git ignore check failed ({code}): {error.strip()}")
                ignored = set(output.split(b"\0")[:-1])
                pending.extend(item for item in directories
                               if os.fsencode(item.relative_to(run.workdir)) not in ignored)
        return excluded

    def _candidates(self, run: _SearchRun, target: Path, reasons: set[str], warnings: list[str]) -> tuple[str, list[Path]]:
        executable = self._capabilities()["git"]
        if executable is None:
            if target.is_file():
                warnings.append("Git 不可用：仅搜索指定文件，未检查 Git 忽略规则")
                return "python_file", [target]
            raise RuntimeError("ripgrep (rg) and Git unavailable: specify a file path or install rg")
        try:
            code, output, error, clipped = self._git(run, executable, ["rev-parse", "--is-inside-work-tree"])
        except FileNotFoundError:
            if not run.workdir.exists():
                raise
            with self._capability_lock:
                self._executables["git"] = None
            if target.is_file():
                warnings.append("Git 已不可用：仅搜索指定文件，未检查 Git 忽略规则")
                return "python_file", [target]
            raise RuntimeError("ripgrep and Git unavailable: specify a file path or install rg") from None
        if code != 0:
            if "not a git repository" not in error.lower():
                raise RuntimeError(f"Git failed ({code}): {error.strip()}")
            if target.is_file():
                warnings.append("不在 Git 工作树中：仅搜索指定文件，未检查 Git 忽略规则")
                return "python_file", [target]
            raise RuntimeError("ripgrep unavailable: directory fallback requires a Git worktree; specify a file")
        if clipped or output.strip() != b"true":
            raise RuntimeError("ripgrep unavailable: cannot confirm Git worktree")
        excluded = self._preflight(run, target, executable)
        if excluded:
            reasons.add("nested_repository")
        selected = target.relative_to(run.workdir).as_posix()
        code, output, error, clipped = self._git(run, executable, [
            "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", f":(literal){selected}"])
        if code != 0:
            raise RuntimeError(f"Git failed ({code}): {error.strip()}")
        if clipped:
            reasons.add("candidate_output")
        # 截断时最后一个 NUL 后的片段不是完整文件名，绝不能据此读取。
        names = sorted(set(output.split(b"\0")[:-1]))
        if len(names) > self._MAX_FILES:
            names = names[:self._MAX_FILES]
            reasons.add("candidate_count")
        if not names:
            return "python_git", []
        data = b"\0".join(names) + b"\0"
        code, ignored, error, clipped = self._git(run, executable,
            ["check-ignore", "--no-index", "-z", "--stdin"], data=data)
        if code not in (0, 1) or clipped:
            raise RuntimeError(f"Git ignore check failed ({code}): {error.strip()}")
        ignored_names = set(ignored.split(b"\0")[:-1])
        candidates = []
        for name in names:
            run.check()
            if name in ignored_names:
                continue
            candidate = run.workdir / os.fsdecode(name)
            if not candidate.is_relative_to(target if target.is_dir() else target.parent):
                raise ValueError("Git returned a path outside the requested search scope")
            if any(candidate.is_relative_to(root) for root in excluded):
                continue
            candidates.append(candidate)
        return "python_git", candidates

    def _read_search_file(self, run: _SearchRun, target: Path, remaining: int) -> bytes:
        chunks = bytearray()
        with target.open("rb") as source:
            while len(chunks) <= min(2 * 1024 * 1024, remaining):
                run.check()
                block = source.read(min(65536, 2 * 1024 * 1024 + 1 - len(chunks), remaining + 1 - len(chunks)))
                if not block:
                    break
                chunks.extend(block)
        return bytes(chunks)

    def _python_search(self, run: _SearchRun, target: Path, pattern: str, file_glob: str | None,
                       regex: bool, case_sensitive: bool, context_lines: int, max_results: int) -> str:
        if regex:
            raise ValueError("ripgrep unavailable: regex=true is unsupported by literal fallback; install rg")
        if file_glob and (any(char in file_glob for char in "/\\{}") or file_glob.startswith("!")):
            raise ValueError("ripgrep unavailable: file_glob supports simple filename patterns only, such as *.py")
        rows = []
        reasons: set[str] = set()
        skipped: dict[str, int] = {}
        warnings = ["rg 不可用，已降级为 Python 字面量搜索；不支持正则和复杂 glob；"
                    "仅解码 UTF-8，忽略大小写使用 Unicode casefold，与 rg 可能不同"]
        backend = "python_git" if target.is_dir() else "python_file"
        read_bytes, output_bytes, matches = 0, 2048, 0
        needle = pattern if case_sensitive else pattern.casefold()
        try:
            backend, candidates = self._candidates(run, target, reasons, warnings)
            for candidate in candidates:
                run.check()
                relative = candidate.relative_to(run.workdir)
                if any(part.startswith(".") for part in relative.parts):
                    continue
                if file_glob and not fnmatch.fnmatchcase(candidate.name, file_glob):
                    continue
                try:
                    checked = self._checked_path(run, str(relative))
                except (ValueError, FileNotFoundError):
                    skipped["unsafe_path"] = skipped.get("unsafe_path", 0) + 1
                    continue
                if not checked.is_file():
                    skipped["not_regular_file"] = skipped.get("not_regular_file", 0) + 1
                    continue
                if checked.stat().st_size > 2 * 1024 * 1024:
                    skipped["file_size"] = skipped.get("file_size", 0) + 1
                    continue
                if read_bytes >= self._MAX_READ_BYTES:
                    reasons.add("read_bytes")
                    break
                data = self._read_search_file(run, checked, self._MAX_READ_BYTES - read_bytes)
                read_bytes += len(data)
                if read_bytes > self._MAX_READ_BYTES:
                    reasons.add("read_bytes")
                    break
                if len(data) > 2 * 1024 * 1024 or b"\0" in data:
                    skipped["binary_or_large"] = skipped.get("binary_or_large", 0) + 1
                    continue
                try:
                    text = data.decode("utf-8-sig")
                except UnicodeDecodeError:
                    skipped["encoding"] = skipped.get("encoding", 0) + 1
                    continue
                lines = text.split("\n")
                if text.endswith("\n"):
                    lines.pop()
                hits = []
                extra_line = len(lines)
                for index, line in enumerate(lines):
                    run.check()
                    if needle in (line if case_sensitive else line.casefold()):
                        if len(hits) == max_results - matches:
                            reasons.add("max_results")
                            extra_line = index
                            break
                        hits.append(index)
                selected = set()
                for index in hits:
                    selected.update(range(max(0, index - context_lines),
                                          min(len(lines), index + context_lines + 1, extra_line)))
                hit_set = set(hits)
                for index in sorted(selected):
                    run.check()
                    line = lines[index].rstrip("\r")
                    row = {"path": (checked if run.external else relative).as_posix(),
                           "line": index + 1, "text": line[:2000],
                           "line_truncated": len(line) > 2000, "is_match": index in hit_set}
                    size = len(json.dumps(row, ensure_ascii=False).encode("utf-8")) + 2
                    if len(rows) == 1000 or output_bytes + size > self._MAX_OUTPUT_BYTES:
                        reasons.add("result_output")
                        break
                    rows.append(row)
                    output_bytes += size
                    matches += int(row["is_match"])
                if reasons.intersection({"max_results", "result_output"}):
                    break
        except TimeoutError:
            reasons.add("time_budget")
        if skipped:
            reasons.add("skipped_files")
            warnings.append("部分文件因类型、大小、编码或路径限制未搜索，不能据空结果认定整个范围无匹配")
        # 状态先于正文，已有界面截断和上下文预览仍能识别降级与不完整覆盖。
        return json.dumps({"backend": backend, "fallback_reason": "rg_unavailable", "truncated": bool(reasons),
                           "truncation_reasons": sorted(reasons), "warnings": warnings, "skipped": skipped,
                           "match_count": matches,
                           "limits": "UTF-8 literal search; files <= 2 MiB; <= 5000 entries; reads <= 20 MiB; "
                                     "<= 1000 rows; lines <= 2000 chars; output <= 2 MB; total budget 30 s.",
                           "results": rows},
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
