"""
内置工具 —— 命令执行、文件读写、文本编辑和文件搜索。

* ``BashTool``    — 运行 shell 命令
* ``ReadFileTool`` — 读取文件内容
* ``WriteFileTool`` — 将内容写入文件
* ``EditFileTool`` — 在文件中做精确字符串替换
* ``GlobTool``    — 按 glob 模式搜索文件
"""

from __future__ import annotations

import asyncio
import glob as _glob
import json
import os
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from coding_agent.tools.base import Tool, ToolCancelledError, ToolResult


class _WindowsProcessJob:
    """将 Shell 及后代绑定到 Windows Job，父进程退出后仍可统一终止。"""

    def __init__(self, pid: int) -> None:
        import ctypes
        from ctypes import wintypes

        self._api = ctypes.WinDLL("kernel32", use_last_error=True)
        for name, arguments, result in (
            ("CreateJobObjectW", [wintypes.LPVOID, wintypes.LPCWSTR], wintypes.HANDLE),
            ("OpenProcess", [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD], wintypes.HANDLE),
            ("AssignProcessToJobObject", [wintypes.HANDLE, wintypes.HANDLE], wintypes.BOOL),
            ("TerminateJobObject", [wintypes.HANDLE, wintypes.UINT], wintypes.BOOL),
            ("QueryInformationJobObject", [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID,
                                           wintypes.DWORD, wintypes.LPVOID], wintypes.BOOL),
            ("CloseHandle", [wintypes.HANDLE], wintypes.BOOL),
        ):
            function = getattr(self._api, name)
            function.argtypes = arguments
            function.restype = result
        self._handle = self._api.CreateJobObjectW(None, None)
        if not self._handle:
            raise ctypes.WinError(ctypes.get_last_error())
        process_handle = self._api.OpenProcess(0x0101, False, pid)
        try:
            if not process_handle or not self._api.AssignProcessToJobObject(self._handle, process_handle):
                raise ctypes.WinError(ctypes.get_last_error())
        except OSError:
            self._api.CloseHandle(self._handle)
            self._handle = None
            raise
        finally:
            if process_handle:
                self._api.CloseHandle(process_handle)

    def stop(self) -> None:
        import ctypes

        if self._handle and not self._api.TerminateJobObject(self._handle, 1):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self) -> None:
        if self._handle:
            self._api.CloseHandle(self._handle)
            self._handle = None

    async def wait(self) -> None:
        import ctypes

        while True:
            accounting = ctypes.create_string_buffer(48)
            if not self._api.QueryInformationJobObject(self._handle, 1, accounting, 48, None):
                raise ctypes.WinError(ctypes.get_last_error())
            if int.from_bytes(accounting.raw[40:44], "little") == 0:
                return
            await asyncio.sleep(0.01)


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
    description = "运行 shell 命令，返回退出码、stdout、stderr、耗时、超时、取消和截断状态；非零退出码表示失败。输出共保留最多 50000 字符。"
    input_schema = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "要运行的 shell 命令。"},
        },
        "required": ["command"],
    }

    def __init__(self, workdir: Path | str | None = None, timeout_seconds: float = 120) -> None:
        self._workdir = Path(workdir) if workdir else Path.cwd()
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._timeout_seconds = timeout_seconds

    def execute(self, command: str) -> ToolResult:
        """同步入口复用同一进程管理逻辑；异步调用方应使用 aexecute。"""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.aexecute(command))
        with ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(lambda: asyncio.run(self.aexecute(command))).result()

    async def aexecute(self, command: str) -> ToolResult:
        """取消和超时均终止进程树并等待回收，随后才报告结果。"""
        started = time.perf_counter()
        if not isinstance(command, str) or not command.strip():
            return self._result(started, error="Error: command must be a non-empty string", failure_stage="validation")
        # 阻止明显危险的命令
        dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
        if any(d in command for d in dangerous):
            return self._result(started, error="Error: Dangerous command blocked", failure_stage="blocked")
        if os.name != "nt" and sys.platform != "linux":
            return self._result(started, error="Error: Verified Shell process-tree cleanup requires Linux or Windows",
                                failure_stage="startup")

        options = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt"
                   else {"start_new_session": True})
        job = None
        control = None
        tree_status = None

        async def spawn():
            nonlocal job, control
            if os.name == "nt":
                child = await asyncio.create_subprocess_exec(
                    sys.executable, "-c",
                    "import subprocess,sys; sys.stdin.buffer.read(1); "
                    "sys.exit(subprocess.call(sys.argv[1], shell=True))", command,
                    cwd=str(self._workdir), stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, **options,
                )
                try:
                    job = _WindowsProcessJob(child.pid)
                except OSError:
                    await self._stop_windows_bootstrap(child)
                    await child.communicate()
                    raise
                child.stdin.write(b"1")
                await child.stdin.drain()
                child.stdin.close()
            else:
                control, peer = socket.socketpair()
                control.setblocking(False)
                try:
                    child = await asyncio.create_subprocess_exec(
                        sys.executable, str(Path(__file__).with_name("_process_supervisor.py")),
                        str(peer.fileno()), command, pass_fds=(peer.fileno(),),
                        cwd=str(self._workdir), stdin=subprocess.DEVNULL,
                        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, **options,
                    )
                finally:
                    peer.close()
            return child

        spawning = asyncio.create_task(spawn())
        process = None
        reading = None
        stdout = bytearray()
        stderr = bytearray()
        output_size = 0
        cancellation_requested = False

        async def drain(stream, target):
            nonlocal output_size
            while chunk := await stream.read(8192):
                output_size += len(chunk)
                target.extend(chunk[:max(0, 200000 - len(target))])

        async def confirm_tree():
            nonlocal tree_status
            if control is None:
                return
            evidence = bytearray()
            while not evidence.endswith(b"\n"):
                chunk = await asyncio.get_running_loop().sock_recv(control, 4096)
                if not chunk or len(evidence) + len(chunk) > 8192:
                    raise RuntimeError("Shell supervisor exited without process-tree completion evidence")
                evidence.extend(chunk)
            try:
                tree_status = json.loads(evidence)
            except ValueError as exc:
                raise RuntimeError("Invalid Shell supervisor completion evidence") from exc
            if not isinstance(tree_status, dict):
                raise RuntimeError("Invalid Shell supervisor completion evidence")
            if not tree_status.get("tree_stopped"):
                raise RuntimeError(tree_status.get("error", "Shell process-tree cleanup unconfirmed"))
            if type(tree_status.get("exit_code")) is not int:
                raise RuntimeError("Shell supervisor did not report the command exit code")

        async def collect():
            collectors = [asyncio.create_task(coroutine) for coroutine in (
                drain(process.stdout, stdout), drain(process.stderr, stderr), confirm_tree())]
            try:
                await asyncio.gather(*collectors)
            finally:
                for collector in collectors:
                    if not collector.done():
                        collector.cancel()
                await asyncio.gather(*collectors, return_exceptions=True)
            await process.wait()
            if job is not None:
                # Shell exit and pipe EOF do not mean detached descendants have
                # exited. Keep the execution cancellable until the Job is empty;
                # the outer command timeout still covers this wait.
                await job.wait()
            if control is not None and process.returncode != 0:
                raise RuntimeError("Shell supervisor failed after reporting completion")

        async def cleanup():
            nonlocal process, reading
            process = await spawning
            if reading is None:
                reading = asyncio.create_task(collect())
            if job is not None:
                job.stop()
            elif control is not None:
                if tree_status is None:
                    try:
                        control.send(b"stop")
                    except (BrokenPipeError, ConnectionResetError):
                        pass  # collect() must still obtain explicit completion evidence.
            await asyncio.wait_for(asyncio.shield(reading), timeout=6)
            if job is not None:
                await asyncio.wait_for(job.wait(), timeout=5)

        async def finish_cleanup():
            cleaning = asyncio.create_task(cleanup())
            while True:
                try:
                    await asyncio.shield(cleaning)
                    return
                except asyncio.CancelledError:
                    if cleaning.cancelled():
                        raise
                    continue

        def result(**status):
            return self._result(started, stdout.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n"),
                                stderr.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n"),
                                output_truncated=output_size > len(stdout) + len(stderr), **status)

        try:
            process = await asyncio.shield(spawning)
            reading = asyncio.create_task(collect())
            await asyncio.wait_for(asyncio.shield(reading), timeout=self._timeout_seconds)
        except asyncio.CancelledError:
            cancellation_requested = True
            try:
                await finish_cleanup()
            except Exception as exc:
                raise ToolCancelledError(result(
                    error=f"Error: Command cancellation cleanup unconfirmed: {exc}",
                    failure_stage="cleanup", termination_confirmed=False)) from None
            raise ToolCancelledError(result(error="Error: Command cancelled; process tree stopped",
                                            cancelled=True, failure_stage="cancelled",
                                            termination_confirmed=True)) from None
        except TimeoutError:
            try:
                await finish_cleanup()
            except Exception as exc:
                return result(error=f"Error: Command timeout cleanup unconfirmed: {exc}",
                              timed_out=True, failure_stage="cleanup", termination_confirmed=False)
            return result(error=f"Error: Timeout ({self._timeout_seconds:g}s)",
                          timed_out=True, failure_stage="timeout", termination_confirmed=True)
        except RuntimeError as exc:
            return result(error=f"Error: {exc}", failure_stage="supervision", termination_confirmed=False)
        except (FileNotFoundError, OSError) as exc:
            return result(error=f"Error: {exc}", failure_stage="supervision" if process else "startup",
                          termination_confirmed=False if process else None)
        finally:
            if control is not None:
                control.close()
            if reading is not None and not reading.done():
                reading.cancel()
                await asyncio.gather(reading, return_exceptions=True)
            if job is not None:
                async def close_job():
                    try:
                        job.stop()
                        await asyncio.wait_for(job.wait(), timeout=5)
                    finally:
                        job.close()

                closing = asyncio.create_task(close_job())
                cancelled_during_close = False
                while True:
                    try:
                        await asyncio.shield(closing)
                        break
                    except asyncio.CancelledError:
                        cancelled_during_close = True
                        if closing.cancelled():
                            raise
                        continue
                    except Exception as exc:
                        failure = result(error=f"Error: Command cleanup unconfirmed: {exc}",
                                         failure_stage="cleanup", termination_confirmed=False)
                        if cancelled_during_close or cancellation_requested:
                            raise ToolCancelledError(failure) from None
                        return failure
                if cancelled_during_close:
                    raise ToolCancelledError(result(error="Error: Command cancelled; process tree stopped",
                                                    cancelled=True, failure_stage="cancelled",
                                                    termination_confirmed=True)) from None

        exit_code = tree_status["exit_code"] if tree_status else process.returncode
        error = f"Error: Command exited with code {exit_code}" if exit_code != 0 else None
        return result(exit_code=exit_code, error=error, failure_stage="nonzero_exit" if error else None,
                      termination_confirmed=True)

    async def _stop_windows_bootstrap(self, process: asyncio.subprocess.Process) -> None:
        """Job 绑定失败时，命令尚未放行；只需终止等待握手的启动进程。"""
        if process.returncode is None:
            process.kill()
        await process.wait()

    def _result(self, started: float, stdout: str = "", stderr: str = "", *,
                exit_code: int | None = None, error: str | None = None, timed_out: bool = False,
                failure_stage: str | None = None, cancelled: bool = False,
                termination_confirmed: bool | None = None,
                output_truncated: bool = False) -> ToolResult:
        """限制输出大小，同时保留执行状态；未正常退出时退出码为空。"""
        truncated = output_truncated or len(stdout) + len(stderr) > 50000
        stdout = stdout[:50000]
        stderr = stderr[:max(0, 50000 - len(stdout))]
        return ToolResult(tool_use_id="", content=(stdout + stderr).strip() or "(no output)", error=error,
                          execution={"exit_code": exit_code, "stdout": stdout, "stderr": stderr,
                                     "duration_ms": round((time.perf_counter() - started) * 1000, 3),
                                     "timed_out": timed_out, "cancelled": cancelled, "truncated": truncated,
                                     "termination_confirmed": termination_confirmed,
                                     "failure_stage": failure_stage})


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
