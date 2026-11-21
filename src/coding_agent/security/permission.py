"""
权限管线 —— 责任链模式，三道门。

通过可插拔的检查器依次校验工具调用::

    Gate 1 ─ DenyListChecker     (硬阻止危险模式)
        │
    Gate 2 ─ WorkspaceChecker    (文件路径必须在工作区内)
        │   ─ DestructiveCmdChecker  (bash rm / chmod 777 / …)
        │
    Gate 3 ─ UserApprovalChecker (暂停并询问 "[y/N]")

检查器返回一个 ``Verdict``：

* ``ALLOW``          — 放行给下一个检查器
* ``DENY``           — 立即停止，工具被阻止
* ``NEEDS_APPROVAL`` — 升级到审批处理器

``PermissionPipeline`` 按插入顺序串联检查器，在
第一个 DENY 或第一个被处理器拒绝的 NEEDS_APPROVAL 处短路。

Deny-list 模式从 YAML 文件加载（见 ``DenyListChecker.from_yaml``）
或直接传入。
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
import logging
import os
import re
import stat
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path, PureWindowsPath
from typing import Any, AsyncIterator, Awaitable, Callable

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ExternalPathGrant:
    workdir: Path
    path: Path
    is_directory: bool = False


_approved_external_read: ContextVar[_ExternalPathGrant | None] = ContextVar(
    "approved_external_read", default=None,
)
_approved_external_search: ContextVar[_ExternalPathGrant | None] = ContextVar(
    "approved_external_search", default=None,
)
_WINDOWS_DRIVE_PATH = re.compile(r"^([a-zA-Z]):[\\/]")


def _wsl_drive_mount(drive: str) -> Path:
    return Path("/mnt") / drive.lower()


def _resolve_wsl_drive_path(path_text: str) -> Path:
    windows_path = PureWindowsPath(path_text)
    mount = _wsl_drive_mount(windows_path.drive[0])
    if not mount.is_dir():
        raise ValueError(
            f"当前运行环境无法访问 Windows 盘符 {windows_path.drive[0]}:；"
            f"未找到挂载目录 {mount}"
        )
    return mount.joinpath(*windows_path.parts[1:]).resolve()


def resolve_read_path(workdir: Path, requested_path: str) -> Path:
    """解析当前宿主机可访问的相对、绝对或 WSL Windows 盘符路径。"""
    if not isinstance(requested_path, str) or not requested_path.strip():
        raise ValueError("文件路径不能为空")

    path_text = requested_path.strip()
    drive_match = _WINDOWS_DRIVE_PATH.match(path_text)
    if drive_match and os.name != "nt":
        return _resolve_wsl_drive_path(path_text)
    else:
        path = Path(path_text)

    return path.resolve() if path.is_absolute() else (workdir / path).resolve()


def approved_read_path(workdir: Path, relative_path: str) -> Path:
    """解析读取路径；工作区外仅允许当前审批上下文中的精确文件。"""
    resolved_workdir = workdir.resolve()
    resolved_path = resolve_read_path(resolved_workdir, relative_path)
    if resolved_path.is_relative_to(resolved_workdir):
        return resolved_path
    if _approved_external_read.get() == _ExternalPathGrant(resolved_workdir, resolved_path):
        return resolved_path
    raise ValueError(f"路径逃逸工作区: {relative_path}")


def resolve_search_path(workdir: Path, requested_path: str) -> Path:
    """搜索不跟随链接；解析后的安全检查不能丢失原始路径中的链接信息。"""
    resolved = resolve_read_path(workdir, requested_path)
    path_text = requested_path.strip()
    if _WINDOWS_DRIVE_PATH.match(path_text) and os.name != "nt":
        windows_path = PureWindowsPath(path_text)
        requested = _wsl_drive_mount(windows_path.drive[0]).joinpath(*windows_path.parts[1:])
    else:
        requested = workdir / path_text
    for candidate in (requested, *requested.parents):
        if candidate == workdir:
            break
        info = candidate.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("Searching through symlinks or junctions is not supported")
    parts = resolved.relative_to(workdir).parts if resolved.is_relative_to(workdir) else resolved.parts
    if any(part.casefold() == ".git" for part in parts):
        raise ValueError("Searching .git metadata is not supported")
    return resolved


def approved_search_path(workdir: Path, requested_path: str) -> Path:
    """只接受当前搜索审批的精确目标，后代文件由本次搜索范围约束。"""
    workdir = workdir.resolve()
    resolved = resolve_read_path(workdir, requested_path)
    if (not resolved.is_relative_to(workdir)
            and _approved_external_search.get() != _ExternalPathGrant(workdir, resolved, resolved.is_dir())):
        raise ValueError(f"路径逃逸工作区: {requested_path}")
    if resolve_search_path(workdir, requested_path) != resolved:
        raise ValueError("搜索路径在校验期间发生变化，请重新审批")
    return resolved


# ---------------------------------------------------------------------------
#  Verdict 枚举
# ---------------------------------------------------------------------------

class Verdict(Enum):
    """单次权限检查的结果。"""

    ALLOW = "allow"
    DENY = "deny"
    NEEDS_APPROVAL = "needs_approval"


# ---------------------------------------------------------------------------
#  抽象检查器
# ---------------------------------------------------------------------------

class BasePermissionChecker(ABC):
    """权限链中的一个环节。

    子类重写 ``check`` 并返回 ``(Verdict, reason_str)``。
    """

    @abstractmethod
    async def check(
        self,
        tool_name: str,
        params: dict[str, Any],
    ) -> tuple[Verdict, str]:
        """检查待执行的工具调用。

        返回:
            ``(Verdict, str)`` —— 裁决结果 + 可展示给用户的可读原因。
        """
        ...


# ---------------------------------------------------------------------------
#  第一道门 —— 硬阻止列表
# ---------------------------------------------------------------------------

class DenyListChecker(BasePermissionChecker):
    """阻止包含任何已配置 deny 模式的命令。

    对 ``bash`` 工具的 ``command`` 参数做子串匹配。
    其他工具一律放行。

    参数:
        deny_patterns: 始终被阻止的子串列表。
    """

    def __init__(self, deny_patterns: list[str]) -> None:
        self.deny_patterns = list(deny_patterns)

    async def check(self, tool_name: str, params: dict[str, Any]) -> tuple[Verdict, str]:
        if tool_name != "bash":
            return Verdict.ALLOW, ""

        command: str = params.get("command", "")
        for pattern in self.deny_patterns:
            if pattern in command:
                msg = f"Blocked: '{pattern}' is on the deny list"
                logger.warning("DenyListChecker blocked bash: %s", msg)
                return Verdict.DENY, msg

        return Verdict.ALLOW, ""

    @classmethod
    def from_yaml(cls, path: str | Path = "config.yaml", *, key: str = "deny_list") -> DenyListChecker:
        """从 YAML 文件加载 deny 模式。

        文件应包含一个顶层列表::

            deny_list:
              - "rm -rf /"
              - "sudo"
              - "shutdown"
              - "reboot"
              - "mkfs"
              - "dd if="
              - "> /dev/sda"

        参数:
            path: YAML 文件路径。
            key: YAML 中包含 deny 列表的顶层键名。

        返回:
            配置好的 ``DenyListChecker``（文件/键缺失时返回空列表）。
        """
        if yaml is None:
            logger.warning("PyYAML 未安装 —— deny list 为空")
            return cls([])

        try:
            resolved = Path(path).resolve()
        except (OSError, ValueError):
            return cls([])

        if not resolved.exists():
            return cls([])

        with open(resolved, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}

        patterns: list[str] = data.get(key, [])
        if not isinstance(patterns, list):
            patterns = []
        return cls(patterns)


# ---------------------------------------------------------------------------
#  第二道门 a —— 工作区边界
# ---------------------------------------------------------------------------

class WorkspaceChecker(BasePermissionChecker):
    """检查文件工具路径不逃逸出工作区。

    适用于: ``read_file``、``write_file``、``edit_file``。

    参数:
        workdir: 项目根目录。所有文件路径必须解析到此目录内。
    """

    def __init__(self, workdir: Path | str = Path.cwd()) -> None:
        self.workdir = Path(workdir).resolve()

    async def check(self, tool_name: str, params: dict[str, Any]) -> tuple[Verdict, str]:
        if tool_name == "apply_patch":
            from coding_agent.tools.patch import parse_patch, patch_path

            try:
                for operation in parse_patch(params.get("patch", "")):
                    patch_path(self.workdir, operation.path)
                    if operation.destination is not None:
                        patch_path(self.workdir, operation.destination)
            except (ValueError, OSError) as exc:
                return Verdict.DENY, str(exc)
            return Verdict.ALLOW, ""
        if tool_name not in ("read_file", "write_file", "edit_file", "search_code", "code_outline", "git_diff"):
            return Verdict.ALLOW, ""

        rel = params.get("path") or "."
        try:
            resolved = (resolve_read_path(self.workdir, rel) if tool_name in ("read_file", "search_code")
                        else (self.workdir / rel).resolve())
        except (OSError, ValueError) as exc:
            if tool_name in ("read_file", "search_code"):
                return Verdict.DENY, f"File path is not accessible from this runtime: {rel}: {exc}"
            return Verdict.NEEDS_APPROVAL, f"Invalid path: {rel}"

        if not resolved.is_relative_to(self.workdir):
            msg = f"Access outside workspace: {rel}"
            logger.debug("WorkspaceChecker flagged: %s", msg)
            return Verdict.NEEDS_APPROVAL, msg

        return Verdict.ALLOW, ""


# ---------------------------------------------------------------------------
#  第二道门 b —— 破坏性 shell 命令
# ---------------------------------------------------------------------------

class DestructiveCommandChecker(BasePermissionChecker):
    """标记潜在破坏性的 bash 命令，需要用户审批。

    参数:
        patterns: 触发 NEEDS_APPROVAL 的子串模式。
    """

    DEFAULT_PATTERNS: list[str] = ["rm ", "> /etc/", "chmod 777"]

    def __init__(self, patterns: list[str] | None = None) -> None:
        self.patterns = patterns or list(self.DEFAULT_PATTERNS)

    async def check(self, tool_name: str, params: dict[str, Any]) -> tuple[Verdict, str]:
        if tool_name != "bash":
            return Verdict.ALLOW, ""

        command: str = params.get("command", "")
        for kw in self.patterns:
            if kw in command:
                msg = f"Potentially destructive command: '{kw}'"
                logger.info("DestructiveCommandChecker flagged: %s", msg)
                return Verdict.NEEDS_APPROVAL, msg

        return Verdict.ALLOW, ""


# ---------------------------------------------------------------------------
#  管线 —— 串联检查器
# ---------------------------------------------------------------------------

# 审批回调类型别名
ApprovalHandler = Callable[[str, dict[str, Any], str], Awaitable[bool]]


class PermissionPipeline:
    """责任链：按顺序运行每个检查器。

    用法::

        pipeline = PermissionPipeline(
            checkers=[
                DenyListChecker(["rm -rf /", "sudo"]),
                WorkspaceChecker(workdir=Path.cwd()),
                DestructiveCommandChecker(),
            ],
            approval_handler=my_approval_fn,   # 异步审批回调：接收工具名、参数和原因，返回是否允许执行
        )

        verdict, reason = await pipeline.check("bash", {"command": "ls"})
    """

    def __init__(
        self,
        checkers: list[BasePermissionChecker] | None = None,
        *,
        approval_handler: ApprovalHandler | None = None,
    ) -> None:
        self.checkers: list[BasePermissionChecker] = checkers or []
        self.approval_handler = approval_handler
        self._approval_tasks: set[asyncio.Task] = set()

    def is_waiting_for_approval(self, task: asyncio.Task) -> bool:
        return task in self._approval_tasks

    def add_checker(self, checker: BasePermissionChecker) -> None:
        """将检查器追加到链尾。"""
        self.checkers.append(checker)

    async def check(
        self,
        tool_name: str,
        params: dict[str, Any],
    ) -> tuple[Verdict, str]:
        verdict, reason, _grant = await self._evaluate(tool_name, params)
        return verdict, reason

    @asynccontextmanager
    async def authorize(
        self,
        tool_name: str,
        params: dict[str, Any],
    ) -> AsyncIterator[tuple[Verdict, str]]:
        """读取与搜索分别持有单次授权，不能互用或改变工作区根目录。"""
        verdict, reason, grant = await self._evaluate(tool_name, params)
        token: Token[_ExternalPathGrant | None] | None = None
        approved_scope = _approved_external_search if tool_name == "search_code" else _approved_external_read
        if verdict == Verdict.ALLOW and grant is not None:
            token = approved_scope.set(grant)
        try:
            yield verdict, reason
        finally:
            if token is not None:
                approved_scope.reset(token)

    async def _evaluate(
        self,
        tool_name: str,
        params: dict[str, Any],
    ) -> tuple[Verdict, str, _ExternalPathGrant | None]:
        """运行权限管线。

        返回:
            ``(Verdict, reason, grant)``，只有全部检查通过才返回本次外部路径授权。
        """
        approved_path: _ExternalPathGrant | None = None
        for checker in self.checkers:
            verdict, reason = await checker.check(tool_name, params)

            if verdict == Verdict.DENY:
                return verdict, reason, None

            if verdict == Verdict.NEEDS_APPROVAL:
                if self.approval_handler is None:
                    # 无法询问 —— 默认拒绝
                    logger.warning(
                        "%s 需要审批但未配置 approval_handler —— 拒绝",
                        tool_name,
                    )
                    return Verdict.DENY, f"Approval required but no handler: {reason}", None

                pending_grant = None
                if tool_name in ("read_file", "search_code") and isinstance(checker, WorkspaceChecker):
                    # 在等待用户之前绑定目标；审批期间改路径或重定向链接不能扩大授权。
                    resolver = resolve_search_path if tool_name == "search_code" else resolve_read_path
                    try:
                        resolved_path = resolver(checker.workdir, params.get("path") or ".")
                    except (OSError, ValueError) as exc:
                        return Verdict.DENY, str(exc), None
                    if not resolved_path.is_relative_to(checker.workdir):
                        pending_grant = _ExternalPathGrant(
                            checker.workdir, resolved_path,
                            is_directory=tool_name == "search_code" and resolved_path.is_dir(),
                        )
                current_task = asyncio.current_task()
                if current_task is not None:
                    self._approval_tasks.add(current_task)
                try:
                    approved = await self.approval_handler(tool_name, params, reason)
                finally:
                    if current_task is not None:
                        self._approval_tasks.discard(current_task)
                if not approved:
                    return Verdict.DENY, f"Permission denied by user: {reason}", None
                if pending_grant is not None:
                    approved_path = pending_grant
                # 用户同意 —— 继续走链中后续检查器

        return Verdict.ALLOW, "", approved_path


# ---------------------------------------------------------------------------
#  内置审批处理器
# ---------------------------------------------------------------------------

async def _terminal_approval_handler(
    tool_name: str,
    params: dict[str, Any],
    reason: str,
) -> bool:
    """通过 stdin 询问用户。围绕 ``input()`` 的异步兼容包装。

    用户输入 'y' 或 'yes' 返回 True。
    """
    print(f"\n\033[33m⚠  {reason}\033[0m")
    print(f"   工具: {tool_name}({_format_params(params)})")
    try:
        choice = await asyncio.to_thread(
            input, "   允许？[y/N] "
        )
    except (EOFError, KeyboardInterrupt):
        return False
    return choice.strip().lower() in ("y", "yes")


def _format_params(params: dict[str, Any], max_len: int = 80) -> str:
    """美化打印前几个参数用于终端展示。"""
    parts = []
    for k, v in params.items():
        s = str(v)
        if len(s) > max_len:
            s = s[:max_len] + "…"
        parts.append(f"{k}={s!r}")
    return ", ".join(parts)


def create_terminal_pipeline(
    workdir: Path | str | None = None,
    deny_list: list[str] | None = None,
    destructive_patterns: list[str] | None = None,
) -> PermissionPipeline:
    """工厂函数：构建基于终端审批的管线。

    参数:
        workdir: ``WorkspaceChecker`` 的工作区根目录。
        deny_list: ``DenyListChecker`` 的模式列表。若为 None
            则使用内置的默认检查规则。
        destructive_patterns: ``DestructiveCommandChecker`` 的模式列表。

    返回:
        开箱即用的 ``PermissionPipeline``。
    """
    if deny_list is None:
        deny_list = [
            "rm -rf /", "sudo", "shutdown", "reboot",
            "mkfs", "dd if=", "> /dev/sda",
        ]

    return PermissionPipeline(
        checkers=[
            DenyListChecker(deny_list),
            WorkspaceChecker(Path(workdir) if workdir else Path.cwd()),
            DestructiveCommandChecker(destructive_patterns),
        ],
        approval_handler=_terminal_approval_handler,
    )
