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
import logging
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Awaitable

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


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
        if tool_name not in ("read_file", "write_file", "edit_file"):
            return Verdict.ALLOW, ""

        rel = params.get("path", "")
        try:
            resolved = (self.workdir / rel).resolve()
        except (OSError, ValueError):
            return Verdict.NEEDS_APPROVAL, f"Invalid path: {rel}"

        if not resolved.is_relative_to(self.workdir):
            msg = f"Access outside workspace: {rel}"
            logger.warning("WorkspaceChecker flagged: %s", msg)
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

    def add_checker(self, checker: BasePermissionChecker) -> None:
        """将检查器追加到链尾。"""
        self.checkers.append(checker)

    async def check(
        self,
        tool_name: str,
        params: dict[str, Any],
    ) -> tuple[Verdict, str]:
        """运行权限管线。

        返回:
            ``(Verdict, reason)`` —— 第一个非 ALLOW 的裁决，
            或 ``(ALLOW, "")`` 如果所有检查器通过。
        """
        for checker in self.checkers:
            verdict, reason = await checker.check(tool_name, params)

            if verdict == Verdict.DENY:
                return verdict, reason

            if verdict == Verdict.NEEDS_APPROVAL:
                if self.approval_handler is None:
                    # 无法询问 —— 默认拒绝
                    logger.warning(
                        "%s 需要审批但未配置 approval_handler —— 拒绝",
                        tool_name,
                    )
                    return Verdict.DENY, f"Approval required but no handler: {reason}"

                approved = await self.approval_handler(tool_name, params, reason)
                if not approved:
                    return Verdict.DENY, f"Permission denied by user: {reason}"
                # 用户同意 —— 继续走链中后续检查器

        return Verdict.ALLOW, ""


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
