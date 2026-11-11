"""
系统提示词组装 —— 运行时、带缓存、确定性。

PROMPT_SECTIONS
    以主题为键的提示词片段字典。片段在运行时根据真实状态选择
    （例如记忆只有在文件存在时才加载）。

assemble_system_prompt(context)
    选择并拼接各片段。``context`` 字典的键决定包含哪些片段。

get_system_prompt(context)
    通过 ``json.dumps(sort_keys=True)`` 实现确定性缓存 —— 避免
    在进程内重复拼接字符串。上下文变化时缓存自动失效。
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  提示词片段注册表
# ---------------------------------------------------------------------------

PROMPT_SECTIONS: dict[str, str] = {
    "identity": "You are a coding agent. Use tools to solve tasks and report the result.",
}

# ---------------------------------------------------------------------------
#  组装
# ---------------------------------------------------------------------------

_SECTION_ORDER = [
    "identity",
    "tools",
    "workspace",
    "skills",
    "memory",
    "mcp",
]


def assemble_system_prompt(context: dict[str, Any]) -> str:
    """根据 *context* 选择片段来构建系统提示词。

    驱动组装的上下文键：

    ================== =============================================
    键                  作用
    ================== =============================================
    ``enabled_tools``  注入 "Available tools: {names}."
    ``workspace``      注入 "Working directory: {path}."
    ``memories``       若非空，注入完整记忆块。
    ``skills``         若非空，注入技能目录。
    ``mcp_servers``    若非空，注入已连接的 MCP 服务器列表。
    ``extra_sections`` ``{name: text}`` 字典，追加在标准片段之后。
    ================== =============================================
    """
    sections: list[str] = []

    sections.append(context.get("identity") or PROMPT_SECTIONS["identity"])
    sections.append(
        "Final response: always deliver a user-facing answer before ending the turn, in the user's language. "
        "After using tools, state the outcome, relevant changed files, actual verification and remaining work. "
        "Progress updates, tool results and complete_task are not a final answer. "
        "If blocked or interrupted by a limit, explain what was done and what remains; never claim unverified success. "
        "Fix incorrect source code before verification; passing tests alone does not prove the real behavior works."
    )

    # 动态：工具
    tools = context.get("enabled_tools", [])
    if tools:
        names = ", ".join(tools)
        sections.append(f"Available tools: {names}.")
        if "search_code" in tools:
            sections.append(
                "Coding workflow: inspect git_status/git_diff before editing. Locate code with search_code "
                "(literal by default; regex=true opts in). Use code_outline for Python symbols, then "
                "read_file(start_line=..., end_line=...) to inspect exact numbered lines. "
                "Use apply_patch when available for contextual edits and preserve existing user changes. "
                "If patch context is missing or ambiguous, read the file again; do not guess. "
                "Run relevant existing tests through bash after editing and inspect git_diff. "
                "Only issue independent reads in the same tool batch; issue dependent edits and checks "
                "in separate batches. Truncated search results require a narrower path or pattern."
            )

    # 动态：工作目录
    workspace = context.get("workspace", "")
    if workspace:
        sections.append(f"Working directory: {workspace}")

    # 条件：技能
    skills = context.get("skills", "")
    if skills:
        sections.append(f"Skills available:\n{skills}")

    # 条件：记忆
    memories = context.get("memories", "")
    if memories:
        sections.append(f"Relevant memories:\n{memories}")

    # 条件：MCP
    mcp_servers = context.get("mcp_servers", [])
    if mcp_servers:
        sections.append(f"Connected MCP servers: {', '.join(mcp_servers)}")

    # 额外片段
    extra = context.get("extra_sections", {})
    if isinstance(extra, dict):
        for key, text in extra.items():
            if text:
                sections.append(str(text))

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
#  确定性缓存
# ---------------------------------------------------------------------------

_last_context_key: str | None = None
_last_prompt: str | None = None


def get_system_prompt(context: dict[str, Any]) -> str:
    """返回组装好的系统提示词；如果 *context* 自上次调用以来未变化，
    则复用缓存值。

    缓存键是 ``json.dumps(context, sort_keys=True)`` —— 不受字典
    顺序影响，也摆脱了 Python 哈希随机化的问题。

    参数：
        context: 携带当前状态的字典（tools、workspace、memories、
            skills、mcp_servers、extra_sections）。
    """
    global _last_context_key, _last_prompt
    key = json.dumps(context, sort_keys=True, ensure_ascii=False, default=str)

    if key == _last_context_key and _last_prompt is not None:
        logger.debug("System prompt cache hit")
        return _last_prompt

    _last_context_key = key
    _last_prompt = assemble_system_prompt(context)
    logger.debug("System prompt assembled (%d chars)", len(_last_prompt))
    return _last_prompt


def invalidate_prompt_cache() -> None:
    """强制下一次调用 :func:`get_system_prompt` 时重新组装。"""
    global _last_context_key, _last_prompt
    _last_context_key = None
    _last_prompt = None
