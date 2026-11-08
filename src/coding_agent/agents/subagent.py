"""
子智能体池 —— 预创建的、上下文隔离的智能体实例。

每个子智能体：
* 拥有自己 **独立** 的 ``messages[]`` 列表（上下文隔离）。
* 获得受限的工具集（没有递归 ``task`` 工具）。
* 有硬性的 30 轮安全限制。
* 只返回 **摘要** —— 中间工具结果被丢弃。

两种智能体角色：
    ``explorer`` — bash、分段读取、代码搜索、Python 提纲和 Git 查看。
    ``worker``   — explorer 工具 + write_file、edit_file、apply_patch。

用法::

    pool = AgentPool(size=4, workdir=Path.cwd())
    result = await pool.dispatch("找出所有 Python 文件并统计行数")
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Awaitable

from coding_agent.tools.base import Tool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  常量
# ---------------------------------------------------------------------------

MAX_SUBAGENT_TURNS = 30  # 硬性安全限制

SUB_SYSTEM_TEMPLATE = (
    "你是一个位于 {workdir} 的编码智能体。"
    "完成交给你的任务，然后返回简洁的摘要。"
    "不要进一步委托。"
)


# ---------------------------------------------------------------------------
#  工具集
# ---------------------------------------------------------------------------

def _explorer_tool_names() -> set[str]:
    return {"bash", "read_file", "glob", "search_code", "code_outline", "git_status", "git_diff"}


def _worker_tool_names() -> set[str]:
    return _explorer_tool_names() | {"write_file", "edit_file", "apply_patch"}


# ---------------------------------------------------------------------------
#  子智能体池（AgentPool）
# ---------------------------------------------------------------------------

# LLM 调用回调类型: (messages, tools, system, max_tokens) → ProviderResponse
LLMCallFn = Callable[[list[dict[str, Any]], list[dict[str, Any]], str, int], Awaitable[Any]]


@dataclass
class SubagentSpec:
    """预配置的子智能体模板。"""

    name: str
    role: str  # 角色类型：探索者或执行者（explorer、worker）
    system_prompt: str = ""
    tools: list[dict[str, Any]] = field(default_factory=list)
    tool_map: dict[str, Tool] = field(default_factory=dict)


class AgentPool:
    """预配置的子智能体规格池，随时可分发。

    池实例很轻量 —— 它们持有工具定义和配置，
    而不是活跃的 LLM 连接。``dispatch()`` 为每次调用
    创建全新的 ``messages[]``，因此没有跨任务的状态泄漏。
    """

    def __init__(
        self,
        *,
        workdir: Path | str | None = None,
        size: int = 4,
        llm_call: LLMCallFn | None = None,
        runner: Callable[..., Awaitable[str]] | None = None,
    ) -> None:
        wd = Path(workdir) if workdir else Path.cwd()
        self._workdir = wd
        self._size = size
        self._llm_call = llm_call
        self._runner = runner

        # ── 一次性构建工具规格 ──────────────────────────────
        all_tools = _build_base_tools(wd)

        explorer_tools = {
            name: all_tools[name]
            for name in _explorer_tool_names()
        }
        worker_tools = {
            name: all_tools[name]
            for name in _worker_tool_names()
        }

        self._specs: list[SubagentSpec] = []
        for i in range(size):
            role = "explorer" if i % 2 == 0 else "worker"
            tool_map = explorer_tools if role == "explorer" else worker_tools
            self._specs.append(SubagentSpec(
                name=f"subagent-{i:02d}",
                role=role,
                system_prompt=SUB_SYSTEM_TEMPLATE.format(workdir=wd),
                tools=[t.to_anthropic_spec() for t in tool_map.values()],
                tool_map=tool_map,
            ))

    # ── 公开 API ─────────────────────────────────────────────

    @property
    def available_roles(self) -> list[str]:
        """返回池中每个规格的角色。"""
        return [s.role for s in self._specs]

    async def dispatch(
        self,
        description: str,
        *,
        role: str = "worker",
        max_turns: int = MAX_SUBAGENT_TURNS,
    ) -> str:
        """启动一次子智能体运行。

        选取第一个匹配 *role* 的可用规格，创建全新的
        ``messages[]`` 列表，运行 agent loop，返回摘要。

        参数:
            description: 子智能体的任务（成为第一条 user 消息）。
            role: ``"explorer"`` 或 ``"worker"``。
            max_turns: 覆盖 30 轮安全限制。

        返回:
            子智能体的最终文本摘要，或达到限制但未得出结论时的错误字符串。
        """
        if self._llm_call is None and self._runner is None:
            return "Error: AgentPool has no llm_call configured"

        # 选取第一个匹配的规格
        spec = next((s for s in self._specs if s.role == role), None)
        if spec is None:
            return f"Error: unknown subagent role: {role}"

        max_turns = max(1, min(max_turns, MAX_SUBAGENT_TURNS))
        if self._runner is not None:
            # 应用入口提供统一运行器，子任务也会经过上下文、Hook 和权限管线。
            return await self._runner(description, spec, max_turns)

        logger.info("Sub-agent [%s / %s] dispatched: %s", spec.name, spec.role, description[:80])

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": description},
        ]

        # ── 子智能体循环 ─────────────────────────────────────
        for turn in range(max_turns):
            try:
                response = await self._llm_call(
                    messages,
                    spec.tools,
                    spec.system_prompt,
                    8000,
                )
            except Exception as exc:
                logger.exception("Sub-agent [%s] LLM call failed", spec.name)
                return f"Error: {exc}"

            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use":
                break

            # 执行工具
            results: list[dict[str, Any]] = []
            for block in response.content:
                if getattr(block, "type", None) != "tool_use" and (
                    isinstance(block, dict) and block.get("type") != "tool_use"
                ):
                    continue

                tool_name = block.get("name") if isinstance(block, dict) else block.name
                tool_input = block.get("input", {}) if isinstance(block, dict) else dict(block.input)
                tool_id = block.get("id", "") if isinstance(block, dict) else block.id

                tool = spec.tool_map.get(tool_name)
                if tool is None:
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": f"Unknown tool: {tool_name}",
                    })
                    continue

                try:
                    output = await tool.aexecute(**tool_input)
                except Exception as exc:
                    output = f"Error: {exc}"

                logger.debug("子智能体 [%s] %s → %s", spec.name, tool_name, str(output)[:100])
                results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": output,
                })

            messages.append({"role": "user", "content": results})

        # ── 提取摘要 ────────────────────────────────────────
        summary = _extract_last_assistant_text(messages)

        if not summary:
            summary = f"Sub-agent stopped after {max_turns} turns without final answer."

        logger.info("Sub-agent [%s] done (%d turns)", spec.name, min(turn + 1, max_turns))
        return summary


# ---------------------------------------------------------------------------
#  工具工厂（构建与内置工具匹配的 Tool 实例）
# ---------------------------------------------------------------------------

def _build_base_tools(workdir: Path) -> dict[str, Tool]:
    """返回子智能体的最小工具实例字典。

    这些是内置工具的轻量版本，不依赖 ToolRegistry。
    """
    from coding_agent.tools import build_builtin_tools

    return {tool.name: tool for tool in build_builtin_tools(workdir)}


def _extract_last_assistant_text(messages: list[dict[str, Any]]) -> str:
    """从后往前遍历消息，提取最后一条 assistant 文本。"""
    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in reversed(content):
            tp = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
            if tp == "text":
                text = block.get("text", "") if isinstance(block, dict) else getattr(block, "text", "")
                if text.strip():
                    return text.strip()
    return ""
