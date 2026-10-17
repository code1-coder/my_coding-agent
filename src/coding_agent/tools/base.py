"""
工具基类。

定义了 Tool（所有工具的抽象基类）和 ToolResult（工具执行的标准化返回值）。
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    """工具执行的标准结果。

    属性:
        tool_use_id: 触发本次执行的 tool_use 块的 ID。
        content: 工具的输出文本（未处理错误时为空字符串）。
        error: 如果不为 None，表示工具以此错误消息失败。
    """

    tool_use_id: str
    content: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """转换为模型协议所需的工具结果，失败时附上 is_error 标记。"""
        d: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": self.tool_use_id,
            "content": self.error if self.error else self.content,
        }
        if self.error:
            d["is_error"] = True
        return d


class Tool(ABC):
    """智能体可调用的工具的抽象基类。

    子类必须设置类级别属性并实现 ``execute`` 方法::

        class BashTool(Tool):
            name = "bash"
            description = "运行一个 shell 命令。"
            input_schema = {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            }

            def execute(self, command: str) -> str: ...
    """

    # ── 子类必须重写以下属性 ──────────────────────────────────
    name: str = ""
    description: str = ""
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    # ── Provider 格式序列化 ───────────────────────────────────

    def to_anthropic_spec(self) -> dict[str, Any]:
        """返回 Anthropic Messages 格式的工具定义。"""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def to_openai_spec(self) -> dict[str, Any]:
        """返回 OpenAI Chat Completions 格式的工具定义。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }

    # ── 执行 ───────────────────────────────────────────────────

    @abstractmethod
    def execute(self, **kwargs: Any) -> str:
        """同步执行工具。

        返回:
            工具输出字符串。抛出异常以表示不可恢复的错误
            （框架会捕获并包装异常）。
        """
        ...

    async def aexecute(self, **kwargs: Any) -> str:
        """异步包装器 —— 默认在线程池中运行 ``execute``。

        如果工具有原生异步实现，可以重写此方法。
        """
        return await asyncio.to_thread(self.execute, **kwargs)


def tool_result(
    tool_use_id: str,
    content: str = "",
    *,
    error: str | None = None,
) -> dict[str, Any]:
    """快速构建 Anthropic 格式的 tool_result 字典。

    >>> tool_result("id_1", "hello")
    {'type': 'tool_result', 'tool_use_id': 'id_1', 'content': 'hello'}
    """
    d: dict[str, Any] = {"type": "tool_result", "tool_use_id": tool_use_id}
    if error:
        d["content"] = error
        d["is_error"] = True
    else:
        d["content"] = content
    return d
