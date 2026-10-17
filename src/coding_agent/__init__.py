"""
coding-agent —— 一个模块化的 AI 编码智能体，支持多家 LLM 后端。

快速上手::

    from coding_agent.config import load_config
    from coding_agent.runtime import AgentRuntime

    cfg = load_config()
    async with AgentRuntime(cfg) as runtime:
        async for ev in runtime.run_turn("列出当前项目的文件"):
            if ev.type == "text_delta":
                print(ev.text, end="")
"""

from coding_agent.config import Config, load_config
from coding_agent.loop import AgentLoop, LoopEvent
from coding_agent.providers import (
    AnthropicProvider,
    LLMProvider,
    OpenAICompatProvider,
)
from coding_agent.tools import BUILTIN_TOOLS, Tool, ToolResult
from coding_agent.runtime import AgentRuntime

__all__ = [
    "Config",
    "load_config",
    "AgentLoop",
    "AgentRuntime",
    "LoopEvent",
    "LLMProvider",
    "AnthropicProvider",
    "OpenAICompatProvider",
    "Tool",
    "ToolResult",
    "BUILTIN_TOOLS",
]
