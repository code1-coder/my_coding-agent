"""内置工具集。"""

from coding_agent.tools.base import Tool, ToolResult, tool_result
from coding_agent.tools.builtin import (
    BashTool,
    EditFileTool,
    GlobTool,
    ReadFileTool,
    WriteFileTool,
)

__all__ = [
    "Tool",
    "ToolResult",
    "tool_result",
    "BashTool",
    "ReadFileTool",
    "WriteFileTool",
    "EditFileTool",
    "GlobTool",
]

# 预构建的五个内置工具列表（使用默认工作目录）。
# 如需自定义工作目录，请自行实例化各个工具。
BUILTIN_TOOLS: list[Tool] = [
    BashTool(),
    ReadFileTool(),
    WriteFileTool(),
    EditFileTool(),
    GlobTool(),
]
