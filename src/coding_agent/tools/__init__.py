"""内置工具集。"""

from pathlib import Path

from coding_agent.tools.base import Tool, ToolResult, tool_result
from coding_agent.tools.builtin import (
    BashTool,
    EditFileTool,
    GlobTool,
    ReadFileTool,
    WriteFileTool,
)
from coding_agent.tools.code import CodeOutlineTool, GitDiffTool, GitStatusTool, SearchCodeTool
from coding_agent.tools.patch import ApplyPatchTool

__all__ = [
    "Tool",
    "ToolResult",
    "tool_result",
    "BashTool",
    "ReadFileTool",
    "WriteFileTool",
    "EditFileTool",
    "GlobTool",
    "SearchCodeTool",
    "CodeOutlineTool",
    "GitStatusTool",
    "GitDiffTool",
    "ApplyPatchTool",
    "build_builtin_tools",
]


def build_builtin_tools(workdir: Path | str | None = None) -> list[Tool]:
    """为主会话及子智能体构建相同的工作区工具实例。"""
    return [tool_type(workdir=workdir) for tool_type in (
        BashTool, ReadFileTool, WriteFileTool, EditFileTool, GlobTool,
        SearchCodeTool, CodeOutlineTool, GitStatusTool, GitDiffTool, ApplyPatchTool,
    )]


BUILTIN_TOOLS: list[Tool] = build_builtin_tools()
