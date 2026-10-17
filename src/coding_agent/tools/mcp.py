"""
MCP（Model Context Protocol）集成。

MCPClient
    包装与一个 MCP 服务器的连接。连接时发现工具，
    并以命名约定::

        mcp__{server}__{tool}

    暴露这些工具。

assemble_tool_pool
    将内置工具 + 所有已连接 MCP 服务器工具合并为
    一个统一列表（兼容 ``ToolRegistry``）。

用法::

    from coding_agent.tools.mcp import MCPClient, assemble_tool_pool

    client = MCPClient("docs")
    client.register(
        tool_defs=[{"name": "search", ...}],
        handlers={"search": lambda query: f"找到: {query}"},
    )

    tools, handlers = assemble_tool_pool(builtin_tools, builtin_handlers, [client])
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  名称规范化
# ---------------------------------------------------------------------------

_DISALLOWED_CHARS = re.compile(r"[^a-zA-Z0-9_-]")


def normalize_mcp_name(name: str) -> str:
    """将 ``[a-zA-Z0-9_-]`` 之外的字符替换为下划线。"""
    return _DISALLOWED_CHARS.sub("_", name)


# ---------------------------------------------------------------------------
#  MCP 客户端（MCPClient）
# ---------------------------------------------------------------------------

Handler = Callable[..., str]


class MCPClient:
    """表示一个已连接的 MCP 服务器。

    教学版本使用 mock 处理器注册表。生产环境会使用 ``mcp`` SDK，
    通过 stdio / HTTP 调用 ``tools/list`` 发现工具以及 ``tools/call`` 调用工具。
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.tools: list[dict[str, Any]] = []
        self._handlers: dict[str, Handler] = {}

    def register(
        self,
        tool_defs: list[dict[str, Any]],
        handlers: dict[str, Handler],
    ) -> None:
        """注册一个或多个工具及其处理函数。

        参数:
            tool_defs: 工具定义列表（必须包含 ``name``、
                       ``description`` 和 ``inputSchema``）。
            handlers: ``{tool_name: callable}`` 映射。
        """
        self.tools = tool_defs
        self._handlers = handlers

    def call_tool(self, tool_name: str, args: dict[str, Any]) -> str:
        """按名称调用已注册的工具。

        返回:
            工具的输出字符串，或 MCP 错误消息。
        """
        handler = self._handlers.get(tool_name)
        if handler is None:
            return f"MCP error: unknown tool '{tool_name}'"
        try:
            return handler(**args)
        except Exception as exc:
            return f"MCP error: {exc}"

    @property
    def tool_names(self) -> list[str]:
        return [t["name"] for t in self.tools]

    def __repr__(self) -> str:
        return f"MCPClient({self.name!r}, 工具数={len(self.tools)})"


# ---------------------------------------------------------------------------
#  Mock MCP 服务器（用于教学 / 无真实 MCP 基础设施时的测试）
# ---------------------------------------------------------------------------

_MOCK_REGISTRY: dict[str, Callable[[], MCPClient]] = {}


def register_mock_server(name: str, factory: Callable[[], MCPClient]) -> None:
    """以 *name* 注册一个 mock MCP 服务器工厂函数。"""
    _MOCK_REGISTRY[name] = factory


def connect_mcp(name: str) -> tuple[str, MCPClient | None]:
    """连接到 MCP 服务器（mock 或真实）。

    返回:
        ``(状态消息, client_or_None)``。
    """
    factory = _MOCK_REGISTRY.get(name)
    if factory is None:
        available = ", ".join(_MOCK_REGISTRY.keys()) or "(none)"
        return f"Unknown server '{name}'. Available: {available}", None

    client = factory()
    tool_names = client.tool_names
    logger.info("MCP connected: %s → %s", name, tool_names)
    return (
        f"Connected to MCP server '{name}'. "
        f"Discovered {len(client.tools)} tools: {', '.join(tool_names)}",
        client,
    )


# ---------------------------------------------------------------------------
#  工具池组装
# ---------------------------------------------------------------------------


def assemble_tool_pool(
    builtin_tools: list[dict[str, Any]],
    builtin_handlers: dict[str, Any],
    mcp_clients: list[MCPClient] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """将内置工具与 MCP 发现的工具合并。

    MCP 工具添加前缀: ``mcp__{server}__{tool}``
    （server 和 tool 名称均通过 :func:`normalize_mcp_name` 规范化）。

    参数:
        builtin_tools: Anthropic 格式的工具定义。
        builtin_handlers: 内置工具的 ``{name: callable}`` 映射。
        mcp_clients: 已连接的 MCP 客户端。

    返回:
        ``(tools, handlers)`` —— 统一后的工具池。
    """
    tools = list(builtin_tools)
    handlers = dict(builtin_handlers)

    for client in (mcp_clients or []):
        safe_server = normalize_mcp_name(client.name)
        for tool_def in client.tools:
            safe_tool = normalize_mcp_name(tool_def["name"])
            prefixed = f"mcp__{safe_server}__{safe_tool}"

            tools.append({
                "name": prefixed,
                "description": tool_def.get("description", ""),
                "input_schema": tool_def.get("inputSchema", {}),
            })

            # 用闭包捕获（默认参数技巧避免延迟绑定）
            handlers[prefixed] = _make_mcp_handler(client, tool_def["name"])

    return tools, handlers


def _make_mcp_handler(client: MCPClient, tool_name: str) -> Handler:
    """构建一个闭包，调用 *client* 执行 *tool_name*。"""
    def _handler(**kwargs: Any) -> str:
        return client.call_tool(tool_name, kwargs)
    _handler.__name__ = f"mcp__{client.name}__{tool_name}"  # 便于调试
    return _handler


# ---------------------------------------------------------------------------
#  注册内置 mock 服务器（用于演示 / 测试）
# ---------------------------------------------------------------------------

def _mock_server_docs() -> MCPClient:
    c = MCPClient("docs")
    c.register(
        tool_defs=[
            {
                "name": "search",
                "description": "搜索文档。（只读）",
                "inputSchema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
            {
                "name": "get_version",
                "description": "获取 API 版本。（只读）",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        ],
        handlers={
            "search": lambda query: f"[docs] Found 3 results for '{query}'",
            "get_version": lambda: "[docs] API v2.1.0",
        },
    )
    return c


def _mock_server_deploy() -> MCPClient:
    c = MCPClient("deploy")
    # 注册模拟部署工具
    c.register(
        tool_defs=[
            {
                "name": "trigger",
                "description": "触发部署。（破坏性操作）",
                "inputSchema": {
                    "type": "object",
                    "properties": {"service": {"type": "string"}},
                    "required": ["service"],
                },
            },
            {
                "name": "status",
                "description": "检查部署状态。（只读）",
                "inputSchema": {
                    "type": "object",
                    "properties": {"service": {"type": "string"}},
                    "required": ["service"],
                },
            },
        ],
        handlers={
            "trigger": lambda service: f"[deploy] Triggered: {service}",
            "status": lambda service: f"[deploy] {service}: running (v1.4.2)",
        },
    )
    return c


register_mock_server("docs", _mock_server_docs)
register_mock_server("deploy", _mock_server_deploy)
