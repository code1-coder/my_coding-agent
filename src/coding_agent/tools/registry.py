"""
工具注册表 —— 基于装饰器注册，自动生成 JSON Schema。

两种注册方式：

1. **装饰器**（推荐用于新编写的纯函数工具）::

       registry = ToolRegistry()

       @registry.register(
           name="bash",
           description="在工作区运行 shell 命令",
           permission="execute",
           risk_level="moderate",
       )
       async def bash(command: str, timeout: int = 120) -> str: ...

2. **适配器**（适用于已有的 ``Tool`` 子类实例）::

       from coding_agent.tools.builtin import BashTool
       registry.register_tool(BashTool(workdir=Path.cwd()))

注册表会根据 Python 类型注解自动生成 JSON Schema，
因此你无需手写 ``input_schema``。
"""

from __future__ import annotations

import asyncio
import inspect
import sys
import typing
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, get_args, get_origin

from coding_agent.tools.base import Tool, ToolResult

# Python 3.10+ 联合类型
if sys.version_info >= (3, 10):
    import types as _stdlib_types  # noqa: E402
    _UNION_TYPES = (typing.Union, _stdlib_types.UnionType)  # type: ignore[attr-defined]
else:
    _UNION_TYPES = (typing.Union,)  # type: ignore[misc]


# ---------------------------------------------------------------------------
#  ToolEntry —— 已注册工具的元数据容器
# ---------------------------------------------------------------------------

@dataclass
class ToolEntry:
    """一个已注册工具的元数据 + 可调用对象。"""

    name: str
    description: str
    fn: Callable[..., Any]
    permission: str = "read"          # 权限类型：读取、写入或执行
    risk_level: str = "low"           # 风险等级：低、中、高或严重
    input_schema: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
#  注册表
# ---------------------------------------------------------------------------

class ToolRegistry:
    """基于装饰器的工具注册表。

    典型用法::

        registry = ToolRegistry()

        @registry.register(name="echo", description="返回问候语。")
        async def echo(message: str) -> str:
            return f"echo: {message}"

        # 喂给 LLM 的工具列表
        tools_for_llm = registry.list_tools()

        # O(1) 分发
        result = await registry.dispatch("echo", {"message": "hi"})
    """

    def __init__(self) -> None:
        self._tools: OrderedDict[str, ToolEntry] = OrderedDict()

    # ── 装饰器 API ──────────────────────────────────────────

    def register(
        self,
        *,
        name: str,
        description: str = "",
        permission: str = "read",
        risk_level: str = "low",
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """装饰器：将普通 async/sync 函数注册为工具。

        函数的类型注解会自动用于生成 JSON Schema。
        没有默认值的参数视为必填参数。
        """
        def _decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            schema = _fn_to_json_schema(fn)
            self._tools[name] = ToolEntry(
                name=name,
                description=description,
                fn=fn,
                permission=permission,
                risk_level=risk_level,
                input_schema=schema,
            )
            return fn

        return _decorator

    # ── 适配器 API ──────────────────────────────────────────

    def register_tool(self, tool: Tool) -> ToolEntry:
        """包装已有的 ``Tool`` 子类实例。

        工具的 ``aexecute`` 在分发时被调用，因此 Tool 自身的
        ``execute`` → ``aexecute`` 线程池逻辑得以保留。
        """
        entry = ToolEntry(
            name=tool.name,
            description=tool.description,
            fn=tool.aexecute,
            permission="write",
            risk_level="moderate",
            input_schema=tool.input_schema,
        )
        self._tools[tool.name] = entry
        return entry

    # ── 查询 API ────────────────────────────────────────────

    def register_handler(
        self, definition: dict[str, Any], handler: Callable[..., Any],
    ) -> ToolEntry:
        """注册已有协议定义及其处理器，供 MCP 等动态工具使用。"""
        entry = ToolEntry(
            name=definition["name"],
            description=definition.get("description", ""),
            fn=handler,
            input_schema=definition.get("input_schema", {}),
        )
        self._tools[entry.name] = entry
        return entry

    def list_tools(self) -> list[dict[str, Any]]:
        """返回 Anthropic 格式的工具定义，可直接喂给 LLM。

        >>> reg.list_tools()
        [{"name": "bash", "description": "...", "input_schema": {...}}, ...]
        """
        return [
            {
                "name": e.name,
                "description": e.description,
                "input_schema": e.input_schema,
            }
            for e in self._tools.values()
        ]

    def get(self, name: str) -> ToolEntry | None:
        """按名称查找单个工具的元数据（未注册返回 None）。"""
        return self._tools.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __iter__(self):
        return iter(self._tools)

    def __len__(self) -> int:
        return len(self._tools)

    # ── 分发 ─────────────────────────────────────────────────

    async def dispatch(self, name: str, args: dict[str, Any], *, tool_use_id: str = "") -> ToolResult:
        """O(1) 查找并执行。

        参数:
            name: 工具名称。
            args: 从 LLM 响应中提取的关键字参数。
            tool_use_id: LLM 响应块中的 ``tool_use.id``。

        返回:
            ``ToolResult``（包装函数的返回值或错误信息）。

        异常:
            KeyError: 如果 *name* 未注册。
        """
        entry = self._tools.get(name)
        if entry is None:
            raise KeyError(f"Unknown tool: {name}")

        try:
            if inspect.iscoroutinefunction(entry.fn):
                output = await entry.fn(**args)
            else:
                output = await asyncio.to_thread(entry.fn, **args)
        except Exception as exc:
            return ToolResult(tool_use_id=tool_use_id, error=str(exc))

        # 标准化返回：str → ToolResult
        if isinstance(output, ToolResult):
            if not output.tool_use_id:
                output.tool_use_id = tool_use_id
            return output
        return ToolResult(tool_use_id=tool_use_id, content=str(output))


# ---------------------------------------------------------------------------
#  JSON Schema 构建器 —— 从 Python 类型注解
# ---------------------------------------------------------------------------

_TYPE_MAP: dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    type(None): "null",
}


def _fn_to_json_schema(fn: Callable[..., Any]) -> dict[str, Any]:
    """从函数签名和类型注解中提取 JSON Schema。"""
    try:
        hints = typing.get_type_hints(fn)
    except Exception:
        hints = {}

    # Python 3.10+ ``inspect.signature`` 支持新式联合类型
    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        return {"type": "object", "properties": {}, "required": []}

    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        json_type = _resolve_schema(param_name, param, hints)
        properties[param_name] = json_type

        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def _resolve_schema(
    name: str,
    param: inspect.Parameter,
    hints: dict[str, Any],
) -> dict[str, Any]:
    """为 *param* 构建单个属性的 JSON Schema 片段。"""
    annot = hints.get(name)

    # 1) 无注解 —— 回退到默认值类型或 string
    if annot is None:
        if param.default is not inspect.Parameter.empty:
            return _python_val_to_schema(param.default)
        return {"type": "string"}

    # 2) 有注解 —— 递归遍历类型树
    schema = _type_to_schema(annot)

    # 3) 从 docstring 添加描述？（暂略 —— 保持最小化）
    return schema


def _type_to_schema(tp: Any) -> dict[str, Any]:
    """递归地将类型转为 JSON Schema。"""
    origin = get_origin(tp)
    args = get_args(tp)

    # ── 简单类型 ────────────────────────────────────────────
    if origin is None:
        if tp in _TYPE_MAP:
            return {"type": _TYPE_MAP[tp]}
        # 回退
        return {"type": "string"}

    # ── Optional[X]  或  X | None ──────────────────────────
    if origin in _UNION_TYPES:
        non_none = [a for a in args if a is not type(None)]  # noqa: E721
        if len(non_none) == 1:
            # Optional[X] → 直接转为 X（工具参数在 API 层总是可选的）
            return _type_to_schema(non_none[0])
        # 真正的联合类型 —— 使用 anyOf
        return {"anyOf": [_type_to_schema(a) for a in args]}

    # ── 枚举类型 Literal["a", "b"] ──────────────────────────────────
    if origin is Literal:
        return {"enum": list(args), "type": _infer_literal_type(args)}

    # ── 序列类型 list[X] / Sequence[X] ──────────────────────────────
    if origin in (list, typing.List):
        items = _type_to_schema(args[0]) if args else {"type": "string"}
        return {"type": "array", "items": items}

    # ── 字典类型 dict[str, X] ───────────────────────────────────────
    if origin in (dict, typing.Dict):
        return {"type": "object"}

    # ── 带附加元数据的类型 typing.Annotated[X, ...] ───────────────────────────
    if origin is typing.Annotated:
        return _type_to_schema(args[0])

    return {"type": "string"}


def _python_val_to_schema(val: Any) -> dict[str, Any]:
    """从默认值推断 JSON Schema 类型。"""
    if isinstance(val, bool):
        return {"type": "boolean"}
    if isinstance(val, int):
        return {"type": "integer"}
    if isinstance(val, float):
        return {"type": "number"}
    if isinstance(val, str):
        return {"type": "string"}
    if isinstance(val, list):
        item_schema = _python_val_to_schema(val[0]) if val else {"type": "string"}
        return {"type": "array", "items": item_schema}
    if val is None:
        return {"type": "null"}
    return {"type": "string"}


def _infer_literal_type(values: tuple[Any, ...]) -> str | None:
    """返回能匹配所有 *values* 的 JSON Schema 类型字符串。"""
    types = {type(v) for v in values}
    if types <= {str}:
        return "string"
    if types <= {int}:
        return "integer"
    if types <= {int, float}:
        return "number"
    if types <= {bool}:
        return "boolean"
    return None  # 混合类型 —— 省略 "type"
