"""
Provider 抽象层。

定义了 ``ProviderResponse``（所有 LLM 调用的统一返回格式）以及
``LLMProvider``（每个后端都必须实现的抽象接口）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Literal


class ModelRequestError(RuntimeError):
    """可向用户展示的模型请求失败。"""


class ModelIdleTimeout(ModelRequestError):
    """模型尝试在空闲期限内未观察到协议或 HTTP 活动。"""


class IncompleteStreamError(ModelRequestError):
    """模型流未提供完整结束标记。"""


# ═══════════════════════════════════════════════════════════════
#  响应类型（跨 Provider 统一格式）
# ═══════════════════════════════════════════════════════════════

@dataclass
class ProviderResponse:
    """``LLMProvider.create_message`` 的统一返回格式。

    ``content`` 是一个字典列表，采用 *Anthropic content-block* 结构::

        {"type": "text", "text": "你好"}
        {"type": "tool_use", "id": "toolu_...", "name": "bash", "input": {...}}

    这也是 agent loop 内部 ``messages`` 列表使用的格式，
    因此内容块可以直接追加。
    """

    content: list[dict[str, Any]]
    stop_reason: str  # 停止原因：轮次结束、工具调用、达到令牌上限或命中停止序列等
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)

    # ── 辅助属性 ─────────────────────────────────────────────

    @property
    def text(self) -> str:
        """所有文本块的拼接结果。"""
        return "".join(b["text"] for b in self.content if b["type"] == "text")

    @property
    def tool_use_blocks(self) -> list[dict[str, Any]]:
        """此响应中的所有 tool_use 块。"""
        return [b for b in self.content if b["type"] == "tool_use"]


@dataclass
class StreamEvent:
    """流式传输过程中发出的单个事件。

    根据 ``type`` 的不同，使用的字段也不同：

    ================ ===========================================
    type             填充的字段
    ================ ===========================================
    ``text_delta``    ``text``
    ``tool_use_start`` ``tool_id``, ``tool_name``
    ``tool_input``    ``tool_id``, ``text``（部分 JSON）
    ``done``          ``response`` (:class:`ProviderResponse`)
    ================ ===========================================
    """

    type: Literal["text_delta", "tool_use_start", "tool_input", "done"]
    text: str = ""
    tool_id: str = ""
    tool_name: str = ""
    response: ProviderResponse | None = None


# ═══════════════════════════════════════════════════════════════
#  抽象接口
# ═══════════════════════════════════════════════════════════════

class LLMProvider(ABC):
    """LLM 后端的抽象接口。

    每个 Provider 接受：
    * **messages** —— Anthropic 格式的消息列表（内容块为字典）
    * **tools** —— Anthropic 格式的工具定义
    * **system** —— 系统提示词字符串

    内部使用不同原生格式的 Provider（如 OpenAI）需要自行负责转换。

    用法::

        provider = AnthropicProvider(model_id="claude-sonnet-5")
        response = await provider.create_message(messages, tools, system)

        async for ev in provider.stream_message(messages, tools, system):
            if ev.type == "text_delta":
                print(ev.text, end="")
            elif ev.type == "done":
                final = ev.response
    """

    def __init__(self, model_id: str, api_key: str | None = None, base_url: str | None = None) -> None:
        self.model_id = model_id
        self.api_key = api_key
        self.base_url = base_url
        self.http_activity_supported = False

    async def aclose(self) -> None:
        """释放异步资源；持有网络客户端的后端应覆盖此方法。"""

    # ── 子类必须实现以下方法 ───────────────────────────────────

    @abstractmethod
    async def create_message(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system: str = "",
        max_tokens: int = 8000,
    ) -> ProviderResponse:
        """非流式调用 —— 一次性返回完整响应。"""
        ...

    @abstractmethod
    async def stream_message(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system: str = "",
        max_tokens: int = 8000,
    ) -> AsyncIterator[StreamEvent]:
        """流式调用 —— 逐项产出 :class:`StreamEvent`。

        最后一个事件的 type 始终为 ``"done"``，且 ``response`` 字段已填充。
        """
        ...
