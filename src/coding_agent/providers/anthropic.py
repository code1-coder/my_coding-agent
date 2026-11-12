"""
Anthropic Provider —— ``anthropic.AsyncAnthropic`` 的轻量封装。

消息和工具格式均为 Anthropic 原生格式，因此无需转换。
"""

from __future__ import annotations

from typing import Any, AsyncIterator

from anthropic import AsyncAnthropic

from coding_agent.providers.activity import observe_http_client
from coding_agent.providers.base import (
    LLMProvider,
    ProviderResponse,
    StreamEvent,
)


class AnthropicProvider(LLMProvider):
    """Anthropic 模型（Claude 系列）的 LLM 后端。

    使用原生 Anthropic Messages API，无需格式转换。
    """

    def __init__(
        self,
        model_id: str = "claude-sonnet-5",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        super().__init__(model_id=model_id, api_key=api_key, base_url=base_url)
        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self._client = AsyncAnthropic(**kwargs)
        self.http_activity_supported = observe_http_client(self._client)

    async def aclose(self) -> None:
        """在所属事件循环关闭前释放 SDK 的异步连接池。"""
        await self._client.close()

    # ── 非流式调用 ────────────────────────────────────────────

    async def create_message(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system: str = "",
        max_tokens: int = 8000,
    ) -> ProviderResponse:
        """发送完整对话与工具定义，并将原生响应转为统一的 ProviderResponse。

        system 和 tools 仅在非空时传给 SDK；请求异常由调用方处理。
        """
        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools

        resp = await self._client.messages.create(**kwargs)
        return _to_provider_response(resp)

    # ── 流式调用 ──────────────────────────────────────────────

    async def stream_message(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system: str = "",
        max_tokens: int = 8000,
    ) -> AsyncIterator[StreamEvent]:
        """将 SDK 流中的文本、工具开始和参数增量转换为统一事件。

        流结束后额外产出携带完整响应的 done 事件，供智能体记录消息
        并判断是否继续调用工具。
        """
        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools

        tool_ids: dict[int, str] = {}
        async with self._client.messages.stream(**kwargs) as stream:
            async for ev in stream:
                if ev.type == "content_block_start":
                    if ev.content_block.type == "tool_use":
                        tool_ids[ev.index] = ev.content_block.id
                        yield StreamEvent(
                            type="tool_use_start",
                            tool_id=ev.content_block.id,
                            tool_name=ev.content_block.name,
                        )
                elif ev.type == "content_block_delta":
                    if ev.delta.type == "text_delta":
                        yield StreamEvent(type="text_delta", text=ev.delta.text)
                    elif ev.delta.type == "input_json_delta":
                        yield StreamEvent(type="tool_input", tool_id=tool_ids.get(ev.index, ""),
                                          text=ev.delta.partial_json)

            final = await stream.get_final_message()
            yield StreamEvent(type="done", response=_to_provider_response(final))


# ═══════════════════════════════════════════════════════════════
#  辅助函数
# ═══════════════════════════════════════════════════════════════

def _to_provider_response(msg: Any) -> ProviderResponse:
    """将 Anthropic SDK Message 转换为我们的 ``ProviderResponse``。"""
    content: list[dict[str, Any]] = []
    for block in msg.content:
        if block.type == "text":
            content.append({"type": "text", "text": block.text})
        elif block.type == "tool_use":
            content.append({
                "type": "tool_use",
                "id": block.id,
                "name": block.name,
                "input": dict(block.input),
            })

    usage = {
        "input_tokens": getattr(msg.usage, "input_tokens", 0),
        "output_tokens": getattr(msg.usage, "output_tokens", 0),
    }

    return ProviderResponse(
        content=content,
        stop_reason=msg.stop_reason,
        model=msg.model,
        usage=usage,
    )
