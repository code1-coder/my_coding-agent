"""
OpenAI 兼容 Provider。

适用于任何使用 OpenAI Chat Completions 协议的端点
（OpenAI、DeepSeek、GLM、Kimi、Ollama、vLLM 等）。
通过设置 ``base_url`` 来区分不同后端。

由于项目其余部分使用 Anthropic 风格的消息和工具格式，
此 Provider 负责以下转换：
* Anthropic 消息  →  OpenAI 消息
* Anthropic 工具定义 → OpenAI ``tools`` 数组
* OpenAI 响应     →  :class:`ProviderResponse`（Anthropic 风格）
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

from openai import AsyncOpenAI

from coding_agent.providers.base import (
    IncompleteStreamError,
    LLMProvider,
    ProviderResponse,
    StreamEvent,
)


class OpenAICompatProvider(LLMProvider):
    """OpenAI 及 OpenAI 兼容 API 的 LLM 后端。

    使用 ``openai`` SDK。工具定义和消息会自动从
    内部 Anthropic 风格格式转换。
    """

    def __init__(
        self,
        model_id: str = "gpt-5",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        super().__init__(model_id=model_id, api_key=api_key, base_url=base_url)
        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        else:
            # 默认使用 OpenAI 公共端点
            pass
        self._client = AsyncOpenAI(**kwargs)

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
        """转换消息和工具协议，调用兼容接口并返回统一的完整响应。

        非空的 system 会作为首条系统消息插入新列表，不修改原始消息列表。
        """
        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "messages": _convert_messages(messages),
            "max_tokens": max_tokens,
        }
        if system:
            kwargs["messages"] = [{"role": "system", "content": system}] + kwargs["messages"]
        if tools:
            kwargs["tools"] = _convert_tools_in(tools)

        resp = await self._client.chat.completions.create(**kwargs)
        return _oai_to_provider_response(resp)

    # ── 流式调用 ──────────────────────────────────────────────

    async def stream_message(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system: str = "",
        max_tokens: int = 8000,
    ) -> AsyncIterator[StreamEvent]:
        """逐项产出文本和工具参数增量，并在流结束后组装完整响应。

        工具调用按索引累积 ID、名称和 JSON 参数片段，待片段收齐后
        才解析参数；无法解析的参数回退为空字典。最终的 done 事件
        包含停止原因、模型名称及服务端返回的令牌用量。
        """
        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "messages": _convert_messages(messages),
            "max_tokens": max_tokens,
            "stream": True,
        }
        if system:
            kwargs["messages"] = [{"role": "system", "content": system}] + kwargs["messages"]
        if tools:
            kwargs["tools"] = _convert_tools_in(tools)

        stream = await self._client.chat.completions.create(**kwargs)

        # OpenAI 流式传输中 tool-call 是增量到达的 —— 按 index 缓冲累积。
        text_parts: list[str] = []
        tool_call_buf: dict[int, dict[str, Any]] = {}  # 按工具索引保存 id、名称和参数片段
        finish_reason: str | None = None
        model: str = ""
        usage: dict[str, int] = {}

        async with stream:
            async for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                finish_reason = finish_reason or (chunk.choices[0].finish_reason if chunk.choices else None)
                model = model or (chunk.model or "")

                if chunk.usage:
                    usage["input_tokens"] = usage.get("input_tokens", 0) + (chunk.usage.prompt_tokens or 0)
                    usage["output_tokens"] = usage.get("output_tokens", 0) + (chunk.usage.completion_tokens or 0)

                if delta is None:
                    continue

                # 文本增量
                if delta.content:
                    text_parts.append(delta.content)
                    yield StreamEvent(type="text_delta", text=delta.content)

                # tool-call 增量
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_call_buf:
                            tool_call_buf[idx] = {"id": "", "name": "", "args_chunks": []}
                        buf = tool_call_buf[idx]
                        if tc.id:
                            buf["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                buf["name"] = tc.function.name
                                yield StreamEvent(
                                    type="tool_use_start",
                                    tool_id=buf["id"] or f"call_{idx}",
                                    tool_name=tc.function.name,
                                )
                            if tc.function.arguments:
                                buf["args_chunks"].append(tc.function.arguments)
                                yield StreamEvent(
                                    type="tool_input",
                                    tool_id=buf["id"] or f"call_{idx}",
                                    text=tc.function.arguments,
                                )

        if finish_reason is None:
            raise IncompleteStreamError("模型响应流提前结束，未收到 finish_reason。")

        # ── 组装最终 ProviderResponse ──────────────────────────
        content: list[dict[str, Any]] = []
        full_text = "".join(text_parts)
        if full_text:
            content.append({"type": "text", "text": full_text})

        # 解析累积的 tool-call JSON 片段
        for idx in sorted(tool_call_buf):
            buf = tool_call_buf[idx]
            raw_args = "".join(buf["args_chunks"])
            try:
                parsed = json.loads(raw_args) if raw_args else {}
            except json.JSONDecodeError:
                parsed = {}
            content.append({
                "type": "tool_use",
                "id": buf["id"] or f"call_{idx}",
                "name": buf["name"],
                "input": parsed,
            })

        stop_reason = "tool_use" if tool_call_buf else _map_finish_reason(finish_reason)

        yield StreamEvent(
            type="done",
            response=ProviderResponse(
                content=content,
                stop_reason=stop_reason,
                model=model,
                usage=usage,
            ),
        )


# ═══════════════════════════════════════════════════════════════
#  格式转换器
# ═══════════════════════════════════════════════════════════════

def _convert_tools_in(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """将 Anthropic 格式的工具定义 → OpenAI ``tools`` 数组。"""
    result: list[dict[str, Any]] = []
    for t in tools:
        result.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {"type": "object", "properties": {}, "required": []}),
            },
        })
    return result


def _convert_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """将 Anthropic 格式的消息 → OpenAI Chat Completions 格式。

    处理：
    * 纯文本 user / assistant 消息
    * 含 tool_use 内容块的 assistant 消息
    * 含 tool_result 内容块的 user 消息 → ``role: "tool"``
    """
    oai_msgs: list[dict[str, Any]] = []
    for msg in messages:
        role = msg["role"]
        raw_content = msg.get("content")

        # ── user 消息 ─────────────────────────────────────
        if role == "user":
            if isinstance(raw_content, str):
                oai_msgs.append({"role": "user", "content": raw_content})
            elif isinstance(raw_content, list):
                # 检测 tool_result 块
                tool_results = [
                    b for b in raw_content
                    if isinstance(b, dict) and b.get("type") == "tool_result"
                ]
                if tool_results:
                    for tr in tool_results:
                        oai_msgs.append({
                            "role": "tool",
                            "tool_call_id": tr["tool_use_id"],
                            "content": tr["content"],
                        })
                else:
                    # 纯 user 消息 —— 拼接 text 块
                    text = "".join(
                        b.get("text", "")
                        for b in raw_content
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                    oai_msgs.append({"role": "user", "content": text})
            continue

        # ── assistant 消息 ────────────────────────────────
        if role == "assistant":
            if isinstance(raw_content, str):
                oai_msgs.append({"role": "assistant", "content": raw_content})
            elif isinstance(raw_content, list):
                text_parts = [
                    b["text"] for b in raw_content
                    if isinstance(b, dict) and b.get("type") == "text"
                ]
                tool_blocks = [
                    b for b in raw_content
                    if isinstance(b, dict) and b.get("type") == "tool_use"
                ]

                oai_msg: dict[str, Any] = {"role": "assistant"}
                if text_parts:
                    oai_msg["content"] = "".join(text_parts)
                else:
                    oai_msg["content"] = None  # type: ignore[assignment]
                if tool_blocks:
                    oai_msg["tool_calls"] = [
                        {
                            "id": tb["id"],
                            "type": "function",
                            "function": {
                                "name": tb["name"],
                                "arguments": json.dumps(tb["input"], ensure_ascii=False),
                            },
                        }
                        for tb in tool_blocks
                    ]
                oai_msgs.append(oai_msg)
            continue

        # ── tool 消息（Anthropic 无此 role，但做安全兜底）──
        if role == "tool":
            oai_msgs.append(msg)
            continue

    return oai_msgs


def _oai_to_provider_response(resp: Any) -> ProviderResponse:
    """将 OpenAI ChatCompletion 转换为 :class:`ProviderResponse`。"""
    choice = resp.choices[0]
    content: list[dict[str, Any]] = []

    if choice.message.content:
        content.append({"type": "text", "text": choice.message.content})

    if choice.message.tool_calls:
        for tc in choice.message.tool_calls:
            try:
                inp = json.loads(tc.function.arguments) if tc.function.arguments else {}
            except json.JSONDecodeError:
                inp = {}
            content.append({
                "type": "tool_use",
                "id": tc.id,
                "name": tc.function.name,
                "input": inp,
            })

    has_tools = bool(choice.message.tool_calls)
    stop_reason = "tool_use" if has_tools else _map_finish_reason(choice.finish_reason)

    usage = {}
    if resp.usage:
        usage = {
            "input_tokens": resp.usage.prompt_tokens or 0,
            "output_tokens": resp.usage.completion_tokens or 0,
        }

    return ProviderResponse(
        content=content,
        stop_reason=stop_reason,
        model=resp.model or "",
        usage=usage,
    )


def _map_finish_reason(reason: str | None) -> str:
    """将 OpenAI finish_reason 映射为 Anthropic stop_reason。"""
    if reason is None:
        return "end_turn"
    mapping = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "end_turn",
    }
    return mapping.get(reason, reason or "end_turn")
