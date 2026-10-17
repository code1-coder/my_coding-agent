"""
智能体循环测试：使用模拟模型服务，避免调用真实 API。
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

import pytest

from coding_agent.loop import AgentLoop, LoopEvent
from coding_agent.providers.base import (
    LLMProvider,
    ProviderResponse,
    StreamEvent,
)
from coding_agent.tools.base import Tool


# ═══════════════════════════════════════════════════════════════
#  测试用占位工具
# ═══════════════════════════════════════════════════════════════

class EchoTool(Tool):
    name = "echo"
    description = "Echo the input back."
    input_schema = {
        "type": "object",
        "properties": {"message": {"type": "string"}},
        "required": ["message"],
    }

    def execute(self, message: str) -> str:
        return f"echo: {message}"


# ═══════════════════════════════════════════════════════════════
#  模拟模型服务
# ═══════════════════════════════════════════════════════════════

class MockProvider(LLMProvider):
    """按预设顺序返回响应的模拟模型服务。"""

    def __init__(self, responses: list[ProviderResponse]) -> None:
        super().__init__(model_id="mock")
        self.responses = responses
        self.call_count = 0

    async def create_message(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system: str = "",
        max_tokens: int = 8000,
    ) -> ProviderResponse:
        resp = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return resp

    async def stream_message(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system: str = "",
        max_tokens: int = 8000,
    ) -> AsyncIterator[StreamEvent]:
        resp = await self.create_message(messages, tools, system, max_tokens)
        # 为每个文本块逐项产出增量文本
        for block in resp.content:
            if block["type"] == "text":
                # 逐字产出，以模拟真实的流式响应
                for ch in block["text"]:
                    yield StreamEvent(type="text_delta", text=ch)
                    await asyncio.sleep(0)
        yield StreamEvent(type="done", response=resp)


# ═══════════════════════════════════════════════════════════════
#  辅助函数
# ═══════════════════════════════════════════════════════════════

async def _collect_events(loop: AgentLoop, messages: list[dict]) -> list[LoopEvent]:
    events: list[LoopEvent] = []
    async for ev in loop.run(messages, stream=True):
        events.append(ev)
    return events


# ═══════════════════════════════════════════════════════════════
#  测试用例
# ═══════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_stops_on_end_turn():
    """模型返回 stop_reason='end_turn' 时结束循环。"""
    provider = MockProvider([
        ProviderResponse(
            content=[{"type": "text", "text": "Hello!"}],
            stop_reason="end_turn",
        ),
    ])
    loop = AgentLoop(provider=provider, tools=[EchoTool()])
    messages: list[dict] = [{"role": "user", "content": "hi"}]

    events = await _collect_events(loop, messages)

    text_output = "".join(e.text for e in events if e.type == "text_delta")
    assert text_output == "Hello!"
    assert events[-1].type == "done"
    assert provider.call_count == 1


@pytest.mark.asyncio
async def test_single_tool_call():
    """模型调用一次工具后结束当前任务。"""
    provider = MockProvider([
        # 第一轮：调用 echo 工具
        ProviderResponse(
            content=[{
                "type": "tool_use",
                "id": "toolu_01",
                "name": "echo",
                "input": {"message": "world"},
            }],
            stop_reason="tool_use",
        ),
        # 第二轮：返回最终文本
        ProviderResponse(
            content=[{"type": "text", "text": "Done."}],
            stop_reason="end_turn",
        ),
    ])
    loop = AgentLoop(provider=provider, tools=[EchoTool()])
    messages: list[dict] = [{"role": "user", "content": "echo world"}]

    events = await _collect_events(loop, messages)

    # 检查消息列表是否按预期构建
    assert len(messages) == 4  # 用户输入、助手工具调用、用户工具结果、助手最终响应
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"][0]["type"] == "tool_use"
    assert messages[2]["role"] == "user"  # 工具执行结果
    assert messages[2]["content"][0]["type"] == "tool_result"
    assert "echo: world" in messages[2]["content"][0]["content"]

    text_output = "".join(e.text for e in events if e.type == "text_delta")
    assert text_output == "Done."
    assert events[-1].type == "done"
    assert provider.call_count == 2


@pytest.mark.asyncio
async def test_concurrent_tool_execution():
    """同一轮中的多个工具调用应并发执行。"""
    provider = MockProvider([
        ProviderResponse(
            content=[
                {
                    "type": "tool_use",
                    "id": "toolu_a",
                    "name": "echo",
                    "input": {"message": "first"},
                },
                {
                    "type": "tool_use",
                    "id": "toolu_b",
                    "name": "echo",
                    "input": {"message": "second"},
                },
            ],
            stop_reason="tool_use",
        ),
        ProviderResponse(
            content=[{"type": "text", "text": "ok"}],
            stop_reason="end_turn",
        ),
    ])
    loop = AgentLoop(provider=provider, tools=[EchoTool()])
    messages: list[dict] = [{"role": "user", "content": "echo two things"}]

    await _collect_events(loop, messages)

    # 两个工具结果都应存在
    results = messages[2]["content"]
    assert len(results) == 2
    contents = {r["content"] for r in results}
    assert contents == {"echo: first", "echo: second"}


@pytest.mark.asyncio
async def test_unknown_tool():
    """模型调用未注册的工具时，应返回错误结果。"""
    provider = MockProvider([
        ProviderResponse(
            content=[{
                "type": "tool_use",
                "id": "toolu_x",
                "name": "nonexistent",
                "input": {},
            }],
            stop_reason="tool_use",
        ),
        ProviderResponse(
            content=[{"type": "text", "text": "fell back"}],
            stop_reason="end_turn",
        ),
    ])
    loop = AgentLoop(provider=provider, tools=[EchoTool()])
    messages: list[dict] = [{"role": "user", "content": "call bad tool"}]

    await _collect_events(loop, messages)

    result = messages[2]["content"][0]
    assert result["type"] == "tool_result"
    assert "Unknown tool" in result["content"]
    assert result.get("is_error") is True


@pytest.mark.asyncio
async def test_max_iterations():
    """即使模型持续调用工具，达到 max_iterations 后也应结束循环。"""
    provider = MockProvider([
        ProviderResponse(
            content=[{
                "type": "tool_use",
                "id": "toolu_loop",
                "name": "echo",
                "input": {"message": "again"},
            }],
            stop_reason="tool_use",
        ),
    ] * 100)  # 提供足够多的预设响应
    loop = AgentLoop(provider=provider, tools=[EchoTool()], max_iterations=3)
    messages: list[dict] = [{"role": "user", "content": "loop forever"}]

    events = await _collect_events(loop, messages)
    assert events[-1].type == "done"
    assert provider.call_count == 3


@pytest.mark.asyncio
async def test_non_streaming_mode():
    """关闭流式输出（stream=False）时循环也应正常运行。"""
    provider = MockProvider([
        ProviderResponse(
            content=[{"type": "text", "text": "non-streamed response"}],
            stop_reason="end_turn",
        ),
    ])
    loop = AgentLoop(provider=provider, tools=[EchoTool()])
    messages: list[dict] = [{"role": "user", "content": "hi"}]

    events: list[LoopEvent] = []
    async for ev in loop.run(messages, stream=False):
        events.append(ev)

    # 非流式模式下，完整文本应通过单个 text_delta 事件返回
    text_parts = [e.text for e in events if e.type == "text_delta"]
    assert "non-streamed response" in "".join(text_parts)
