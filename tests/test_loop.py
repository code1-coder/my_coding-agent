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
    ModelRequestError,
    ProviderResponse,
    StreamEvent,
)
from coding_agent.tools.base import Tool
from coding_agent.tools.registry import ToolRegistry


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
    assert provider.call_count == 4
    assert sum(event.type == "tool_done" for event in events) == 3
    assert events[-1].data["stop_reason"] == "max_iterations"
    assert "轮次上限" in events[-1].text
    assert "不能确认" in events[-1].text


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [True, False])
async def test_empty_final_response_gets_one_tool_free_summary(stream):
    class Provider(MockProvider):
        async def create_message(self, messages, tools=None, system="", max_tokens=8000):
            if self.call_count == 2:
                assert tools == []
                assert "交付本轮结果" in messages[-1]["content"]
                assert "echo: verified" in str(messages)
            return await super().create_message(messages, tools, system, max_tokens)

    provider = Provider([
        ProviderResponse(content=[{"type": "text", "text": "正在检查。"}, {
            "type": "tool_use", "id": "read", "name": "echo", "input": {"message": "verified"},
        }], stop_reason="tool_use"),
        ProviderResponse(content=[], stop_reason="end_turn"),
        ProviderResponse(content=[{"type": "text", "text": "检查结果：verified。"}], stop_reason="end_turn"),
    ])
    loop = AgentLoop(provider=provider, tools=[EchoTool()])
    messages = [{"role": "user", "content": "检查并报告结果"}]
    events = [event async for event in loop.run(messages, stream=stream)]
    visible = "".join(event.text for event in events if event.type == "text_delta")
    assert visible.count("检查结果：verified。") == 1
    assert events[-1].text == visible
    assert messages[-1]["content"][0]["text"] == "检查结果：verified。"
    assert all(message.get("content") for message in messages)
    assert sum(event.type == "tool_done" for event in events) == 1
    assert provider.call_count == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["empty", "error", "tools"])
async def test_final_summary_fallback_is_bounded_and_does_not_execute_tools(outcome):
    class Provider(MockProvider):
        async def create_message(self, messages, tools=None, system="", max_tokens=8000):
            if self.call_count == 1:
                assert tools == []
                if outcome == "error":
                    self.call_count += 1
                    raise RuntimeError("summary unavailable")
            return await super().create_message(messages, tools, system, max_tokens)

    final = ProviderResponse(content=[], stop_reason="end_turn")
    if outcome == "tools":
        final = ProviderResponse(content=[{
            "type": "tool_use", "id": "unexpected", "name": "echo", "input": {"message": "do not run"},
        }], stop_reason="tool_use")
    provider = Provider([ProviderResponse(content=[], stop_reason="end_turn"), final])
    events = await _collect_events(AgentLoop(provider, [EchoTool()]), [{"role": "user", "content": "hi"}])
    assert provider.call_count == 2
    assert not any(event.type == "tool_start" for event in events)
    assert "未能自动生成" in events[-1].text
    assert events[-1].data["final_answer_generated"] is False


@pytest.mark.asyncio
async def test_final_response_without_stream_deltas_is_visible():
    class Provider(MockProvider):
        async def stream_message(self, *arguments, **options):
            yield StreamEvent(type="done", response=await self.create_message(*arguments, **options))

    provider = Provider([ProviderResponse(content=[{"type": "text", "text": "实际结果"}], stop_reason="end_turn")])
    events = await _collect_events(AgentLoop(provider, []), [{"role": "user", "content": "hi"}])
    assert [event.text for event in events if event.type == "text_delta"] == ["实际结果"]
    assert events[-1].text == "实际结果"
    assert provider.call_count == 1


@pytest.mark.asyncio
async def test_model_request_failure_emits_failure_without_success_completion():
    class FailingProvider(MockProvider):
        async def create_message(self, *arguments, **options):
            raise ModelRequestError("模型服务暂不可用")

    messages = [{"role": "user", "content": "检查项目"}]
    events = await _collect_events(AgentLoop(FailingProvider([]), []), messages)

    assert [event.type for event in events] == ["error"]
    assert events[0].data["failed"] is True
    assert "模型服务暂不可用" in events[0].text
    assert "本轮未完成" in events[0].text
    assert not any(event.type == "done" for event in events)
    assert messages[-1]["role"] == "assistant"
    assert "模型请求失败" in messages[-1]["content"][0]["text"]


@pytest.mark.asyncio
async def test_unexpected_provider_bug_is_not_converted_to_request_failure():
    class BuggyProvider(MockProvider):
        async def create_message(self, *arguments, **options):
            raise RuntimeError("provider programming bug")

    with pytest.raises(RuntimeError, match="provider programming bug"):
        await _collect_events(AgentLoop(BuggyProvider([]), []), [{"role": "user", "content": "检查"}])


@pytest.mark.asyncio
async def test_unexpected_provider_bug_after_partial_text_is_not_relabelled():
    class BuggyProvider(MockProvider):
        async def stream_message(self, *arguments, **options):
            yield StreamEvent(type="text_delta", text="部分输出")
            raise RuntimeError("provider programming bug")

    events = []
    loop = AgentLoop(BuggyProvider([]), [])
    with pytest.raises(RuntimeError, match="provider programming bug"):
        async for event in loop.run([{"role": "user", "content": "检查"}]):
            events.append(event)
    assert [event.text for event in events if event.type == "text_delta"] == ["部分输出"]
    assert not any(event.type == "error" for event in events)


@pytest.mark.asyncio
async def test_cancelling_summary_propagates_without_fake_completion():
    started = asyncio.Event()

    class Provider(MockProvider):
        async def create_message(self, messages, tools=None, system="", max_tokens=8000):
            if self.call_count == 1:
                started.set()
                await asyncio.Event().wait()
            return await super().create_message(messages, tools, system, max_tokens)

    provider = Provider([ProviderResponse(content=[], stop_reason="end_turn")])
    messages = [{"role": "user", "content": "hi"}]
    task = asyncio.create_task(_collect_events(AgentLoop(provider, []), messages))
    await asyncio.wait_for(started.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert messages == [{"role": "user", "content": "hi"}]


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


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_large_batch_reports_completed_tools_before_slow_tool_and_cleans_up(cancel):
    release = asyncio.Event()
    stopped = asyncio.Event()
    registry = ToolRegistry()

    @registry.register(name="controlled")
    async def controlled(slow: bool):
        if slow:
            try:
                await release.wait()
            finally:
                stopped.set()
        return "slow" if slow else "fast"

    blocks = [{"type": "tool_use", "id": str(index), "name": "controlled",
               "input": {"slow": index == 0}} for index in range(41)]
    provider = MockProvider([
        ProviderResponse(content=blocks, stop_reason="tool_use"),
        ProviderResponse(content=[{"type": "text", "text": "完成"}], stop_reason="end_turn"),
    ])
    messages = [{"role": "user", "content": "批量执行"}]
    events = AgentLoop(provider, [], registry=registry, progress_interval=0.02).run(messages)
    try:
        for _ in blocks:
            assert (await anext(events)).type == "tool_start"
        completed_ids = []
        for _ in range(40):
            event = await asyncio.wait_for(anext(events), 2)
            assert event.type == "tool_done"
            completed_ids.append(event.tool_id)
        assert set(completed_ids) == {str(index) for index in range(1, 41)}
        assert not stopped.is_set()
        waiting = await asyncio.wait_for(anext(events), 2)
        assert waiting.type == "warning"
        assert waiting.data["pending_tools"] == [{"tool_name": "controlled", "tool_id": "0"}]
        if cancel:
            await events.aclose()
        else:
            release.set()
            remaining = [event async for event in events]
            assert [event.tool_id for event in remaining if event.type == "tool_done"] == ["0"]
            assert remaining[-1].type == "done"
        assert stopped.is_set()
        results = messages[2]["content"]
        assert [result["tool_use_id"] for result in results] == [str(index) for index in range(41)]
        assert all(result["content"] == "fast" for result in results[1:])
        assert bool(results[0].get("is_error")) is cancel
    finally:
        release.set()
        await events.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [True, False])
async def test_stalled_model_has_visible_progress_and_total_timeout(stream):
    from coding_agent.context.recovery import RecoveryManager

    cancelled = asyncio.Event()

    class Provider(MockProvider):
        async def create_message(self, *arguments, **options):
            self.call_count += 1
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

    provider = Provider([])
    loop = AgentLoop(provider, [], model_timeout=0.1, progress_interval=0.02,
                     recovery=RecoveryManager())
    events = loop.run([{"role": "user", "content": "开始"}], stream=stream)
    observed = []
    async with asyncio.timeout(2):
        async for event in events:
            observed.append(event)
    assert any(event.type == "warning" and event.data["phase"] == "model" for event in observed)
    assert observed[-1].type == "error"
    assert "已停止等待" in observed[-1].text
    assert not any(event.type == "done" for event in observed)
    assert cancelled.is_set()
    assert provider.call_count == 1


@pytest.mark.asyncio
async def test_tool_argument_stream_is_counted_as_activity_without_exposing_arguments():
    tool_response = ProviderResponse(
        content=[{"type": "tool_use", "id": "tool-1", "name": "echo",
                  "input": {"message": "private argument"}}],
        stop_reason="tool_use",
    )
    final_response = ProviderResponse(
        content=[{"type": "text", "text": "完成"}], stop_reason="end_turn",
    )

    class Provider(MockProvider):
        async def stream_message(self, *arguments, **options):
            self.call_count += 1
            if self.call_count == 1:
                for _ in range(16):
                    yield StreamEvent(type="tool_input", tool_id="tool-1", text="secret")
                    await asyncio.sleep(0.01)
                yield StreamEvent(type="done", response=tool_response)
            else:
                yield StreamEvent(type="done", response=final_response)

    provider = Provider([tool_response, final_response])
    events = await _collect_events(
        AgentLoop(provider, [EchoTool()], progress_interval=0.025),
        [{"role": "user", "content": "运行工具"}],
    )

    model_warnings = [event for event in events
                      if event.type == "warning" and event.data.get("phase") == "model"]
    assert model_warnings == []
    assert events[-1].type == "done"


@pytest.mark.asyncio
async def test_model_wait_warning_reports_recent_tool_argument_activity():
    release = asyncio.Event()
    response = ProviderResponse(content=[], stop_reason="end_turn")

    class Provider(MockProvider):
        async def stream_message(self, *arguments, **options):
            self.call_count += 1
            yield StreamEvent(type="tool_input", tool_id="tool-1", text="private argument")
            await release.wait()
            yield StreamEvent(type="done", response=response)

    provider = Provider([response])
    events = AgentLoop(provider, [], progress_interval=0.02).run(
        [{"role": "user", "content": "等待工具参数流"}]
    )
    try:
        warning = await asyncio.wait_for(anext(events), timeout=1)
        assert warning.type == "warning"
        assert warning.data["last_protocol_event"] == "tool_input"
        assert warning.data["seconds_since_protocol_event"] >= 0.01
        assert warning.data["request_attempts"] == 1
        assert warning.data["application_retry_count"] == 0
        assert "工具参数增量" in warning.text
        assert "private argument" not in warning.text
    finally:
        release.set()
        await events.aclose()


@pytest.mark.asyncio
async def test_active_model_stream_can_exceed_soft_budget_within_hard_limit():
    response = ProviderResponse(
        content=[{"type": "text", "text": "长响应完成"}], stop_reason="end_turn",
    )

    class Provider(MockProvider):
        async def stream_message(self, *arguments, **options):
            self.call_count += 1
            for _ in range(8):
                yield StreamEvent(type="tool_input", tool_id="tool-1", text="{}")
                await asyncio.sleep(0.015)
            yield StreamEvent(type="done", response=response)

    events = await asyncio.wait_for(
        _collect_events(AgentLoop(
            Provider([response]), [], model_timeout=0.3, model_idle_timeout=0.04,
            progress_interval=0.01,
        ), [{"role": "user", "content": "等待持续响应"}]),
        timeout=1,
    )
    assert events[-1].type == "done"
    assert events[-1].text == "长响应完成"


@pytest.mark.asyncio
async def test_idle_model_stream_stops_before_total_budget_and_cleans_up():
    stopped = asyncio.Event()
    response = ProviderResponse(content=[], stop_reason="end_turn")

    class Provider(MockProvider):
        async def stream_message(self, *arguments, **options):
            try:
                yield StreamEvent(type="tool_input", tool_id="tool-1", text="{}")
                await asyncio.Event().wait()
                yield StreamEvent(type="done", response=response)
            finally:
                stopped.set()

    loop = AgentLoop(Provider([response]), [], model_timeout=1,
                     model_idle_timeout=0.04, progress_interval=0.01)
    events = await asyncio.wait_for(
        _collect_events(loop, [{"role": "user", "content": "静默流"}]), timeout=0.5,
    )
    assert events[-1].type == "error"
    assert "空闲" in events[-1].text
    assert not any(event.type == "done" for event in events)
    assert stopped.is_set()


@pytest.mark.asyncio
async def test_active_model_stream_respects_finite_hard_limit_and_cancels_provider():
    stopped = asyncio.Event()
    response = ProviderResponse(content=[], stop_reason="end_turn")

    class Provider(MockProvider):
        async def stream_message(self, *arguments, **options):
            try:
                while True:
                    yield StreamEvent(type="tool_input", tool_id="tool-1", text="{}")
                    await asyncio.sleep(0.01)
            finally:
                stopped.set()

    loop = AgentLoop(Provider([response]), [], model_timeout=0.08,
                     model_idle_timeout=0.05, progress_interval=0.01)
    events = await asyncio.wait_for(
        _collect_events(loop, [{"role": "user", "content": "持续流"}]), timeout=0.5,
    )
    assert events[-1].type == "error"
    assert "总时限" in events[-1].text
    assert not any(event.type == "done" for event in events)
    assert stopped.is_set()
