"""公共运行层集成测试：通过真实模块验证执行流程，而不调用外部模型。"""

import asyncio
import json
from copy import deepcopy
from unittest.mock import AsyncMock

import pytest

from coding_agent.config import Config
from coding_agent.hooks.bus import EventBus, EventType
from coding_agent.providers.base import LLMProvider, ProviderResponse, StreamEvent
from coding_agent.runtime import AgentRuntime


def text_response(text="完成", reason="end_turn"):
    return ProviderResponse(content=[{"type": "text", "text": text}], stop_reason=reason,
                            usage={"input_tokens": 3, "output_tokens": 2})


def tool_response(name, args, tool_id="call-1"):
    return ProviderResponse(content=[{"type": "tool_use", "id": tool_id,
                                     "name": name, "input": args}], stop_reason="tool_use")


class ScriptedProvider(LLMProvider):
    def __init__(self, responses=(), *, handler=None, maintenance=None):
        super().__init__("mock")
        self.responses = list(responses)
        self.handler = handler
        self.maintenance = maintenance or (lambda prompt: "摘要" if "请摘要" in prompt else "[]")
        self.calls = []
        self.closed = False

    async def create_message(self, messages, tools=None, system="", max_tokens=8000):
        self.calls.append({"messages": deepcopy(messages), "tools": deepcopy(tools),
                           "system": system, "max_tokens": max_tokens})
        if tools is None:
            return text_response(self.maintenance(messages[0]["content"]))
        response = self.handler(messages, tools, system) if self.handler else self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    async def stream_message(self, *args, **kwargs):
        response = await self.create_message(*args, **kwargs)
        if response.text:
            yield StreamEvent(type="text_delta", text=response.text)
        yield StreamEvent(type="done", response=response)

    async def aclose(self):
        self.closed = True


async def collect(runtime, prompt="开始", **kwargs):
    return [event async for event in runtime.run_turn(prompt, **kwargs)]


@pytest.mark.asyncio
async def test_code_tools_runtime_round_trip(tmp_path):
    (tmp_path / "example.py").write_text("def value():\n    return 1\n", encoding="utf-8")
    patch = ("*** Begin Patch\n*** Update File: example.py\n@@\n def value():\n"
             "-    return 1\n+    return 2\n*** End Patch")
    provider = ScriptedProvider([
        tool_response("code_outline", {"path": "example.py"}, "outline"),
        tool_response("read_file", {"path": "example.py", "start_line": 1, "end_line": 2}, "read"),
        tool_response("apply_patch", {"patch": patch}, "patch"),
        text_response(),
    ])
    observed = []

    async def record(event):
        observed.append(event.data["tool_name"])

    async with AgentRuntime(Config(workdir=str(tmp_path)), provider=provider) as runtime:
        runtime.bus.subscribe(EventType.POST_TOOL_USE, record)
        await collect(runtime)
        assert observed == ["code_outline", "read_file", "apply_patch"]
        assert "1: def value():" in str(runtime.messages)
        assert "Applied patch:" in str(runtime.messages)
        assert "Coding workflow:" in next(call["system"] for call in provider.calls if call["tools"])
        assert {"search_code", "apply_patch", "git_status", "git_diff", "code_outline"} <= set(runtime.registry)
    assert (tmp_path / "example.py").read_text() == "def value():\n    return 2\n"


@pytest.mark.asyncio
async def test_runtime_hooks_registry_skills_tasks_and_dynamic_mcp(tmp_path):
    skill = tmp_path / "skills" / "review"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: review\ndescription: 审查\n---\n先运行测试。", encoding="utf-8")
    provider = ScriptedProvider([
        tool_response("load_skill", {"name": "review"}, "skill"),
        tool_response("create_task", {"subject": "写测试"}, "task"),
        tool_response("connect_mcp", {"name": "docs"}, "connect"),
        tool_response("mcp__docs__search", {"query": "hooks"}, "mcp"),
        text_response(),
    ])
    bus = EventBus()
    observed = []

    async def record(event):
        observed.append((event.type, dict(event.data), event.context))

    for kind in EventType:
        bus.subscribe(kind, record)
    async with AgentRuntime(Config(workdir=str(tmp_path)), provider=provider, event_bus=bus) as runtime:
        events = await collect(runtime)
        assert events[-1].type == "done"
        assert len(runtime.tasks.list_all()) == 1
        assert runtime.tasks.list_all()[0].subject == "写测试"
        assert "review" in provider.calls[0]["system"]
        assert "先运行测试" in str(provider.calls[1]["messages"])
        agent_calls = [c for c in provider.calls if c["tools"] is not None]
        assert any(t["name"] == "mcp__docs__search" for t in agent_calls[-1]["tools"])
        assert "Found 3 results" in str(runtime.messages)
        assert runtime.usage.total_tokens == 10  # 最终响应与记忆提取两次调用的用量。
        assert all(context is runtime.context for _, _, context in observed)
    assert provider.closed
    assert {kind for kind, _, _ in observed} == set(EventType) - {EventType.ERROR}
    assert sum(kind == EventType.PRE_TOOL_USE for kind, _, _ in observed) == 4
    assert sum(kind == EventType.POST_TOOL_USE for kind, _, _ in observed) == 4


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked_by", ["hook", "permission"])
async def test_runtime_blocks_tools_before_any_side_effect(tmp_path, blocked_by):
    provider = ScriptedProvider([
        tool_response("write_file", {"path": "blocked.txt", "content": "不应写入"}),
        text_response(),
    ])
    bus = EventBus()
    if blocked_by == "hook":
        bus.subscribe(EventType.PRE_TOOL_USE, AsyncMock(return_value="禁止写入"))
    runtime = AgentRuntime(Config(workdir=str(tmp_path)), provider=provider, event_bus=bus)
    if blocked_by == "permission":
        from coding_agent.security.permission import BasePermissionChecker, Verdict

        class Deny(BasePermissionChecker):
            async def check(self, tool_name, params):
                return Verdict.DENY, "禁止写入"

        runtime.permissions.add_checker(Deny())
    async with runtime:
        await collect(runtime)
        assert not (tmp_path / "blocked.txt").exists()
        assert runtime.messages[2]["content"][0]["is_error"]


@pytest.mark.asyncio
async def test_runtime_applies_output_hook_and_prevents_prompt_call(tmp_path):
    from coding_agent.hooks.bus import OutputTruncateHook

    provider = ScriptedProvider([
        tool_response("read_file", {"path": "input.txt"}), text_response(),
    ])
    (tmp_path / "input.txt").write_text("x" * 500, encoding="utf-8")
    async with AgentRuntime(Config(workdir=str(tmp_path)), provider=provider) as runtime:
        runtime.bus.subscribe(EventType.POST_TOOL_USE, OutputTruncateHook(max_chars=30, truncate=True))
        await collect(runtime)
        assert "truncated at 30" in runtime.messages[2]["content"][0]["content"]
        runtime.bus.subscribe(EventType.USER_PROMPT_SUBMIT, AsyncMock(return_value=False))
        previous = len(provider.calls)
        with pytest.raises(PermissionError):
            await collect(runtime, "被阻止的输入")
        assert len(provider.calls) == previous


@pytest.mark.asyncio
async def test_runtime_compacts_real_history_before_model_call(tmp_path):
    provider = ScriptedProvider([text_response()])
    cfg = Config(workdir=str(tmp_path), context_limit=40, max_messages=10)
    async with AgentRuntime(cfg, provider=provider) as runtime:
        runtime.messages.extend({"role": "user", "content": "历史" * 100} for _ in range(30))
        await collect(runtime, "记住当前目标")
        model_call = next(c for c in provider.calls if c["tools"] is not None)
        assert model_call["messages"][0]["content"].startswith("[Compacted]")
        assert list((tmp_path / ".transcripts").glob("*.jsonl"))
        memory_call = provider.calls[-1]["messages"][0]["content"]
        assert "记住当前目标" in memory_call


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [True, False])
async def test_runtime_retries_rate_limit_and_reactively_compacts(tmp_path, monkeypatch, stream):
    monkeypatch.setattr("coding_agent.context.recovery.asyncio.sleep", AsyncMock())
    provider = ScriptedProvider([
        RuntimeError("429 rate limit"), RuntimeError("context_length_exceeded"), text_response(),
    ])
    async with AgentRuntime(Config(workdir=str(tmp_path)), provider=provider) as runtime:
        runtime.messages.extend({"role": "user", "content": "历史"} for _ in range(8))
        events = await collect(runtime, stream=stream)
        assert events[-1].text == "完成"
        assert any("[Reactive compact]" in str(call["messages"]) for call in provider.calls)
        assert sum(event.type == "text_delta" for event in events) == 1


@pytest.mark.asyncio
async def test_runtime_recovers_output_limit_without_replaying_tools(tmp_path):
    provider = ScriptedProvider([text_response("截断", "max_tokens"), text_response("完整回答")])
    async with AgentRuntime(Config(workdir=str(tmp_path)), provider=provider) as runtime:
        events = await collect(runtime)
        assert any(e.type == "text_reset" for e in events)
        assert events[-1].text == "完整回答"
        agent_calls = [c for c in provider.calls if c["tools"] is not None]
        assert agent_calls[1]["max_tokens"] == 64000
        assert "截断" not in str(runtime.messages)


@pytest.mark.asyncio
async def test_subagent_uses_real_registry_permissions_and_isolated_history(tmp_path):
    def handle(messages, tools, system):
        if "不要进一步委托" not in system:
            return tool_response("task", {"description": "读取文件", "role": "worker"}) if len(messages) == 1 else text_response()
        assert not any(t["name"] in {"task", "spawn_teammate"} for t in tools)
        if len(messages) == 1:
            return tool_response("read_file", {"path": "note.txt"}, "child-call")
        return text_response("子任务完成")

    (tmp_path / "note.txt").write_text("只在子任务出现", encoding="utf-8")
    provider = ScriptedProvider(handler=handle)
    bus = EventBus()
    tools_seen = []

    async def record(event):
        tools_seen.append(event.data)

    bus.subscribe(EventType.POST_TOOL_USE, record)
    async with AgentRuntime(Config(workdir=str(tmp_path)), provider=provider, event_bus=bus) as runtime:
        await collect(runtime)
        assert "只在子任务出现" not in str(runtime.messages)
        assert "子任务完成" in str(runtime.messages)
        assert any(e["agent"].startswith("subagent-") for e in tools_seen)


@pytest.mark.asyncio
async def test_team_lifecycle_consumes_mail_claims_tasks_and_shuts_down(tmp_path):
    def handle(messages, tools, system):
        return text_response("队友工作完成")

    provider = ScriptedProvider(handler=handle)
    cfg = Config(workdir=str(tmp_path), team_idle_timeout=0.02)
    async with AgentRuntime(cfg, provider=provider) as runtime:
        queued = runtime.tasks.create("空闲后认领此任务")
        await runtime.registry.dispatch("spawn_teammate", {
            "name": "worker", "role": "tester", "prompt": "完成测试",
        })
        result = await asyncio.wait_for(runtime.team.wait("worker"), 1)
        assert "队友工作完成" in result
        inbox = runtime.team.check_inbox("lead")
        assert inbox[0]["type"] == "result"
        assert not runtime.team.active_names
        assert any("worker" in c["system"] for c in provider.calls)
        assert runtime.tasks.get(queued.id).owner == "worker"


@pytest.mark.asyncio
async def test_runtime_close_cancels_background_lifecycle(tmp_path):
    provider = ScriptedProvider(handler=lambda *args: text_response())
    runtime = AgentRuntime(Config(workdir=str(tmp_path)), provider=provider)
    await runtime.start()
    await runtime.registry.dispatch("spawn_teammate", {"name": "worker", "role": "tester", "prompt": "任务"})
    await asyncio.sleep(0.01)
    assert runtime.team.active_names == ["worker"]
    await runtime.aclose()
    assert not runtime.team.active_names
    assert provider.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("response", ["[]", '[null]', '[{"name":"bad"}]'])
async def test_invalid_consolidation_preserves_existing_memories(tmp_path, response):
    from coding_agent.memory import MemoryStore

    store = MemoryStore(workdir=tmp_path, summarize=AsyncMock(return_value=response), consolidate_threshold=1)
    store.write_file("keep-me", "user", "保留", "重要内容")
    await store.consolidate()
    assert "重要内容" in store.read_file("keep-me.md")


@pytest.mark.asyncio
async def test_runtime_switches_to_fallback_and_closes_both_clients(tmp_path, monkeypatch):
    monkeypatch.setattr("coding_agent.context.recovery.asyncio.sleep", AsyncMock())
    primary = ScriptedProvider([RuntimeError("529 overloaded")] * 3)
    fallback = ScriptedProvider([text_response("备用模型的回复")])
    fallback.model_id = "backup"
    cfg = Config(workdir=str(tmp_path), fallback_model="backup")
    factory = lambda config: fallback
    async with AgentRuntime(cfg, provider=primary, provider_factory=factory) as runtime:
        events = await collect(runtime)
        assert events[-1].text == "备用模型的回复"
        assert "backup" in runtime._providers
    assert primary.closed and fallback.closed


@pytest.mark.asyncio
async def test_cancelling_turn_repairs_unmatched_tool_calls(tmp_path):
    provider = ScriptedProvider([tool_response("write_file", {"path": "pending.txt", "content": "不应写入"})])
    async with AgentRuntime(Config(workdir=str(tmp_path)), provider=provider) as runtime:
        events = runtime.run_turn("取消执行")
        event = await anext(events)
        assert event.type == "tool_start"
        await events.aclose()
        assert not (tmp_path / "pending.txt").exists()
        assert runtime.messages[-1]["content"][0]["is_error"]
        assert runtime.messages[-1]["content"][0]["tool_use_id"] == "call-1"


@pytest.mark.asyncio
async def test_stream_failure_after_text_is_not_replayed(tmp_path):
    class InterruptedProvider(ScriptedProvider):
        async def stream_message(self, *args, **kwargs):
            yield StreamEvent(type="text_delta", text="已输出")
            raise RuntimeError("429 rate limit")

    provider = InterruptedProvider()
    async with AgentRuntime(Config(workdir=str(tmp_path)), provider=provider) as runtime:
        events = runtime.run_turn("测试")
        assert (await anext(events)).text == "已输出"
        with pytest.raises(RuntimeError, match="流式响应中断"):
            await anext(events)


@pytest.mark.asyncio
async def test_compaction_persists_outputs_at_documented_threshold(tmp_path):
    from coding_agent.context.compaction import CompactionPipeline

    messages = [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "small-id",
                                               "content": "x" * 40000}]}]
    await CompactionPipeline(workdir=tmp_path).run(messages)
    assert (tmp_path / ".task_outputs" / "tool-results" / "small-id.txt").exists()
    assert "<persisted-output>" in messages[0]["content"][0]["content"]


@pytest.mark.asyncio
async def test_failed_summary_preserves_original_history(tmp_path):
    from coding_agent.context.compaction import compact_history

    messages = [{"role": "user", "content": "必须保留"}]
    result = await compact_history(messages, summarize=AsyncMock(side_effect=RuntimeError("失败")), workdir=tmp_path)
    assert result == messages


@pytest.mark.asyncio
async def test_lead_receives_team_inbox_before_next_model_call(tmp_path):
    provider = ScriptedProvider([text_response()])
    async with AgentRuntime(Config(workdir=str(tmp_path)), provider=provider) as runtime:
        runtime.mailbox.send("worker", "lead", "测试全部通过", msg_type="result")
        await collect(runtime)
        first = next(c for c in provider.calls if c["tools"] is not None)
        assert "测试全部通过" in str(first["messages"])
        assert not runtime.mailbox.peek("lead")


@pytest.mark.asyncio
async def test_large_tool_output_is_persisted_in_actual_runtime(tmp_path):
    (tmp_path / "large.txt").write_text("x" * 40000, encoding="utf-8")
    provider = ScriptedProvider([tool_response("read_file", {"path": "large.txt"}), text_response()])
    async with AgentRuntime(Config(workdir=str(tmp_path)), provider=provider) as runtime:
        await collect(runtime)
        agent_calls = [c for c in provider.calls if c["tools"] is not None]
        assert "<persisted-output>" in str(agent_calls[1]["messages"])
        path = tmp_path / ".task_outputs" / "tool-results" / "call-1.txt"
        assert path.read_text(encoding="utf-8") == "x" * 40000


@pytest.mark.parametrize("changes", [
    {"context_limit": 0}, {"max_messages": 2}, {"keep_recent": -1},
    {"max_iterations": False}, {"team_idle_timeout": -1}, {"mcp_servers": "docs"},
])
def test_invalid_runtime_options_fail_before_creating_provider(tmp_path, changes):
    from unittest.mock import Mock

    factory = Mock()
    with pytest.raises(ValueError):
        AgentRuntime(Config(workdir=str(tmp_path), **changes), provider_factory=factory)
    factory.assert_not_called()


def test_non_transient_generation_error_is_not_retried(monkeypatch):
    from coding_agent.context.recovery import RecoveryState, with_retry

    sleep = AsyncMock()
    monkeypatch.setattr("coding_agent.context.recovery.asyncio.sleep", sleep)
    call = AsyncMock(side_effect=RuntimeError("generate request failed: 401 Unauthorized"))
    with pytest.raises(RuntimeError, match="401"):
        asyncio.run(with_retry(call, RecoveryState()))
    assert call.await_count == 1
    sleep.assert_not_called()
