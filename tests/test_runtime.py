"""公共运行层集成测试：通过真实模块验证执行流程，而不调用外部模型。"""

import asyncio
import json
import os
import shutil
import subprocess
from copy import deepcopy
from unittest.mock import AsyncMock

import pytest
import httpx
import openai

from coding_agent.config import Config
from coding_agent.hooks.bus import EventBus, EventType
from coding_agent.providers.base import LLMProvider, ModelRequestError, ProviderResponse, StreamEvent
from coding_agent.runtime import AgentRuntime, RetryUnavailable


def model_http_error(status, headers=None, message=None):
    response = httpx.Response(status, headers=headers,
                              request=httpx.Request("POST", "https://example.invalid"))
    return openai.APIStatusError(message or f"HTTP {status}", response=response, body=None)


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


def single_role_analysis_provider():
    """模型尝试缩成职责一维，之后只提交一条有真实源码支撑的结论。"""
    def respond(messages, tools, system):
        match len(provider.calls):
            case 1:
                return tool_response("analysis_checkpoint", {"action": "plan", "dimensions": ["role"]}, "plan")
            case 2:
                return tool_response("read_file", {"path": "entry.py"}, "read")
            case 3:
                state = json.loads(messages[-1]["content"][-1]["content"].rsplit("\n", 1)[1])
                return tool_response("analysis_checkpoint", {"action": "record", "dimension": "role",
                                     "finding": "提供入口函数。", "evidence_ids": list(state["evidence"])}, "record")
            case 4:
                return tool_response("analysis_checkpoint", {"action": "finish"}, "finish")
        return text_response()

    provider = ScriptedProvider(handler=respond)
    return provider


@pytest.mark.asyncio
async def test_analysis_cannot_narrow_default_scope_via_checkpoint(tmp_path):
    (tmp_path / "entry.py").write_text("def start(): pass\n", encoding="utf-8")
    async with AgentRuntime(Config(workdir=str(tmp_path)), provider=single_role_analysis_provider()) as runtime:
        events = await collect(runtime, "检查职责、调用关系、入口用法和生命周期", analysis=True)

    assert events[-1].data["analysis_status"] == "partial"
    assert "提供入口函数" in events[-1].text
    for label in ("调用关系", "入口与用法", "生命周期与边界"):
        assert f"{label}：未确认" in events[-1].text
    assert any(event.type == "tool_done" and event.tool_id == "plan" and event.data["is_error"] for event in events)


@pytest.mark.asyncio
async def test_analysis_does_not_claim_completion_without_evidence(tmp_path):
    provider = ScriptedProvider([text_response("已经验证全部行为")])
    async with AgentRuntime(Config(workdir=str(tmp_path)), provider=provider) as runtime:
        events = await collect(runtime, "解释运行层的作用", analysis=True)
    assert events[-1].type == "done"
    assert events[-1].data["analysis_status"] == "partial"
    assert "未确认" in events[-1].text
    assert "已经验证全部行为" not in "".join(event.text for event in events)
    assert not list((tmp_path / ".memory").iterdir())


@pytest.mark.asyncio
@pytest.mark.parametrize("evidence_tool", ["search_code", "code_outline"])
async def test_analysis_revokes_findings_when_source_version_changes(tmp_path, evidence_tool):
    source = tmp_path / "entry.py"
    source.write_text("def old_entry(): pass\n", encoding="utf-8")
    args = {"path": source.name}
    evidence_args = {**args, "pattern": "old_entry"} if evidence_tool == "search_code" else args
    old_id = None

    def respond(messages, tools, system):
        match len(provider.calls):
            case 1:
                return tool_response("read_file", args, "baseline")
            case 2:
                return tool_response(evidence_tool, evidence_args, "evidence")
            case 3:
                state = json.loads(messages[-1]["content"][-1]["content"].rsplit("\n", 1)[1])
                old_id = next(ref for ref, label in state["evidence"].items() if evidence_tool in label)
                return tool_response("analysis_checkpoint", {"action": "record", "dimension": "role",
                                     "finding": "入口为 old_entry。", "evidence_ids": [old_id]}, "record")
            case 4:
                source.write_text("def new_entry(): pass\n", encoding="utf-8")
                return tool_response("read_file", args, "updated")
            case 5:
                return tool_response("analysis_checkpoint", {"action": "finish"}, "finish")
        return text_response()

    provider = ScriptedProvider(handler=respond)
    async with AgentRuntime(Config(workdir=str(tmp_path)), provider=provider) as runtime:
        events = await collect(runtime, "解释入口的职责", analysis=True, analysis_dimensions=["role"])
    assert any(event.type == "analysis_progress" and "入口为 old_entry" in event.text for event in events)
    assert any(event.tool_id == "finish" and event.data.get("is_error") for event in events)
    assert events[-1].data["analysis_status"] == "partial"
    assert "入口为 old_entry" not in events[-1].text


@pytest.mark.asyncio
@pytest.mark.parametrize(("phase", "change"), [
    ("finish", "none"), ("finish", "write"), ("finish", "delete"),
    ("resume", "write"), ("resume", "delete"),
])
async def test_analysis_checks_cited_source_without_model_reread(tmp_path, phase, change):
    source = tmp_path / "entry.py"
    source.write_text("def old_entry(): pass\n", encoding="utf-8")
    finding = "入口为 old_entry。"

    def respond(messages, tools, system):
        match len(provider.calls):
            case 1:
                return tool_response("read_file", {"path": "entry.py"}, "read")
            case 2:
                state = json.loads(messages[-1]["content"][-1]["content"].rsplit("\n", 1)[1])
                return tool_response("analysis_checkpoint", {"action": "record", "dimension": "role",
                                     "finding": finding, "evidence_ids": list(state["evidence"])}, "record")
            case 3:
                if phase == "resume":
                    return ConnectionError("断开")
                return tool_response("analysis_checkpoint", {"action": "finish"}, "finish")
            case 4:
                state = json.loads(messages[-1]["content"][-1]["content"].rsplit("\n", 1)[1])
                assert finding not in state["checks"]
                assert "待复核" in state["checks"]
                assert not state["evidence"]
        return text_response()

    def change_source():
        if change == "write":
            source.write_text("def new_entry(): pass\n", encoding="utf-8")
        elif change == "delete":
            source.unlink()

    provider = ScriptedProvider(handler=respond)
    cfg = Config(workdir=str(tmp_path), model_max_attempts=1)
    async with AgentRuntime(cfg, provider=provider) as runtime:
        events = []
        async for event in runtime.run_turn("检查入口职责", analysis=True, analysis_dimensions=["role"]):
            events.append(event)
            if event.type == "tool_done" and event.tool_id == "finish":
                change_source()
        if phase == "resume":
            assert events[-1].type == "error"
            change_source()
            events = [event async for event in runtime.retry_turn(events[-1].data["recovery_id"])]
    assert events[-1].type == "done"
    if change == "none":
        assert events[-1].data["analysis_status"] == "complete"
        assert finding in events[-1].text
    else:
        assert events[-1].data["analysis_status"] == "partial"
        assert finding not in events[-1].text
        assert "待复核" in events[-1].text


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", ["search_code", "code_outline"])
async def test_analysis_cannot_finish_on_unversioned_source_leads(tmp_path, tool):
    (tmp_path / "entry.py").write_text("def locate_entry(): pass\n", encoding="utf-8")
    params = {"path": "entry.py"}
    if tool == "search_code":
        params["pattern"] = "locate_entry"

    def respond(messages, tools, system):
        match len(provider.calls):
            case 1:
                return tool_response(tool, params, "lead")
            case 2:
                state = json.loads(messages[-1]["content"][-1]["content"].rsplit("\n", 1)[1])
                return tool_response("analysis_checkpoint", {"action": "record", "dimension": "role",
                                     "finding": "入口已经确认。", "evidence_ids": list(state["evidence"])}, "record")
            case 3:
                return tool_response("analysis_checkpoint", {"action": "finish"}, "finish")
        return text_response()

    provider = ScriptedProvider(handler=respond)
    async with AgentRuntime(Config(workdir=str(tmp_path)), provider=provider) as runtime:
        events = await collect(runtime, "检查入口的职责", analysis=True, analysis_dimensions=["role"])
    assert events[-1].data["analysis_status"] == "partial"
    assert "待复核" in events[-1].text
    assert "入口已经确认" not in events[-1].text


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", ["approve", "deny", "cancel", "hook_denies"])
async def test_analysis_revalidates_external_evidence_through_read_permissions(tmp_path, decision):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("def outside_entry(): pass\n", encoding="utf-8")
    finding = "已定位外部入口。"
    approvals = []
    waiting = asyncio.Event()
    cleaned = asyncio.Event()

    async def approve(name, params, reason):
        approvals.append((name, params))
        if len(approvals) == 1:
            return True
        if decision == "cancel":
            waiting.set()
            try:
                await asyncio.Future()
            finally:
                cleaned.set()
        return decision == "approve"

    async def block_revalidation(event):
        if decision == "hook_denies" and event.data.get("purpose") == "analysis_revalidation":
            return "禁止再次访问工作区外文件。"

    def respond(messages, tools, system):
        match len(provider.calls):
            case 1:
                return tool_response("read_file", {"path": str(outside)}, "read")
            case 2:
                state = json.loads(messages[-1]["content"][-1]["content"].rsplit("\n", 1)[1])
                return tool_response("analysis_checkpoint", {"action": "record", "dimension": "role",
                                     "finding": finding, "evidence_ids": list(state["evidence"])}, "record")
            case 3:
                return tool_response("analysis_checkpoint", {"action": "finish"}, "finish")
        pytest.fail("收尾不应再次请求模型。")

    provider = ScriptedProvider(handler=respond)
    events = []
    async with AgentRuntime(Config(workdir=str(workspace)), provider=provider, approval_handler=approve) as runtime:
        runtime.bus.subscribe(EventType.PRE_TOOL_USE, block_revalidation)

        async def run():
            async for event in runtime.run_turn("检查外部入口", analysis=True, analysis_dimensions=["role"]):
                events.append(event)

        if decision == "cancel":
            task = asyncio.create_task(run())
            try:
                await asyncio.wait_for(waiting.wait(), timeout=5)
            finally:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            assert cleaned.is_set()
            assert not runtime.turn_lock.locked()
        else:
            await run()

    assert approvals == [("read_file", {"path": str(outside)})] * (1 if decision == "hook_denies" else 2)
    if decision == "cancel":
        assert not any(event.type in {"done", "error"} for event in events)
    elif decision == "approve":
        assert events[-1].data["analysis_status"] == "complete"
        assert finding in events[-1].text
    else:
        assert events[-1].data["analysis_status"] == "partial"
        assert "无法验证源码版本" in events[-1].text
        assert finding not in events[-1].text


@pytest.mark.asyncio
@pytest.mark.parametrize("compaction", ["proactive", "reactive"])
async def test_analysis_does_not_persist_reading_or_transcripts(tmp_path, compaction):
    source = tmp_path / "sample.py"
    source.write_text("synthetic_value = 1\n" * 1800, encoding="utf-8")
    replies = [tool_response("read_file", {"path": source.name}, "large-read"), text_response()]
    cfg = Config(workdir=str(tmp_path))
    if compaction == "proactive":
        cfg.context_limit = 25
    else:
        replies.insert(1, model_http_error(400, message="context_length_exceeded"))
    provider = ScriptedProvider(replies)
    async with AgentRuntime(cfg, provider=provider) as runtime:
        events = await collect(runtime, "解释大文件", analysis=True)
    assert events[-1].type == "done"
    assert any(call["tools"] is None for call in provider.calls)
    read = next(event for event in events if event.type == "tool_done" and event.tool_id == "large-read")
    assert read.data["is_error"] and "缩小" in read.tool_output
    assert "synthetic_value" not in str(provider.calls)
    assert not (tmp_path / ".task_outputs").exists()
    assert not (tmp_path / ".transcripts").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("context_limit", [50_000, 200])
async def test_analysis_delivers_all_four_reads_before_compacting_them(tmp_path, context_limit):
    blocks = []
    for index in range(4):
        path = tmp_path / f"part_{index}.py"
        path.write_text("value = 1\n" * 60 + f"tail_marker = {index}\n", encoding="utf-8")
        blocks.extend(tool_response("read_file", {"path": path.name}, f"read-{index}").content)
    provider = ScriptedProvider([ProviderResponse(content=blocks, stop_reason="tool_use"), text_response()])
    cfg = Config(workdir=str(tmp_path), context_limit=context_limit)
    async with AgentRuntime(cfg, provider=provider) as runtime:
        await collect(runtime, "检查四个文件", analysis=True)
    request = [call for call in provider.calls if call["tools"]][-1]
    results = [block for message in request["messages"] if isinstance(message["content"], list)
               for block in message["content"] if block["type"] == "tool_result"]
    assert [block["tool_use_id"] for block in results] == [f"read-{i}" for i in range(4)]
    assert all(f"tail_marker = {i}" in block["content"] for i, block in enumerate(results))


@pytest.mark.asyncio
async def test_analysis_can_record_blocker_after_three_failures_in_one_batch(tmp_path):
    failed_reads = [tool_response("read_file", {"path": f"missing_{i}.py"}, f"missing-{i}").content[0]
                    for i in range(3)]
    provider = ScriptedProvider([
        ProviderResponse(content=failed_reads, stop_reason="tool_use"),
        tool_response("analysis_checkpoint", {"action": "record", "dimension": "usage", "blocked": True,
                     "finding": "三个候选入口均不存在。"}, "blocked"),
        tool_response("analysis_checkpoint", {"action": "finish"}, "finish"),
    ])
    async with AgentRuntime(Config(workdir=str(tmp_path)), provider=provider) as runtime:
        events = await collect(runtime, "只检查入口", analysis=True, analysis_dimensions=["usage"])
    assert events[-1].data["stop_reason"] == "analysis_finished"
    assert "三个候选入口均不存在" in events[-1].text


@pytest.mark.asyncio
@pytest.mark.parametrize("exhausted", [False, True])
async def test_analysis_recovers_truncated_output_without_replaying_reads(tmp_path, exhausted):
    (tmp_path / "entry.py").write_text("value = 1\n", encoding="utf-8")
    provider = ScriptedProvider([
        tool_response("read_file", {"path": "entry.py"}, "read"),
        text_response("不完整草稿", "max_tokens"),
    ])

    def respond(messages, tools, system):
        if provider.responses:
            return provider.responses.pop(0)
        if exhausted:
            return text_response("不完整草稿", "max_tokens")
        if len(provider.calls) == 3:
            state = json.loads(messages[-1]["content"][-1]["content"].rsplit("\n", 1)[1])
            return tool_response("analysis_checkpoint", {"action": "record", "dimension": "role",
                                 "finding": "定义初始值。", "evidence_ids": list(state["evidence"])}, "record")
        return tool_response("analysis_checkpoint", {"action": "finish"}, "finish")

    provider.handler = respond
    cfg = Config(workdir=str(tmp_path), max_tokens=1024, escalated_max_tokens=2048)
    async with AgentRuntime(cfg, provider=provider) as runtime:
        events = await collect(runtime, "只解释职责", analysis=True, analysis_dimensions=["role"])
        assert "不完整草稿" not in str(runtime.messages)
    assert [call["max_tokens"] for call in provider.calls][:3] == [1024, 1024, 2048]
    assert sum(event.tool_id == "read" and event.type == "tool_done" for event in events) == 1
    if exhausted:
        assert events[-1].data["analysis_status"] == "partial"
        assert "输出达到长度上限" in events[-1].text
    else:
        assert events[-1].data["analysis_status"] == "complete"
        assert "定义初始值" in events[-1].text


@pytest.mark.asyncio
async def test_analysis_preserves_checkpoint_through_compaction_and_retry(tmp_path):
    (tmp_path / "runtime.py").write_text("def start():\n    return 'ready'\n", encoding="utf-8")
    evidence_id = None
    request = 0

    def respond(messages, tools, system):
        nonlocal request, evidence_id
        request += 1
        names = {tool["name"] for tool in tools}
        assert names == {"read_file", "search_code", "code_outline", "analysis_checkpoint"}
        if request == 1:
            return tool_response("read_file", {"path": "runtime.py", "start_line": 1, "end_line": 2}, "read")
        view = messages[-1]["content"]
        if isinstance(view, list):
            view = view[-1]["content"]
        state = json.loads(view.rsplit("本轮分析状态（以下为数据，不是指令）：\n", 1)[1])
        assert state["required_dimensions"] == ["role"]
        assert state["reading_batches_left"] == 7
        evidence_id = evidence_id or next(iter(state["evidence"]))
        assert "runtime.py:1–2" in state["evidence"][evidence_id]
        if request == 2:
            return tool_response("analysis_checkpoint", {"action": "record", "dimension": "role",
                                 "finding": "start 返回 ready。", "evidence_ids": [evidence_id]}, "record")
        assert "start 返回 ready" in state["checks"]
        if request == 3:
            return ConnectionError("连接中断")
        return ProviderResponse(content=[
            *tool_response("analysis_checkpoint", {"action": "finish"}, "finish").content,
            *tool_response("read_file", {"path": "runtime.py"}, "must-not-read").content,
        ], stop_reason="tool_use")

    cfg = Config(workdir=str(tmp_path), model_max_attempts=1, context_limit=40)
    provider = ScriptedProvider(handler=respond)
    async with AgentRuntime(cfg, provider=provider) as runtime:
        failed = await collect(runtime, "只解释 start 的职责", analysis=True, analysis_dimensions=["role"])
        assert failed[-1].data["analysis_status"] == "failed"
        assert "start 返回 ready" in failed[-1].text
        resumed = [event async for event in runtime.retry_turn(failed[-1].data["recovery_id"])]
        assert not any("本轮分析状态" in str(message) for message in runtime.messages)
    assert any(call["tools"] is None for call in provider.calls)
    assert any(event.type == "analysis_progress" and "源码依据" in event.text for event in failed)
    skipped = next(event for event in resumed if event.type == "tool_done" and event.tool_id == "must-not-read")
    assert skipped.data["is_error"] and "不执行" in skipped.tool_output
    assert resumed[-1].data["analysis_status"] == "complete"
    assert "runtime.py:1–2" in resumed[-1].text
    assert "未运行项目代码" in resumed[-1].text


@pytest.mark.asyncio
async def test_analysis_rejects_forged_evidence_and_side_effects(tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("private", encoding="utf-8")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    providers = ScriptedProvider([
        tool_response("write_file", {"path": "must-not-exist", "content": "wrong"}, "write"),
        tool_response("read_file", {"path": str(outside)}, "denied"),
        tool_response("analysis_checkpoint", {"action": "record", "dimension": "role", "finding": "wrong",
                                               "evidence_ids": ["denied"]}, "forged"),
        text_response("已经核查所有内容"),
    ])
    approvals = []

    async def reject(name, params, reason):
        approvals.append((name, params))
        return False

    async with AgentRuntime(Config(workdir=str(workspace)), provider=providers, approval_handler=reject) as runtime:
        events = await collect(runtime, "检查项目", analysis=True)
    assert approvals == [("read_file", {"path": str(outside)})]
    assert all(event.data["is_error"] for event in events if event.type == "tool_done")
    assert "本轮已登记的证据 ID" in next(event.tool_output for event in events if event.tool_id == "forged"
                                           and event.type == "tool_done")
    assert events[-1].data["analysis_status"] == "partial"
    assert "wrong" not in events[-1].text
    assert not (workspace / "must-not-exist").exists()


@pytest.mark.asyncio
async def test_analysis_stops_after_two_batches_of_duplicate_evidence(tmp_path):
    (tmp_path / "sample.py").write_text("value = 1\n", encoding="utf-8")
    responses = [tool_response("read_file", {"path": "sample.py", "show_line_numbers": numbered}, str(index))
                 for index, numbered in enumerate([True, False, True])]
    async with AgentRuntime(Config(workdir=str(tmp_path)), provider=ScriptedProvider(responses)) as runtime:
        events = await collect(runtime, "解释 sample", analysis=True)
    assert events[-1].data["stop_reason"] == "analysis_no_progress"
    assert events[-1].data["analysis_status"] == "partial"
    assert sum(event.type == "tool_done" for event in events) == 3


@pytest.mark.asyncio
async def test_analysis_budget_stops_before_dispatching_more_reads(tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("\n".join(f"item_{line} = {line}" for line in range(1, 11)), encoding="utf-8")
    responses = [tool_response("read_file", {"path": path.name, "start_line": line, "end_line": line}, f"read-{line}")
                 for line in range(1, 10)]
    bus = EventBus()
    dispatched = []

    async def observe(event):
        dispatched.append(event.data["tool_id"])

    bus.subscribe(EventType.PRE_TOOL_USE, observe)
    async with AgentRuntime(Config(workdir=str(tmp_path)), provider=ScriptedProvider(responses), event_bus=bus) as runtime:
        events = await collect(runtime, "检查 sample.py", analysis=True)
    assert dispatched == [f"read-{line}" for line in range(1, 9)]
    assert events[-1].data["stop_reason"] == "analysis_budget"
    assert events[-1].data["analysis_status"] == "partial"


@pytest.mark.asyncio
async def test_analysis_stops_when_model_repeats_checkpoint_without_new_evidence(tmp_path):
    args = {"action": "record", "dimension": "usage", "blocked": True, "finding": "缺少入口文件"}
    responses = [tool_response("analysis_checkpoint", args, f"record-{index}") for index in range(3)]
    async with AgentRuntime(Config(workdir=str(tmp_path)), provider=ScriptedProvider(responses)) as runtime:
        events = await collect(runtime, "解释入口", analysis=True)
    assert sum(event.type == "analysis_progress" for event in events) == 1
    assert events[-1].data["stop_reason"] == "analysis_stalled_controls"
    assert "缺少入口文件" in events[-1].text


@pytest.mark.asyncio
async def test_analysis_cancel_before_external_read_approval_completes(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    external = tmp_path / "external.py"
    external.write_text("private = True", encoding="utf-8")
    approval_started = asyncio.Event()
    approval_cancelled = asyncio.Event()

    async def approve(*args):
        approval_started.set()
        try:
            await asyncio.Future()
        finally:
            approval_cancelled.set()

    provider = ScriptedProvider([tool_response("read_file", {"path": str(external)}, "external")])
    async with AgentRuntime(Config(workdir=str(workspace)), provider=provider, approval_handler=approve) as runtime:
        events = []

        async def consume():
            async for event in runtime.run_turn("读取外部源码", analysis=True):
                events.append(event)

        task = asyncio.create_task(consume())
        try:
            await asyncio.wait_for(approval_started.wait(), timeout=3)
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert approval_cancelled.is_set() and not runtime.turn_lock.locked()
        assert runtime.messages[-1]["content"][0]["is_error"]
        with pytest.raises(RetryUnavailable):
            await anext(runtime.retry_turn())
    assert not any(event.type in {"done", "analysis_progress", "memory_saved"} for event in events)
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_search_fallback_runtime_delivers_results_and_final_answer(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", "")
    (tmp_path / "example.py").write_text("first = 1\nsecond = 2\n", encoding="utf-8")
    provider = ScriptedProvider([
        tool_response("search_code", {"path": "example.py", "pattern": "first"}, "search-1"),
        tool_response("search_code", {"path": "example.py", "pattern": "second"}, "search-2"),
        text_response("两个定义已找到"),
    ])
    async with AgentRuntime(Config(workdir=str(tmp_path)), provider=provider) as runtime:
        events = await collect(runtime)
        completed = [event for event in events if event.type == "tool_done"]
        assert len(completed) == 2
        assert all(not event.data["is_error"] for event in completed)
        assert [json.loads(event.tool_output)["results"][0]["line"] for event in completed] == [1, 2]
        assert all(json.loads(event.tool_output)["backend"] == "python_file" for event in completed)
        assert events[-1].type == "done" and events[-1].text == "两个定义已找到"


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked_by", ["hook", "permission"])
async def test_search_fallback_is_blocked_before_reading(tmp_path, monkeypatch, blocked_by):
    from coding_agent.security.permission import BasePermissionChecker, Verdict
    from coding_agent.tools.code import SearchCodeTool

    monkeypatch.setenv("PATH", "")
    (tmp_path / "private.py").write_text("private_value\n", encoding="utf-8")
    read = False
    original = SearchCodeTool._read_search_file

    def observe(*args, **kwargs):
        nonlocal read
        read = True
        return original(*args, **kwargs)

    monkeypatch.setattr(SearchCodeTool, "_read_search_file", observe)
    provider = ScriptedProvider([
        tool_response("search_code", {"path": "private.py", "pattern": "private_value"}), text_response()])
    bus = EventBus()
    if blocked_by == "hook":
        bus.subscribe(EventType.PRE_TOOL_USE, AsyncMock(return_value="禁止读取"))
    async with AgentRuntime(Config(workdir=str(tmp_path)), provider=provider, event_bus=bus) as runtime:
        if blocked_by == "permission":
            class Deny(BasePermissionChecker):
                async def check(self, tool_name, params):
                    return Verdict.DENY, "禁止读取"

            runtime.permissions.add_checker(Deny())
        events = await collect(runtime)
        assert any(event.type == "tool_done" and event.data["is_error"] for event in events)
        assert not read


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
async def test_runtime_reads_external_file_after_explicit_approval(tmp_path):
    external_file = tmp_path.parent / f"{tmp_path.name}-approved.md"
    external_file.write_text("审批后可读取", encoding="utf-8")
    provider = ScriptedProvider([
        tool_response("read_file", {"path": str(external_file)}, "external-read"),
        text_response("已读取"),
    ])
    approvals = []
    approval_pending = False
    warnings_during_approval = []

    async def approve(tool_name, params, reason):
        nonlocal approval_pending
        approvals.append((tool_name, params, reason))
        approval_pending = True
        try:
            await asyncio.sleep(0.15)
            return True
        finally:
            approval_pending = False

    async with AgentRuntime(Config(workdir=str(tmp_path), progress_interval=0.05), provider=provider) as runtime:
        runtime.permissions.approval_handler = approve
        async def run_and_observe():
            observed = []
            async for event in runtime.run_turn("开始"):
                observed.append(event)
                if approval_pending and event.type == "warning":
                    warnings_during_approval.append(event.text)
            return observed

        events = await asyncio.wait_for(run_and_observe(), timeout=2)
        result = runtime.messages[2]["content"][0]
        assert result["content"] == "审批后可读取"
        assert not result.get("is_error")
        assert events[-1].type == "done"
        assert warnings_during_approval == []
        assert len(approvals) == 1
        assert approvals[0][0] == "read_file"
        reused_grant = await runtime.registry.dispatch("read_file", {"path": str(external_file)})
        assert reused_grant.error and "路径逃逸工作区" in reused_grant.error


@pytest.mark.asyncio
@pytest.mark.parametrize(("backend", "scope"), [
    ("rg", "directory"), ("rg", "file"),
    ("git", "directory"), ("git", "file"), ("git", "subdirectory"), ("file", "file"),
])
async def test_runtime_searches_external_scope_after_explicit_approval(tmp_path, monkeypatch, backend, scope):
    executable = shutil.which("rg" if backend == "rg" else "git")
    if backend != "file" and executable is None:
        pytest.skip(f"requires {backend}")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    external = tmp_path / "外部 文档"
    external.mkdir()
    directory = external / "子目录" if scope == "subdirectory" else external
    directory.mkdir(exist_ok=True)
    source = directory / "report.md"
    source.write_text("before\n4.7 approved scope\nafter\n", encoding="utf-8")
    (external / "ignored.md").write_text("4.7 private\n", encoding="utf-8")
    if backend == "git":
        subprocess.run([executable, "init", "-q"], cwd=external, check=True, capture_output=True)
        (external / ".gitignore").write_text("ignored.md\n", encoding="utf-8")
    elif backend == "rg":
        (external / "ignored.md").unlink()
    monkeypatch.setenv("PATH", os.path.dirname(executable) if backend != "file" else "")
    target = source if scope == "file" else directory
    provider = ScriptedProvider([
        tool_response("search_code", {"pattern": "4.7", "path": str(target)}, "approved-search"),
        text_response("已找到批准范围内的内容"),
    ])
    approvals = []

    async def approve(tool_name, params, reason):
        approvals.append((tool_name, params["path"]))
        return True

    async with AgentRuntime(Config(workdir=str(workspace)), provider=provider) as runtime:
        runtime.permissions.approval_handler = approve
        events = await collect(runtime)
        completed = next(event for event in events if event.type == "tool_done")
        assert not completed.data["is_error"], completed.tool_output
        result = json.loads(completed.tool_output)
        assert result["match_count"] == 1
        assert result["results"][0]["path"] == source.as_posix()
        assert result["results"][0]["line"] == 2
        assert approvals == [("search_code", str(target))]
        assert events[-1].type == "done"
        reused = await runtime.registry.dispatch("search_code", {"pattern": "4.7", "path": str(target)})
        assert reused.error and "路径逃逸工作区" in reused.error
        assert runtime.workdir == workspace.resolve()


@pytest.mark.asyncio
@pytest.mark.parametrize("approval", ["reject", "no_handler"])
async def test_runtime_denies_external_search_without_approval(tmp_path, approval):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    external = tmp_path / "outside.md"
    external.write_text("unapproved value\n", encoding="utf-8")
    provider = ScriptedProvider([
        tool_response("search_code", {"pattern": "value", "path": str(external)}),
        text_response("没有批准，未搜索外部文件"),
    ])

    async def reject(*_args):
        return False

    async with AgentRuntime(Config(workdir=str(workspace)), provider=provider) as runtime:
        runtime.permissions.approval_handler = reject if approval == "reject" else None
        events = await collect(runtime)
        result = next(event for event in events if event.type == "tool_done")
        assert result.data["is_error"]
        assert ("Permission denied by user" if approval == "reject" else "no handler") in result.tool_output
        assert "unapproved value" not in result.tool_output
        assert events[-1].type == "done"


@pytest.mark.asyncio
async def test_runtime_finishes_external_read_when_approval_is_rejected(tmp_path):
    external_file = tmp_path.parent / f"{tmp_path.name}-rejected.md"
    external_file.write_text("不可读取", encoding="utf-8")
    provider = ScriptedProvider([
        tool_response("read_file", {"path": str(external_file)}, "external-read"),
        text_response("已说明无法读取"),
    ])

    async def reject(_tool_name, _params, _reason):
        return False

    async with AgentRuntime(Config(workdir=str(tmp_path)), provider=provider) as runtime:
        runtime.permissions.approval_handler = reject
        events = await asyncio.wait_for(collect(runtime), timeout=2)
        result = runtime.messages[2]["content"][0]
        assert result["is_error"] is True
        assert "Permission denied by user" in result["content"]
        assert "不可读取" not in result["content"]
        assert events[-1].type == "done"


@pytest.mark.asyncio
async def test_runtime_continues_after_mixed_reads_fail_on_different_targets(tmp_path):
    batches = []
    for index in range(3):
        name = f"available-{index}.txt"
        (tmp_path / name).write_text(f"内容 {index}", encoding="utf-8")
        read = tool_response("read_file", {"path": name}, f"read-{index}")
        missing = tool_response("read_file", {"path": f"missing-{index}.txt"}, f"missing-{index}")
        read.content.extend(missing.content)
        batches.append(read)
    (tmp_path / "followup.txt").write_text("后续步骤的结果", encoding="utf-8")
    provider = ScriptedProvider([
        *batches, tool_response("read_file", {"path": "followup.txt"}, "followup"),
        text_response("已完成后续检查"),
    ])

    async with AgentRuntime(Config(workdir=str(tmp_path)), provider=provider) as runtime:
        events = await collect(runtime)

    completed = {event.tool_id: event for event in events if event.type == "tool_done"}
    for index in range(3):
        assert completed[f"read-{index}"].tool_output == f"内容 {index}"
        assert completed[f"missing-{index}"].data["is_error"]
    assert events[-1].data["stop_reason"] == "end_turn"
    assert completed["followup"].tool_output == "后续步骤的结果"


@pytest.mark.asyncio
async def test_runtime_stops_after_equivalent_read_failures_even_with_other_successes(tmp_path):
    missing = tmp_path / "missing.txt"
    batches = []
    for index, path in enumerate((missing.name, "./missing.txt", str(missing))):
        name = f"available-{index}.txt"
        (tmp_path / name).write_text("其他目标的内容", encoding="utf-8")
        failed = tool_response("read_file", {"path": path}, f"missing-{index}")
        failed.content.extend(tool_response("read_file", {"path": name}, f"read-{index}").content)
        batches.append(failed)
    provider = ScriptedProvider(batches)

    async with AgentRuntime(Config(workdir=str(tmp_path)), provider=provider) as runtime:
        events = await collect(runtime)

    assert sum(event.type == "tool_done" and event.data["is_error"] for event in events) == 3
    assert sum(event.type == "tool_done" and not event.data["is_error"] for event in events) == 3
    assert events[-1].data["stop_reason"] == "repeated_tool_failure"
    assert "missing.txt" in events[-1].text


@pytest.mark.asyncio
async def test_runtime_clears_read_failure_streak_after_same_target_recovers(tmp_path):
    source = tmp_path / "changing.txt"
    other = tool_response("read_file", {"path": "unrelated.txt"}, "unrelated")
    recovered = tool_response("read_file", {"path": str(source)}, "recovered")
    recovered.content.extend(other.content)
    provider = ScriptedProvider([
        tool_response("read_file", {"path": source.name}, "before-1"),
        tool_response("read_file", {"path": source.name}, "before-2"),
        recovered,
        tool_response("read_file", {"path": source.name}, "after-1"),
        tool_response("read_file", {"path": source.name}, "after-2"),
        text_response("已记录恢复和后续读取错误"),
    ])
    seen = {}
    events = []
    async with AgentRuntime(Config(workdir=str(tmp_path)), provider=provider) as runtime:
        async for event in runtime.run_turn("检查变化中的文件"):
            events.append(event)
            if event.type != "tool_done":
                continue
            seen[event.tool_id] = event
            if event.tool_id == "before-2":
                source.write_text("文件已恢复", encoding="utf-8")
            if event.tool_id == "recovered":
                source.unlink()

    assert events[-1].data["stop_reason"] == "end_turn"
    assert seen["recovered"].tool_output == "文件已恢复"
    assert seen["unrelated"].data["is_error"]
    assert all(seen[call_id].data["is_error"] for call_id in ("before-1", "before-2", "after-1", "after-2"))


@pytest.mark.asyncio
@pytest.mark.parametrize(("commands", "exit_codes", "stop_reason"), [
    (["demo"] * 3, [1, 1, 1], "repeated_tool_failure"),
    (["demo-a", "demo-b", "demo-c"], [1, 1, 1], "end_turn"),
    (["demo"] * 3, [1, 2, 1], "end_turn"),
])
async def test_runtime_distinguishes_shell_targets_and_structured_failures(tmp_path, commands, exit_codes, stop_reason):
    provider = ScriptedProvider([
        *(tool_response("bash", {"command": command}, f"shell-{index}")
          for index, command in enumerate(commands)),
        text_response("已记录各次命令失败"),
    ])
    async with AgentRuntime(Config(workdir=str(tmp_path)), provider=provider) as runtime:
        from coding_agent.tools.base import ToolResult

        attempts = []

        def fail(command):
            attempts.append(command)
            index = len(attempts) - 1
            return ToolResult("", error=f"Error: demo command failed (attempt {index + 1})",
                              execution={"failure_stage": "nonzero_exit", "exit_code": exit_codes[index]})

        runtime.registry.register_handler(
            {"name": "bash", "description": "", "input_schema": {"type": "object"}},
            fail,
        )
        events = await asyncio.wait_for(collect(runtime), timeout=2)
    assert attempts == commands
    assert sum(event.type == "tool_done" and event.data["is_error"] for event in events) == 3
    assert events[-1].type == "done"
    assert events[-1].data["stop_reason"] == stop_reason
    if stop_reason == "repeated_tool_failure":
        assert "bash" in events[-1].text
        assert "attempt 3" in events[-1].text
        assert "停止重复调用" in events[-1].text


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
async def test_runtime_reuses_unchanged_read_result_without_repeating_full_output(tmp_path):
    source = tmp_path / "note.txt"
    source.write_text("证据内容" * 30, encoding="utf-8")
    provider = ScriptedProvider([
        tool_response("read_file", {"path": "note.txt"}, "read-first"),
        tool_response("read_file", {"path": "note.txt"}, "read-again"),
        text_response("证据已核对"),
    ])
    bus = EventBus()
    observed = []

    async def record(event):
        observed.append((event.type, event.data.get("tool_id"), event.data.get("reused_result")))

    bus.subscribe(EventType.PRE_TOOL_USE, record)
    bus.subscribe(EventType.POST_TOOL_USE, record)
    async with AgentRuntime(Config(workdir=str(tmp_path)), provider=provider, event_bus=bus) as runtime:
        await collect(runtime, "检查 note.txt")
        first_result = runtime.messages[2]["content"][0]["content"]
        repeated_result = runtime.messages[4]["content"][0]["content"]
        assert first_result == "证据内容" * 30
        assert "read-first" in repeated_result
        assert "与当前文件内容一致" in repeated_result
        assert len(repeated_result) < len(first_result)
        assert sum(item[0] == EventType.PRE_TOOL_USE for item in observed) == 2
        assert sum(item[0] == EventType.POST_TOOL_USE for item in observed) == 2
        assert observed[-1][2] is True
        assert "metadata" not in provider.calls[1]["messages"][2]["content"][0]


@pytest.mark.asyncio
async def test_runtime_does_not_reuse_different_read_ranges(tmp_path):
    (tmp_path / "note.txt").write_text("first\nsecond\n", encoding="utf-8")
    provider = ScriptedProvider([
        tool_response("read_file", {"path": "note.txt", "start_line": 1, "end_line": 1}, "line-one"),
        tool_response("read_file", {"path": "note.txt", "start_line": 2, "end_line": 2}, "line-two"),
        text_response(),
    ])
    async with AgentRuntime(Config(workdir=str(tmp_path)), provider=provider) as runtime:
        await collect(runtime)
        results = [message["content"][0]["content"] for message in runtime.messages
                   if message.get("role") == "user" and isinstance(message.get("content"), list)
                   and message["content"] and message["content"][0].get("type") == "tool_result"]
        assert results[0].startswith("1: first\n")
        assert results[1] == "2: second"
        assert all("reused_from" not in result.get("metadata", {}) for message in runtime.messages
                   if isinstance(message.get("content"), list) for result in message["content"]
                   if isinstance(result, dict) and result.get("type") == "tool_result")


@pytest.mark.asyncio
async def test_runtime_does_not_reuse_hook_truncated_read_result(tmp_path):
    from coding_agent.hooks.bus import OutputTruncateHook

    (tmp_path / "note.txt").write_text("长内容" * 100, encoding="utf-8")
    provider = ScriptedProvider([
        tool_response("read_file", {"path": "note.txt"}, "truncated-first"),
        tool_response("read_file", {"path": "note.txt"}, "truncated-again"),
        text_response(),
    ])
    observed = []

    async def record(event):
        observed.append(event.data.get("reused_result"))

    bus = EventBus()
    bus.subscribe(EventType.POST_TOOL_USE, OutputTruncateHook(max_chars=20, truncate=True), priority=10)
    bus.subscribe(EventType.POST_TOOL_USE, record)
    async with AgentRuntime(Config(workdir=str(tmp_path)), provider=provider, event_bus=bus) as runtime:
        await collect(runtime)
        assert observed == [False, False]
        assert all("truncated at 20" in message["content"][0]["content"]
                   for message in runtime.messages if message.get("role") == "user"
                   and isinstance(message.get("content"), list)
                   and message["content"] and message["content"][0].get("type") == "tool_result")


@pytest.mark.asyncio
async def test_runtime_never_reuses_repeated_write_tools(tmp_path):
    provider = ScriptedProvider([
        tool_response("write_file", {"path": "note.txt", "content": "value"}, "write-first"),
        tool_response("write_file", {"path": "note.txt", "content": "value"}, "write-again"),
        text_response(),
    ])
    post_calls = []

    async def record(event):
        post_calls.append(event.data["tool_name"])

    bus = EventBus()
    bus.subscribe(EventType.POST_TOOL_USE, record)
    async with AgentRuntime(Config(workdir=str(tmp_path)), provider=provider, event_bus=bus) as runtime:
        await collect(runtime)
        assert (tmp_path / "note.txt").read_text(encoding="utf-8") == "value"
        assert post_calls == ["write_file", "write_file"]
        assert all(not result.get("metadata", {}).get("reused_from")
                   for message in runtime.messages if isinstance(message.get("content"), list)
                   for result in message["content"] if isinstance(result, dict)
                   and result.get("type") == "tool_result")


@pytest.mark.asyncio
async def test_runtime_rereads_file_when_contents_change_between_calls(tmp_path):
    source = tmp_path / "note.txt"
    source.write_text("旧内容", encoding="utf-8")
    tool_calls = 0

    def handle(messages, tools, system):
        nonlocal tool_calls
        if tools is None:
            return text_response("维护完成")
        if tool_calls == 0:
            tool_calls += 1
            return tool_response("read_file", {"path": "note.txt"}, "read-old")
        if tool_calls == 1:
            source.write_text("新内容", encoding="utf-8")
            tool_calls += 1
            return tool_response("read_file", {"path": "note.txt"}, "read-new")
        return text_response("已读取更新内容")

    provider = ScriptedProvider(handler=handle)
    async with AgentRuntime(Config(workdir=str(tmp_path)), provider=provider) as runtime:
        await collect(runtime, "检查 note.txt")

    results = [message["content"][0]["content"] for message in runtime.messages
               if message.get("role") == "user" and isinstance(message.get("content"), list)
               and message["content"] and message["content"][0].get("type") == "tool_result"]
    assert results == ["旧内容", "新内容"]


@pytest.mark.asyncio
async def test_runtime_reuses_read_evidence_across_user_turns(tmp_path):
    (tmp_path / "note.txt").write_text("跨轮证据", encoding="utf-8")
    provider = ScriptedProvider([
        tool_response("read_file", {"path": "note.txt"}, "turn-one-read"),
        text_response("第一轮完成"),
        tool_response("read_file", {"path": "note.txt"}, "turn-two-read"),
        text_response("第二轮完成"),
    ])
    async with AgentRuntime(Config(workdir=str(tmp_path)), provider=provider) as runtime:
        await collect(runtime, "读取 note.txt")
        await collect(runtime, "再次检查 note.txt")
        repeated = next(message["content"][0]["content"] for message in reversed(runtime.messages)
                        if message.get("role") == "user" and isinstance(message.get("content"), list)
                        and message["content"] and message["content"][0].get("tool_use_id") == "turn-two-read")
        assert "turn-one-read" in repeated
        assert "跨轮证据" not in repeated
        assert "与当前文件内容一致" in repeated


@pytest.mark.asyncio
async def test_runtime_does_not_reuse_failed_file_read_result(tmp_path):
    read_count = 0

    def handle(messages, tools, system):
        nonlocal read_count
        read_count += 1
        if read_count == 1:
            return tool_response("read_file", {"path": "later.txt"}, "missing-read")
        if read_count == 2:
            (tmp_path / "later.txt").write_text("现在可读", encoding="utf-8")
            return tool_response("read_file", {"path": "later.txt"}, "recovered-read")
        return text_response()

    provider = ScriptedProvider(handler=handle)
    async with AgentRuntime(Config(workdir=str(tmp_path)), provider=provider) as runtime:
        await collect(runtime)
        results = [message["content"][0] for message in runtime.messages
                   if message.get("role") == "user" and isinstance(message.get("content"), list)
                   and message["content"] and message["content"][0].get("type") == "tool_result"]
        assert len(results) == 2
        assert results[0]["content"].startswith("Error:")
        assert "metadata" not in results[0]
        assert results[1]["content"] == "现在可读"
        assert results[1]["metadata"]["repeat_policy"] == "fresh_read"


@pytest.mark.asyncio
async def test_runtime_tracks_cross_turn_reuse_without_stopping_new_requests(tmp_path):
    (tmp_path / "note.txt").write_text("跨轮稳定证据", encoding="utf-8")
    provider = ScriptedProvider([
        tool_response("read_file", {"path": "note.txt"}, "cross-turn-1"), text_response("第一轮"),
        tool_response("read_file", {"path": "note.txt"}, "cross-turn-2"), text_response("第二轮"),
        tool_response("read_file", {"path": "note.txt"}, "cross-turn-3"), text_response("第三轮"),
    ])
    async with AgentRuntime(Config(workdir=str(tmp_path), max_iterations=10), provider=provider) as runtime:
        for turn in range(3):
            events = await collect(runtime, f"第 {turn + 1} 次检查 note.txt")
        reads = [message["content"][0] for message in runtime.messages
                 if message.get("role") == "user" and isinstance(message.get("content"), list)
                 and message["content"] and message["content"][0].get("type") == "tool_result"]
        assert len(reads) == 3
        assert reads[-1]["metadata"]["repeat_count"] == 2
        assert events[-1].data["stop_reason"] == "end_turn"


@pytest.mark.asyncio
async def test_runtime_stops_after_three_read_only_rounds_without_new_evidence(tmp_path):
    (tmp_path / "note.txt").write_text("稳定证据", encoding="utf-8")

    def handle(messages, tools, system):
        if tools == []:
            return text_response("根据已有证据，文件内容为稳定证据。")
        call_number = sum(1 for message in messages if message.get("role") == "assistant"
                          for block in message.get("content", [])
                          if isinstance(block, dict) and block.get("type") == "tool_use")
        return tool_response("read_file", {"path": "note.txt"}, f"repeat-{call_number}")

    provider = ScriptedProvider(handler=handle)
    async with AgentRuntime(Config(workdir=str(tmp_path), max_iterations=10), provider=provider) as runtime:
        events = await collect(runtime, "读取 note.txt 并总结")
        tool_calls = [block for message in runtime.messages if message.get("role") == "assistant"
                      for block in message.get("content", [])
                      if isinstance(block, dict) and block.get("type") == "tool_use"]
        assert len(tool_calls) == 4
        assert events[-1].data["stop_reason"] == "no_progress"
        assert "连续只读调用未产生新证据" in events[-1].text
        assert "稳定证据" in events[-1].text


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
@pytest.mark.parametrize("mode", ["snip", "proactive", "reactive"])
async def test_runtime_preserves_current_request_through_compaction_and_resume(tmp_path, mode):
    constraint = "唯一约束：仅核对接口，不要写入文件。"
    prompt = "请核对接口。" + "前置资料 " * 700 + constraint + " 后置资料" * 700
    responses = []
    for index in range(7):
        path = f"note-{index}.txt"
        (tmp_path / path).write_text("接口说明", encoding="utf-8")
        responses.append(tool_response("read_file", {"path": path}, f"read-{index}"))
    if mode == "reactive":
        responses.append(model_http_error(400, message="context_length_exceeded"))
    responses.extend([ConnectionError("断开"), text_response("已核对")])
    provider = ScriptedProvider(responses)
    cfg = Config(workdir=str(tmp_path), max_messages=8,
                 context_limit=1000 if mode == "proactive" else 50000,
                 model_max_attempts=2 if mode == "reactive" else 1)
    async with AgentRuntime(cfg, provider=provider) as runtime:
        runtime.messages.extend([
            {"role": "user", "content": "历史请求"},
            {"role": "assistant", "content": "历史答复"},
        ] * 2)
        events = await collect(runtime, prompt, stream=False)
        failure = events[-1]
        assert failure.type == "error"
        resumed = [event async for event in runtime.retry_turn(failure.data["recovery_id"], stream=False)]
    assert resumed[-1].text == "已核对"
    assert not any(event.type == "tool_start" for event in resumed)
    calls = [call["messages"] for call in provider.calls if call["tools"] is not None]
    markers = {"snip": "[snipped ", "proactive": "[Compacted]", "reactive": "[Reactive compact]"}
    assert any(message["content"].startswith(markers[mode]) for call in calls for message in call
               if isinstance(message["content"], str))
    if mode != "snip":
        summaries = [call["messages"][0]["content"] for call in provider.calls
                     if call["tools"] is None and call["messages"][0]["content"].startswith("请摘要")]
        assert any(constraint in summary for summary in summaries)
    for messages in calls:
        assert str(messages).count(constraint) == 1
        assert sum(message.get("content") == prompt for message in messages) == 1
        blocks = [block for message in messages if isinstance(message["content"], list)
                  for block in message["content"]]
        assert [block["id"] for block in blocks if block["type"] == "tool_use"] == [
            block["tool_use_id"] for block in blocks if block["type"] == "tool_result"
        ]


@pytest.mark.asyncio
async def test_runtime_reports_context_limit_if_provider_rejects_preserved_request(tmp_path):
    prompt = "请核对接口。" + "前置资料 " * 700 + "只读检查" + " 后置资料" * 700
    provider = ScriptedProvider([
        model_http_error(400, message="context_length_exceeded"),
        model_http_error(400, message="context_length_exceeded"),
    ])
    cfg = Config(workdir=str(tmp_path), model_max_attempts=3)
    async with AgentRuntime(cfg, provider=provider) as runtime:
        events = await collect(runtime, prompt, stream=False)

    assert events[-1].type == "error"
    assert "上下文超出限制" in events[-1].text
    assert not any(event.type in {"done", "tool_start"} for event in events)
    calls = [call["messages"] for call in provider.calls if call["tools"] is not None]
    assert len(calls) == 2
    assert all(sum(message.get("content") == prompt for message in messages) == 1 for messages in calls)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [True, False])
async def test_runtime_retries_rate_limit_and_reactively_compacts(tmp_path, monkeypatch, stream):
    monkeypatch.setattr("coding_agent.context.recovery.asyncio.sleep", AsyncMock())
    provider = ScriptedProvider([
        model_http_error(429), model_http_error(400, message="context_length_exceeded"), text_response(),
    ])
    async with AgentRuntime(Config(workdir=str(tmp_path)), provider=provider) as runtime:
        runtime.messages.extend({"role": "user", "content": "历史"} for _ in range(8))
        events = await collect(runtime, stream=stream)
        assert events[-1].text == "完成"
        assert any("[Reactive compact]" in str(call["messages"]) for call in provider.calls)
        assert sum(event.type == "text_delta" for event in events) == 1


@pytest.mark.asyncio
async def test_connection_recovers_on_fourth_model_attempt(tmp_path, monkeypatch):
    monkeypatch.setattr("coding_agent.context.recovery.asyncio.sleep", AsyncMock())
    provider = ScriptedProvider([ConnectionError("断开")] * 3 + [text_response("已恢复")])
    async with AgentRuntime(Config(workdir=str(tmp_path)), provider=provider) as runtime:
        events = await collect(runtime)
    assert all(event.type != "error" for event in events)
    assert events[-1].type == "done" and events[-1].text == "已恢复"
    waits = [event.data for event in events if event.data.get("phase") == "retry_wait"]
    assert [data["next_attempt"] for data in waits] == [2, 3, 4]


@pytest.mark.asyncio
async def test_resume_uses_completed_results_and_remaining_iterations(tmp_path):
    provider = ScriptedProvider([
        tool_response("write_file", {"path": "once.txt", "content": "first"}, "write-once"),
        ConnectionError("断开"),
        text_response("已核对写入"),
    ])
    cfg = Config(workdir=str(tmp_path), model_max_attempts=1, max_iterations=2)
    async with AgentRuntime(cfg, provider=provider) as runtime:
        events = await collect(runtime, "写入并核对")
        failed = events[-1]
        assert failed.type == "error" and failed.data["attempt"] == 1
        assert (tmp_path / "once.txt").read_text(encoding="utf-8") == "first"
        resumed = [event async for event in runtime.retry_turn(failed.data["recovery_id"])]
        assert not any(event.type == "tool_start" for event in resumed)
        assert resumed[-1].type == "done" and resumed[-1].text == "已核对写入"
        request_history = [call["messages"] for call in provider.calls if call["tools"] is not None][-1]
        assert request_history[0]["content"] == "写入并核对"
        assert request_history[-1]["content"][0]["tool_use_id"] == "write-once"
        assert "本轮未完成" not in str(request_history)
        with pytest.raises(RetryUnavailable):
            await anext(runtime.retry_turn(failed.data["recovery_id"]))


@pytest.mark.asyncio
@pytest.mark.parametrize("field, new_value", [("model_id", "new-model"), ("base_url", "https://example.invalid")])
async def test_resume_rejects_changed_provider(tmp_path, field, new_value):
    provider = ScriptedProvider([ConnectionError("断开")])
    async with AgentRuntime(Config(workdir=str(tmp_path), model_max_attempts=1), provider=provider) as runtime:
        recovery_id = (await collect(runtime))[-1].data["recovery_id"]
        setattr(provider, field, new_value)
        with pytest.raises(RetryUnavailable, match="历史或配置已变化"):
            await anext(runtime.retry_turn(recovery_id))
        assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_resume_rejects_parallel_claim_and_stale_history(tmp_path):
    started = asyncio.Event()
    release = asyncio.Event()

    class WaitingProvider(ScriptedProvider):
        async def stream_message(self, *args, **kwargs):
            if len(self.calls) == 1:
                started.set()
                await release.wait()
            async for event in super().stream_message(*args, **kwargs):
                yield event

    provider = WaitingProvider([ConnectionError("断开"), ConnectionError("再次断开")])
    async with AgentRuntime(Config(workdir=str(tmp_path), model_max_attempts=1), provider=provider) as runtime:
        first_id = (await collect(runtime))[-1].data["recovery_id"]

        async def resume():
            return [event async for event in runtime.retry_turn(first_id)]

        resumed = asyncio.create_task(resume())
        try:
            await asyncio.wait_for(started.wait(), timeout=2)
            with pytest.raises(RetryUnavailable, match="仍在执行"):
                await anext(runtime.retry_turn(first_id))
        finally:
            release.set()
            events = await resumed
        second_id = events[-1].data["recovery_id"]
        assert second_id != first_id
        runtime.messages[0]["content"] = "会话历史已被更新"
        with pytest.raises(RetryUnavailable, match="历史或配置已变化"):
            await anext(runtime.retry_turn(second_id))
        assert len(provider.calls) == 2


@pytest.mark.asyncio
async def test_cooldown_prevents_resume_and_new_turn_invalidates_id(tmp_path):
    provider = ScriptedProvider([model_http_error(429, {"Retry-After": "25"}), text_response()])
    cfg = Config(workdir=str(tmp_path), model_timeout=3)
    async with AgentRuntime(cfg, provider=provider) as runtime:
        failed = (await collect(runtime))[-1]
        assert failed.type == "error" and failed.data["retry_after_seconds"] >= 24
        with pytest.raises(RetryUnavailable, match="冷却"):
            await anext(runtime.retry_turn(failed.data["recovery_id"]))
        assert len(provider.calls) == 1
        assert (await collect(runtime, "新任务"))[-1].type == "done"
        with pytest.raises(RetryUnavailable, match="没有匹配"):
            await anext(runtime.retry_turn(failed.data["recovery_id"]))


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
    primary = ScriptedProvider([model_http_error(529)] * 3)
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
async def test_model_failure_after_tool_execution_is_delivered_and_skips_memory(tmp_path):
    from unittest.mock import AsyncMock

    provider = ScriptedProvider([
        tool_response("write_file", {"path": "result.txt", "content": "written"}),
        ModelRequestError("模型服务暂不可用"),
    ])
    async with AgentRuntime(Config(workdir=str(tmp_path)), provider=provider) as runtime:
        runtime.memory.extract = AsyncMock(return_value=0)
        events = await collect(runtime, "写入结果并总结")

        assert any(event.type == "tool_done" and event.tool_name == "write_file"
                   and not event.data["is_error"] for event in events)
        failures = [event for event in events if event.type == "error"]
        assert len(failures) == 1
        assert "write_file" in failures[0].text
        assert "执行成功" in failures[0].text
        assert "模型服务暂不可用" in failures[0].text
        assert not any(event.type == "done" for event in events)
        runtime.memory.extract.assert_not_awaited()
        assert (tmp_path / "result.txt").read_text(encoding="utf-8") == "written"
        assert runtime.messages[-1]["role"] == "assistant"
        assert not runtime.turn_lock.locked()

        provider.responses.append(text_response("下一轮正常完成"))
        next_events = await collect(runtime, "继续检查")
        assert any(event.type == "done" and "下一轮正常完成" in event.text for event in next_events)


@pytest.mark.asyncio
async def test_child_runtime_returns_model_failure_instead_of_empty_success(tmp_path):
    from coding_agent.tools.registry import ToolRegistry

    provider = ScriptedProvider([ModelRequestError("模型服务暂不可用")])
    async with AgentRuntime(Config(workdir=str(tmp_path)), provider=provider) as runtime:
        result = await runtime.run_child(
            [{"role": "user", "content": "执行子任务"}], "",
            agent_name="worker", registry=ToolRegistry(), max_iterations=2,
        )

    assert "子任务模型请求失败" in result
    assert "结果未确认" in result
    assert "模型服务暂不可用" in result


@pytest.mark.asyncio
async def test_stream_failure_after_text_is_not_replayed(tmp_path):
    class InterruptedProvider(ScriptedProvider):
        async def stream_message(self, *args, **kwargs):
            yield StreamEvent(type="text_delta", text="已输出")
            raise ModelRequestError("模型响应连接中断")

    provider = InterruptedProvider()
    async with AgentRuntime(Config(workdir=str(tmp_path)), provider=provider) as runtime:
        events = await collect(runtime, "测试")
        assert [event.type for event in events] == ["text_delta", "error"]
        assert events[0].text == "已输出"
        assert "已显示部分内容，未自动重放" in events[-1].text
        assert not any(event.type == "done" for event in events)


@pytest.mark.asyncio
@pytest.mark.parametrize(("deltas", "final_text", "expected_text", "expected_events"), [
    (["  ", "\n"], "answer", "answer", [("text_delta", "answer")]),
    (["  \n", "answer\n", "  code"], "  \nanswer\n  code", "  \nanswer\n  code",
     [("text_delta", "  \nanswer\n"), ("text_delta", "  code")]),
    (["answer"], "answer extended", "answer extended",
     [("text_delta", "answer"), ("text_delta", " extended")]),
    (["answer"], "answer", "answer", [("text_delta", "answer")]),
    (["partial"], "authoritative", "authoritative",
     [("text_delta", "partial"), ("text_reset", "authoritative")]),
])
async def test_stream_reconciles_deltas_with_final_response(
        tmp_path, deltas, final_text, expected_text, expected_events):
    class ChunkedProvider(ScriptedProvider):
        async def stream_message(self, *args, **kwargs):
            response = text_response(final_text)
            for delta in deltas:
                yield StreamEvent(type="text_delta", text=delta)
            yield StreamEvent(type="done", response=response)

    provider = ChunkedProvider()
    async with AgentRuntime(Config(workdir=str(tmp_path)), provider=provider) as runtime:
        events = await collect(runtime, "检查流式正文")

    actual_events = [(event.type, event.text) for event in events
                     if event.type in {"text_delta", "text_reset"}]
    assert actual_events == expected_events
    assert events[-1].type == "done"
    assert events[-1].text == expected_text
    if expected_events[-1][0] == "text_reset":
        reset_event = next(event for event in events if event.type == "text_reset")
        assert reset_event.data["reason"] == "response_reconciliation"


@pytest.mark.asyncio
@pytest.mark.parametrize("deltas, final_text", [
    (["  ", "\n"], "  \n"),
    (["旧正文"], ""),
    (["旧正文"], "  \n"),
])
async def test_blank_final_response_uses_nonblank_final_delivery(tmp_path, deltas, final_text):
    class WhitespaceProvider(ScriptedProvider):
        async def stream_message(self, *args, **kwargs):
            for delta in deltas:
                yield StreamEvent(type="text_delta", text=delta)
            response = text_response(final_text)
            yield StreamEvent(type="done", response=response)

        async def create_message(self, messages, tools=None, system="", max_tokens=8000):
            if tools == []:
                return text_response("空白响应未作为正文显示。")
            return await super().create_message(messages, tools, system, max_tokens)

    async with AgentRuntime(Config(workdir=str(tmp_path)), provider=WhitespaceProvider()) as runtime:
        events = await collect(runtime, "检查空白响应")

    visible_events = [(event.type, event.text) for event in events
                      if event.type in {"text_delta", "text_reset"}]
    expected = [("text_delta", "空白响应未作为正文显示。")]
    if any(delta.strip() for delta in deltas):
        expected[:0] = [("text_delta", "旧正文"), ("text_reset", "")]
    assert visible_events == expected
    assert events[-1].type == "done"
    assert events[-1].text == "空白响应未作为正文显示。"


@pytest.mark.asyncio
async def test_connection_retry_discards_partial_tool_call_and_executes_once(tmp_path, monkeypatch):
    import httpx
    import openai

    monkeypatch.setattr("coding_agent.context.recovery.asyncio.sleep", AsyncMock())

    class InterruptedProvider(ScriptedProvider):
        attempts = 0

        async def stream_message(self, *args, **kwargs):
            self.attempts += 1
            if self.attempts == 1:
                yield StreamEvent(type="tool_use_start", tool_name="write_file", tool_id="incomplete")
                yield StreamEvent(type="tool_input", tool_id="incomplete", text='{"path":')
                raise openai.APIConnectionError(request=httpx.Request("POST", "https://example.invalid"))
            async for event in super().stream_message(*args, **kwargs):
                yield event

    provider = InterruptedProvider([
        tool_response("write_file", {"path": "result.txt", "content": "complete"}, "complete"),
        text_response(),
    ])
    async with AgentRuntime(Config(workdir=str(tmp_path)), provider=provider) as runtime:
        events = await collect(runtime)
        assert sum(event.type == "tool_start" for event in events) == 1
        assert sum(event.type == "tool_done" for event in events) == 1
        assert "incomplete" not in str(runtime.messages)
        assert provider.attempts == 3
        assert (tmp_path / "result.txt").read_text() == "complete"


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


@pytest.mark.asyncio
async def test_runtime_keeps_persisted_output_reference_after_micro_compaction(tmp_path):
    (tmp_path / "large.txt").write_text("evidence-" + "x" * 40000, encoding="utf-8")
    provider = ScriptedProvider([
        tool_response("read_file", {"path": "large.txt"}, "first-read"),
        tool_response("read_file", {"path": "large.txt"}, "second-read"),
        tool_response("read_file", {"path": "large.txt"}, "third-read"),
        tool_response("read_file", {"path": "large.txt"}, "fourth-read"),
        text_response(),
    ])
    async with AgentRuntime(Config(workdir=str(tmp_path)), provider=provider) as runtime:
        await collect(runtime)
        agent_calls = [call for call in provider.calls if call["tools"] is not None]
        old_result = str(agent_calls[-1]["messages"])

    assert "<persisted-output>" in old_result
    assert ".task_outputs/tool-results/first-read.txt" in old_result
    assert "完整结果已保存" in old_result


@pytest.mark.parametrize("changes", [
    {"context_limit": 0}, {"max_messages": 2}, {"keep_recent": -1},
    {"max_iterations": False}, {"team_idle_timeout": -1}, {"mcp_servers": "docs"},
    {"model_timeout": 0}, {"model_idle_timeout": 0}, {"model_connect_timeout": float("inf")},
    {"final_answer_timeout": float("nan")}, {"maintenance_timeout": float("inf")},
    {"progress_interval": float("nan")}, {"model_timeout": True},
    {"model_max_attempts": True}, {"model_max_attempts": 0}, {"model_max_attempts": 21},
])
def test_invalid_runtime_options_fail_before_creating_provider(tmp_path, changes):
    from unittest.mock import Mock

    factory = Mock()
    with pytest.raises(ValueError):
        AgentRuntime(Config(workdir=str(tmp_path), **changes), provider_factory=factory)
    factory.assert_not_called()


def test_model_connection_and_idle_timeouts_are_independent():
    from coding_agent.config import validate_config

    validate_config(Config(model_connect_timeout=5, model_idle_timeout=1))


def test_non_transient_generation_error_is_not_retried(monkeypatch):
    from coding_agent.context.recovery import RecoveryState, with_retry

    sleep = AsyncMock()
    monkeypatch.setattr("coding_agent.context.recovery.asyncio.sleep", sleep)
    call = AsyncMock(side_effect=RuntimeError("generate request failed: 401 Unauthorized"))
    with pytest.raises(RuntimeError, match="401"):
        asyncio.run(with_retry(call, RecoveryState()))
    assert call.await_count == 1
    sleep.assert_not_called()


@pytest.mark.asyncio
async def test_many_tools_continue_after_stalled_compaction_and_memory_requests(tmp_path):
    stopped = []

    class Provider(ScriptedProvider):
        async def create_message(self, messages, tools=None, **options):
            if tools is None:
                try:
                    await asyncio.Event().wait()
                finally:
                    stopped.append(True)
            return await super().create_message(messages, tools=tools, **options)

    (tmp_path / "note.txt").write_text("真实文件内容", encoding="utf-8")
    blocks = [{"type": "tool_use", "id": f"read-{index}", "name": "read_file",
               "input": {"path": "note.txt"}} for index in range(40)]
    provider = Provider([ProviderResponse(content=blocks, stop_reason="tool_use"), text_response("读取完成")])
    cfg = Config(workdir=str(tmp_path), context_limit=1000, maintenance_timeout=0.12,
                 progress_interval=0.02)
    async with AgentRuntime(cfg, provider=provider) as runtime:
        errors = []

        async def record(event):
            errors.append(event.data)

        runtime.bus.subscribe(EventType.ERROR, record)
        async with asyncio.timeout(3):
            events = await collect(runtime)
        assert sum(event.type == "tool_done" for event in events) == 40
        assert events[-1].text == "读取完成"
        assert events[-1].type == "done"
        assert len(stopped) == 2
        assert any(event.data.get("phase") == "整理工具输出和压缩上下文" for event in events)
        assert any(event.data.get("phase") == "提取和保存记忆" for event in events)
        assert any("已停止等待" in event.get("error", "") for event in errors)
        assert not runtime.workspace_lock.locked()
        assert "真实文件内容" in str(provider.calls[-1]["messages"])


@pytest.mark.asyncio
async def test_cancelling_memory_phase_releases_workspace_and_turn_locks(tmp_path):
    started = asyncio.Event()
    stopped = asyncio.Event()

    class Provider(ScriptedProvider):
        async def create_message(self, messages, tools=None, **options):
            if tools is None:
                started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    stopped.set()
            return await super().create_message(messages, tools=tools, **options)

    async with AgentRuntime(Config(workdir=str(tmp_path), progress_interval=0.02),
                            provider=Provider([text_response()])) as runtime:
        events = runtime.run_turn("开始")
        assert (await anext(events)).type == "text_delta"
        event = await asyncio.wait_for(anext(events), 2)
        assert event.type == "warning"
        assert started.is_set()
        await events.aclose()
        assert stopped.is_set()
        assert not runtime.workspace_lock.locked()
        assert not runtime.turn_lock.locked()
