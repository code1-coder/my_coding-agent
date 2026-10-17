"""
工具注册表、权限检查流水线和事件总线测试。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Literal

import pytest

from coding_agent.tools.base import Tool, ToolResult
from coding_agent.tools.registry import ToolRegistry


# ====================================================================
#  工具注册表（ToolRegistry）
# ====================================================================

@pytest.fixture
def registry() -> ToolRegistry:
    reg = ToolRegistry()

    @reg.register(name="echo", description="Echo back", permission="read")
    async def echo(message: str) -> str:
        return f"echo: {message}"

    @reg.register(name="add", description="Add two numbers", permission="read")
    async def add(a: int, b: int = 0) -> str:
        return str(a + b)

    @reg.register(name="greet", description="Greet with optional style")
    async def greet(name: str, style: str = "friendly") -> str:
        return f"Hello {name} ({style})"

    return reg


class TestToolRegistry:
    async def test_list_tools_returns_anthropic_format(self, registry):
        tools = registry.list_tools()
        assert len(tools) == 3
        names = {t["name"] for t in tools}
        assert names == {"echo", "add", "greet"}
        # 每个工具都应具有有效的 JSON Schema
        for t in tools:
            assert t["input_schema"]["type"] == "object"
            assert isinstance(t["input_schema"]["properties"], dict)

    async def test_dispatch_returns_tool_result(self, registry):
        result = await registry.dispatch("echo", {"message": "hello"}, tool_use_id="id_1")
        assert isinstance(result, ToolResult)
        assert result.content == "echo: hello"
        assert result.tool_use_id == "id_1"
        assert result.error is None

    async def test_dispatch_unknown_tool_raises(self, registry):
        with pytest.raises(KeyError, match="Unknown tool"):
            await registry.dispatch("no_such_tool", {})

    async def test_dispatch_catches_exceptions(self, registry):
        @registry.register(name="crash", description="Always crashes")
        async def crash() -> str:
            raise RuntimeError("boom")

        result = await registry.dispatch("crash", {})
        assert result.error == "boom"

    async def test_json_schema_required_vs_optional(self, registry):
        # add 函数的参数 a 必填，参数 b 可选
        entry = registry.get("add")
        assert entry is not None
        assert entry.input_schema["required"] == ["a"]
        assert "b" in entry.input_schema["properties"]

    async def test_json_schema_literal(self):
        reg = ToolRegistry()

        @reg.register(name="pick", description="Pick a color")
        async def pick(color: Literal["red", "green", "blue"]) -> str:
            return color

        schema = reg.get("pick").input_schema
        assert schema["properties"]["color"]["enum"] == ["red", "green", "blue"]

    async def test_contains_and_len(self, registry):
        assert "echo" in registry
        assert "nope" not in registry
        assert len(registry) == 3

    async def test_register_tool_adapter(self):
        """现有的 Tool 子类可以通过 register_tool() 注册。"""
        class FakeTool(Tool):
            name = "fake"
            description = "A fake tool"
            input_schema = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}

            def execute(self, x: int) -> str:
                return f"got {x}"

        reg = ToolRegistry()
        reg.register_tool(FakeTool())

        entry = reg.get("fake")
        assert entry is not None
        assert entry.name == "fake"
        assert entry.permission == "write"

        result = await reg.dispatch("fake", {"x": 42})
        assert result.content == "got 42"

    def test_sync_function(self, registry):
        """同步函数应直接调用，不通过 to_thread 转发。"""
        reg = ToolRegistry()

        @reg.register(name="sync_echo", description="Sync echo")
        def sync_echo(text: str) -> str:
            return f"sync: {text}"

        result = asyncio.run(reg.dispatch("sync_echo", {"text": "hi"}))
        assert result.content == "sync: hi"


# ====================================================================
#  权限检查流水线（PermissionPipeline）
# ====================================================================

from coding_agent.security.permission import (
    DenyListChecker,
    DestructiveCommandChecker,
    PermissionPipeline,
    Verdict,
    WorkspaceChecker,
    create_terminal_pipeline,
)


class TestDenyListChecker:
    async def test_blocks_matching_pattern(self):
        checker = DenyListChecker(["rm -rf /", "sudo"])
        # “rm -rf /”是“rm -rf /tmp”的子串，优先采用首个匹配规则
        verdict, reason = await checker.check("bash", {"command": "sudo rm -rf /tmp"})
        assert verdict == Verdict.DENY
        assert "rm -rf /" in reason  # 按遍历顺序采用第一个匹配规则

    async def test_allows_clean_command(self):
        checker = DenyListChecker(["rm -rf /", "sudo"])
        verdict, reason = await checker.check("bash", {"command": "ls -la"})
        assert verdict == Verdict.ALLOW

    async def test_ignores_non_bash_tools(self):
        checker = DenyListChecker(["rm -rf /"])
        verdict, reason = await checker.check("read_file", {"path": "foo.txt"})
        assert verdict == Verdict.ALLOW


class TestWorkspaceChecker:
    async def test_allows_internal_path(self, tmp_path):
        checker = WorkspaceChecker(workdir=tmp_path)
        (tmp_path / "a.txt").write_text("hi")
        verdict, reason = await checker.check("read_file", {"path": "a.txt"})
        assert verdict == Verdict.ALLOW

    async def test_flags_external_path(self, tmp_path):
        checker = WorkspaceChecker(workdir=tmp_path)
        verdict, reason = await checker.check("write_file", {"path": "../../etc/passwd"})
        assert verdict == Verdict.NEEDS_APPROVAL
        assert "outside workspace" in reason

    async def test_ignores_non_file_tools(self):
        checker = WorkspaceChecker()
        verdict, reason = await checker.check("bash", {"command": "ls"})
        assert verdict == Verdict.ALLOW


class TestDestructiveCommandChecker:
    async def test_flags_rm(self):
        checker = DestructiveCommandChecker()
        verdict, reason = await checker.check("bash", {"command": "rm -rf build/"})
        assert verdict == Verdict.NEEDS_APPROVAL
        assert "rm " in reason

    async def test_allows_safe_command(self):
        checker = DestructiveCommandChecker()
        verdict, reason = await checker.check("bash", {"command": "echo hello"})
        assert verdict == Verdict.ALLOW


class TestPermissionPipeline:
    async def test_chain_short_circuits_on_deny(self):
        """第一道拒绝检查命中时，应在进入第二道检查前终止。"""
        pipeline = PermissionPipeline(
            checkers=[
                DenyListChecker(["sudo"]),
                DestructiveCommandChecker(),
            ],
        )
        verdict, reason = await pipeline.check("bash", {"command": "sudo ls"})
        assert verdict == Verdict.DENY
        assert "sudo" in reason

    async def test_pipeline_calls_approval_handler(self):
        called_with: list = []

        async def handler(tool_name, params, reason):
            called_with.append((tool_name, params, reason))
            return True  # 允许执行

        pipeline = PermissionPipeline(
            checkers=[DestructiveCommandChecker(["rm "])],
            approval_handler=handler,
        )
        verdict, reason = await pipeline.check("bash", {"command": "rm file.txt"})
        assert verdict == Verdict.ALLOW
        assert len(called_with) == 1

    async def test_pipeline_denies_if_approval_rejected(self):
        async def handler(*_):
            return False

        pipeline = PermissionPipeline(
            checkers=[DestructiveCommandChecker(["rm "])],
            approval_handler=handler,
        )
        verdict, reason = await pipeline.check("bash", {"command": "rm file.txt"})
        assert verdict == Verdict.DENY
        assert "denied by user" in reason

    async def test_no_approval_handler_denies_by_default(self):
        """未配置 approval_handler 时，NEEDS_APPROVAL 应转为 DENY。"""
        pipeline = PermissionPipeline(
            checkers=[DestructiveCommandChecker()],
            approval_handler=None,
        )
        verdict, reason = await pipeline.check("bash", {"command": "rm file.txt"})
        assert verdict == Verdict.DENY
        assert "no handler" in reason.lower()

    async def test_factory_creates_working_pipeline(self):
        pipeline = create_terminal_pipeline(
            workdir=Path.cwd(),
            deny_list=["mkfs"],
            destructive_patterns=["chmod 777"],
        )
        assert len(pipeline.checkers) == 3


# ====================================================================
#  事件总线（EventBus）
# ====================================================================

from coding_agent.hooks.bus import (
    Event,
    EventBus,
    EventType,
    OutputTruncateHook,
    TokenUsageHook,
)


class TestEventBus:
    async def test_emits_to_subscribers(self):
        bus = EventBus()
        seen: list[str] = []

        async def hook_a(event: Event):
            seen.append("a")
            return None

        bus.subscribe(EventType.PRE_TOOL_USE, hook_a)
        await bus.emit(Event(EventType.PRE_TOOL_USE, data={}))
        assert seen == ["a"]

    async def test_priority_ordering(self):
        bus = EventBus()
        order: list[str] = []

        async def hook_high(event: Event):
            order.append("high")
            return None

        async def hook_low(event: Event):
            order.append("low")
            return None

        bus.subscribe(EventType.PRE_TOOL_USE, hook_low, priority=80)
        bus.subscribe(EventType.PRE_TOOL_USE, hook_high, priority=10)

        await bus.emit(Event(EventType.PRE_TOOL_USE))
        assert order == ["high", "low"]

    async def test_short_circuit_on_non_none(self):
        bus = EventBus()
        called: list[str] = []

        async def blocker(event: Event):
            called.append("blocker")
            return "DENIED"

        async def never_reached(event: Event):
            called.append("never")
            return None

        bus.subscribe(EventType.PRE_TOOL_USE, blocker, priority=10)
        bus.subscribe(EventType.PRE_TOOL_USE, never_reached, priority=20)

        result = await bus.emit(Event(EventType.PRE_TOOL_USE))
        assert result == "DENIED"
        assert called == ["blocker"]

    async def test_on_decorator(self):
        bus = EventBus()
        called = False

        @bus.on(EventType.USER_PROMPT_SUBMIT, priority=5)
        async def my_hook(event: Event):
            nonlocal called
            called = True
            return None

        await bus.emit(Event(EventType.USER_PROMPT_SUBMIT))
        assert called

    async def test_unsubscribe(self):
        bus = EventBus()
        calls: list[str] = []

        async def h(event: Event):
            calls.append("h")
            return None

        bus.subscribe(EventType.STOP, h)
        await bus.emit(Event(EventType.STOP))
        assert len(calls) == 1

        bus.unsubscribe(EventType.STOP, h)
        await bus.emit(Event(EventType.STOP))
        assert len(calls) == 1  # 不应再次调用

    async def test_hook_exception_is_suppressed(self):
        bus = EventBus()
        survived = False

        async def crashy(event: Event):
            raise RuntimeError("boom")

        async def survivor(event: Event):
            nonlocal survived
            survived = True
            return None

        bus.subscribe(EventType.ERROR, crashy, priority=0)
        bus.subscribe(EventType.ERROR, survivor, priority=99)

        await bus.emit(Event(EventType.ERROR))
        assert survived

    async def test_registered_events(self):
        bus = EventBus()

        async def h(e): return None

        bus.subscribe(EventType.PRE_TOOL_USE, h)
        bus.subscribe(EventType.POST_TOOL_USE, h)
        assert EventType.PRE_TOOL_USE in bus.registered_events
        assert bus.hook_count() == 2


class TestTokenUsageHook:
    async def test_accumulates_usage(self):
        hook = TokenUsageHook()
        event = Event(
            EventType.POST_LLM_CALL,
            data={"usage": {"input_tokens": 100, "output_tokens": 50}},
        )
        await hook(event)
        assert hook.input_tokens == 100
        assert hook.output_tokens == 50
        assert hook.total_tokens == 150

        # 第二次调用
        await hook(Event(EventType.POST_LLM_CALL, data={"usage": {"input_tokens": 20, "output_tokens": 10}}))
        assert hook.input_tokens == 120
        assert hook.output_tokens == 60
        assert hook.total_tokens == 180

    async def test_writes_to_context(self):
        hook = TokenUsageHook()
        ctx: dict = {}
        event = Event(EventType.POST_LLM_CALL, data={"usage": {"input_tokens": 30, "output_tokens": 15}}, context=ctx)
        await hook(event)
        assert ctx["total_input_tokens"] == 30
        assert ctx["total_output_tokens"] == 15

    async def test_ignores_non_llm_events(self):
        hook = TokenUsageHook()
        await hook(Event(EventType.USER_PROMPT_SUBMIT, data={}))
        assert hook.total_tokens == 0


class TestOutputTruncateHook:
    async def test_no_warning_for_small_output(self, caplog):
        hook = OutputTruncateHook(max_chars=100, truncate=False)
        await hook(Event(EventType.POST_TOOL_USE, data={"output": "short"}))
        # 不应产生警告级别的日志
        assert not any(r.levelname == "WARNING" for r in caplog.records)

    async def test_warns_for_large_output(self, caplog):
        import logging as _logging
        hook = OutputTruncateHook(max_chars=10, truncate=False)
        with caplog.at_level(_logging.WARNING):
            await hook(Event(EventType.POST_TOOL_USE, data={"tool_name": "bash", "output": "x" * 100}))
        assert any("Large output" in r.message for r in caplog.records)

    async def test_truncate_mode(self):
        hook = OutputTruncateHook(max_chars=20, truncate=True)
        data = {"tool_name": "bash", "output": "x" * 100}
        event = Event(EventType.POST_TOOL_USE, data=data)
        await hook(event)
        assert len(event.data["output"]) < 100
        assert "truncated" in event.data["output"]
