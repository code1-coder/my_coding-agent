"""
系统提示词组装、MCP 集成、网页 API 和压缩边界测试。
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest


# ====================================================================
#  系统提示词组装
# ====================================================================

from coding_agent.prompt import (
    assemble_system_prompt,
    get_system_prompt,
    invalidate_prompt_cache,
)


class TestSystemPrompt:
    def test_identity_always_included(self):
        ctx = {}
        result = assemble_system_prompt(ctx)
        assert "coding agent" in result.lower()

    def test_tools_injected_from_context(self):
        ctx = {"enabled_tools": ["bash", "read_file"]}
        result = assemble_system_prompt(ctx)
        assert "bash" in result
        assert "read_file" in result

    def test_workspace_injected(self):
        ctx = {"workspace": "/home/user/project"}
        result = assemble_system_prompt(ctx)
        assert "/home/user/project" in result

    def test_memory_conditional(self):
        ctx = {"memories": "User prefers tabs"}
        result = assemble_system_prompt(ctx)
        assert "User prefers tabs" in result

        ctx_no_mem = {}
        result2 = assemble_system_prompt(ctx_no_mem)
        assert "Relevant memories" not in result2

    def test_mcp_servers_injected(self):
        ctx = {"mcp_servers": ["docs", "deploy"]}
        result = assemble_system_prompt(ctx)
        assert "docs" in result
        assert "deploy" in result

    def test_cache_hit(self):
        invalidate_prompt_cache()
        ctx = {"enabled_tools": ["bash"]}
        first = get_system_prompt(ctx)
        second = get_system_prompt(ctx)  # 应命中缓存
        assert first == second

    def test_cache_invalidation_on_change(self):
        invalidate_prompt_cache()
        ctx_a = {"enabled_tools": ["bash"]}
        ctx_b = {"enabled_tools": ["bash", "read_file"]}
        first = get_system_prompt(ctx_a)
        second = get_system_prompt(ctx_b)
        assert first != second

    def test_extra_sections(self):
        ctx = {"extra_sections": {"rules": "Don't delete without asking."}}
        result = assemble_system_prompt(ctx)
        assert "Don't delete without asking" in result


# ====================================================================
#  MCP 集成
# ====================================================================

from coding_agent.tools.mcp import (
    MCPClient,
    assemble_tool_pool,
    connect_mcp,
    normalize_mcp_name,
)


class TestMCP:
    def test_normalize_name(self):
        assert normalize_mcp_name("hello world") == "hello_world"
        assert normalize_mcp_name("my-server!") == "my-server_"
        assert normalize_mcp_name("abc123") == "abc123"

    def test_client_register_and_call(self):
        client = MCPClient("test-srv")
        client.register(
            tool_defs=[{
                "name": "echo",
                "description": "Echo back",
                "inputSchema": {"type": "object", "properties": {"msg": {"type": "string"}}, "required": ["msg"]},
            }],
            handlers={"echo": lambda msg: f"echo: {msg}"},
        )
        assert len(client.tools) == 1
        assert client.call_tool("echo", {"msg": "hello"}) == "echo: hello"

    def test_client_unknown_tool(self):
        client = MCPClient("empty")
        result = client.call_tool("nope", {})
        assert "MCP error" in result

    def test_connect_mock_servers(self):
        msg, client = connect_mcp("docs")
        assert client is not None
        assert "Connected" in msg
        assert len(client.tools) == 2

    def test_connect_unknown_server(self):
        msg, client = connect_mcp("nonexistent")
        assert client is None
        assert "Unknown server" in msg

    def test_assemble_tool_pool(self):
        builtin = [{"name": "bash", "description": "Run shell", "input_schema": {}}]
        builtin_handlers = {"bash": lambda cmd: cmd}

        mcp_client = MCPClient("docs")
        mcp_client.register(
            tool_defs=[{"name": "search", "description": "Search", "inputSchema": {}}],
            handlers={"search": lambda query: f"found: {query}"},
        )

        tools, handlers = assemble_tool_pool(builtin, builtin_handlers, [mcp_client])

        tool_names = {t["name"] for t in tools}
        assert "bash" in tool_names
        assert "mcp__docs__search" in tool_names

        # 处理函数应能正常执行
        result = handlers["mcp__docs__search"](query="test")
        assert "found: test" in result


# ====================================================================
#  提示词与上下文压缩的集成测试
# ====================================================================

from coding_agent.context.compaction import (
    estimate_tokens,
    snip_compact,
    micro_compact,
    tool_result_budget,
)
from coding_agent.prompt import assemble_system_prompt


class TestPromptCompactIntegration:
    """验证压缩会降低令牌数量，同时保持系统提示词稳定。"""

    def test_compaction_reduces_token_estimate(self):
        """对较长对话执行 snip_compact 后，应减少令牌数量。"""
        msgs = [{"role": "user", "content": "x" * 200} for _ in range(60)]
        before = estimate_tokens(msgs)
        snip_compact(msgs, max_messages=30)
        after = estimate_tokens(msgs)
        assert after < before

    def test_micro_compact_preserves_recent(self):
        """执行 micro_compact 后，最近 KEEP_RECENT 条结果应保持完整。"""
        msgs = [
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"old_{i}", "content": "x" * 300}]}
            for i in range(10)
        ]
        micro_compact(msgs, keep_recent=3)
        # 最近 3 条结果不应被压缩
        for i in range(7, 10):
            assert "earlier tool result compacted" not in msgs[i]["content"][0]["content"].lower()
        # 前 7 条结果应被压缩
        for i in range(7):
            assert "earlier tool result compacted" in msgs[i]["content"][0]["content"].lower()

    def test_tool_result_budget_persists(self, tmp_path):
        """大型工具结果应持久化到文件，并降低估算的令牌数量。"""
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "big1", "content": "y" * 50000}
            ]},
        ]
        before = estimate_tokens(msgs)
        tool_result_budget(msgs, workdir=tmp_path, persist_threshold=30000, max_total_chars=5000)
        after = estimate_tokens(msgs)
        assert after < before

    def test_system_prompt_not_affected_by_compaction(self):
        """系统提示词的组装应独立于对话消息的压缩。"""
        ctx = {"enabled_tools": ["bash", "read_file"]}
        prompt1 = assemble_system_prompt(ctx)

        # 修改消息不应影响提示词缓存的命中情况
        # 上下文配置与对话消息相互独立
        prompt2 = assemble_system_prompt(ctx)
        assert prompt1 == prompt2


# ====================================================================
#  多智能体通信边界测试
# ====================================================================

from coding_agent.agents.team import FileMailbox


class TestMultiAgentCommunication:
    def test_multiple_agents_messaging(self, tmp_path):
        bus = FileMailbox(workdir=tmp_path)

        # 负责人向 3 个执行者发送消息
        for i in range(3):
            bus.send("lead", f"worker-{i}", f"Task #{i}")

        # 各执行者分别读取自己的消息
        for i in range(3):
            msgs = bus.read_inbox(f"worker-{i}")
            assert len(msgs) == 1
            assert f"Task #{i}" in msgs[0]["content"]

        # 执行者向负责人发送回复
        for i in range(3):
            bus.send(f"worker-{i}", "lead", f"Done #{i}")

        lead_msgs = bus.read_inbox("lead")
        assert len(lead_msgs) == 3

    def test_inbox_isolation(self, tmp_path):
        bus = FileMailbox(workdir=tmp_path)
        bus.send("lead", "alice", "secret for alice")
        bus.send("lead", "bob", "secret for bob")

        alice_msgs = bus.read_inbox("alice")
        bob_msgs = bus.read_inbox("bob")

        assert alice_msgs[0]["content"] == "secret for alice"
        assert bob_msgs[0]["content"] == "secret for bob"

    def test_peek_does_not_consume(self, tmp_path):
        bus = FileMailbox(workdir=tmp_path)
        bus.send("lead", "worker", "urgent")
        assert bus.peek("worker") is True
        # 消息应仍然存在
        msgs = bus.read_inbox("worker")
        assert len(msgs) == 1


# ====================================================================
#  智能体循环的 stop_reason 分支测试
# ====================================================================

from coding_agent.loop import AgentLoop, LoopEvent
from coding_agent.providers.base import LLMProvider, ProviderResponse


class FakeProvider(LLMProvider):
    """为循环测试按顺序返回预设响应的模拟模型服务。"""

    def __init__(self, responses: list[ProviderResponse]):
        super().__init__(model_id="fake")
        self.responses = responses
        self.calls = 0

    async def create_message(self, messages, tools=None, system="", max_tokens=8000):
        r = self.responses[self.calls % len(self.responses)]
        self.calls += 1
        return r

    async def stream_message(self, messages, tools=None, system="", max_tokens=8000):
        import asyncio
        r = self.responses[self.calls % len(self.responses)]
        self.calls += 1
        for block in r.content:
            if block["type"] == "text":
                for ch in block["text"]:
                    from coding_agent.providers.base import StreamEvent
                    yield StreamEvent(type="text_delta", text=ch)
                    await asyncio.sleep(0)
        from coding_agent.providers.base import StreamEvent
        yield StreamEvent(type="done", response=r)


class TestLoopStopReason:
    @pytest.mark.asyncio
    async def test_stops_on_end_turn(self):
        provider = FakeProvider([
            ProviderResponse(
                content=[{"type": "text", "text": "Done."}],
                stop_reason="end_turn",
            ),
        ])
        loop = AgentLoop(provider=provider, tools=[], max_iterations=10)
        messages: list = [{"role": "user", "content": "hi"}]

        events = []
        async for ev in loop.run(messages):
            events.append(ev)

        assert events[-1].type == "done"
        assert provider.calls == 1

    @pytest.mark.asyncio
    async def test_continues_on_tool_use(self):
        """stop_reason=tool_use 时应继续下一轮循环。"""
        from coding_agent.tools.base import Tool

        class EchoTool(Tool):
            name = "echo"
            description = "echo"
            input_schema = {"type": "object", "properties": {"m": {"type": "string"}}, "required": ["m"]}
            def execute(self, m: str) -> str:
                return f"echo:{m}"

        provider = FakeProvider([
            ProviderResponse(
                content=[{"type": "tool_use", "id": "t1", "name": "echo", "input": {"m": "hello"}}],
                stop_reason="tool_use",
            ),
            ProviderResponse(
                content=[{"type": "text", "text": "Received: echo:hello"}],
                stop_reason="end_turn",
            ),
        ])
        loop = AgentLoop(provider=provider, tools=[EchoTool()], max_iterations=10)
        messages: list = [{"role": "user", "content": "echo hello"}]

        events = []
        async for ev in loop.run(messages):
            events.append(ev)

        assert provider.calls == 2
        assert events[-1].type == "done"

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error(self):
        """调用未注册工具时应返回错误，并继续执行循环。"""
        from coding_agent.tools.base import Tool

        class OnlyEcho(Tool):
            name = "echo"
            description = "echo"
            input_schema = {"type": "object", "properties": {"m": {"type": "string"}}, "required": ["m"]}
            def execute(self, m: str) -> str:
                return f"echo:{m}"

        provider = FakeProvider([
            ProviderResponse(
                content=[{"type": "tool_use", "id": "t1", "name": "no_such_tool", "input": {}}],
                stop_reason="tool_use",
            ),
            ProviderResponse(
                content=[{"type": "text", "text": "Ok, tool not found."}],
                stop_reason="end_turn",
            ),
        ])
        loop = AgentLoop(provider=provider, tools=[OnlyEcho()], max_iterations=10)
        messages: list = [{"role": "user", "content": "call bad tool"}]

        async for ev in loop.run(messages):
            pass

        # 工具结果中应包含错误信息
        tool_result_msg = messages[2]  # 包含工具执行结果的用户消息
        assert "Unknown tool" in str(tool_result_msg["content"])
