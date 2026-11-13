"""
错误恢复、子智能体池、任务看板、文件邮箱和生命周期测试。
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# ====================================================================
#  恢复管理器（RecoveryManager）与重试装饰器（with_retry）
# ====================================================================

from coding_agent.context.recovery import (
    CONTINUATION_PROMPT,
    RecoveryManager,
    RecoveryState,
    is_prompt_too_long_error,
    with_retry,
)


class TestIsPromptTooLong:
    def test_detects_prompt_too_long(self):
        err = Exception("prompt_is_too_long: context exceeds limit")
        assert is_prompt_too_long_error(err)

    def test_detects_context_length_exceeded(self):
        err = Exception("context_length_exceeded: max 200K tokens")
        assert is_prompt_too_long_error(err)

    def test_rejects_other_errors(self):
        err = Exception("something went wrong")
        assert not is_prompt_too_long_error(err)


class TestWithRetry:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("kind", ["openai", "anthropic", "httpx", "timeout", "incomplete"])
    async def test_retries_transport_errors(self, monkeypatch, kind):
        import anthropic
        import httpx
        import openai
        from coding_agent.providers.base import IncompleteStreamError

        request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
        errors = {
            "openai": openai.APIConnectionError(request=request),
            "anthropic": anthropic.APIConnectionError(request=request),
            "httpx": httpx.RemoteProtocolError("incomplete chunked read"),
            "timeout": TimeoutError("timed out"),
            "incomplete": IncompleteStreamError("missing finish_reason"),
        }
        sleep = AsyncMock()
        monkeypatch.setattr("coding_agent.context.recovery.asyncio.sleep", sleep)
        call = AsyncMock(side_effect=[errors[kind], "ok"])
        assert await with_retry(call, RecoveryState()) == "ok"
        assert call.await_count == 2
        sleep.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_transport_retries_are_bounded(self, monkeypatch):
        import httpx
        import openai
        from coding_agent.providers.base import ModelRequestError

        sleep = AsyncMock()
        monkeypatch.setattr("coding_agent.context.recovery.asyncio.sleep", sleep)
        error = openai.APIConnectionError(request=httpx.Request("POST", "https://example.invalid"))
        call = AsyncMock(side_effect=error)
        with pytest.raises(ModelRequestError, match="3 次尝试") as failure:
            await with_retry(call, RecoveryState())
        assert failure.value.__cause__ is error
        assert call.await_count == 3 and sleep.await_count == 2

    @pytest.mark.asyncio
    async def test_cancellation_is_not_retried(self, monkeypatch):
        sleep = AsyncMock()
        monkeypatch.setattr("coding_agent.context.recovery.asyncio.sleep", sleep)
        call = AsyncMock(side_effect=asyncio.CancelledError())
        with pytest.raises(asyncio.CancelledError):
            await with_retry(call, RecoveryState())
        assert call.await_count == 1
        sleep.assert_not_called()

    @pytest.mark.asyncio
    async def test_success_first_try(self, monkeypatch):
        monkeypatch.setattr("asyncio.sleep", AsyncMock())
        state = RecoveryState(current_model="test-model")
        fn = AsyncMock(return_value="ok")
        result = await with_retry(fn, state)
        assert result == "ok"
        assert fn.call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_429(self, monkeypatch):
        monkeypatch.setattr("asyncio.sleep", AsyncMock())
        state = RecoveryState(current_model="test-model")
        call_count = 0

        async def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("429 rate limit exceeded")
            return "ok"

        result = await with_retry(flaky, state)
        assert result == "ok"
        assert call_count == 3
        assert state.total_429_retries == 2

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [408, 409, 500, 503, 599])
    async def test_retries_transient_http_statuses_with_a_bound(self, monkeypatch, status):
        from unittest.mock import AsyncMock
        from coding_agent.providers.base import ModelRequestError

        sleep = AsyncMock()
        monkeypatch.setattr("coding_agent.context.recovery.asyncio.sleep", sleep)

        class StatusError(Exception):
            status_code = status

        call = AsyncMock(side_effect=[StatusError(f"HTTP {status}"), "ok"])
        assert await with_retry(call, RecoveryState()) == "ok"
        assert call.await_count == 2
        sleep.assert_awaited_once()

        failing_call = AsyncMock(side_effect=StatusError(f"HTTP {status}"))
        with pytest.raises(ModelRequestError, match=f"HTTP {status}"):
            await with_retry(failing_call, RecoveryState())
        assert failing_call.await_count == 3

    @pytest.mark.asyncio
    async def test_529_fallback_model(self, monkeypatch):
        monkeypatch.setattr("asyncio.sleep", AsyncMock())
        state = RecoveryState(current_model="primary")

        async def always_overloaded():
            raise Exception("529 overloaded")

        with pytest.raises(RuntimeError, match="Max retries"):
            await with_retry(always_overloaded, state, fallback_model="fallback")

        # 连续出现 3 次 529 错误后，应切换模型
        assert state.current_model == "fallback"

    @pytest.mark.asyncio
    async def test_raises_non_transient_immediately(self, monkeypatch):
        monkeypatch.setattr("asyncio.sleep", AsyncMock())
        state = RecoveryState(current_model="test")

        async def auth_error():
            raise Exception("401 Unauthorized")

        with pytest.raises(Exception, match="401"):
            await with_retry(auth_error, state)


class TestRecoveryManager:
    @pytest.mark.asyncio
    async def test_path2_compact_called_once(self):
        compact_fn = AsyncMock(return_value=[{"role": "user", "content": "[compacted]"}])
        mgr = RecoveryManager(compact_fn=compact_fn)
        state = RecoveryState(current_model="test")

        call_count = 0

        async def llm():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("prompt_is_too_long: too many tokens")
            return "ok"

        result = await mgr.call(llm, state, [{"role": "user", "content": "x" * 10000}])
        assert result == "ok"
        assert state.total_compacts == 1

    @pytest.mark.asyncio
    async def test_path2_unrecoverable_after_compact(self):
        compact_fn = AsyncMock(return_value=[{"role": "user", "content": "[compacted]"}])
        mgr = RecoveryManager(compact_fn=compact_fn)
        state = RecoveryState(current_model="test")

        async def always_long():
            raise Exception("prompt_is_too_long")

        result = await mgr.call(always_long, state, [{"role": "user", "content": "x"}])
        # 首次调用时执行压缩；再次调用仍失败时，应判定为不可恢复。
        assert result is None

    def test_path1_escalation(self):
        mgr = RecoveryManager()
        state = RecoveryState()

        # 首先提高输出令牌上限
        assert mgr.handle_max_tokens(state, [], MagicMock()) is True
        assert state.has_escalated is True
        assert mgr.current_max_tokens(state) == 64000

    def test_path1_continuation_prompt(self):
        mgr = RecoveryManager()
        state = RecoveryState(has_escalated=True)
        messages: list = [{"role": "user", "content": "hi"}]

        # 提高上限至 64K 后仍被截断时，应注入续写提示词
        response = MagicMock()
        response.content = [{"type": "text", "text": "truncated"}]

        assert mgr.handle_max_tokens(state, messages, response) is True
        assert state.recovery_count == 1
        # 续写提示词应已注入
        assert CONTINUATION_PROMPT in messages[-1]["content"]

    def test_path1_exhausted(self):
        mgr = RecoveryManager(max_recovery_retries=1)
        state = RecoveryState(has_escalated=True, recovery_count=1)
        messages: list = [{"role": "user", "content": "hi"}]

        response = MagicMock()
        response.content = [{"type": "text", "text": "truncated again"}]

        assert mgr.handle_max_tokens(state, messages, response) is False


# ====================================================================
#  子智能体池（AgentPool）
# ====================================================================

from coding_agent.agents.subagent import AgentPool


class TestAgentPool:
    def test_pool_creates_specs(self, tmp_path):
        pool = AgentPool(workdir=tmp_path, size=3)
        assert len(pool.available_roles) == 3
        # 按探索者、执行者、探索者的顺序交替分配角色
        assert pool.available_roles == ["explorer", "worker", "explorer"]

    @pytest.mark.asyncio
    async def test_dispatch_without_llm_call_returns_error(self, tmp_path):
        pool = AgentPool(workdir=tmp_path, size=1)
        result = await pool.dispatch("test task")
        assert "Error" in result
        assert "llm_call" in result.lower()

    @pytest.mark.asyncio
    async def test_dispatch_runs_subagent_loop(self, tmp_path):
        """端到端测试：子智能体调用模型一次，并返回 end_turn。"""
        async def mock_llm(messages, tools, system, max_tokens):
            class Resp:
                content = [{"type": "text", "text": "Found 3 files."}]  # 使用字典格式的响应，而非 SDK 响应对象
                stop_reason = "end_turn"
            return Resp()

        pool = AgentPool(workdir=tmp_path, size=1, llm_call=mock_llm)
        result = await pool.dispatch("list files", role="explorer")
        assert "Found 3 files." in result

    @pytest.mark.asyncio
    async def test_dispatch_executes_tools(self, tmp_path):
        """子智能体调用工具、接收结果，再返回文本。"""
        (tmp_path / "a.txt").write_text("hello")

        call_count = 0

        async def mock_llm(messages, tools, system, max_tokens):
            nonlocal call_count
            call_count += 1
            class Resp:
                if call_count == 1:
                    # 返回对 read_file 的工具调用
                    content = [{"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "a.txt"}}]
                    stop_reason = "tool_use"
                else:
                    content = [{"type": "text", "text": "The file says hello."}]
                    stop_reason = "end_turn"
            return Resp()

        pool = AgentPool(workdir=tmp_path, size=1, llm_call=mock_llm)
        result = await pool.dispatch("read file a.txt", role="explorer")
        assert "The file says hello." in result
        assert call_count == 2


# ====================================================================
#  任务看板（TaskBoard）
# ====================================================================

from coding_agent.agents.tasks import Task, TaskBoard


class TestTaskBoard:
    @pytest.fixture
    def board(self, tmp_path):
        return TaskBoard(workdir=tmp_path)

    def test_create_and_get(self, board):
        task = board.create("Test subject", "Test description")
        assert task.subject == "Test subject"
        assert task.status == "pending"
        loaded = board.get(task.id)
        assert loaded.subject == task.subject

    def test_list_all(self, board):
        board.create("A")
        board.create("B")
        assert len(board.list_all()) == 2

    def test_claim_and_complete(self, board):
        task = board.create("Do work")
        result = board.claim(task.id, owner="agent")
        assert "Claimed" in result
        t = board.get(task.id)
        assert t.status == "in_progress"
        assert t.owner == "agent"

        result = board.complete(task.id)
        assert "Completed" in result
        t = board.get(task.id)
        assert t.status == "completed"

    def test_cannot_claim_claimed_task(self, board):
        task = board.create("Do work")
        board.claim(task.id, owner="agent")
        result = board.claim(task.id, owner="other")
        # 任务已处于 in_progress 状态，因此应先触发状态检查
        assert "cannot claim" in result.lower()

    def test_blockedBy_dependency(self, board):
        dep = board.create("Dependency")
        board.claim(dep.id, owner="agent")
        board.complete(dep.id)

        blocked = board.create("Blocked", blockedBy=[dep.id])
        assert board.can_start(blocked.id) is True

    def test_cannot_start_with_uncompleted_dep(self, board):
        dep = board.create("Dep")
        blocked = board.create("Blocked", blockedBy=[dep.id])
        assert board.can_start(blocked.id) is False

    def test_complete_reports_unblocked(self, board):
        dep = board.create("Dep")
        board.claim(dep.id, owner="agent")

        child = board.create("Child", blockedBy=[dep.id])
        result = board.complete(dep.id)
        assert "Completed" in result

    def test_find_unclaimed(self, board):
        dep = board.create("Dep")
        board.claim(dep.id, "agent")
        board.complete(dep.id)

        ready = board.create("Ready", blockedBy=[dep.id])
        blocked = board.create("StillBlocked", blockedBy=["nonexistent"])

        unclaimed = board.find_unclaimed()
        ids = [t.id for t in unclaimed]
        assert ready.id in ids
        assert blocked.id not in ids

    def test_delete(self, board):
        task = board.create("To delete")
        assert board.delete(task.id) is True
        assert board.delete(task.id) is False
        with pytest.raises(FileNotFoundError):
            board.get(task.id)


# ====================================================================
#  文件邮箱（FileMailbox）
# ====================================================================

from coding_agent.agents.team import FileMailbox


class TestFileMailbox:
    @pytest.fixture
    def bus(self, tmp_path):
        return FileMailbox(workdir=tmp_path)

    def test_send_and_read(self, bus):
        bus.send("alice", "bob", "Hello Bob!")
        msgs = bus.read_inbox("bob")
        assert len(msgs) == 1
        assert msgs[0]["from"] == "alice"
        assert msgs[0]["content"] == "Hello Bob!"
        # 消息应已被读取并移除
        assert len(bus.read_inbox("bob")) == 0

    def test_peek_non_destructive(self, bus):
        bus.send("alice", "bob", "hi")
        assert bus.peek("bob") is True
        # 消息应仍然存在
        assert len(bus.read_inbox("bob")) == 1

    def test_peek_empty_inbox(self, bus):
        assert bus.peek("nobody") is False

    def test_message_metadata(self, bus):
        bus.send("lead", "worker", "Please run tests", msg_type="request", metadata={"priority": "high"})
        msgs = bus.read_inbox("worker")
        assert msgs[0]["type"] == "request"
        assert msgs[0]["metadata"]["priority"] == "high"


# ====================================================================
#  智能体生命周期（AgentLifecycle）
# ====================================================================

from coding_agent.agents.lifecycle import AgentLifecycle
from coding_agent.agents.team import FileMailbox


class TestAgentLifecycle:
    def test_instantiation(self, tmp_path):
        bus = FileMailbox(workdir=tmp_path)
        llm = MagicMock()

        lc = AgentLifecycle(
            name="test-agent",
            role="tester",
            bus=bus,
            llm_call=llm,
            workdir=tmp_path,
        )
        assert lc.name == "test-agent"
        assert lc.role == "tester"

    def test_run_end_turn_stops(self, tmp_path):
        bus = FileMailbox(workdir=tmp_path)

        class Resp:
            content = [{"type": "text", "text": "All done."}]
            stop_reason = "end_turn"

        llm = MagicMock(return_value=Resp)

        lc = AgentLifecycle(
            name="test-agent",
            role="tester",
            bus=bus,
            llm_call=llm,
            workdir=tmp_path,
            poll_interval=0.01,
            idle_timeout=0.02,
        )
        result = lc.run("Run tests")
        # 应完成任务并发送摘要
        assert "All done." in result
        llm.assert_called_once()

    def test_identity_injection(self, tmp_path):
        """消息较少时应注入身份说明。"""
        bus = FileMailbox(workdir=tmp_path)

        captured_messages = []

        class Resp:
            content = [{"type": "text", "text": "ok"}]
            stop_reason = "end_turn"

        def llm(messages, tools, system, max_tokens):
            captured_messages.extend(messages)
            return Resp()

        lc = AgentLifecycle(
            name="worker-05",
            role="backend-dev",
            bus=bus,
            llm_call=llm,
            workdir=tmp_path,
            poll_interval=0.01,
            idle_timeout=0.02,
        )
        lc.run("Do something")

        # 身份说明应位于首条消息中
        identity_found = any("worker-05" in str(m.get("content", "")) for m in captured_messages)
        assert identity_found

    def test_tool_use_cycle(self, tmp_path):
        """智能体调用工具后，应将执行结果回传到对话。"""
        bus = FileMailbox(workdir=tmp_path)
        (tmp_path / "out.txt").write_text("data")

        call_order = [0]

        def llm(messages, tools, system, max_tokens):
            call_order[0] += 1
            class Resp:
                if call_order[0] == 1:
                    content = [{"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "out.txt"}}]
                    stop_reason = "tool_use"
                else:
                    content = [{"type": "text", "text": "Read successfully."}]
                    stop_reason = "end_turn"
            return Resp()

        lc = AgentLifecycle(
            name="reader",
            role="reader",
            bus=bus,
            llm_call=llm,
            workdir=tmp_path,
            poll_interval=0.01,
            idle_timeout=0.02,
        )
        result = lc.run("read out.txt")
        assert "Read" in result
        assert call_order[0] == 2
