"""
Agent 主循环 —— 编码智能体的核心大脑。

核心能力：

* **async** —— 所有 LLM 调用都是异步的，工具通过 :func:`asyncio.gather` 并发执行。
* **SSE 流式输出** —— 文本增量实时逐字产出，方便 CLI 逐字符打印。
* **工具并发执行** —— 当模型一次请求多个工具时，它们同时运行。
* **停止原因检查** —— 仅在 ``stop_reason == "tool_use"`` 时继续调用工具，
  其他停止原因结束当前循环。

用法::

    from coding_agent.loop import AgentLoop
    from coding_agent.providers import AnthropicProvider
    from coding_agent.tools import BUILTIN_TOOLS

    provider = AnthropicProvider(model_id="claude-sonnet-5")
    loop = AgentLoop(provider=provider, tools=BUILTIN_TOOLS)

    messages = [{"role": "user", "content": "List Python files"}]
    async for event in loop.run(messages):
        if event.type == "text_delta":
            print(event.text, end="", flush=True)
        elif event.type == "tool_start":
            print(f"\\n[{event.tool_name}] ", end="")
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import aclosing
from typing import Any, AsyncIterator, Callable

from coding_agent.providers.base import LLMProvider, ProviderResponse
from coding_agent.tools.base import Tool, ToolCancelledError, ToolResult, tool_result
from coding_agent.tools.registry import ToolRegistry
from coding_agent.hooks.bus import Event, EventBus, EventType
from coding_agent.security.permission import PermissionPipeline, Verdict
from coding_agent.context.compaction import CompactionPipeline
from coding_agent.context.recovery import RecoveryManager, RecoveryState
from coding_agent.providers.base import ModelRequestError, StreamEvent

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  循环事件（比 provider 的 StreamEvent 更丰富）
# ═══════════════════════════════════════════════════════════════

class LoopEvent:
    """:meth:`AgentLoop.run` 发出的事件。"""

    def __init__(
        self,
        type: str,
        text: str = "",
        tool_name: str = "",
        tool_id: str = "",
        tool_output: str = "",
        data: dict[str, Any] | None = None,
    ) -> None:
        self.type = type
        self.text = text
        self.tool_name = tool_name
        self.tool_id = tool_id
        self.tool_output = tool_output
        self.data = data or {}


# ═══════════════════════════════════════════════════════════════

class AgentLoop:
    """编排智能体循环：LLM → 工具 → LLM → … 。

    实例很轻量 —— 每个对话轮次都可以新建一个。
    """

    def __init__(
        self,
        provider: LLMProvider,
        tools: list[Tool],
        system_prompt: str = "",
        max_tokens: int = 8000,
        max_iterations: int = 50,
        registry: ToolRegistry | None = None,
        event_bus: EventBus | None = None,
        event_context: dict[str, Any] | None = None,
        permissions: PermissionPipeline | None = None,
        compaction: CompactionPipeline | None = None,
        recovery: RecoveryManager | None = None,
        agent_name: str = "lead",
        provider_for_model: Callable[[str], LLMProvider] | None = None,
        model_timeout: float = 180.0,
        progress_interval: float = 10.0,
    ) -> None:
        self.provider = provider
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.max_iterations = max_iterations
        self.registry = registry if registry is not None else ToolRegistry()
        for tool in tools:
            self.registry.register_tool(tool)
        self.event_bus = event_bus if event_bus is not None else EventBus()
        self.event_context = event_context if event_context is not None else {}
        self.permissions = permissions
        self.compaction = compaction
        self.recovery = recovery
        self.recovery_state = RecoveryState(current_model=provider.model_id)
        self.agent_name = agent_name
        self.provider_for_model = provider_for_model
        self.model_timeout = model_timeout
        self.progress_interval = progress_interval

    # ── 公共 API ──────────────────────────────────────────

    async def run(
        self, messages: list[dict[str, Any]], *, stream: bool = True,
    ) -> AsyncIterator[LoopEvent]:
        """执行本轮循环，并在取消或中断时修复尚未配对的工具结果。"""
        try:
            async with aclosing(self._run_impl(messages, stream=stream)) as events:
                async for event in events:
                    yield event
        finally:
            if messages and messages[-1].get("role") == "assistant":
                content = messages[-1].get("content", [])
                pending = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_use"] if isinstance(content, list) else []
                if pending:
                    messages.append({"role": "user", "content": [
                        tool_result(block["id"], error="本轮请求已中断，工具结果不可用。")
                        for block in pending
                    ]})

    async def _run_impl(
        self,
        messages: list[dict[str, Any]],
        *,
        stream: bool = True,
    ) -> AsyncIterator[LoopEvent]:
        """执行智能体循环。

        参数：
            messages: 对话历史（**原地修改**）。初始列表至少应包含
                       用户的起始消息。
            stream: 为 True 时，文本会实时产出（SSE 风格）。

        逐项产出：
            :class:`LoopEvent` 项，包括文本增量和最终的 ``done`` 事件。

        每轮先把模型响应追加到 messages，再并发执行请求的工具，并将
        工具结果作为用户消息写回。仅工具调用会触发下一轮；达到
        max_iterations 后也会产出 done，以防模型持续调用工具。
        """
        self.recovery_state = RecoveryState(current_model=self.provider.model_id)
        text = ""
        for iteration in range(self.max_iterations):
            if self.compaction is not None:
                async with aclosing(self.wait_events(
                    self.compaction.run(messages), "整理工具输出和压缩上下文",
                )) as events:
                    async for event in events:
                        yield event
            response = None
            prefix = text
            async with aclosing(self._model_events(messages, stream)) as events:
                async for ev in events:
                    if ev.type == "text_delta":
                        text += ev.text
                        yield LoopEvent(type="text_delta", text=ev.text)
                    elif ev.type == "done":
                        response = ev.response
                    elif ev.type == "warning":
                        yield ev
            if response is None:
                raise RuntimeError("模型未返回完整响应")

            if response.stop_reason == "max_tokens" and self.recovery is not None:
                escalated = self.recovery_state.has_escalated
                continuing = self.recovery.handle_max_tokens(
                    self.recovery_state, messages, response,
                )
                if continuing:
                    if not escalated:
                        text = prefix
                        yield LoopEvent(type="text_reset", text=prefix)
                    continue
                # 恢复管理器已追加最后一次截断响应。
            else:
                messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use" or not response.tool_use_blocks:
                if not response.text.strip() or response.stop_reason in ("max_tokens", "tool_use"):
                    reason = response.stop_reason if response.stop_reason != "end_turn" else "empty_response"
                    async for event in self._finish_without_answer(messages, text, reason):
                        yield event
                    return
                await self.emit(EventType.STOP, stop_reason=response.stop_reason, response=response)
                yield LoopEvent(type="done", text=text, data={"stop_reason": response.stop_reason})
                return

            blocks = response.tool_use_blocks
            execution_ids = [uuid.uuid4().hex for _ in blocks]
            for block, execution_id in zip(blocks, execution_ids):
                yield LoopEvent(type="tool_start", tool_name=block["name"],
                                tool_id=block["id"], data={"input": block.get("input", {}),
                                "execution_id": execution_id, "agent": self.agent_name})
            async with aclosing(self._tool_events(messages, blocks, execution_ids)) as events:
                async for event in events:
                    yield event

        async for event in self._finish_without_answer(messages, text, "max_iterations"):
            yield event

    async def _finish_without_answer(self, messages: list[dict[str, Any]], text: str,
                                     reason: str) -> AsyncIterator[LoopEvent]:
        """最多补充一次无工具总结；失败时明确告知缺少交付，不能静默结束。"""
        notices = {
            "max_iterations": "已达到本轮执行轮次上限，已停止继续调用工具；任务是否完成仍需依据执行结果确认。",
            "max_tokens": "模型输出达到长度上限，回答可能不完整。",
            "tool_use": "模型请求使用工具，但未提供有效工具调用。",
        }
        notice = notices.get(reason, "")
        request_messages = list(messages)
        if request_messages and request_messages[-1].get("role") == "assistant":
            last_content = request_messages[-1].get("content")
            empty = (not last_content.strip() if isinstance(last_content, str) else not last_content or
                     all(isinstance(block, dict) and block.get("type") == "text" and
                         not block.get("text", "").strip() for block in last_content))
            if empty:
                request_messages.pop()
                messages.pop()
        request_messages.append({"role": "user", "content": (
            "请立即向用户交付本轮结果，不要调用任何工具或继续执行任务。"
            "根据本轮对话和已有工具结果，说明实际完成事项、相关文件、实际验证、失败或未完成事项。"
            "不要把计划、测试通过或工具调用结束当作任务已完成，不要编造结果。"
            "请使用用户的语言。" + notice
        )})
        answer = ""
        try:
            async with asyncio.timeout(30), aclosing(
                self._model_events(request_messages, False, final_only=True),
            ) as events:
                async for event in events:
                    if event.type == "done":
                        response = event.response
                        if response.stop_reason == "end_turn" and not response.tool_use_blocks:
                            answer = response.text.strip()
                    elif event.type == "warning":
                        yield event
        except Exception:
            logger.exception("补充最终答复失败")
        generated = bool(answer)
        if not answer:
            answer = "未能自动生成本轮结果说明，不能确认任务已完整交付。请核对本轮工具执行记录与工作区改动。"
        delivery = "\n\n".join(part for part in (notice, answer) if part)
        messages.append({"role": "assistant", "content": [{"type": "text", "text": delivery}]})
        rendered = ("\n\n" if text else "") + delivery
        yield LoopEvent(type="text_delta", text=rendered)
        await self.emit(EventType.STOP, stop_reason=reason)
        yield LoopEvent(type="done", text=text + rendered,
                        data={"stop_reason": reason, "final_answer_generated": generated})

    async def emit(self, event_type: EventType, **data: Any) -> Any:
        """用同一会话上下文分发事件，使内置 Hook 和自定义 Hook 实际生效。"""
        return await self.event_bus.emit(Event(
            event_type, data={"agent": self.agent_name, **data}, context=self.event_context,
        ))

    async def wait_events(self, operation, phase: str) -> AsyncIterator[LoopEvent]:
        """等待异步阶段时定期报告状态，关闭生成器时回收操作。"""
        task = asyncio.create_task(operation)
        try:
            while not task.done():
                completed, _ = await asyncio.wait({task}, timeout=self.progress_interval)
                if not completed:
                    yield LoopEvent(type="warning", text=f"仍在{phase}，可取消当前请求。",
                                    data={"phase": phase, "agent": self.agent_name})
            await task
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def _tool_events(self, messages, blocks, execution_ids) -> AsyncIterator[LoopEvent]:
        tasks = [asyncio.create_task(self._execute_one(block, execution_id=execution_id))
                 for block, execution_id in zip(blocks, execution_ids)]
        indexes = {task: index for index, task in enumerate(tasks)}
        pending = set(tasks)
        try:
            while pending:
                completed, pending = await asyncio.wait(
                    pending, timeout=self.progress_interval, return_when=asyncio.FIRST_COMPLETED,
                )
                if not completed:
                    waiting = [{"tool_name": blocks[indexes[task]]["name"],
                                "tool_id": blocks[indexes[task]]["id"]} for task in tasks if task in pending]
                    labels = ", ".join(f"{item['tool_name']} ({item['tool_id']})" for item in waiting[:5])
                    yield LoopEvent(type="warning", text=f"仍在等待 {len(pending)} 个工具：{labels}；可取消当前请求。",
                                    data={"phase": "tools", "pending_tools": waiting, "agent": self.agent_name})
                for task in sorted(completed, key=indexes.__getitem__):
                    index = indexes[task]
                    block, result = blocks[index], task.result()
                    yield LoopEvent(type="tool_done", tool_name=block["name"],
                                    tool_id=block["id"], tool_output=result.error or result.content,
                                    data={"input": block.get("input", {}), "is_error": bool(result.error),
                                          "execution": result.execution,
                                          "execution_id": execution_ids[index], "agent": self.agent_name})
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            outcomes = await asyncio.gather(*tasks, return_exceptions=True)
            results = [outcome.to_dict() if isinstance(outcome, ToolResult) else
                       tool_result(block["id"], error="本轮请求已中断，工具结果不可用。")
                       for block, outcome in zip(blocks, outcomes)]
            messages.append({"role": "user", "content": results})

    async def _model_events(self, messages: list[dict[str, Any]], stream: bool, *, final_only: bool = False):
        """在恢复管理器中执行模型请求，同时通过队列实时转发文本增量。"""
        queue: asyncio.Queue = asyncio.Queue()
        sentinel = object()

        async def attempt():
            sent_text = False
            try:
                provider = (self.provider_for_model(self.recovery_state.current_model)
                            if self.provider_for_model else self.provider)
                blocked = await self.emit(EventType.PRE_LLM_CALL, messages=messages,
                                          purpose="agent", model=provider.model_id)
                if blocked is not None:
                    raise PermissionError(f"模型调用被 Hook 阻止：{blocked}")
                kwargs = dict(messages=messages, tools=[] if final_only else self.registry.list_tools(),
                              system=self.system_prompt, max_tokens=(
                                  self.recovery.current_max_tokens(self.recovery_state)
                                  if self.recovery else self.max_tokens))
                response = None
                if stream:
                    async with aclosing(provider.stream_message(**kwargs)) as events:
                        async for event in events:
                            if event.type == "text_delta":
                                sent_text = sent_text or bool(event.text)
                                queue.put_nowait(event)
                            elif event.type == "done":
                                response = event.response
                else:
                    response = await provider.create_message(**kwargs)
                    if response.text:
                        queue.put_nowait(StreamEvent(type="text_delta", text=response.text))
                if response is None:
                    raise RuntimeError("模型响应流未正常结束")
                if stream and not sent_text and response.text:
                    queue.put_nowait(StreamEvent(type="text_delta", text=response.text))
                await self.emit(EventType.POST_LLM_CALL, response=response,
                                usage=response.usage, purpose="agent")
                return response
            except Exception as exc:
                await self.emit(EventType.ERROR, error=str(exc), purpose="agent")
                if sent_text:
                    # 已显示的增量无法安全重放，避免重试后产生重复文本。
                    raise ModelRequestError("流式响应中断，已显示部分内容，未自动重放。请重试本轮请求。") from exc
                raise

        async def request():
            try:
                try:
                    async with asyncio.timeout(self.model_timeout):
                        if self.recovery is not None and not final_only:
                            response = await self.recovery.call(
                                attempt, self.recovery_state, messages, max_tokens=self.max_tokens,
                            )
                        else:
                            response = await attempt()
                except TimeoutError as exc:
                    raise ModelRequestError(
                        f"模型请求超过 {self.model_timeout:g} 秒（含重试与恢复），已停止等待。"
                        "请检查模型服务或调整 model_timeout 后重试。"
                    ) from exc
                if response is None:
                    raise RuntimeError("上下文压缩后请求仍超出模型限制")
                queue.put_nowait(StreamEvent(type="done", response=response))
            finally:
                queue.put_nowait(sentinel)

        task = asyncio.create_task(request())
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=self.progress_interval)
                except TimeoutError:
                    yield LoopEvent(type="warning", text="仍在等待模型响应或恢复，可取消当前请求。",
                                    data={"phase": "model", "agent": self.agent_name})
                    continue
                if event is sentinel:
                    break
                yield event
            await task
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    # ── 内部辅助方法 ────────────────────────────────────

    async def _execute_one(self, block: dict[str, Any], *, execution_id: str = "") -> ToolResult:
        """执行单个 tool_use 块，保留执行信息与原调用 ID。

        未注册的工具和执行异常均转换为错误结果，交由模型在后续轮次
        处理，避免单个工具失败直接中断同轮的其他工具调用。
        """
        tool_name = block["name"]
        tool_id = block["id"]
        tool_input = block.get("input", {})

        error = None
        output = ""
        execution = {}
        cancellation = None
        try:
            blocked = await self.emit(EventType.PRE_TOOL_USE, tool_name=tool_name,
                                      params=tool_input, tool_id=tool_id, execution_id=execution_id)
            if blocked is not None:
                raise PermissionError(f"工具被 Hook 阻止：{blocked}")
            if self.permissions is not None:
                verdict, reason = await self.permissions.check(tool_name, tool_input)
                if verdict != Verdict.ALLOW:
                    raise PermissionError(reason)
            result = await self.registry.dispatch(tool_name, tool_input, tool_use_id=tool_id)
            output, error = result.content, result.error
            execution = result.execution
        except ToolCancelledError as exc:
            cancellation = exc
            output, error = exc.result.content, exc.result.error
            execution = exc.result.execution
        except Exception as exc:
            error = str(exc)
        if error:
            await self.emit(EventType.ERROR, tool_name=tool_name, error=error)
        event = Event(EventType.POST_TOOL_USE, data={
            "agent": self.agent_name, "tool_name": tool_name, "tool_id": tool_id,
            "execution_id": execution_id,
            "output": error or output, "error": error,
            "execution": execution,
        }, context=self.event_context)
        await self.event_bus.emit(event)
        if cancellation is not None:
            raise cancellation
        return ToolResult(tool_id, content=str(event.data["output"]),
                          error=str(event.data["output"]) if error else None, execution=execution)
