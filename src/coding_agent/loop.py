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
from coding_agent.context.recovery import RecoveryManager, RecoveryState, is_model_request_failure
from coding_agent.providers.base import ModelIdleTimeout, ModelRequestError, StreamEvent
from coding_agent.providers.activity import RequestActivity, _current_request_activity

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
        model_timeout: float = 900.0,
        model_connect_timeout: float = 20.0,
        model_idle_timeout: float = 120.0,
        final_answer_timeout: float = 30.0,
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
        self.model_connect_timeout = model_connect_timeout
        self.model_idle_timeout = model_idle_timeout
        self.final_answer_timeout = final_answer_timeout
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
        tool_outcomes: dict[tuple[str, str, bool], int] = {}
        for iteration in range(self.max_iterations):
            if self.compaction is not None:
                async with aclosing(self.wait_events(
                    self.compaction.run(messages), "整理工具输出和压缩上下文",
                )) as events:
                    async for event in events:
                        yield event
            response = None
            prefix = text
            try:
                async with aclosing(self._model_events(messages, stream)) as events:
                    async for ev in events:
                        if ev.type == "text_delta":
                            text += ev.text
                            yield LoopEvent(type="text_delta", text=ev.text)
                        elif ev.type == "done":
                            response = ev.response
                        elif ev.type == "warning":
                            yield ev
            except Exception as exc:
                if not is_model_request_failure(exc):
                    raise
                delivery = self._request_failure_delivery(exc, tool_outcomes)
                history_text = (f"模型中断前的未完成输出：\n{text}\n\n" if text else "") + delivery
                messages.append({"role": "assistant", "content": [{"type": "text", "text": history_text}]})
                yield LoopEvent(type="error", text=delivery, data={
                    "failed": True, "failure_type": type(exc).__name__,
                })
                return
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
                    if event.type == "tool_done":
                        arguments = event.data.get("input")
                        target = arguments.get("path", "") if isinstance(arguments, dict) else ""
                        target = target if isinstance(target, str) else ""
                        outcome = (event.tool_name, target, bool(event.data.get("is_error")))
                        tool_outcomes[outcome] = tool_outcomes.get(outcome, 0) + 1
                    yield event

        async for event in self._finish_without_answer(messages, text, "max_iterations"):
            yield event

    @staticmethod
    def _request_failure_delivery(exc: Exception, tool_outcomes: dict[tuple[str, str, bool], int]) -> str:
        if isinstance(exc, ModelRequestError):
            reason = str(exc)
        else:
            status = getattr(exc, "status_code", None)
            if isinstance(status, int):
                reason = f"模型服务返回 HTTP {status}"
            elif isinstance(exc, (ConnectionError, TimeoutError)):
                reason = "模型服务连接或请求超时"
            else:
                reason = "模型服务 API 请求失败"

        lines = [f"模型请求失败，本轮未完成。失败原因：{reason}"]
        if tool_outcomes:
            lines.append("本轮已记录的工具执行：")
            lines.extend(
                f"- {name}{f'（{target}）' if target else ''}："
                f"{'执行失败' if failed else '执行成功'}（{count} 次）"
                for (name, target, failed), count in sorted(tool_outcomes.items())
            )
        else:
            lines.append("本轮没有已完成的工具执行。")
        lines.append("未完成事项：模型没有确认任务结果，请核对上述执行记录后再判断是否完成。")
        return "\n".join(lines)

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
            async with asyncio.timeout(self.final_answer_timeout), aclosing(
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
        loop = asyncio.get_running_loop()
        request_started_at = loop.time()
        request_attempts = 0
        request_activity = RequestActivity()

        async def attempt():
            nonlocal request_attempts
            sent_text = False
            request_attempts += 1
            provider = (self.provider_for_model(self.recovery_state.current_model)
                        if self.provider_for_model else self.provider)
            request_activity.begin_provider_attempt(
                http_observed=bool(getattr(provider, "http_activity_supported", False)),
            )
            activity_token = _current_request_activity.set(request_activity)
            idle_timeout = asyncio.timeout(None)
            try:
                blocked = await self.emit(EventType.PRE_LLM_CALL, messages=messages,
                                          purpose="agent", model=provider.model_id)
                if blocked is not None:
                    raise PermissionError(f"模型调用被 Hook 阻止：{blocked}")
                kwargs = dict(messages=messages, tools=[] if final_only else self.registry.list_tools(),
                              system=self.system_prompt, max_tokens=(
                                  self.recovery.current_max_tokens(self.recovery_state)
                                  if self.recovery else self.max_tokens))
                response = None
                async with idle_timeout:
                    request_activity.monitor_idle_timeout(
                        idle_timeout, self.model_idle_timeout,
                        wait_for_http_request=bool(getattr(provider, "http_activity_supported", False)),
                    )
                    request_activity.mark_request_started()
                    if stream:
                        async with aclosing(provider.stream_message(**kwargs)) as events:
                            async for event in events:
                                if event.type != "done":
                                    request_activity.mark_model_event(event.type)
                                    queue.put_nowait(event)
                                if event.type == "text_delta":
                                    sent_text = sent_text or bool(event.text)
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
                error = ModelIdleTimeout(
                    f"模型请求空闲超时：连续 {self.model_idle_timeout:g} 秒没有可观测活动，"
                    "单次尝试已停止。"
                ) if idle_timeout.expired() and isinstance(exc, TimeoutError) else exc
                await self.emit(EventType.ERROR, error=str(error), purpose="agent")
                if sent_text:
                    # 已显示的增量无法安全重放，避免重试后产生重复文本。
                    if isinstance(error, ModelIdleTimeout):
                        raise error from exc
                    if is_model_request_failure(error):
                        raise ModelRequestError(
                            "流式响应中断，已显示部分内容，未自动重放。请重试本轮请求。"
                        ) from error
                if error is not exc:
                    raise error from exc
                raise
            finally:
                request_activity.stop_idle_timeout_monitoring()
                _current_request_activity.reset(activity_token)

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
                        f"模型请求超过总时限 {self.model_timeout:g} 秒（含重试与恢复），已停止等待。"
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
                    now = loop.time()
                    if (request_activity.last_activity_type in {
                            "text_delta", "tool_use_start", "tool_input",
                    } and request_activity.last_activity_at is not None
                            and now - request_activity.last_activity_at < self.progress_interval):
                        continue
                    elapsed = now - request_started_at
                    if request_activity.last_activity_type is None:
                        activity_text = (
                            "尚未收到可观测的模型流或 HTTP 活动，服务端状态未知"
                            if stream else "当前为非流式请求，调用完成前没有可观测的中间事件"
                        )
                        idle_seconds = None
                        activity_type = None
                    else:
                        activity_type = request_activity.last_activity_type
                        idle_seconds = now - request_activity.last_activity_at
                        activity_label = {
                            "text_delta": "文本增量",
                            "tool_use_start": "工具调用声明",
                            "tool_input": "工具参数增量",
                            "http_request_started": f"HTTP 请求已发出，尚未收到响应（距发送 {idle_seconds:.1f} 秒）",
                            "http_response": (
                                f"已收到 HTTP {request_activity.last_http_status} 响应，"
                                f"尚未收到响应数据（距响应 {idle_seconds:.1f} 秒）"
                            ),
                            "http_response_data": "HTTP 响应数据",
                            "http_error": f"HTTP 请求发生连接错误（距错误 {idle_seconds:.1f} 秒）",
                            "request_started": "模型请求处理中，尚无可观测的模型响应",
                        }.get(activity_type, "模型流事件")
                        if activity_type in {
                                "http_request_started", "http_response", "http_error", "request_started",
                        }:
                            activity_text = activity_label
                        else:
                            activity_text = f"最近收到{activity_label}，{idle_seconds:.1f}秒前"
                    retry_count = max(0, request_attempts - 1)
                    http_retry_count = request_activity.http_retry_count
                    remaining_total = max(0.0, self.model_timeout - elapsed)
                    retry_summary = (
                        f"HTTP 请求{request_activity.http_request_attempts}次（SDK HTTP 重试"
                        f"{http_retry_count}次）"
                        if request_activity.http_observation_supported
                        else "HTTP 网络计数不可用（当前 Provider 未接入 HTTP 观测）"
                    )
                    yield LoopEvent(
                        type="warning",
                        text=(f"模型请求已等待{elapsed:.1f}秒；{activity_text}；应用层共调用"
                              f"{request_attempts}次（额外恢复尝试{retry_count}次），{retry_summary}；"
                              f"连接上限 {self.model_connect_timeout:.1f} 秒、空闲上限 "
                              f"{self.model_idle_timeout:.1f} 秒；总预算 {self.model_timeout:.1f} 秒，"
                              f"剩余 {remaining_total:.1f} 秒，可取消当前请求。"),
                        data={
                            "phase": "model",
                            "agent": self.agent_name,
                            "request_elapsed_seconds": elapsed,
                            "connect_timeout_seconds": self.model_connect_timeout,
                            "idle_timeout_seconds": self.model_idle_timeout,
                            "total_timeout_seconds": self.model_timeout,
                            "remaining_total_seconds": remaining_total,
                            "last_protocol_event": (
                                activity_type if request_activity.last_activity_source == "model" else None
                            ),
                            "last_activity_type": activity_type,
                            "last_activity_source": request_activity.last_activity_source,
                            "seconds_since_protocol_event": (
                                idle_seconds if request_activity.last_activity_source == "model" else None
                            ),
                            "request_attempts": request_attempts,
                            "application_retry_count": retry_count,
                            "http_request_attempts": request_activity.http_request_attempts,
                            "http_retry_count": http_retry_count,
                            "http_observation_supported": request_activity.http_observation_supported,
                            "last_http_status": request_activity.last_http_status,
                        },
                    )
                    continue
                if event is sentinel:
                    break
                if event.type == "text_delta":
                    yield event
                elif event.type == "done":
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
