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
import hashlib
import json
import logging
import os
import uuid
from collections.abc import Iterable
from contextlib import aclosing
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable

from coding_agent.providers.base import LLMProvider, ProviderResponse
from coding_agent.tools.base import Tool, ToolCancelledError, ToolResult, tool_result
from coding_agent.tools.registry import ToolRegistry
from coding_agent.hooks.bus import Event, EventBus, EventType
from coding_agent.security.permission import PermissionPipeline, Verdict
from coding_agent.context.compaction import CompactionPipeline, INTERNAL_CONTEXT_KEY
from coding_agent.context.recovery import (
    DEFAULT_MODEL_ATTEMPTS, RETRY_LABELS, RecoveryManager, RecoveryState, RetryBudget,
    is_model_request_failure, retry_after_deadline, retry_category,
)
from coding_agent.providers.base import (
    IncompleteStreamError, ModelIdleTimeout, ModelRequestError, RecoverableModelRequestError, StreamEvent,
)
from coding_agent.providers.activity import RequestActivity, _current_request_activity

logger = logging.getLogger(__name__)
_MAX_NO_PROGRESS_READ_ROUNDS = 3
_MAX_REPEATED_TOOL_FAILURES = 3
ANALYSIS_READ_TOOLS = frozenset({"read_file", "search_code", "code_outline"})
_ANALYSIS_CHECKPOINT = "analysis_checkpoint"
_ANALYSIS_DIMENSIONS = {"role": "职责", "flow": "调用关系", "usage": "入口与用法", "limits": "生命周期与边界"}
ANALYSIS_DIMENSION_IDS = tuple(_ANALYSIS_DIMENSIONS)
_ANALYSIS_BATCHES = 8
_ANALYSIS_BATCH_SIZE = 4
_ANALYSIS_NO_PROGRESS = 2
_ANALYSIS_MAX_RESULT_CHARS = 30_000
_ToolCallKey = tuple[str, str]
_ToolErrorKey = tuple[str, int | None] | str


def _analysis_display(value: str, limit: int) -> str:
    return "".join(char if char.isprintable() else " " for char in value[:limit]).strip()


def resolve_analysis_dimensions(analysis: bool, dimensions: object = None) -> tuple[str, ...] | None:
    """只校验用户入口提供的结构化范围，不从对话或工具调用推断范围。"""
    if dimensions is None:
        return ANALYSIS_DIMENSION_IDS if analysis else None
    if not analysis:
        raise ValueError("只有静态分析模式可以选择分析维度。")
    if (not isinstance(dimensions, (list, tuple)) or not 1 <= len(dimensions) <= len(ANALYSIS_DIMENSION_IDS)
            or any(not isinstance(item, str) or item not in _ANALYSIS_DIMENSIONS for item in dimensions)
            or len(set(dimensions)) != len(dimensions)):
        raise ValueError("请选择至少一个不重复的分析维度：" + "、".join(ANALYSIS_DIMENSION_IDS) + "。")
    return tuple(dimensions)


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

@dataclass(frozen=True)
class AnalysisEvidence:
    label: str
    key: str
    paths: frozenset[str]
    call_id: str
    source_path: str = ""
    version: str = ""


@dataclass
class AnalysisProgress:
    """保存本轮用户选定的必查维度、证据引用和阶段结论。"""

    goal: str
    dimensions: tuple[str, ...] = ANALYSIS_DIMENSION_IDS
    evidence: dict[str, AnalysisEvidence] = field(default_factory=dict)
    findings: dict[str, tuple[str, str, tuple[str, ...]]] = field(default_factory=dict)
    review_gaps: dict[str, str] = field(default_factory=dict)
    batches: int = 0
    stalled: int = 0
    stalled_controls: int = 0
    revision: int = 0
    finished: bool = False
    versions: dict[str, str] = field(default_factory=dict)

    def register(self, registry: ToolRegistry) -> None:
        registry.register(
            name=_ANALYSIS_CHECKPOINT,
            description=("提交本轮分析检查点。必查维度由用户入口确定，工具不能更改；"
                         "record 记录一个 dimension、finding 及 read_file 的 evidence_ids，或用 blocked=true 记录缺口；"
                         "finish 仅在要求的维度都有记录后使用。每次单独调用。"),
        )(self.checkpoint)

    async def checkpoint(self, action: str, dimension: str = "", finding: str = "", evidence_ids: list[str] | None = None,
                         blocked: bool = False) -> str:
        if action == "finish":
            if not all(item in self.findings for item in self.dimensions):
                raise ValueError("还有未记录的分析维度；无证据时请说明具体阻塞。")
            self.finished = True
            return "已关闭分析工具阶段。"
        refs = [] if evidence_ids is None else evidence_ids
        if (action != "record" or not isinstance(dimension, str) or dimension not in self.dimensions
                or not isinstance(finding, str) or not 0 < len(finding.strip()) <= 400
                or type(blocked) is not bool or not isinstance(refs, list) or len(refs) > 4
                or any(not isinstance(ref, str) or ref not in self.evidence for ref in refs)
                or (not blocked and not refs)):
            raise ValueError("请提供有效的维度、简短结论和本轮已登记的证据 ID；无证据时标记阻塞。")
        finding = _analysis_display(finding, 400)
        if not finding:
            raise ValueError("结论或阻塞原因不能为空。")
        update = ("blocked" if blocked else "supported", finding, tuple(sorted(set(refs))))
        previous = self.findings.get(dimension)
        if previous and (previous[0], previous[2]) == (update[0], update[2]):
            return "已有相同状态和证据；仅改写文字不更新检查点。"
        self.findings[dimension] = update
        self.review_gaps.pop(dimension, None)
        self.revision += 1
        return self.describe(dimension)

    def describe(self, dimension: str) -> str:
        label = _ANALYSIS_DIMENSIONS[dimension]
        if dimension not in self.findings:
            if dimension in self.review_gaps:
                return f"{label}：待复核，{self.review_gaps[dimension]}"
            return f"{label}：未确认，尚未记录结论。"
        status, finding, refs = self.findings[dimension]
        locations = "；".join(self.evidence[ref].label for ref in refs)
        confidence = "未确认"
        if status == "supported":
            confidence = "源码依据" if all(self.evidence[ref].version for ref in refs) else "定位线索，待复核"
        return f"{label}（{confidence}）：{finding}" + (
            f"［{locations}］" if locations else "")

    def summary(self) -> str:
        return "\n".join(f"- {self.describe(item)}" for item in self.dimensions)

    def view(self) -> str:
        return json.dumps({"goal": self.goal, "required_dimensions": self.dimensions, "checks": self.summary(),
                           "evidence": {ref: item.label for ref, item in self.evidence.items()},
                           "reading_batches_left": _ANALYSIS_BATCHES - self.batches}, ensure_ascii=False)

    def add_evidence(self, block: dict, result: dict, execution_id: str) -> bool:
        content, meta = result.get("content", ""), result.get("metadata", {})
        if meta.get("reused_from"):
            return False
        name, args = block["name"], block.get("input", {})
        path = str(args.get("path") or ".")
        paths = {path}
        span = ""
        version = ""
        if name == "read_file":
            if "read_span" not in meta:
                return False
            selected = json.loads(meta["repeat_key"])
            path, version = selected["path"], meta["freshness_token"]
            self._update_file_version(path, version)
            paths = {path}
        if result.get("is_error") or not content.strip() or "... (truncated at " in content:
            return False
        if name == "read_file":
            if hashlib.sha256(content.encode()).hexdigest() != meta.get("output_digest"):
                return False
            first, last = meta["read_span"]
            if last < first:
                return False
            span = f":{first}–{last}"
        elif name in {"search_code", "code_outline"}:
            try:
                payload = json.loads(content)
            except json.JSONDecodeError:
                return False
            if not isinstance(payload, dict):
                return False
            items = payload.get("results" if name == "search_code" else "symbols", [])
            if payload.get("truncated") or not items or any(item.get("line_truncated") for item in items):
                return False
            path = payload.get("path") or path
            paths = {path}
            if name == "search_code":
                matches = sorted({(item["path"], item["line"], item["text"]) for item in items})
                content = json.dumps(matches, ensure_ascii=False)
                paths = {file for file, _, _ in matches}
                path = "、".join(f"{file}:{line}" for file, line, _ in matches[:3])
        # 读文件用规范化路径、实际行区间和文件版本判重，不因切换行号样式产生新证据。
        digest = version or hashlib.sha256(content.encode()).hexdigest()
        key = json.dumps((name, path, span, digest), ensure_ascii=False)
        if any(item.key == key for item in self.evidence.values()):
            return False
        label = f"{_analysis_display(path, 200)}{span}（{name}）"
        self.evidence[execution_id] = AnalysisEvidence(
            label, key, frozenset(os.path.normcase(file) for file in paths), block["id"],
            source_path=path if version else "", version=version,
        )
        return True

    def _update_file_version(self, path: str, version: str) -> None:
        path = os.path.normcase(path)
        # 搜索和提纲不含文件版本；首次拿到指纹时也不能确认这些旧结果仍有效。
        if self.versions.get(path) != version:
            stale = {ref for ref, item in self.evidence.items() if path in item.paths}
            self.invalidate_evidence(dict.fromkeys(stale, "源码版本已更新，请重新读取。"))
        self.versions[path] = version

    def invalidate_evidence(self, reasons: dict[str, str]) -> None:
        for ref in reasons:
            del self.evidence[ref]
        for dimension, (_, _, refs) in list(self.findings.items()):
            unavailable = next((reasons[ref] for ref in refs if ref in reasons), None)
            if unavailable is not None:
                del self.findings[dimension]
                self.review_gaps[dimension] = unavailable
                self.revision += 1


@dataclass
class TurnProgress:
    recovery: RecoveryState
    turn_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    iteration: int = 0
    tool_outcomes: dict[tuple[str, str, bool], int] = field(default_factory=dict)
    tool_failure_streaks: dict[_ToolCallKey, dict[_ToolErrorKey, int]] = field(default_factory=dict)
    no_progress_read_rounds: int = 0
    uncertain_tools: bool = False
    analysis: AnalysisProgress | None = None


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
        model_max_attempts: int = DEFAULT_MODEL_ATTEMPTS,
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
        self.model_max_attempts = model_max_attempts

    # ── 公共 API ──────────────────────────────────────────

    async def run(
        self, messages: list[dict[str, Any]], *, stream: bool = True, resume: TurnProgress | None = None,
        analysis: AnalysisProgress | None = None,
    ) -> AsyncIterator[LoopEvent]:
        """执行本轮循环，并在取消或中断时修复尚未配对的工具结果。"""
        try:
            async with aclosing(self._run_impl(messages, stream=stream, resume=resume, analysis=analysis)) as events:
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
        resume: TurnProgress | None = None,
        analysis: AnalysisProgress | None = None,
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
        self.progress = progress = resume or TurnProgress(
            RecoveryState(current_model=self.provider.model_id), analysis=analysis,
        )
        analysis = progress.analysis
        if analysis is not None:
            analysis.register(self.registry)
        self.request_failure: RecoverableModelRequestError | None = None
        self.resuming_request = resume is not None
        self.recovery_state = progress.recovery
        if analysis is not None and resume is not None:
            async with aclosing(self.wait_events(self._revalidate_analysis(analysis), "复核分析证据")) as events:
                async for event in events:
                    yield event
        text = ""
        tool_outcomes = progress.tool_outcomes
        for iteration in range(progress.iteration, self.max_iterations):
            progress.iteration = iteration
            if self.compaction is not None and resume is None:
                async with aclosing(self.wait_events(
                    self.compaction.run(messages), "整理工具输出和压缩上下文",
                )) as events:
                    async for event in events:
                        yield event
            resume = None
            response = None
            prefix = text
            try:
                async with aclosing(self._model_events(messages, stream)) as events:
                    async for ev in events:
                        if ev.type == "text_delta" and analysis is None:
                            text += ev.text
                            yield LoopEvent(type="text_delta", text=ev.text)
                        elif ev.type == "text_reset" and analysis is None:
                            text = prefix + ev.text
                            yield LoopEvent(type="text_reset", text=text, data=ev.data)
                        elif ev.type == "done":
                            response = ev.response
                        elif ev.type == "warning":
                            yield ev
            except Exception as exc:
                if not is_model_request_failure(exc):
                    raise
                if isinstance(exc, RecoverableModelRequestError):
                    self.request_failure = exc
                delivery = self._request_failure_delivery(exc, tool_outcomes, progress.uncertain_tools)
                if analysis is not None:
                    delivery += "\n\n本轮分析检查点：\n" + analysis.summary()
                history_text = (f"模型中断前的未完成输出：\n{text}\n\n" if text else "") + delivery
                messages.append({"role": "assistant", "content": [{"type": "text", "text": history_text}]})
                yield LoopEvent(type="error", text=delivery, data={
                    "failed": True, "failure_type": type(exc).__name__,
                    "partial_output": bool(text), **self.request_details,
                    "error_category": getattr(exc, "category", "request"),
                    "http_status": getattr(exc, "status", getattr(exc, "status_code", None)),
                    **({"analysis_status": "failed"} if analysis is not None else {}),
                })
                return
            if response is None:
                raise RuntimeError("模型未返回完整响应")
            self.resuming_request = False

            if analysis is not None:
                if response.stop_reason == "max_tokens":
                    # 截断的工具参数和草稿都不能入历史，升级预算后重新生成同一步。
                    if (iteration + 1 < self.max_iterations and self.recovery is not None
                            and self.recovery.handle_max_tokens(
                                self.recovery_state, messages, response, allow_continuation=False)):
                        continue
                    reason = "max_tokens"
                elif response.stop_reason != "tool_use" or not response.tool_use_blocks:
                    reason = "analysis_not_finished"
                else:
                    reason = None
                if reason:
                    async for event in self._finish_analysis(messages, reason):
                        yield event
                    return

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
                messages.append({"role": "assistant", "content": response.tool_use_blocks
                                 if analysis is not None else response.content})

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
            rejected: dict[int, str] = {}
            reads = False
            if analysis is not None:
                rejected, reads = self._analysis_batch(blocks, analysis)
                if reads:
                    analysis.batches += 1
                revision = analysis.revision
            for block, execution_id in zip(blocks, execution_ids):
                yield LoopEvent(type="tool_start", tool_name=block["name"],
                                tool_id=block["id"], data={"input": block.get("input", {}),
                                "execution_id": execution_id, "agent": self.agent_name})
            round_results: dict[str, tuple[LoopEvent, str | None]] = {}
            async with aclosing(self._tool_events(
                messages, blocks, execution_ids, round_results, rejected=rejected,
            )) as events:
                async for event in events:
                    if event.type == "tool_done":
                        execution = event.data.get("execution") or {}
                        if execution.get("termination_confirmed") is False and execution.get("failure_stage") in (
                            "cleanup", "supervision",
                        ):
                            progress.uncertain_tools = True
                        arguments = event.data.get("input")
                        target = arguments.get("path", "") if isinstance(arguments, dict) else ""
                        target = target if isinstance(target, str) else ""
                        outcome = (event.tool_name, target, bool(event.data.get("is_error")))
                        tool_outcomes[outcome] = tool_outcomes.get(outcome, 0) + 1
                    yield event
            if analysis is not None:
                results = messages[-1]["content"]
                if reads:
                    new_evidence = sum(
                        analysis.add_evidence(block, result, execution_id)
                        for block, result, execution_id in zip(blocks, results, execution_ids)
                    )
                    analysis.stalled = 0 if new_evidence else analysis.stalled + 1
                changed = analysis.revision != revision
                analysis.stalled_controls = 0 if reads or changed else analysis.stalled_controls + 1
                if changed:
                    remaining = _ANALYSIS_BATCHES - analysis.batches
                    next_step = ("请说明缺口或补查未确认的维度。"
                                 if len(analysis.findings) < len(analysis.dimensions) else "各维度均已记录，等待收尾。")
                    yield LoopEvent(type="analysis_progress", text=analysis.summary() + "\n" + next_step, data={
                        "turn_id": progress.turn_id, "revision": analysis.revision,
                        "remaining_batches": remaining,
                    })
                reason = ("analysis_finished" if analysis.finished else
                          "analysis_no_progress" if analysis.stalled >= _ANALYSIS_NO_PROGRESS else
                          "analysis_budget" if analysis.batches >= _ANALYSIS_BATCHES and rejected and not reads else
                          "analysis_stalled_controls" if analysis.stalled_controls >= _ANALYSIS_NO_PROGRESS else None)
                if reason:
                    async for event in self._finish_analysis(messages, reason):
                        yield event
                    return
                # 静态分析按批次判断进展，同批多项失败不能抢先终止整轮。
                continue
            failure = self._record_tool_failures(progress, (
                round_results[execution_id] for execution_id in execution_ids
            ))
            if failure is not None:
                arguments = failure.data["input"]
                target = arguments.get("path") if isinstance(arguments, dict) else None
                async for event in self._finish_without_answer(
                    messages, text, "repeated_tool_failure",
                    failure=(failure.tool_name, target if isinstance(target, str) else "", failure.tool_output),
                ):
                    yield event
                return
            if (round_results and all(result.tool_name == "read_file" and not result.data["is_error"]
                                      and result.data["metadata"].get("reused_from")
                                      for result, _ in round_results.values())):
                progress.no_progress_read_rounds += 1
            else:
                progress.no_progress_read_rounds = 0
            if progress.no_progress_read_rounds >= _MAX_NO_PROGRESS_READ_ROUNDS:
                async for event in self._finish_without_answer(messages, text, "no_progress"):
                    yield event
                return

        async for event in self._finish_without_answer(messages, text, "max_iterations"):
            yield event

    @staticmethod
    def _record_tool_failures(
        progress: TurnProgress, results: Iterable[tuple[LoopEvent, str | None]],
    ) -> LoopEvent | None:
        """按调用目标与错误类型累计；同批同目标成功优先，不依赖并发结束顺序。"""
        successes: set[_ToolCallKey] = set()
        failures: dict[_ToolCallKey, dict[_ToolErrorKey, tuple[int, LoopEvent]]] = {}
        for event, repeat_key in results:
            if repeat_key is None:
                # 没有专用规范化时按完整参数区分目标，不重写命令或保留大段写入内容。
                serialized = json.dumps(event.data["input"], sort_keys=True, separators=(",", ":"))
                repeat_key = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
            key = (event.tool_name, repeat_key)
            if not event.data["is_error"]:
                successes.add(key)
                continue
            execution = event.data["execution"]
            stage = execution.get("failure_stage")
            exit_code = execution.get("exit_code")
            error_key: _ToolErrorKey = event.tool_output
            if isinstance(stage, str) and stage and (exit_code is None or type(exit_code) is int):
                error_key = (stage, exit_code)
            errors = failures.setdefault(key, {})
            count = errors[error_key][0] if error_key in errors else 0
            errors[error_key] = (count + 1, event)

        streaks = progress.tool_failure_streaks
        for key in successes:
            streaks.pop(key, None)
        for key, errors in failures.items():
            if key in successes:
                continue
            previous = streaks.get(key, {})
            current = {error: previous.get(error, 0) + count for error, (count, _) in errors.items()}
            streaks[key] = current
            for error, count in current.items():
                if count >= _MAX_REPEATED_TOOL_FAILURES:
                    return errors[error][1]
        return None

    def _analysis_batch(self, blocks: list[dict], analysis: AnalysisProgress) -> tuple[dict[int, str], bool]:
        controls = [index for index, block in enumerate(blocks) if block["name"] == _ANALYSIS_CHECKPOINT]
        if len(controls) == 1:
            position = controls[0]
            arguments = blocks[position].get("input")
            if isinstance(arguments, dict) and arguments.get("action") == "plan":
                rejected = dict.fromkeys(range(len(blocks)), "检查点须单独提交，本批未执行。")
                rejected[position] = "模型不能更改必查维度，请按本轮分析状态记录结论或缺口。"
                return rejected, False
            if len(blocks) == 1:
                return {}, False
            if isinstance(arguments, dict) and arguments.get("action") == "finish":
                return {index: "收尾请求同批不执行其他工具。" for index in range(len(blocks))
                        if index != position}, False
        if not controls and len(blocks) <= _ANALYSIS_BATCH_SIZE and analysis.batches < _ANALYSIS_BATCHES:
            if all(block["name"] in ANALYSIS_READ_TOOLS for block in blocks):
                return {}, True
            return dict.fromkeys(range(len(blocks)), "分析模式只允许内置只读工具，本批未执行。"), False
        reason = ("已达读取上限；只允许单独提交检查点。" if analysis.batches >= _ANALYSIS_BATCHES else
                  "分析批次不合法；每批最多四个只读工具，检查点须单独提交。")
        return dict.fromkeys(range(len(blocks)), reason), False

    async def _revalidate_analysis(self, analysis: AnalysisProgress, *, stop_reason: str | None = None) -> None:
        cited = {ref: analysis.evidence[ref] for _, _, refs in analysis.findings.values() for ref in refs}
        expected = {item.source_path: item.version for item in cited.values() if item.version}
        versions: dict[str, str | None] = {}
        checks: dict[str, asyncio.Task[str | None]] = {}
        ready = asyncio.Event()
        async with asyncio.TaskGroup() as checks_group:
            for path, expected_version in expected.items():
                version, check = await self._prepare_analysis_source_check(path, expected_version, checks_group, ready)
                versions[path] = version
                if check is not None:
                    checks[path] = check
            if stop_reason is not None:
                await self.emit(EventType.STOP, stop_reason=stop_reason)
            # 所有文件的逐次审批及 Hook 都结束后，再校验通过初检的版本集合。
            ready.set()
        versions.update((path, check.result()) for path, check in checks.items())

        unavailable: dict[str, str] = {}
        for ref, item in cited.items():
            if not item.version:
                unavailable[ref] = "搜索或提纲缺少源码版本，请使用 read_file 读取对应行。"
                continue
            current_version = versions[item.source_path]
            if current_version != item.version:
                problem = "源码已变化" if current_version else "无法验证源码版本"
                unavailable[ref] = f"{problem}［{item.label}］，请重新读取。"
        analysis.invalidate_evidence(unavailable)

    async def _prepare_analysis_source_check(
        self, path: str, expected_version: str, checks: asyncio.TaskGroup, ready: asyncio.Event,
    ) -> tuple[str | None, asyncio.Task[str | None] | None]:
        entry = self.registry.get("read_file")
        if entry is None or entry.freshness_token is None or self.permissions is None:
            return None, None
        fingerprint = entry.freshness_token

        async def read_version() -> str | None:
            try:
                return await asyncio.to_thread(fingerprint, {"path": path, "_evidence_path": path})
            except (OSError, ValueError, RuntimeError):
                # 文件可能被删除、无法访问，或解析到循环链接。
                return None

        async def check_after_hooks() -> str | None:
            await ready.wait()
            return await read_version()

        params = {"path": path}
        check_id = uuid.uuid4().hex
        blocked = await self.emit(EventType.PRE_TOOL_USE, tool_name="read_file", params=params,
                                  tool_id=check_id, execution_id=check_id, purpose="analysis_revalidation")
        version = None
        check = None
        if blocked is None:
            async with self.permissions.authorize("read_file", params) as (verdict, _):
                if verdict == Verdict.ALLOW:
                    version = await read_version()
                    if version == expected_version:
                        # 子任务只继承本次、该路径的授权，后续 Hook 和审批在父任务中执行。
                        check = checks.create_task(check_after_hooks())
        status = "已读取源码指纹。" if version else "无法校验源码版本。"
        await self.emit(EventType.POST_TOOL_USE, tool_name="read_file", tool_id=check_id,
                        execution_id=check_id, purpose="analysis_revalidation", output=status,
                        error=None if version else status)
        return version, check

    async def _finish_analysis(self, messages: list[dict[str, Any]], reason: str) -> AsyncIterator[LoopEvent]:
        analysis = self.progress.analysis
        async with aclosing(self.wait_events(
            self._revalidate_analysis(analysis, stop_reason=reason), "复核分析证据",
        )) as events:
            async for event in events:
                yield event
        complete = analysis.finished and all(
            item in analysis.findings and analysis.findings[item][0] == "supported"
            for item in analysis.dimensions
        )
        notice = {
            "analysis_finished": "已结束源码检查。",
            "analysis_not_finished": "模型结束了检查，但未确认分析范围已覆盖。",
            "analysis_no_progress": "连续读取未产生新证据，已停止检查。",
            "analysis_budget": "已达本轮读取批次上限，已停止检查。",
            "analysis_stalled_controls": "连续请求无效或未更新检查点，已停止检查。",
            "max_tokens": "模型输出达到长度上限，无法继续检查。",
            "max_iterations": "已达本轮迭代上限。",
            "repeated_tool_failure": "工具连续失败，已停止检查。",
            "no_progress": "未获取新的读取证据。",
        }[reason]
        text = f"{'分析完成' if complete else '分析部分交付'}：{notice}\n\n{analysis.summary()}\n\n仅作静态检查，未运行项目代码。"
        messages.append({"role": "assistant", "content": [{"type": "text", "text": text}]})
        yield LoopEvent(type="text_delta", text=text)
        yield LoopEvent(type="done", text=text, data={"stop_reason": reason,
                                                        "analysis_status": "complete" if complete else "partial"})

    def _request_failure_delivery(self, exc: Exception, tool_outcomes: dict[tuple[str, str, bool], int],
                                  uncertain_tools: bool) -> str:
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

        lines = [f"模型请求失败，本轮未完成。失败原因：{reason}",
                 f"失败阶段：模型请求，已发起 {self.request_details['attempt']} 次应用层尝试。"]
        if tool_outcomes:
            lines.append("本轮已记录的工具执行：")
            lines.extend(
                f"- {name}{f'（{target}）' if target else ''}："
                f"{'执行失败' if failed else '执行成功'}（{count} 次）"
                for (name, target, failed), count in list(tool_outcomes.items())[-8:]
            )
            if len(tool_outcomes) > 8:
                lines.append(f"另有 {len(tool_outcomes) - 8} 项工具统计未展示。")
        else:
            lines.append("本轮没有已完成的工具执行。")
        lines.append("未完成事项：模型没有确认任务结果，请核对上述执行记录后再判断是否完成。")
        if uncertain_tools:
            lines.append("部分工具的停止状态未确认；请先核对工作区或外部状态，不能直接续接。")
        return "\n".join(lines)

    async def _finish_without_answer(self, messages: list[dict[str, Any]], text: str,
                                     reason: str, *,
                                     failure: tuple[str, str, str] | None = None) -> AsyncIterator[LoopEvent]:
        """最多补充一次无工具总结；失败时明确告知缺少交付，不能静默结束。"""
        if self.progress.analysis is not None:
            async for event in self._finish_analysis(messages, reason):
                yield event
            return
        notices = {
            "max_iterations": "已达到本轮执行轮次上限，已停止继续调用工具；任务是否完成仍需依据执行结果确认。",
            "no_progress": "连续只读调用未产生新证据，已停止重复读取并根据已有结果整理结论。",
            "max_tokens": "模型输出达到长度上限，回答可能不完整。",
            "tool_use": "模型请求使用工具，但未提供有效工具调用。",
            "repeated_tool_failure": (
                f"工具 {failure[0]} 连续失败 {_MAX_REPEATED_TOOL_FAILURES} 次，已停止重复调用。"
                if failure else "工具连续失败，已停止重复调用。"
            ),
        }
        notice = notices.get(reason, "")
        if failure:
            tool_name, target, error = failure
            notice += (f"\n最后一次失败：{tool_name}{f'（{target}）' if target else ''}：{error}"
                       "\n请确认运行环境和目标路径后重试。")
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
        generated = False
        if reason == "repeated_tool_failure":
            answer = "本轮未能完成文件检查，已停止继续尝试。请确认当前运行环境与目标路径可访问后重试。"
        else:
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

    async def _tool_events(
        self, messages, blocks, execution_ids, round_results: dict[str, tuple[LoopEvent, str | None]],
        *, rejected: dict[int, str],
    ) -> AsyncIterator[LoopEvent]:
        tasks = [asyncio.create_task(self._execute_one(block, messages, execution_id=execution_id,
                                                      rejection=rejected.get(index)))
                 for index, (block, execution_id) in enumerate(zip(blocks, execution_ids))]
        indexes = {task: index for index, task in enumerate(tasks)}
        pending = set(tasks)
        try:
            while pending:
                completed, pending = await asyncio.wait(
                    pending, timeout=self.progress_interval, return_when=asyncio.FIRST_COMPLETED,
                )
                waiting_tasks = [task for task in pending if not (
                    self.permissions and self.permissions.is_waiting_for_approval(task))]
                if not completed and waiting_tasks:
                    waiting = [{"tool_name": blocks[indexes[task]]["name"],
                                "tool_id": blocks[indexes[task]]["id"]} for task in tasks
                               if task in waiting_tasks]
                    labels = ", ".join(f"{item['tool_name']} ({item['tool_id']})" for item in waiting[:5])
                    yield LoopEvent(type="warning", text=f"仍在等待 {len(waiting_tasks)} 个工具：{labels}；可取消当前请求。",
                                    data={"phase": "tools", "pending_tools": waiting, "agent": self.agent_name})
                for task in sorted(completed, key=indexes.__getitem__):
                    index = indexes[task]
                    block = blocks[index]
                    result, repeat_key = task.result()
                    event = LoopEvent(type="tool_done", tool_name=block["name"],
                                    tool_id=block["id"], tool_output=result.error or result.content,
                                    data={"input": block.get("input", {}), "is_error": bool(result.error),
                                          "execution": result.execution,
                                          "metadata": {key: result.metadata[key]
                                                       for key in ("reused_from", "repeat_count")
                                                       if key in result.metadata},
                                          "execution_id": execution_ids[index], "agent": self.agent_name})
                    round_results[execution_ids[index]] = (event, repeat_key)
                    yield event
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            outcomes = await asyncio.gather(*tasks, return_exceptions=True)
            results = [outcome[0].to_dict() if isinstance(outcome, tuple) else
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
        deadline = request_started_at + self.model_timeout
        retry_until = None
        self.request_details = {"attempt": 0, "max_attempts": 1 if final_only else self.model_max_attempts,
                                "turn_id": self.progress.turn_id}

        def retry_wait(category: str, status: int | None, delay: float) -> None:
            nonlocal retry_until
            now = loop.time()
            retry_until = now + delay
            label = f"HTTP {status}" if status else RETRY_LABELS[category]
            queue.put_nowait(LoopEvent(type="warning", text=(
                f"{label}；将在 {delay:.1f} 秒后尝试第 {request_attempts + 1}/{budget.maximum} 次，"
                f"总预算剩余 {max(0., deadline - now):.1f} 秒，可取消。"
            ), data={**self.request_details, "phase": "retry_wait", "next_attempt": request_attempts + 1,
                     "delay_seconds": delay, "remaining_total_seconds": max(0., deadline - now),
                     "error_category": category, "http_status": status, "retryable": True}))

        budget = RetryBudget(self.request_details["max_attempts"], deadline, on_wait=retry_wait)
        response_submitted = False

        async def attempt():
            nonlocal request_attempts, retry_until, response_submitted
            sent_text = False
            delivered_text = ""
            # 正文出现前的纯空白可能只是传输占位，正常完成后再与完整响应核对。
            pending_whitespace = ""
            has_visible_text = False
            retry_until = None
            provider = (self.provider_for_model(self.recovery_state.current_model)
                        if self.provider_for_model else self.provider)
            request_activity.begin_provider_attempt(
                http_observed=bool(getattr(provider, "http_activity_supported", False)),
            )
            activity_token = _current_request_activity.set(request_activity)
            idle_timeout = asyncio.timeout(None)
            try:
                blocked = await self.emit(EventType.PRE_LLM_CALL, messages=messages,
                                          purpose="agent", model=provider.model_id,
                                          analysis=self.progress.analysis is not None)
                if blocked is not None:
                    raise PermissionError(f"模型调用被 Hook 阻止：{blocked}")
                # 历史消息来源和工具结果指纹都是运行时元数据，不属于 Provider 协议。
                provider_messages = [dict(message) for message in messages]
                for message in provider_messages:
                    message.pop(INTERNAL_CONTEXT_KEY, None)
                    content = message.get("content")
                    if isinstance(content, list):
                        message["content"] = [
                            {key: value for key, value in block.items() if key != "metadata"}
                            if isinstance(block, dict) and block.get("type") == "tool_result" else block
                            for block in content
                        ]
                analysis = self.progress.analysis
                if analysis is not None:
                    self._attach_analysis_view(provider_messages, analysis)
                kwargs = dict(messages=provider_messages, tools=[] if final_only else self.registry.list_tools(),
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
                    request_attempts += 1
                    self.request_details.update(attempt=request_attempts, request_id=uuid.uuid4().hex)
                    if request_attempts > 1 or self.resuming_request:
                        queue.put_nowait(LoopEvent(type="warning", text=(
                            f"已开始第 {request_attempts}/{budget.maximum} 次模型调用，尚未确认响应。"
                        ), data={**self.request_details, "phase": "requesting"}))
                    if stream:
                        async with aclosing(provider.stream_message(**kwargs)) as events:
                            async for event in events:
                                if event.type == "done":
                                    response = event.response
                                    continue
                                request_activity.mark_model_event(event.type)
                                if event.type != "text_delta":
                                    queue.put_nowait(event)
                                    continue
                                if analysis is not None:
                                    continue  # 未经过检查点校验的草稿不算已向用户提交。
                                delta = event.text or ""
                                if not delta:
                                    continue
                                if not has_visible_text and not delta.strip():
                                    pending_whitespace += delta
                                    continue
                                output = pending_whitespace + delta
                                pending_whitespace = ""
                                has_visible_text = True
                                delivered_text += output
                                sent_text = sent_text or bool(output)
                                response_submitted = sent_text
                                queue.put_nowait(StreamEvent(type="text_delta", text=output))
                    else:
                        response = await provider.create_message(**kwargs)
                        if response.text and analysis is None:
                            sent_text = response_submitted = True
                            queue.put_nowait(StreamEvent(type="text_delta", text=response.text))
                if response is None:
                    raise IncompleteStreamError("模型响应流未正常结束")
                if stream and analysis is None:
                    # 最终响应是完整文本的权威值，只补前缀差额，分歧则走 reset。
                    final_text = response.text or ""
                    if not has_visible_text:
                        if final_text.strip():
                            queue.put_nowait(StreamEvent(type="text_delta", text=final_text))
                            delivered_text = final_text
                            sent_text = True
                            response_submitted = True
                    elif final_text.startswith(delivered_text):
                        suffix = final_text[len(delivered_text):]
                        if suffix:
                            queue.put_nowait(StreamEvent(type="text_delta", text=suffix))
                            delivered_text += suffix
                    else:
                        queue.put_nowait(LoopEvent(
                            type="text_reset", text=final_text if final_text.strip() else "",
                            data={"reason": "response_reconciliation"},
                        ))
                await self.emit(EventType.POST_LLM_CALL, response=response,
                                usage=response.usage, purpose="agent")
                return response
            except Exception as exc:
                error = ModelIdleTimeout(
                    f"模型请求空闲超时：连续 {self.model_idle_timeout:g} 秒没有可观测活动，"
                    "单次尝试已停止。"
                ) if idle_timeout.expired() and isinstance(exc, TimeoutError) else exc
                category, status = retry_category(error)
                await self.emit(EventType.ERROR, error=type(error).__name__, purpose="agent")
                if sent_text and category is not None:
                    raise RecoverableModelRequestError(
                        f"{RETRY_LABELS[category]}，已显示部分内容，未自动重放。",
                        category=category, status=status, retry_at=retry_after_deadline(error, loop.time()),
                    ) from error
                if sent_text and is_model_request_failure(error):
                    status = getattr(error, "status_code", None)
                    reason = f"模型服务返回 HTTP {status}" if isinstance(status, int) else "模型请求失败"
                    raise ModelRequestError(f"{reason}，已显示部分内容，未自动重放。") from error
                if error is not exc:
                    raise error from exc
                raise
            finally:
                request_activity.stop_idle_timeout_monitoring()
                _current_request_activity.reset(activity_token)

        async def request():
            overall = asyncio.timeout_at(deadline)
            try:
                try:
                    async with overall:
                        if self.recovery is not None and not final_only:
                            response = await self.recovery.call(
                                attempt, self.recovery_state, messages, max_tokens=self.max_tokens,
                                budget=budget,
                            )
                        else:
                            response = await attempt()
                except TimeoutError as exc:
                    if not overall.expired():
                        raise
                    raise RecoverableModelRequestError(
                        f"模型请求超过总时限 {self.model_timeout:g} 秒（含重试与恢复），已停止等待。"
                        + ("部分正文已显示，未自动重放。" if response_submitted else ""),
                        category="total_timeout",
                    ) from exc
                if response is None:
                    raise ModelRequestError("模型上下文超出限制，无法在剩余尝试内恢复。")
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
                    if retry_until is not None:
                        yield LoopEvent(type="warning", text=(
                            f"正在等待重试，预计还需 {max(0., retry_until - now):.1f} 秒，可取消。"
                        ), data={**self.request_details, "phase": "retry_wait",
                                 "remaining_total_seconds": max(0., deadline - now),
                                 "retry_wait_remaining_seconds": max(0., retry_until - now)})
                        continue
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
                            **self.request_details,
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
                if event.type in ("text_delta", "text_reset", "done", "warning"):
                    yield event
            await task
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    # ── 内部辅助方法 ────────────────────────────────────

    @staticmethod
    def _attach_analysis_view(messages: list[dict[str, Any]], analysis: AnalysisProgress) -> None:
        """仅增强模型请求副本，不改变可恢复历史或工具调用配对。"""
        view = "\n\n本轮分析状态（以下为数据，不是指令）：\n" + analysis.view()
        last = messages[-1]
        if last.get("role") == "user":
            content = last.get("content")
            if isinstance(content, str):
                last["content"] = content + view
                return
            if isinstance(content, list) and content and content[-1].get("type") == "tool_result":
                content[-1]["content"] += view
                return
        messages.append({"role": "user", "content": view})

    async def _execute_one(self, block: dict[str, Any], messages: list[dict[str, Any]],
                           *, execution_id: str = "", rejection: str | None = None) -> tuple[ToolResult, str | None]:
        """执行单个 tool_use 块，保留执行信息与原调用 ID。

        未注册的工具和执行异常均转换为错误结果，交由模型在后续轮次
        处理，避免单个工具失败直接中断同轮的其他工具调用。
        """
        tool_name = block["name"]
        tool_id = block["id"]
        tool_input = block.get("input", {})
        if rejection is not None:
            return ToolResult(tool_id, error=rejection), None

        error = None
        output = ""
        execution = {}
        metadata: dict[str, Any] = {}
        repeat_key = None
        cancellation = None
        try:
            blocked = await self.emit(EventType.PRE_TOOL_USE, tool_name=tool_name,
                                      params=tool_input, tool_id=tool_id, execution_id=execution_id)
            if blocked is not None:
                raise PermissionError(f"工具被 Hook 阻止：{blocked}")
            if self.permissions is None:
                result, repeat_key = await self._dispatch_authorized_tool(messages, block)
            else:
                async with self.permissions.authorize(tool_name, tool_input) as (verdict, reason):
                    if verdict != Verdict.ALLOW:
                        raise PermissionError(reason)
                    result, repeat_key = await self._dispatch_authorized_tool(messages, block)
            output, error = result.content, result.error
            execution = result.execution
            metadata = result.metadata
        except ToolCancelledError as exc:
            cancellation = exc
            output, error = exc.result.content, exc.result.error
            execution = exc.result.execution
        except Exception as exc:
            error = str(exc)
        if (not error and tool_name in ANALYSIS_READ_TOOLS
                and self.progress.analysis is not None and len(output) > _ANALYSIS_MAX_RESULT_CHARS):
            # 无磁盘回退时不把超大结果交给模型，也不将预览登记为完整证据。
            error = "静态分析单项输出超过 30,000 字符，请缩小读取或搜索范围后重试。"
        if error:
            await self.emit(EventType.ERROR, tool_name=tool_name, error=error)
        event = Event(EventType.POST_TOOL_USE, data={
            "agent": self.agent_name, "tool_name": tool_name, "tool_id": tool_id,
            "execution_id": execution_id,
            "output": error or output, "error": error,
            "execution": execution,
            "metadata": {key: metadata[key] for key in ("reused_from", "repeat_count")
                         if key in metadata},
            "reused_result": bool(metadata.get("reused_from")),
        }, context=self.event_context)
        await self.event_bus.emit(event)
        if cancellation is not None:
            raise cancellation
        return ToolResult(tool_id, content=str(event.data["output"]),
                          error=str(event.data["output"]) if error else None, execution=execution,
                          metadata=metadata), repeat_key

    async def _dispatch_authorized_tool(
        self, messages: list[dict[str, Any]], block: dict[str, Any],
    ) -> tuple[ToolResult, str | None]:
        cached, repeat_key = await self._reuse_read_result(messages, block, block["id"])
        if cached is not None:
            return cached, repeat_key
        result = await self.registry.dispatch(block["name"], block.get("input", {}), tool_use_id=block["id"])
        # 成功时以实际读取的范围为准。失败时使用授权范围内已解析的键，不写入会话历史。
        actual_key = result.metadata.get("repeat_key")
        return result, actual_key if isinstance(actual_key, str) else repeat_key

    async def _reuse_read_result(self, messages: list[dict[str, Any]], block: dict[str, Any],
                                 tool_id: str) -> tuple[ToolResult | None, str | None]:
        """复用仍有效的读取结果，并保留已有的规范化参数用于失败计数。"""
        entry = self.registry.get(block.get("name", ""))
        if (entry is None or entry.repeat_policy != "fresh_read"
                or entry.repeat_key is None or entry.freshness_token is None):
            return None, None
        arguments = block.get("input", {})
        if not isinstance(arguments, dict):
            return None, None
        repeat_key = None
        try:
            repeat_key = entry.repeat_key(arguments)
            if repeat_key is None:
                return None, None
            freshness = await asyncio.to_thread(entry.freshness_token, arguments)
        except (OSError, ValueError, KeyError, TypeError):
            return None, repeat_key

        results: dict[str, dict[str, Any]] = {}
        analysis = getattr(getattr(self, "progress", None), "analysis", None)
        analysis_calls = {item.call_id for item in analysis.evidence.values()} if analysis is not None else None
        for message in reversed(messages):
            content = message.get("content")
            if not isinstance(content, list):
                continue
            if message.get("role") == "user":
                for item in reversed(content):
                    if isinstance(item, dict) and item.get("type") == "tool_result":
                        results[item.get("tool_use_id", "")] = item
                continue
            if message.get("role") != "assistant":
                continue
            for item in reversed(content):
                if (not isinstance(item, dict) or item.get("type") != "tool_use"
                        or item.get("name") != block.get("name") or item.get("id") == tool_id):
                    continue
                source_id = item.get("id", "")
                if analysis_calls is not None and source_id not in analysis_calls:
                    continue
                prior_result = results.get(source_id)
                if prior_result is None or prior_result.get("is_error"):
                    continue
                metadata = prior_result.get("metadata", {})
                output = str(prior_result.get("content", ""))
                if (not isinstance(metadata, dict) or metadata.get("repeat_policy") != "fresh_read"
                        or metadata.get("repeat_key") != repeat_key
                        or metadata.get("freshness_token") != freshness
                        or hashlib.sha256(output.encode("utf-8")).hexdigest() != metadata.get("output_digest")
                        or metadata.get("reused_from")):
                    continue
                repeat_count = 1 + sum(
                    1 for prior_result in results.values()
                    if isinstance(prior_result.get("metadata"), dict)
                    and prior_result["metadata"].get("reused_from") == source_id
                )
                return ToolResult(
                    tool_use_id=tool_id,
                    content=(f"已复用同一会话中工具调用 {source_id} 的文件读取结果；"
                             "该结果与当前文件内容一致。若需要不同范围，请明确新的读取范围。"),
                    metadata={**metadata, "reused_from": source_id, "repeat_count": repeat_count},
                ), repeat_key
        return None, repeat_key
