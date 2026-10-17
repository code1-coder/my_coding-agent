"""
事件总线 —— 优先级排序的 Hook 分发。

支持 9 种事件类型，并按优先级排序。返回非 None 值的 Hook 会 **短路**
链 —— 下游 Hook 被跳过，返回值传播给调用方。

内置 Hook
    * ``TokenUsageHook``  — 累计整个会话的 token 消耗。
    * ``OutputTruncateHook`` — 工具输出超过阈值时发出警告。

用法::

    bus = EventBus()

    # 装饰器风格
    @bus.on(EventType.PRE_TOOL_USE, priority=10)
    async def log_tool(event: Event) -> None:
        name = event.data["tool_name"]
        print(f"[HOOK] {name}")
        return None   # None = 继续

    # 订阅者风格
    bus.subscribe(EventType.POST_TOOL_USE, my_hook, priority=5)

    # 发射 —— 返回第一个非 None 结果（或 None）
    blocked = await bus.emit(Event(EventType.PRE_TOOL_USE, data={...}))
    if blocked:
        ...  # 工具被某个 hook 拒绝了
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  事件类型枚举
# ---------------------------------------------------------------------------


class EventType(Enum):
    """agent loop 触发的所有生命周期事件。"""

    # ── 会话 ────────────────────────────────────────────────
    SESSION_START = "session_start"        # agent loop 创建
    SESSION_END = "session_end"            # agent loop 即将退出

    # ── 用户输入 ───────────────────────────────────────────
    USER_PROMPT_SUBMIT = "user_prompt_submit"  # 原始查询字符串

    # ── LLM 调用 ───────────────────────────────────────────
    PRE_LLM_CALL = "pre_llm_call"              # provider.create_message / stream 之前
    POST_LLM_CALL = "post_llm_call"            # 收到响应之后

    # ── 工具使用 ───────────────────────────────────────────
    PRE_TOOL_USE = "pre_tool_use"              # 执行工具之前
    POST_TOOL_USE = "post_tool_use"            # 工具返回之后

    # ── 循环控制 ───────────────────────────────────────────
    STOP = "stop"                              # 循环即将退出 (end_turn)

    # ── 错误 ────────────────────────────────────────────────
    ERROR = "error"                            # 捕获到异常


# ---------------------------------------------------------------------------
#  事件载荷
# ---------------------------------------------------------------------------


@dataclass
class Event:
    """agent loop 发出的生命周期事件。

    属性:
        type: 生命周期事件类型。
        data: 事件携带的不可变载荷。
        context: 在单次会话中 **跨事件持久化** 的可变字典。
                 Hook 可以通过此字典读写共享状态。
    """

    type: EventType
    data: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
#  回调类型
# ---------------------------------------------------------------------------

HookCallback = Callable[["Event"], Awaitable[Any | None]]


# ---------------------------------------------------------------------------
#  事件总线（EventBus）
# ---------------------------------------------------------------------------


class EventBus:
    """agent 生命周期事件的优先级排序发布/订阅。

    * **subscribe** / **on** 注册回调。
    * **emit** 按优先级顺序触发所有匹配的回调。
    * 优先级为整数 —— **越小越先执行**（0 最高）。
    * 同优先级 → 按注册顺序（FIFO）。
    * **第一个非 None 返回值**短路链。
    """

    def __init__(self) -> None:
        self._hooks: dict[EventType, list[tuple[int, HookCallback]]] = defaultdict(list)

    # ── 注册 ──────────────────────────────────────────────

    def subscribe(
        self,
        event_type: EventType,
        callback: HookCallback,
        *,
        priority: int = 50,
    ) -> None:
        """为 *event_type* 注册一个回调。

        参数:
            event_type: 要监听的生命周期事件。
            callback: 异步 ``(event) → Any | None``。
            priority: 越低越先执行（默认 50）。典型范围：

                ======== =====
                范围     用途
                ======== =====
                0–9      系统关键（权限、认证）
                10–29    基础设施（日志、指标）
                30–59    内置 hook（token 计数、截断）
                60–99    用户扩展
                ======== =====
        """
        self._hooks[event_type].append((priority, callback))
        self._hooks[event_type].sort(key=lambda pair: pair[0])

    def on(self, event_type: EventType, *, priority: int = 50):
        """:meth:`subscribe` 的装饰器版本::

            @bus.on(EventType.PRE_TOOL_USE, priority=10)
            async def my_hook(event: Event) -> None: ...
        """
        def _decorator(fn: HookCallback) -> HookCallback:
            self.subscribe(event_type, fn, priority=priority)
            return fn
        return _decorator

    def unsubscribe(self, event_type: EventType, callback: HookCallback) -> None:
        """移除之前注册的回调。"""
        self._hooks[event_type] = [
            (p, cb) for (p, cb) in self._hooks[event_type] if cb is not callback
        ]

    # ── 分发 ──────────────────────────────────────────────

    async def emit(self, event: Event) -> Any | None:
        """按优先级顺序触发 *event.type* 的所有 hook。

        返回:
            第一个非 ``None`` 返回值（短路），
            或所有 hook 都返回 ``None`` 时的 ``None``。
        """
        for _priority, callback in self._hooks[event.type]:
            try:
                result = await callback(event)
                if result is not None:
                    logger.debug(
                        "EventBus 短路: %s 由 %s 触发",
                        event.type.value, getattr(callback, "__name__", callback),
                    )
                    return result
            except Exception:
                logger.exception(
                    "Hook %s 对事件 %s 崩溃",
                    getattr(callback, "__name__", callback), event.type.value,
                )
        return None

    # ── 内省 ──────────────────────────────────────────────

    @property
    def registered_events(self) -> set[EventType]:
        """至少有一个订阅者的事件类型。"""
        return {et for et, hooks in self._hooks.items() if hooks}

    def hook_count(self, event_type: EventType | None = None) -> int:
        """已注册 hook 的总数（可按类型过滤）。"""
        if event_type is not None:
            return len(self._hooks[event_type])
        return sum(len(h) for h in self._hooks.values())


# ---------------------------------------------------------------------------
#  内置 Hook
# ---------------------------------------------------------------------------


class TokenUsageHook:
    """累计整个会话中所有 LLM 调用的 token 消耗。

    从 ``event.data["usage"]`` 中读取，该字段是包含
    ``input_tokens`` 和 ``output_tokens`` 键的字典。

    用法::

        token_hook = TokenUsageHook()
        bus.subscribe(EventType.POST_LLM_CALL, token_hook)

        # 之后:
        print(token_hook.total_tokens)   # 输入令牌数与输出令牌数之和
    """

    def __init__(self) -> None:
        self.input_tokens: int = 0
        self.output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    async def __call__(self, event: Event) -> None:
        if event.type != EventType.POST_LLM_CALL:
            return None

        usage: dict[str, Any] = event.data.get("usage", {})
        inp = usage.get("input_tokens", 0)
        out = usage.get("output_tokens", 0)

        self.input_tokens += inp
        self.output_tokens += out

        # 同时累积到共享会话上下文中
        ctx_input = event.context.setdefault("total_input_tokens", 0)
        ctx_output = event.context.setdefault("total_output_tokens", 0)
        event.context["total_input_tokens"] = ctx_input + inp
        event.context["total_output_tokens"] = ctx_output + out

        return None


class OutputTruncateHook:
    """工具返回超大输出时发出警告（并可选择截断）。

    参数:
        max_chars: 超过此字符数时发出警告。
        truncate: 若为 ``True``，则原地截断 ``event.data["output"]``，
                  使 LLM 不会看到完整的大块输出。

    用法::

        trunc_hook = OutputTruncateHook(max_chars=100_000, truncate=True)
        bus.subscribe(EventType.POST_TOOL_USE, trunc_hook)
    """

    def __init__(self, max_chars: int = 100_000, *, truncate: bool = False) -> None:
        self.max_chars = max_chars
        self.truncate = truncate

    async def __call__(self, event: Event) -> None:
        if event.type != EventType.POST_TOOL_USE:
            return None

        raw = event.data.get("output", "")
        text = str(raw)
        if len(text) <= self.max_chars:
            return None

        tool_name = event.data.get("tool_name", "?")
        logger.warning(
            "Large output from %s: %d chars (threshold %d)",
            tool_name, len(text), self.max_chars,
        )

        if self.truncate:
            event.data["output"] = (
                text[:self.max_chars]
                + f"\n\n... (truncated at {self.max_chars} chars, "
                + f"original was {len(text)} chars)"
            )

        return None
