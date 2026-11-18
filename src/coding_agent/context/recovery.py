"""
错误恢复 —— 三个精确的恢复路径，带指数退避。

针对不同的 API 错误分别选择输出续写、上下文压缩或退避重试策略。

三条恢复路径：

**路径 1 — max_tokens（输出被截断）**
    1. 首次：升级 ``max_tokens`` 8K → 64K，**不追加**截断输出
       —— 用更高的上限重试同一请求。
    2. 在 64K 下仍然截断：追加截断输出 + 一条
       ``CONTINUATION_PROMPT``（不道歉、不复述）。最多重复 3 次。
    3. 3 次续写后放弃并返回。

**路径 2 — prompt_too_long（上下文过长）**
    1. 触发 ``reactive_compact``（紧急摘要）。
    2. 重试一次。仍然失败则追加错误消息并返回。

**路径 3 — 429/529 和连接中断（瞬时错误）**
    1. 指数退避从 2 秒起步，最大 30 秒，遵守服务端冷却时间。
    2. 529 过载：跟踪连续 529 次数。连续 3 次后切换
       fallback 模型（若已配置），不重置预算。
    3. 瞬时错误默认共享 5 次总尝试；调用方阻止已显示文本的重放。
    4. 非瞬时错误立即向上抛出。

``RecoveryState`` 追踪所有恢复计数器（has_escalated、
recovery_count、consecutive_529、has_attempted_reactive_compact、
current_model），贯穿整个循环生命周期。
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
import ssl
import time
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Awaitable

import anthropic
import httpx
import openai

from coding_agent.providers.activity import _sdk_http_module
from coding_agent.providers.base import (
    IncompleteStreamError, ModelIdleTimeout, ModelRequestError, RecoverableModelRequestError,
)

logger = logging.getLogger(__name__)
_HTTP_MODULES = (httpx, _sdk_http_module("openai"), _sdk_http_module("anthropic"))
_TRANSPORT_ERRORS = tuple(module.TransportError for module in _HTTP_MODULES)
_CONFIG_ERRORS = (ssl.SSLError,) + tuple(
    kind for module in _HTTP_MODULES for kind in (module.UnsupportedProtocol, module.LocalProtocolError)
)
_STATUS_ERRORS = (openai.APIStatusError, anthropic.APIStatusError)

# ---------------------------------------------------------------------------
#  错误恢复常量
# ---------------------------------------------------------------------------

DEFAULT_MAX_TOKENS = 8000
ESCALATED_MAX_TOKENS = 64000
MAX_RECOVERY_RETRIES = 3           # 续写提示词的最大次数（路径 1）
DEFAULT_MODEL_ATTEMPTS = 5
MAX_CONSECUTIVE_529 = 3            # 连续 529 超过此次数后切换模型

CONTINUATION_PROMPT = (
    "Output token limit hit. Resume directly — "
    "no apology, no recap. Pick up mid-thought."
)

# ---------------------------------------------------------------------------
#  错误恢复状态（RecoveryState）
# ---------------------------------------------------------------------------


@dataclass
class RecoveryState:
    """可变状态容器，追踪整个循环中所有恢复计数器。

    每个 agent 会话创建一次，传递给每次 LLM 调用包装器。
    """

    has_escalated: bool = False
    recovery_count: int = 0                    # 路径 1 续写次数
    consecutive_529: int = 0                   # 路径 3 过载计数器
    has_attempted_reactive_compact: bool = False  # 路径 2 标记
    current_model: str = ""                    # 529 fallback 时可能改变

    # 可观测性计数器
    total_429_retries: int = 0
    total_529_retries: int = 0
    total_escalations: int = 0
    total_compacts: int = 0

    def reset_transient(self) -> None:
        """成功调用后重置瞬时计数器。"""
        self.consecutive_529 = 0


# ---------------------------------------------------------------------------
#  退避辅助函数（路径 3）
# ---------------------------------------------------------------------------


@dataclass
class RetryBudget:
    maximum: int = DEFAULT_MODEL_ATTEMPTS
    deadline: float = math.inf
    attempts: int = 0
    on_wait: Callable[[str, int | None, float], None] | None = None


def _retry_delay(attempt: int) -> float:
    base = min(2 ** attempt, 30)
    return min(30, base + random.uniform(0, base * 0.25))


def retry_after_deadline(exc: Exception, now: float) -> float | None:
    value = exc.response.headers.get("retry-after") if isinstance(exc, _STATUS_ERRORS) else None
    if value is None:
        return None
    value = value.strip()
    try:
        if value.isascii() and value.isdecimal():
            delay = float(value)
        else:
            date = parsedate_to_datetime(value)
            if date.tzinfo is None:
                raise ValueError("HTTP 日期缺少时区")
            delay = date.timestamp() - time.time()
        if not math.isfinite(delay):
            raise ValueError("等待时间不是有限值")
    except (ValueError, OverflowError, OSError):
        logger.debug("模型服务返回了无效的 Retry-After")
        return None
    return now + max(0., delay)


# ---------------------------------------------------------------------------
#  错误判断函数（路径 2）
# ---------------------------------------------------------------------------


def is_prompt_too_long_error(exc: Exception) -> bool:
    """判断异常是否表示上下文过大。"""
    msg = str(exc).lower()
    name = type(exc).__name__.lower()
    return any([
        ("prompt" in msg and "long" in msg),
        "prompt_is_too_long" in msg,
        "context_length_exceeded" in msg,
        "max_context_window" in msg,
        "too_many_tokens" in msg,
        "too many tokens" in msg,
        "token" in msg and "exceed" in msg,
    ])


def is_max_tokens_error(exc: Exception) -> bool:
    """判断异常是否与 token 限制相关（供外层处理）。"""
    msg = str(exc).lower()
    return "max_tokens" in msg or "token_limit" in msg


# ---------------------------------------------------------------------------
#  异步 LLM 调用包装器 —— with_retry（路径 3）
# ---------------------------------------------------------------------------

LLMCall = Callable[[], Awaitable[Any]]


def is_connection_error(exc: Exception) -> bool:
    """识别 SDK 连接错误和超时，避免依赖异常文案进行重试。"""
    if not isinstance(exc, (openai.APIConnectionError, anthropic.APIConnectionError,
                            ConnectionError, TimeoutError, IncompleteStreamError) + _TRANSPORT_ERRORS):
        return False
    cause: BaseException | None = exc
    seen: set[int] = set()
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        if isinstance(cause, _CONFIG_ERRORS):
            return False
        cause = cause.__cause__ or cause.__context__
    return True


RETRY_LABELS = {
    "connection": "模型连接中断或超时",
    "idle_timeout": "模型流空闲超时",
    "rate_limit": "模型服务限流",
    "overload": "模型服务过载",
    "server": "模型服务暂时不可用",
}


def retry_category(exc: Exception) -> tuple[str | None, int | None]:
    if isinstance(exc, ModelIdleTimeout):
        return "idle_timeout", None
    if is_connection_error(exc):
        return "connection", None
    if not isinstance(exc, (openai.APIError, anthropic.APIError)):
        return None, None
    if isinstance(exc, _STATUS_ERRORS) and exc.response.headers.get("x-should-retry") == "false":
        return None, None
    status = getattr(exc, "status_code", None)
    if status is None or status == 200:
        body = exc.body
        if isinstance(body, dict):
            error = body.get("error", body)
            if isinstance(error, dict):
                status = {"overloaded_error": 529, "rate_limit_error": 429,
                          "rate_limit_exceeded": 429, "server_error": 500}.get(
                    str(error.get("type") or error.get("code")),
                )
    if status == 429:
        return "rate_limit", status
    if status == 529:
        return "overload", status
    # 两个 SDK 均把 408/409 纳入重试条件，但 408/409 不适用于普通业务异常。
    if status in (408, 409, 500, 502, 503, 504):
        return "server", status
    return None, None


def is_model_request_failure(exc: Exception) -> bool:
    """识别可安全转为失败终态的模型请求异常，不包含程序缺陷和取消。"""
    return isinstance(exc, (ModelRequestError, openai.APIError, anthropic.APIError,
                            ConnectionError, TimeoutError) + _TRANSPORT_ERRORS)


async def with_retry(
    fn: LLMCall,
    state: RecoveryState,
    *,
    fallback_model: str | None = None,
    budget: RetryBudget | None = None,
) -> Any:
    """对限流、过载、连接中断和超时做有界指数退避。

    非瞬时错误 **向上抛出**，由外层处理器应用路径 1 或路径 2。

    参数:
        fn: 返回 LLM 响应的异步可调用对象。
        state: 可变 :class:`RecoveryState`。
        fallback_model: 连续 3 次 529 后将 ``state.current_model``
            切换为此值。

    返回:
        成功时的 LLM 响应。

    异常:
        非瞬时错误保持原样；预算耗尽时保留最后一次失败的 cause。
    """
    budget = budget or RetryBudget()
    clock = asyncio.get_running_loop().time
    while budget.attempts < budget.maximum:
        budget.attempts += 1
        try:
            result = await fn()
            state.reset_transient()
            return result
        except Exception as exc:
            category, status = retry_category(exc)
            if status != 529:
                state.reset_transient()
            if category is None:
                raise

            if status == 429:
                state.total_429_retries += 1
            elif status == 529:
                state.consecutive_529 += 1
                state.total_529_retries += 1
            now = clock()
            retry_at = retry_after_deadline(exc, now)
            delay = max(_retry_delay(budget.attempts), (retry_at or now) - now)
            if budget.attempts == budget.maximum or now + delay >= budget.deadline:
                reason = "已达到尝试上限" if budget.attempts == budget.maximum else "剩余总预算不足以等待重试"
                label = f"模型服务返回 HTTP {status}" if status else RETRY_LABELS[category]
                raise RecoverableModelRequestError(
                    f"{label}，{reason}（{budget.attempts} 次尝试）。",
                    category=category, status=status, retry_at=retry_at,
                ) from exc
            if fallback_model and state.consecutive_529 >= MAX_CONSECUTIVE_529:
                state.current_model = fallback_model
                state.reset_transient()
            if budget.on_wait:
                budget.on_wait(category, status, delay)
            await asyncio.sleep(delay)
    raise ValueError("重试预算必须包含至少一次可用尝试")


# ---------------------------------------------------------------------------
#  恢复编排器
# ---------------------------------------------------------------------------


class RecoveryManager:
    """在一次 LLM 调用周围编排全部三条错误恢复路径。

    在 agent loop 中的用法::

        mgr = RecoveryManager(
            compact_fn=my_reactive_compact,       # 路径 2
            fallback_model="claude-haiku-4-5",     # 路径 3 fallback
        )

        state = RecoveryState(current_model="claude-sonnet-5")

        while True:
            result = await mgr.call(
                llm_fn=my_create_message,
                state=state,
                messages=messages,
                max_tokens=8000,
            )
            if result is None:  # 不可恢复
                break
            # result 是 ProviderResponse
    """

    def __init__(
        self,
        *,
        compact_fn: Callable[[list[dict[str, Any]]], Awaitable[list[dict[str, Any]]]] | None = None,
        fallback_model: str | None = None,
        default_max_tokens: int = DEFAULT_MAX_TOKENS,
        escalated_max_tokens: int = ESCALATED_MAX_TOKENS,
        max_recovery_retries: int = MAX_RECOVERY_RETRIES,
    ) -> None:
        self._compact_fn = compact_fn
        self._fallback_model = fallback_model
        self._default_max_tokens = default_max_tokens
        self._escalated_max_tokens = escalated_max_tokens
        self._max_recovery_retries = max_recovery_retries

    # ── 公开 API ─────────────────────────────────────────────

    async def call(
        self,
        llm_fn: LLMCall,
        state: RecoveryState,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        budget: RetryBudget | None = None,
    ) -> Any | None:
        """包装一次 LLM 调用，提供完整的三路径恢复。

        参数:
            llm_fn: 异步可调用对象 ``() → ProviderResponse``。闭包应捕获
                    ``messages``、``tools``、``system`` 等。
            state: 可变恢复状态（跨循环迭代追踪）。
            messages: 消息列表（路径 2 的 reactive compact 需要）。
            max_tokens: 起始 token 预算。通过 ``state.has_escalated``
                        在多次调用中原地变更。

        返回:
            成功时的响应，不可恢复时返回 ``None``。

        上下文过长时最多调用一次压缩回调，并通过切片赋值原地更新
        messages，使调用闭包仍能访问同一个列表。输出被截断的响应
        需要调用方另行交给 handle_max_tokens 处理；其他异常继续抛出。
        """
        budget = budget or RetryBudget()
        while True:
            try:
                response = await with_retry(
                    llm_fn,
                    state,
                    fallback_model=self._fallback_model,
                    budget=budget,
                )
                # 成功 —— 重置路径 1 状态
                return response

            except Exception as exc:
                # ── 路径 2: prompt_too_long ─────────────────
                if (isinstance(exc, _STATUS_ERRORS) and exc.status_code in (400, 413, 422)
                        and is_prompt_too_long_error(exc)):
                    if budget.attempts >= budget.maximum:
                        return None
                    if not state.has_attempted_reactive_compact:
                        if self._compact_fn is None:
                            logger.warning(
                                "prompt_too_long 但未配置 compact_fn —— 不可恢复"
                            )
                            return None
                        state.total_compacts += 1
                        new_msgs = await self._compact_fn(messages)
                        messages[:] = new_msgs
                        state.has_attempted_reactive_compact = True
                        continue

                    logger.error("[不可恢复] 压缩后仍然过长")
                    return None

                # ── 向上抛出：由外层处理器处理 max_tokens ──
                raise

    # ── 路径 1 处理器（由 agent loop 在检查 stop_reason 后调用）──

    def handle_max_tokens(
        self,
        state: RecoveryState,
        messages: list[dict[str, Any]],
        response: Any,
    ) -> bool:
        """处理 ``stop_reason == "max_tokens"``。

        在 LLM 调用成功后，当 ``stop_reason`` 为 ``"max_tokens"`` 时调用此方法。

        参数:
            state: 可变恢复状态。
            messages: 对话历史（续写时追加到此列表）。
            response: 包含截断输出的 provider 响应。

        返回:
            如果循环应 **继续**（重试/升级），返回 ``True``。
            如果恢复已用尽 —— 调用方应停止，返回 ``False``。
        """
        # ── 首次升级：8K → 64K，丢弃截断输出 ──
        if not state.has_escalated:
            state.total_escalations += 1
            state.has_escalated = True
            logger.info(
                "[max_tokens] 升级 %d → %d",
                self._default_max_tokens, self._escalated_max_tokens,
            )
            # 不追加截断输出 —— 重试同一请求
            return True

        # ── 64K 仍截断：追加 + 续写提示词 ──
        # 追加截断输出以保留上下文
        messages.append({"role": "assistant", "content": response.content})

        if state.recovery_count < self._max_recovery_retries:
            messages.append({"role": "user", "content": CONTINUATION_PROMPT})
            state.recovery_count += 1
            logger.info(
                "[max_tokens] 续写 %d/%d",
                state.recovery_count, self._max_recovery_retries,
            )
            return True

        logger.error("[max_tokens] 恢复次数已用尽")
        return False

    def current_max_tokens(self, state: RecoveryState) -> int:
        """返回下一次调用的有效 max_tokens。"""
        return self._escalated_max_tokens if state.has_escalated else self._default_max_tokens
