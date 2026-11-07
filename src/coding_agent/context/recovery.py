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
    1. 指数退避：``500ms × 2^attempt``（最大 32s）+ 0–25% 随机抖动。
    2. 529 过载：跟踪连续 529 次数。连续 3 次后切换
       fallback 模型（若已配置）。最多 10 次总重试。
    3. 连接中断、超时和不完整流最多尝试 3 次；调用方阻止已显示文本的重放。
    4. 非瞬时错误立即向上抛出。

``RecoveryState`` 追踪所有恢复计数器（has_escalated、
recovery_count、consecutive_529、has_attempted_reactive_compact、
current_model），贯穿整个循环生命周期。
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

import anthropic
import httpx
import openai

from coding_agent.providers.base import IncompleteStreamError, ModelRequestError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  错误恢复常量
# ---------------------------------------------------------------------------

DEFAULT_MAX_TOKENS = 8000
ESCALATED_MAX_TOKENS = 64000
MAX_RECOVERY_RETRIES = 3           # 续写提示词的最大次数（路径 1）
MAX_RETRIES = 10                   # 退避重试的最大次数（路径 3）
MAX_CONNECTION_ATTEMPTS = 3
BASE_DELAY_MS = 500                # 退避起始延迟
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


def _retry_delay(attempt: int, retry_after: float | None = None) -> float:
    """指数退避 + 随机抖动。

    ``retry_after``（来自 ``Retry-After`` header）优先使用。
    """
    if retry_after is not None:
        return float(retry_after)
    base = min(BASE_DELAY_MS * (2 ** attempt), 32000) / 1000.0
    jitter = random.uniform(0, base * 0.25)
    return base + jitter


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
    return isinstance(exc, (openai.APIConnectionError, anthropic.APIConnectionError,
                            httpx.TransportError, ConnectionError, TimeoutError, IncompleteStreamError))


async def with_retry(
    fn: LLMCall,
    state: RecoveryState,
    *,
    fallback_model: str | None = None,
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
        如果不是 429/529 错误，或 ``MAX_RETRIES`` 用尽后，
        抛出原始异常。
    """
    connection_attempts = 0
    for attempt in range(MAX_RETRIES):
        try:
            result = await fn()
            state.reset_transient()
            return result
        except Exception as exc:
            if is_connection_error(exc):
                connection_attempts += 1
                if connection_attempts >= MAX_CONNECTION_ATTEMPTS:
                    raise ModelRequestError(
                        f"模型连接中断或超时，重试后仍失败（{connection_attempts} 次尝试）。"
                        "请检查网络和模型接口后重试。"
                    ) from exc
                delay = _retry_delay(connection_attempts - 1)
                logger.warning("[连接中断] 重试 %d/%d，等待 %.1fs",
                               connection_attempts, MAX_CONNECTION_ATTEMPTS - 1, delay)
                await asyncio.sleep(delay)
                continue
            ename = type(exc).__name__.lower()
            emsg = str(exc).lower()
            status = getattr(exc, "status_code", None)

            # ── 429 限流 ──────────────────────────────────
            if status == 429 or "ratelimit" in ename or re.search(r"\b429\b", emsg) or "rate limit" in emsg or "rate_limit" in emsg:
                state.total_429_retries += 1
                delay = _retry_delay(attempt)
                logger.warning(
                    "[429 限流] 重试 %d/%d，等待 %.1fs",
                    attempt + 1, MAX_RETRIES, delay,
                )
                await asyncio.sleep(delay)
                continue

            # ── 529 过载 ──────────────────────────────────
            if status == 529 or "overloaded" in ename or re.search(r"\b529\b", emsg) or "overloaded" in emsg:
                state.consecutive_529 += 1
                state.total_529_retries += 1

                if state.consecutive_529 >= MAX_CONSECUTIVE_529:
                    if fallback_model:
                        state.current_model = fallback_model
                        state.consecutive_529 = 0
                        logger.warning(
                            "[529 overload x%d] switching to %s",
                            MAX_CONSECUTIVE_529, fallback_model,
                        )
                    else:
                        state.consecutive_529 = 0
                        logger.warning(
                            "[529 overload x%d] no fallback configured, continuing",
                            MAX_CONSECUTIVE_529,
                        )

                delay = _retry_delay(attempt)
                logger.warning(
                    "[529 overload] retry %d/%d, wait %.1fs",
                    attempt + 1, MAX_RETRIES, delay,
                )
                await asyncio.sleep(delay)
                continue

            # ── 非瞬时错误 → 向上抛出，由外层处理器处理 ──
            raise

    raise RuntimeError(f"Max retries ({MAX_RETRIES}) exceeded")


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
        while True:
            try:
                response = await with_retry(
                    llm_fn,
                    state,
                    fallback_model=self._fallback_model,
                )
                # 成功 —— 重置路径 1 状态
                return response

            except Exception as exc:
                # ── 路径 2: prompt_too_long ─────────────────
                if is_prompt_too_long_error(exc):
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
