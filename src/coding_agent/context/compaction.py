"""
上下文压缩 — 4+1 层压缩管道。

压缩执行顺序为::

    L3 tool_result_budget → L1 snip_compact → L2 micro_compact
    → [estimate_tokens > CONTEXT_LIMIT?] → L4 compact_history
    → LLM 调用 → [prompt_too_long?] → Emergency reactive_compact

关键常量（可通过构造函数调优）：
    * ``CONTEXT_LIMIT`` = 50000 tokens
    * ``KEEP_RECENT``  = 3        — L2 中保留的最新 tool_result 数量
    * ``PERSIST_THRESHOLD`` = 30000 字符 — L3 中超过此值则写入磁盘
    * ``MAX_REACTIVE_RETRIES`` = 1

各层说明：
    **L3** tool_result_budget — 将超过 30000 字符的工具结果持久化到磁盘
          （``.task_outputs/tool-results/{id}.txt``），返回 2000 字符预览
          + 文件路径。
    **L1** snip_compact — 消息数超过 50 条时裁剪中间部分。**保护
          tool_use / tool_result 配对** —— 不会在两者之间裁剪。
    **L2** micro_compact — 收集所有 tool_result 块，保留
          ``KEEP_RECENT`` 条最新的，将更旧的超过 120 字符的内容替换为占位符。
    **L4** compact_history — 将对话记录写入 ``.transcripts/``，调用
          LLM 生成摘要（保留：目标、发现、文件、剩余工作、约束）。
    **Emergency** reactive_compact — 在 ``prompt_too_long`` API 错误时：
          摘要前半部分，保持尾部 5 条完整。最多重试 1 次。

Token 计数优先使用 ``tiktoken``（``cl100k_base``）；
不可用时回退到 ``len(text) // 4``。
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  公开类型
# ---------------------------------------------------------------------------

# 异步回调: (prompt: str) → 摘要文本
SummarizeFn = Callable[[str], Awaitable[str]]

# ---------------------------------------------------------------------------
#  常量
# ---------------------------------------------------------------------------

DEFAULT_CONTEXT_LIMIT = 50000       # tokens —— 触发 L4 compact_history
DEFAULT_KEEP_RECENT = 3             # L2 中保持完整的最新 tool_result 数
DEFAULT_PERSIST_THRESHOLD = 30000   # 字符 —— L3 超过此值则写入磁盘
DEFAULT_MAX_MESSAGES = 50           # 条数 —— L1 超过此值触发裁剪
DEFAULT_MAX_REACTIVE_RETRIES = 1    # 紧急压缩重试次数

# 在 workdir 下创建的子目录
TRANSCRIPT_DIR_NAME = ".transcripts"
TOOL_RESULTS_DIR_NAME = ".task_outputs/tool-results"


# ---------------------------------------------------------------------------
#  Token 估算
# ---------------------------------------------------------------------------

try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")

    def estimate_tokens(messages: list[dict[str, Any]]) -> int:
        """通过 tiktoken (cl100k_base) 精确计数 token。"""
        text = json.dumps(messages, default=str, ensure_ascii=False)
        return len(_ENC.encode(text))

except ImportError:
    def estimate_tokens(messages: list[dict[str, Any]]) -> int:
        """回退方案：~4 字符 ≈ 1 token。"""
        text = json.dumps(messages, default=str, ensure_ascii=False)
        return len(text) // 4


# ---------------------------------------------------------------------------
#  内部辅助函数
# ---------------------------------------------------------------------------

def _block_type(block: Any) -> str | None:
    """从内容块（dict 或 SDK 对象）中提取 'type'。"""
    if isinstance(block, dict):
        return block.get("type")
    return getattr(block, "type", None)


def _message_has_tool_use(msg: dict[str, Any]) -> bool:
    """如果 *msg* 是包含至少一个 tool_use 块的 assistant 回合，返回 True。"""
    if msg.get("role") != "assistant":
        return False
    content = msg.get("content")
    if not isinstance(content, list):
        return False
    return any(_block_type(b) == "tool_use" for b in content)


def _is_tool_result_message(msg: dict[str, Any]) -> bool:
    """如果 *msg* 是内容列表包含 tool_result 块的 user 回合，返回 True。"""
    if msg.get("role") != "user":
        return False
    content = msg.get("content")
    if not isinstance(content, list):
        return False
    return any(
        isinstance(b, dict) and b.get("type") == "tool_result"
        for b in content
    )


def _collect_tool_results(messages: list[dict[str, Any]]) -> list[tuple[int, int, dict[str, Any]]]:
    """返回每个 tool_result 块的 ``(msg_idx, block_idx, block)``。"""
    blocks: list[tuple[int, int, dict[str, Any]]] = []
    for mi, msg in enumerate(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for bi, block in enumerate(content):
            if isinstance(block, dict) and block.get("type") == "tool_result":
                blocks.append((mi, bi, block))
    return blocks


def _clone_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """浅克隆消息列表，以便安全地进行原地修改。"""
    return json.loads(json.dumps(messages, default=str))


# ---------------------------------------------------------------------------
#  L3 — tool_result_budget（首先执行）
# ---------------------------------------------------------------------------

def tool_result_budget(
    messages: list[dict[str, Any]],
    *,
    workdir: Path | str | None = None,
    max_total_chars: int = 200_000,
    persist_threshold: int = DEFAULT_PERSIST_THRESHOLD,
) -> list[dict[str, Any]]:
    """将超过 *persist_threshold* 字符的工具结果持久化到磁盘。

    只检查 **最后一条** user 消息（携带最新工具结果的那条）。
    大的结果写入::

        {workdir}/.task_outputs/tool-results/{tool_use_id}.txt

    并在原位置替换为 2000 字符预览 + 文件路径。

    返回 *messages*（原地修改）。
    """
    wd = Path(workdir) if workdir else Path.cwd()

    last = messages[-1] if messages else None
    if not last or last.get("role") != "user":
        return messages

    content = last.get("content")
    if not isinstance(content, list):
        return messages

    # 找出所有 tool_result 块
    blocks: list[tuple[int, dict[str, Any]]] = [
        (i, b) for i, b in enumerate(content)
        if isinstance(b, dict) and b.get("type") == "tool_result"
    ]
    if not blocks:
        return messages

    total = sum(len(str(b.get("content", ""))) for _, b in blocks)
    if total <= max_total_chars and all(len(str(b.get("content", ""))) <= persist_threshold for _, b in blocks):
        return messages

    # 按体积从大到小排序——优先持久化最大的
    ranked = sorted(blocks, key=lambda pair: len(str(pair[1].get("content", ""))), reverse=True)

    for _idx, block in ranked:
        raw = str(block.get("content", ""))
        if len(raw) <= persist_threshold and total <= max_total_chars:
            continue
        if len(raw) <= 2000:
            continue
        tid = block.get("tool_use_id", "unknown")
        output_path = _persist_large_output(wd, tid, raw)
        block["content"] = f"<persisted-output>\n完整输出: {output_path}\n预览:\n{raw[:2000]}\n</persisted-output>"
        total = sum(len(str(b.get("content", ""))) for _, b in blocks)

    return messages


def _persist_large_output(workdir: Path, tool_use_id: str, output: str) -> Path:
    """将 *output* 写入磁盘，返回文件路径。"""
    out_dir = workdir / TOOL_RESULTS_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_id = re.sub(r"[^\w-]+", "-", str(tool_use_id)).strip("-_")[:100] or "output"
    # 前缀也避免 Windows 的 CON、AUX 等保留设备名。
    if safe_id.upper() in {"CON", "PRN", "AUX", "NUL", *(f"{p}{n}" for p in ("COM", "LPT") for n in range(1, 10))}:
        safe_id = "output-" + safe_id
    path = out_dir / f"{safe_id}.txt"
    if path.exists() and path.read_text(encoding="utf-8") != output:
        path = out_dir / f"{safe_id}_{uuid.uuid4().hex[:8]}.txt"
    if not path.exists():
        path.write_text(output, encoding="utf-8")
    logger.debug("Persisted tool result %s → %s", tool_use_id, path)
    return path


# ---------------------------------------------------------------------------
#  L1 — snip_compact（第二步执行）
# ---------------------------------------------------------------------------

def snip_compact(
    messages: list[dict[str, Any]],
    *,
    max_messages: int = DEFAULT_MAX_MESSAGES,
) -> list[dict[str, Any]]:
    """当消息列表超过 *max_messages* 时裁剪中间部分。

    **tool_use / tool_result 配对保护**：

    * 如果裁剪边界前的最后一条消息是含 tool_use 的 assistant 回合，
      则将边界后移，跳过后续的 tool_result 消息。
    * 如果裁剪边界后的第一条消息是 tool_result 消息，且其前面的
      assistant 回合含 tool_use，则将边界前移一条。

    返回 *messages*（原地修改）。
    """
    if len(messages) <= max_messages:
        return messages

    keep_head = 3
    keep_tail = max_messages - keep_head
    head_end = keep_head
    tail_start = len(messages) - keep_tail

    # ── 前向保护 ──────────────────────────────────────────────
    if head_end > 0 and _message_has_tool_use(messages[head_end - 1]):
        while head_end < len(messages) and _is_tool_result_message(messages[head_end]):
            head_end += 1

    # ── 后向保护 ──────────────────────────────────────────────
    if (
        tail_start > 0
        and tail_start < len(messages)
        and _is_tool_result_message(messages[tail_start])
        and _message_has_tool_use(messages[tail_start - 1])
    ):
        tail_start -= 1

    if head_end >= tail_start:
        return messages

    snipped = tail_start - head_end
    placeholder: dict[str, Any] = {
        "role": "user",
        "content": f"[snipped {snipped} messages]",
    }
    messages[:] = messages[:head_end] + [placeholder] + messages[tail_start:]

    logger.debug("snip_compact: removed %d messages (head=%d tail=%d)", snipped, head_end, tail_start)
    return messages


# ---------------------------------------------------------------------------
#  L2 — micro_compact（第三步执行）
# ---------------------------------------------------------------------------

def micro_compact(
    messages: list[dict[str, Any]],
    *,
    keep_recent: int = DEFAULT_KEEP_RECENT,
) -> list[dict[str, Any]]:
    """将旧的 tool_result 内容替换为占位符。

    收集对话中所有的 tool_result 块。保留最新 *keep_recent* 条的完整内容；
    更旧的且内容超过 120 字符的块替换为::

        "[更早的工具结果已被压缩。如需使用请重新运行。]"

    返回 *messages*（原地修改）。
    """
    tool_results = _collect_tool_results(messages)
    if len(tool_results) <= keep_recent:
        return messages

    # 从旧到新排列，保留最后 N 条
    old_results = tool_results[:-keep_recent] if keep_recent > 0 else tool_results
    for _mi, _bi, block in old_results:
        if len(block.get("content", "")) > 120:
            block["content"] = "[Earlier tool result compacted. Re-run if needed.]"

    logger.debug(
        "micro_compact: %d total tool_results, kept %d",
        len(tool_results), keep_recent,
    )
    return messages


# ---------------------------------------------------------------------------
#  L4 — compact_history（第四步执行，仅当前三步后仍超预算时触发）
# ---------------------------------------------------------------------------

async def compact_history(
    messages: list[dict[str, Any]],
    *,
    summarize: SummarizeFn,
    workdir: Path | str | None = None,
) -> list[dict[str, Any]]:
    """通过 LLM 生成完整对话摘要。

    1. 将当前消息写入 ``.transcripts/transcript_{ts}.jsonl``。
    2. 请求 LLM 生成摘要，保留：目标、发现、已读/已改文件、
       剩余工作、用户约束。
    3. 将 **所有** 消息替换为一条摘要消息::

           [{"role": "user", "content": "[已压缩]\\n\\n{摘要}"}]

    返回一个 **新** 列表（调用方应重新绑定，如
    ``messages = await compact_history(...)``）。
    """
    wd = Path(workdir) if workdir else Path.cwd()
    transcript_path = _write_transcript(messages, wd)
    logger.info("Transcript saved: %s", transcript_path)

    conversation = json.dumps(messages, default=str, ensure_ascii=False)[:80000]

    prompt = (
        "请摘要以下 coding-agent 对话，以便继续工作。\n"
        "保留：1. 当前目标，2. 关键发现/决策，3. 已读/已改的文件，"
        "4. 剩余工作，5. 用户约束。\n"
        "要简洁但具体。\n\n"
        + conversation
    )

    try:
        summary = await summarize(prompt)
        if not summary:
            return _clone_messages(messages)
    except Exception:
        logger.exception("compact_history 摘要生成失败")
        return _clone_messages(messages)

    return [{"role": "user", "content": f"[Compacted]\n\n{summary}"}]


# ---------------------------------------------------------------------------
#  Emergency — reactive_compact（在 prompt_too_long 时触发）
# ---------------------------------------------------------------------------

async def reactive_compact(
    messages: list[dict[str, Any]],
    *,
    summarize: SummarizeFn,
    workdir: Path | str | None = None,
) -> list[dict[str, Any]]:
    """当 API 返回 ``prompt_too_long`` 时的紧急摘要。

    1. 写入对话记录。
    2. 摘要较旧的部分（除最后 5 条外的所有消息）。
    3. 完整保留尾部（约最后 5 条消息）—— 这些是 LLM
       正在尝试操作的最新工具调用上下文。
    4. 应用与 L1 相同的 tool_use / tool_result 配对保护。

    返回一个 **新** 列表。
    """
    wd = Path(workdir) if workdir else Path.cwd()
    _write_transcript(messages, wd)

    tail_start = max(0, len(messages) - 5)

    # 配对保护（与 L1 逻辑相同）
    if (
        tail_start > 0
        and tail_start < len(messages)
        and _is_tool_result_message(messages[tail_start])
        and _message_has_tool_use(messages[tail_start - 1])
    ):
        tail_start -= 1

    old_part = messages[:tail_start]
    tail = messages[tail_start:]
    if not old_part:
        # 少量超大消息也需要真正缩减，不能只在原文前追加一个空摘要。
        return await compact_history(messages, summarize=summarize, workdir=wd)

    # 只摘要旧的部分
    conv = json.dumps(old_part, default=str, ensure_ascii=False)[:80000]
    prompt = (
        "请摘要以下 coding-agent 对话，以便继续工作。\n"
        "保留：1. 当前目标，2. 关键发现/决策，3. 已读/已改的文件，"
        "4. 剩余工作，5. 用户约束。\n"
        "要简洁但具体。\n\n"
        + conv
    )

    try:
        summary = await summarize(prompt)
        if not summary:
            return _clone_messages(messages)
    except Exception:
        logger.exception("reactive_compact 摘要生成失败")
        return _clone_messages(messages)

    logger.info("reactive_compact: 已将 %d 条消息压缩为摘要 + %d 条尾部", len(old_part), len(tail))
    return [{"role": "user", "content": f"[Reactive compact]\n\n{summary}"}, *tail]


# ---------------------------------------------------------------------------
#  管道编排器
# ---------------------------------------------------------------------------

class CompactionPipeline:
    """按正确顺序编排全部 4 层压缩。

    在 agent loop 中的典型用法::

        pipe = CompactionPipeline(summarize=provider.summarize)

        while True:
            # LLM 调用前：执行 L3 → L1 → L2 → [若超限则 L4]
            await pipe.run(messages)

            try:
                response = await provider.create_message(...)
                retries = 0
            except PromptTooLongError:
                if retries < pipe.max_reactive_retries:
                    await pipe.reactive(messages)
                    retries += 1
                    continue
                raise
    """

    def __init__(
        self,
        *,
        summarize: SummarizeFn | None = None,
        workdir: Path | str | None = None,
        context_limit: int = DEFAULT_CONTEXT_LIMIT,
        keep_recent: int = DEFAULT_KEEP_RECENT,
        persist_threshold: int = DEFAULT_PERSIST_THRESHOLD,
        max_messages: int = DEFAULT_MAX_MESSAGES,
        max_reactive_retries: int = DEFAULT_MAX_REACTIVE_RETRIES,
    ) -> None:
        self.summarize = summarize
        self.workdir = Path(workdir) if workdir else Path.cwd()
        self.context_limit = context_limit
        self.keep_recent = keep_recent
        self.persist_threshold = persist_threshold
        self.max_messages = max_messages
        self.max_reactive_retries = max_reactive_retries

    # ── LLM 调用前管道（除非 L4 触发，否则不调 API）──────

    async def run(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """执行 L3 → L1 → L2 → [若超限则 L4]。

        原地修改 *messages*。返回 messages 以便链式调用。

        前三层优先通过持久化和裁剪减少上下文；只有仍超出预算且提供
        summarize 回调时才调用模型生成完整摘要。大型工具输出及
        摘要前的对话记录可能写入工作目录下的持久化文件。
        """
        # L3 → 将大工具输出持久化到磁盘
        tool_result_budget(
            messages,
            workdir=self.workdir,
            persist_threshold=self.persist_threshold,
        )

        # L1 → 裁剪中间消息（配对保护）
        snip_compact(messages, max_messages=self.max_messages)

        # L2 → 旧工具结果替换为占位符
        micro_compact(messages, keep_recent=self.keep_recent)

        # L4 → 若仍超预算，用 LLM 生成摘要
        if estimate_tokens(messages) > self.context_limit:
            if self.summarize is None:
                logger.warning(
                    "上下文超出限制 (%d tokens) 但未配置 summarize 回调 —— 跳过 L4",
                    estimate_tokens(messages),
                )
            else:
                logger.info("上下文超出限制 —— 触发 L4 compact_history")
                new_msgs = await compact_history(
                    messages,
                    summarize=self.summarize,
                    workdir=self.workdir,
                )
                messages[:] = new_msgs

        return messages

    # ── 紧急压缩（1 次 API 调用）────────────────────────────

    async def reactive(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """紧急摘要 —— 在 ``prompt_too_long`` 错误后调用。

        返回新列表；调用方必须重新绑定。"""
        if self.summarize is None:
            raise RuntimeError("reactive_compact 需要 summarize 回调")
        return await reactive_compact(
            messages,
            summarize=self.summarize,
            workdir=self.workdir,
        )

    # ── 辅助方法 ────────────────────────────────────────────

    def estimate_tokens(self, messages: list[dict[str, Any]]) -> int:
        """公开的 token 计数快捷方法。"""
        return estimate_tokens(messages)


# ---------------------------------------------------------------------------
#  内部辅助函数
# ---------------------------------------------------------------------------

def _write_transcript(messages: list[dict[str, Any]], workdir: Path) -> Path:
    """将消息写入带时间戳的 .jsonl 文件，返回路径。"""
    tx_dir = workdir / TRANSCRIPT_DIR_NAME
    tx_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    path = tx_dir / f"transcript_{ts}_{uuid.uuid4().hex[:8]}.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for msg in messages:
            fh.write(json.dumps(msg, default=str, ensure_ascii=False) + "\n")
    return path
