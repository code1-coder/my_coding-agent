"""上下文管理 —— 4+1 层压缩管道 + 错误恢复。"""

from coding_agent.context.compaction import (
    CompactionPipeline,
    compact_history,
    estimate_tokens,
    micro_compact,
    reactive_compact,
    snip_compact,
    tool_result_budget,
)
from coding_agent.context.recovery import (
    CONTINUATION_PROMPT,
    RecoveryManager,
    RecoveryState,
    is_prompt_too_long_error,
    with_retry,
)

__all__ = [
    # 压缩
    "CompactionPipeline",
    "estimate_tokens",
    "tool_result_budget",
    "snip_compact",
    "micro_compact",
    "compact_history",
    "reactive_compact",
    # 恢复
    "RecoveryState",
    "RecoveryManager",
    "with_retry",
    "is_prompt_too_long_error",
    "CONTINUATION_PROMPT",
]
