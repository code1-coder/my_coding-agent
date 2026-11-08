"""
自主智能体生命周期 —— WORK → IDLE → SHUTDOWN。

生命周期状态::

    WORK      — 活跃处理消息、调用 LLM、执行工具。
    IDLE      — 每 ``POLL_INTERVAL`` 秒轮询收件箱 + 任务看板。
                收到收件箱消息 → WORK，有未认领任务 → 认领 → WORK。
                超过 ``IDLE_TIMEOUT`` 秒 → SHUTDOWN。
    SHUTDOWN  — 优雅退出，向 lead 发送最终摘要。

身份重新注入
    上下文压缩后，agent 的身份（名称、角色）可能从
    系统提示词中丢失。每次进入 WORK 阶段前，且当
    消息数 ≤ 3 时，注入一条 ``<identity>`` 消息
    以让 agent 记住自己的身份。

常量:
    * ``POLL_INTERVAL`` = 5 秒
    * ``IDLE_TIMEOUT``  = 60 秒
    * ``MAX_WORK_TURNS`` = 10  （每个 WORK 阶段最多执行轮数）

用法::

    lifecycle = AgentLifecycle(
        name="worker-01",
        role="tester",
        bus=FileMailbox(),
        task_board=TaskBoard(),
        llm_call=provider.create_message,
    )

    await lifecycle.run(prompt="运行测试套件并报告结果")
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  生命周期常量
# ---------------------------------------------------------------------------

POLL_INTERVAL = 5     # IDLE 状态下检查收件箱/任务看板的间隔秒数
IDLE_TIMEOUT = 60      # IDLE → SHUTDOWN 的超时秒数
MAX_WORK_TURNS = 10    # 每个 WORK 阶段的最大 LLM 轮数


# ---------------------------------------------------------------------------
#  生命周期状态
# ---------------------------------------------------------------------------


class LifecycleState(Enum):
    WORK = "work"
    IDLE = "idle"
    SHUTDOWN = "shutdown"


# ---------------------------------------------------------------------------
#  自主智能体的工具集
# ---------------------------------------------------------------------------

AUTONOMOUS_TOOL_NAMES = {
    "bash", "read_file", "write_file", "edit_file",
    "send_message", "list_tasks", "claim_task", "complete_task",
}


# ---------------------------------------------------------------------------
#  智能体生命周期（AgentLifecycle）
# ---------------------------------------------------------------------------


@dataclass
class IdlePollResult:
    action: str  # 轮询结果：恢复工作、关闭或超时（work、shutdown、timeout）
    inbox_messages: list[dict[str, Any]] = field(default_factory=list)
    claimed_task: dict[str, Any] | None = None


class AgentLifecycle:
    """管理单个自主智能体的 WORK→IDLE→SHUTDOWN 循环。

    使用同步模式运行，适合放在后台线程中。在原生异步环境中，将阻塞调用
    包装在 ``asyncio.to_thread`` 中即可。

    用法::

        lifecycle = AgentLifecycle(
            name="worker-01",
            role="backend-dev",
            bus=mailbox,
            task_board=board,
            llm_call=provider.create_message,
            tools=my_tools,
        )
        result = lifecycle.run(initial_prompt="运行测试套件")
    """

    def __init__(
        self,
        *,
        name: str,
        role: str,
        bus: Any,                # 消息总线（FileMailbox 文件邮箱）
        llm_call: Callable[..., Any],
        tools: list[dict[str, Any]] | None = None,
        tool_handlers: dict[str, Any] | None = None,
        task_board: Any | None = None,  # 任务看板（TaskBoard）
        workdir: Path | str | None = None,
        poll_interval: float = POLL_INTERVAL,
        idle_timeout: float = IDLE_TIMEOUT,
        max_work_turns: int = MAX_WORK_TURNS,
    ) -> None:
        self.name = name
        self.role = role
        self.bus = bus
        self._llm_call = llm_call
        self._task_board = task_board
        self._workdir = Path(workdir) if workdir else Path.cwd()
        self._poll_interval = poll_interval
        self._idle_timeout = idle_timeout
        self._max_work_turns = max_work_turns

        # 构建工具规格（如果未提供）
        if tools is not None:
            self._tools = tools
        else:
            self._tools = self._build_default_tools()

        if tool_handlers is not None:
            self._tool_handlers = tool_handlers
        else:
            self._tool_handlers = self._build_default_handlers()

        self._state = LifecycleState.WORK
        self._messages: list[dict[str, Any]] = []

    # ── 公开 API ─────────────────────────────────────────────

    def run(self, initial_prompt: str) -> str:
        """执行完整生命周期。阻塞直到 SHUTDOWN。

        返回发送给 lead 的最终摘要文本。
        """
        self._messages = [{"role": "user", "content": initial_prompt}]

        while self._state != LifecycleState.SHUTDOWN:
            if self._state == LifecycleState.WORK:
                self._run_work_phase()
            elif self._state == LifecycleState.IDLE:
                self._run_idle_phase()

        # ── 发送最终摘要 ─────────────────────────────────────
        summary = self._extract_summary()
        self.bus.send(self.name, "lead", summary, msg_type="result")
        logger.info("[%s] shutdown complete", self.name)
        return summary

    # ── WORK 阶段 ─────────────────────────────────────────────

    async def run_async(self, initial_prompt: str, *, run_turn: Callable[..., Awaitable[str]]) -> str:
        """在应用所属事件循环运行生命周期，共用带 Hook 和权限检查的运行器。"""
        self._messages = [{"role": "user", "content": initial_prompt}]
        self._state = LifecycleState.WORK
        try:
            while self._state != LifecycleState.SHUTDOWN:
                inbox = self.bus.read_inbox(self.name)
                if self._handle_inbox(inbox):
                    break
                if inbox:
                    self._messages.append({"role": "user", "content": f"<inbox>{self._as_json(inbox)}</inbox>"})
                    self._state = LifecycleState.WORK
                if self._state == LifecycleState.WORK:
                    # 身份放在系统提示词中，压缩消息后也不会丢失。
                    await run_turn(self._messages, self._build_system(), self._max_work_turns)
                    self._state = LifecycleState.IDLE
                    deadline = time.monotonic() + self._idle_timeout
                elif self._task_board is not None:
                    tasks = self._task_board.find_unclaimed()
                    if tasks:
                        task = tasks[0]
                        claimed = self._task_board.claim(task.id, owner=self.name)
                        if "Claimed" in claimed:
                            self._messages.append({"role": "user", "content": (
                                f"<auto-claimed>任务 {task.id}: {task.subject}\n{task.description}</auto-claimed>"
                            )})
                            self._state = LifecycleState.WORK
                            continue
                if self._state == LifecycleState.IDLE:
                    if time.monotonic() >= deadline:
                        break
                    await asyncio.sleep(min(self._poll_interval, max(0, deadline - time.monotonic())))
        finally:
            self._state = LifecycleState.SHUTDOWN
            summary = self._extract_summary()
            self.bus.send(self.name, "lead", summary, msg_type="result")
        return summary

    def _run_work_phase(self) -> None:
        """运行最多 MAX_WORK_TURNS 轮的 agent loop。"""
        # 历史消息较少时重新注入身份，帮助模型保持角色一致
        if len(self._messages) <= 3:
            self._messages.insert(0, {
                "role": "user",
                "content": (
                    f"<identity>你是 '{self.name}'，角色: {self.role}。"
                    f"继续你的工作。</identity>"
                ),
            })

        for _turn in range(self._max_work_turns):
            # 消费收件箱
            inbox = self.bus.read_inbox(self.name)
            if inbox:
                shutdown = self._handle_inbox(inbox)
                if shutdown:
                    self._state = LifecycleState.SHUTDOWN
                    return
                if inbox:
                    self._messages.append({
                        "role": "user",
                        "content": f"<inbox>{self._as_json(inbox)}</inbox>",
                    })

            # ── LLM 调用 ───────────────────────────────────────
            try:
                response = self._llm_call(
                    self._messages[-20:],
                    self._tools,
                    self._build_system(),
                    8000,
                )
            except Exception as exc:
                logger.exception("[%s] LLM 调用失败", self.name)
                self._messages.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": f"错误: {exc}"}],
                })
                break

            self._messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use":
                self._state = LifecycleState.IDLE
                return

            # ── 工具执行 ─────────────────────────────────────
            results: list[dict[str, Any]] = []
            for block in response.content:
                tp = getattr(block, "type", None) if hasattr(block, "type") else block.get("type", "")
                if tp != "tool_use":
                    continue

                if hasattr(block, "name"):
                    tool_name, tool_input, tool_id = block.name, dict(block.input), block.id
                else:
                    tool_name = block.get("name", "")
                    tool_input = block.get("input", {})
                    tool_id = block.get("id", "")

                output = self._execute_tool(tool_name, tool_input)
                output_dict = output.to_dict() if hasattr(output, "to_dict") else None
                results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": output_dict["content"] if output_dict else output,
                    **({"is_error": True} if output_dict and output_dict.get("is_error") else {}),
                })

            self._messages.append({"role": "user", "content": results})

        # 达到最大轮数 → IDLE
        self._state = LifecycleState.IDLE

    # ── IDLE 阶段 ─────────────────────────────────────────────

    def _run_idle_phase(self) -> None:
        """轮询收件箱 + 任务看板，直到有工作或超时。"""
        deadline = time.monotonic() + self._idle_timeout

        while time.monotonic() < deadline:
            time.sleep(self._poll_interval)

            # ── 检查收件箱 ────────────────────────────────────
            inbox = self.bus.read_inbox(self.name)
            if inbox:
                shutdown = self._handle_inbox(inbox)
                if shutdown:
                    self._state = LifecycleState.SHUTDOWN
                    return
                if inbox:
                    self._messages.append({
                        "role": "user",
                        "content": f"<inbox>{self._as_json(inbox)}</inbox>",
                    })
                    self._state = LifecycleState.WORK
                    logger.info("[%s] IDLE → WORK (inbox)", self.name)
                    return

            # ── 扫描任务看板 ─────────────────────────────────
            if self._task_board:
                unclaimed = self._task_board.find_unclaimed()
                if unclaimed:
                    task = unclaimed[0]
                    result = self._task_board.claim(task.id, owner=self.name)
                    if "Claimed" in result:
                        self._messages.append({
                            "role": "user",
                            "content": f"<auto-claimed>任务 {task.id}: {task.subject}</auto-claimed>",
                        })
                        logger.info("[%s] IDLE → WORK（已认领: %s）", self.name, task.subject)
                        self._state = LifecycleState.WORK
                        return

        # 超时
        logger.info("[%s] IDLE 超时（%ss），关闭中", self.name, self._idle_timeout)
        self._state = LifecycleState.SHUTDOWN

    # ── 辅助方法 ────────────────────────────────────────────

    def _handle_inbox(self, msgs: list[dict[str, Any]]) -> bool:
        """处理协议消息。返回 True 表示收到关闭请求。"""
        for msg in msgs:
            msg_type = msg.get("type", "message")
            if msg_type == "shutdown_request":
                req_id = msg.get("metadata", {}).get("request_id", "")
                self.bus.send(
                    self.name, "lead",
                    "正在优雅关闭。",
                    msg_type="shutdown_response",
                    metadata={"request_id": req_id, "approve": True},
                )
                logger.info("[%s] 关闭已批准 (%s)", self.name, req_id)
                return True
        return False

    def _execute_tool(self, tool_name: str, tool_input: dict[str, Any]) -> Any:
        """将工具调用分发给处理器或 MessageBus 发送。"""
        if tool_name == "send_message":
            self.bus.send(
                self.name,
                tool_input.get("to", "lead"),
                tool_input.get("content", ""),
            )
            return "Sent"

        if tool_name in ("claim_task", "complete_task", "list_tasks") and self._task_board:
            if tool_name == "claim_task":
                return self._task_board.claim(tool_input.get("task_id", ""), owner=self.name)
            if tool_name == "complete_task":
                return self._task_board.complete(tool_input.get("task_id", ""))
            if tool_name == "list_tasks":
                tasks = self._task_board.list_all()
                if not tasks:
                    return "No tasks."
                return "\n".join(f"  {t.id}: {t.subject} [{t.status}]" for t in tasks)

        handler = self._tool_handlers.get(tool_name)
        if handler:
            try:
                return handler(**tool_input)
            except Exception as exc:
                return f"Error: {exc}"
        return f"Unknown tool: {tool_name}"

    def _extract_summary(self) -> str:
        """从后往前遍历消息，找到最后一条 assistant 文本。"""
        for msg in reversed(self._messages):
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for block in reversed(content):
                tp = block.get("type", "") if isinstance(block, dict) else getattr(block, "type", None)
                if tp == "text":
                    text = block.get("text", "") if isinstance(block, dict) else getattr(block, "text", "")
                    if text.strip():
                        return text.strip()
        return "Done."

    def _build_system(self) -> str:
        """根据智能体名称、角色和工作目录组装当前系统提示词。"""
        return (
            f"你是 '{self.name}'，一名 {self.role}。"
            f"使用工具完成任务。"
            f"检查收件箱中的协议消息。"
            f"通过 send_message 向 'lead' 发送结果。"
        )

    # ── 默认工具 ──────────────────────────────────────────

    def _build_default_tools(self) -> list[dict[str, Any]]:
        """返回最小化的工具定义（Anthropic 格式）。"""
        return [
            {
                "name": "bash", "description": "运行 shell 命令。",
                "input_schema": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
            {
                "name": "read_file", "description": "读取文件内容。",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
            {
                "name": "write_file", "description": "将内容写入文件。",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                    "required": ["path", "content"],
                },
            },
            {
                "name": "send_message", "description": "向另一个智能体发送消息。",
                "input_schema": {
                    "type": "object",
                    "properties": {"to": {"type": "string"}, "content": {"type": "string"}},
                    "required": ["to", "content"],
                },
            },
        ]

    def _build_default_handlers(self) -> dict[str, Any]:
        """构建与 ``_build_default_tools`` 匹配的工具处理器。"""
        from coding_agent.tools.builtin import BashTool, ReadFileTool, WriteFileTool

        bash = BashTool(workdir=self._workdir)
        reader = ReadFileTool(workdir=self._workdir)
        writer = WriteFileTool(workdir=self._workdir)

        return {
            "bash": bash.execute,
            "read_file": reader.execute,
            "write_file": writer.execute,
        }

    @staticmethod
    def _as_json(obj: Any) -> str:
        """序列化收件箱等上下文数据，保留中文并将不支持的对象转为字符串。"""
        import json as _json
        return _json.dumps(obj, default=str, ensure_ascii=False)
