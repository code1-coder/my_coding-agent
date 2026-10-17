"""
智能体团队基础设施 —— MessageBus + 团队协作。

基于文件的 MessageBus
    每个智能体有一个 ``.team/inbox/{name}.jsonl`` 邮箱。消息以
    JSON lines 逐行追加。读取是 **破坏性** 的：``read_inbox``
    读取并 **消费** 文件。``peek`` 只检查不消费。

Lead 工具
    ``spawn_teammate`` —— 创建一个后台队友线程，拥有自己的 agent loop、
    受限工具集和 MessageBus 集成。
    ``send_message`` / ``check_inbox`` —— 智能体间通信。

用法::

    bus = FileMailbox(workdir=Path.cwd())
    bus.send("lead", "worker-01", "请运行测试")

    msgs = bus.read_inbox("worker-01")
"""

from __future__ import annotations

import asyncio
import json
import re
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  常量
# ---------------------------------------------------------------------------

TEAM_DIR_NAME = ".team"
INBOX_DIR_NAME = "inbox"


# ---------------------------------------------------------------------------
#  MessageBus 接口
# ---------------------------------------------------------------------------

class MessageBus:
    """智能体间消息传递的抽象接口。

    子类实现传输层（文件、Redis、gRPC 等）。
    """

    def send(
        self,
        from_agent: str,
        to_agent: str,
        content: str,
        msg_type: str = "message",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """向指定智能体发送正文、消息类型和可选元数据，由子类实现传输。"""
        raise NotImplementedError

    def read_inbox(self, agent: str) -> list[dict[str, Any]]:
        """读取并消费指定智能体的消息；具体存储和清理方式由子类实现。"""
        raise NotImplementedError

    def peek(self, agent: str) -> bool:
        """检查指定智能体是否有未读消息，不消费消息。"""
        raise NotImplementedError


# ---------------------------------------------------------------------------
#  FileMailbox —— 默认的基于文件的实现
# ---------------------------------------------------------------------------


@dataclass
class _Envelope:
    """总线上的一条序列化消息。"""
    __slots__ = ("from_", "to", "content", "type", "ts", "metadata")

    def __init__(
        self,
        from_: str,
        to: str,
        content: str,
        msg_type: str = "message",
        ts: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.from_ = from_
        self.to = to
        self.content = content
        self.type = msg_type
        self.ts = ts if ts is not None else time.time()
        self.metadata = metadata or {}

    def to_dict(self) -> dict[str, Any]:
        """转换为邮箱文件使用的字典，保留发送方、接收方及消息元数据。"""
        return {
            "from": self.from_,
            "to": self.to,
            "content": self.content,
            "type": self.type,
            "ts": self.ts,
            "metadata": self.metadata,
        }


class FileMailbox(MessageBus):
    """基于文件系统的消息总线，由 ``.team/inbox/{name}.jsonl`` 支撑。

    写入是追加模式。读取是 **破坏性** 的（读取 + 删除文件）。

    教学版本：无文件锁。生产环境应使用
    ``proper-lockfile`` 或类似的 advisory-lock 库。
    """

    def __init__(self, workdir: Path | str | None = None, *, namespace: str | None = None) -> None:
        wd = Path(workdir) if workdir else Path.cwd()
        self._inbox_dir = wd / TEAM_DIR_NAME / INBOX_DIR_NAME
        if namespace:
            if not re.fullmatch(r"[\w-]+", namespace):
                raise ValueError("邮箱命名空间包含无效字符")
            self._inbox_dir /= namespace
        self._inbox_dir.mkdir(parents=True, exist_ok=True)

    def _inbox_path(self, agent: str) -> Path:
        if not re.fullmatch(r"[\w-]+", agent):
            raise ValueError("智能体名称只允许字母、数字、下划线和连字符")
        return self._inbox_dir / f"{agent}.jsonl"

    # ── MessageBus 接口 ───────────────────────────────────

    def send(
        self,
        from_agent: str,
        to_agent: str,
        content: str,
        msg_type: str = "message",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """将消息以一行 JSON 追加到接收方的邮箱文件。

        不覆盖现有消息；同一邮箱的并发读写需要调用方自行协调。
        """
        env = _Envelope(
            from_=from_agent,
            to=to_agent,
            content=content,
            msg_type=msg_type,
            metadata=metadata,
        )
        inbox = self._inbox_path(to_agent)
        with open(inbox, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(env.to_dict(), ensure_ascii=False) + "\n")
        logger.debug("[bus] %s → %s (%s): %s", from_agent, to_agent, msg_type, content[:50])

    def read_inbox(self, agent: str) -> list[dict[str, Any]]:
        """读取并消费 *agent* 的所有消息，返回成功解析的消息字典列表。

        邮箱不存在时返回空列表，格式损坏的行会被跳过；读取完成后
        删除邮箱文件，因此再次读取不会返回同一批消息。
        """
        inbox = self._inbox_path(agent)
        if not inbox.exists():
            return []
        raw = inbox.read_text(encoding="utf-8")
        msgs: list[dict[str, Any]] = []
        for line in raw.splitlines():
            line = line.strip()
            if line:
                try:
                    msgs.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("%s 的收件箱中有损坏的行: %s", agent, line[:80])
        inbox.unlink()  # 消费
        return msgs

    def peek(self, agent: str) -> bool:
        """非破坏性检查：*agent* 是否有未读消息。"""
        inbox = self._inbox_path(agent)
        return inbox.exists() and inbox.stat().st_size > 0


# ---------------------------------------------------------------------------
#  团队注册表
# ---------------------------------------------------------------------------


class TeamRegistry:
    """追踪活跃队友并支持协作。

    用法::

        team = TeamRegistry(bus=mailbox, llm_call=provider.create_message)
        team.spawn("worker-01", role="tester", prompt="运行所有测试并报告")
        # 之后:
        inbox = team.check_inbox("lead")
    """

    def __init__(
        self,
        *,
        bus: MessageBus,
        llm_call: Callable[..., Any] | None = None,
        workdir: Path | str | None = None,
    ) -> None:
        self.bus = bus
        self._llm_call = llm_call
        self._workdir = Path(workdir) if workdir else Path.cwd()
        self._active: dict[str, threading.Thread] = {}
        self._async_active: dict[str, asyncio.Task] = {}

    @property
    def active_names(self) -> list[str]:
        return sorted(set(self._active) | {n for n, t in self._async_active.items() if not t.done()})

    def spawn_async(self, name: str, run: Callable[[], Any]) -> str:
        """调度异步队友，避免在线程中新建循环后复用已有 HTTP 客户端。"""
        if name in self.active_names:
            return f"Teammate '{name}' already exists"

        async def guarded():
            try:
                return await run()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.bus.send(name, "lead", f"队友执行失败：{exc}", msg_type="error")
                return f"Error: {exc}"

        self._async_active[name] = asyncio.create_task(guarded(), name=f"teammate:{name}")
        return f"Teammate '{name}' spawned"

    async def wait(self, name: str) -> str:
        """等待指定队友的完整生命周期结束。"""
        if name not in self._async_active:
            return f"Unknown teammate: {name}"
        return await asyncio.shield(self._async_active[name])

    async def aclose(self) -> None:
        """会话退出时取消队友，并等待其清理完成。"""
        tasks = list(self._async_active.values())
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def spawn(
        self,
        name: str,
        role: str,
        prompt: str,
        *,
        max_turns: int = 10,
    ) -> str:
        """在后台守护线程中启动一个队友。

        参数:
            name: 唯一的智能体名称。
            role: 人类可读的角色（如 "tester"、"reviewer"）。
            prompt: 初始任务 / 指令。
            max_turns: 安全限制（默认 10）。

        返回:
            人类可读的状态字符串。
        """
        if name in self._active:
            return f"Teammate '{name}' already exists"

        if self._llm_call is None:
            return "Error: no llm_call configured for team"

        from coding_agent.agents.subagent import (
            SUB_SYSTEM_TEMPLATE,
            _build_base_tools,
            _extract_last_assistant_text,
        )

        all_tools = _build_base_tools(self._workdir)
        sub_tools = [t.to_anthropic_spec() for t in all_tools.values()]
        sub_handlers = dict(all_tools)

        system = SUB_SYSTEM_TEMPLATE.format(workdir=self._workdir)
        llm_call = self._llm_call
        bus = self.bus

        def _run() -> None:
            messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]

            for _turn in range(max_turns):
                # 检查收件箱
                inbox = bus.read_inbox(name)
                if inbox:
                    messages.append({
                        "role": "user",
                        "content": f"<inbox>{json.dumps(inbox)}</inbox>",
                    })

                try:
                    response = llm_call(
                        messages[-20:], sub_tools, system, 8000,
                    )
                except Exception:
                    break

                messages.append({"role": "assistant", "content": response.content})

                if response.stop_reason != "tool_use":
                    break

                results: list[dict[str, Any]] = []
                for block in response.content:
                    tp = getattr(block, "type", None)
                    if tp != "tool_use":
                        continue
                    tool_name = block.name
                    tool_input = dict(block.input)
                    tool_id = block.id

                    if tool_name == "send_message":
                        bus.send(name, tool_input.get("to", "lead"), tool_input.get("content", ""))
                        results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": "Sent",
                        })
                        continue

                    tool = sub_handlers.get(tool_name)
                    if tool:
                        try:
                            output = tool.execute(**tool_input)
                        except Exception as exc:
                            output = f"Error: {exc}"
                    else:
                        output = f"Unknown tool: {tool_name}"

                    results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": str(output),
                    })

                messages.append({"role": "user", "content": results})

            # 发送最终摘要
            summary = _extract_last_assistant_text(messages) or "Done."
            bus.send(name, "lead", summary, msg_type="result")
            logger.info("Teammate %s finished", name)

        t = threading.Thread(target=_run, daemon=True)
        self._active[name] = t
        t.start()
        logger.info("Teammate spawned: %s (%s)", name, role)
        return f"Teammate '{name}' spawned as {role}"

    def check_inbox(self, agent: str = "lead") -> list[dict[str, Any]]:
        """读取并返回 *agent* 的所有消息。"""
        return self.bus.read_inbox(agent)

    def send(self, to_agent: str, content: str, from_agent: str = "lead") -> str:
        """发送消息并返回确认字符串。"""
        self.bus.send(from_agent, to_agent, content)
        return f"已发送给 {to_agent}"
