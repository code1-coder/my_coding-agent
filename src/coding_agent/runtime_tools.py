"""把已有技能、任务、子智能体、团队和 MCP 模块适配到统一工具注册表。"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from coding_agent.agents.lifecycle import AgentLifecycle
from coding_agent.tools.builtin import BashTool, ReadFileTool, WriteFileTool, EditFileTool, GlobTool

if TYPE_CHECKING:
    from coding_agent.runtime import AgentRuntime
    from coding_agent.tools.registry import ToolRegistry


def register_runtime_tools(runtime: AgentRuntime, registry: ToolRegistry, *, agent_name: str,
                           delegation: bool, allowed_tools: set[str] | None) -> None:
    """只负责能力适配；所有实际调用由 AgentLoop 统一检查、分发并触发 Hook。"""
    for cls in (BashTool, ReadFileTool, WriteFileTool, EditFileTool, GlobTool):
        tool = cls(workdir=runtime.workdir)
        if allowed_tools is None or tool.name in allowed_tools:
            registry.register_tool(tool)

    @registry.register(name="load_skill", description="按技能目录中的名称加载完整技能说明。")
    async def load_skill(name: str) -> str:
        return runtime.skills.load(name) or f"未找到技能：{name}"

    # 探索者仅提供读取工具和技能；执行者与团队成员可以管理任务、发送消息。
    if allowed_tools is not None and "write_file" not in allowed_tools:
        return

    def as_json(data):
        return json.dumps(data, ensure_ascii=False)

    @registry.register(name="create_task", description="创建带依赖的持久化任务。", permission="write")
    async def create_task(subject: str, description: str = "", blockedBy: list[str] | None = None) -> str:
        async with runtime.workspace_lock:
            return as_json(runtime.tasks.create(subject, description, blockedBy).to_dict())

    @registry.register(name="list_tasks", description="列出任务看板中的任务与状态。")
    async def list_tasks() -> str:
        return as_json([task.to_dict() for task in runtime.tasks.list_all()])

    @registry.register(name="get_task", description="读取指定任务。")
    async def get_task(task_id: str) -> str:
        return as_json(runtime.tasks.get(task_id).to_dict())

    @registry.register(name="claim_task", description="依赖完成后认领任务。", permission="write")
    async def claim_task(task_id: str) -> str:
        async with runtime.workspace_lock:
            return runtime.tasks.claim(task_id, owner=agent_name)

    @registry.register(name="complete_task", description="完成任务并解除后续任务的依赖阻塞。", permission="write")
    async def complete_task(task_id: str) -> str:
        async with runtime.workspace_lock:
            return runtime.tasks.complete(task_id)

    @registry.register(name="delete_task", description="删除指定任务。", permission="write")
    async def delete_task(task_id: str) -> str:
        async with runtime.workspace_lock:
            return as_json({"deleted": runtime.tasks.delete(task_id)})

    @registry.register(name="update_task", description="更新任务说明或依赖；状态通过认领、完成工具变更。", permission="write")
    async def update_task(task_id: str, subject: str | None = None, description: str | None = None,
                          blockedBy: list[str] | None = None) -> str:
        async with runtime.workspace_lock:
            entry = runtime.tasks.get(task_id)
            if subject is not None:
                entry.subject = subject
            if description is not None:
                entry.description = description
            if blockedBy is not None:
                entry.blockedBy = blockedBy
            runtime.tasks.update(entry)
            return as_json(entry.to_dict())

    @registry.register(name="send_message", description="向当前会话的队友或 lead 发送消息。", permission="write")
    async def send_message(to: str, content: str) -> str:
        return runtime.team.send(to, content, from_agent=agent_name)

    @registry.register(name="check_inbox", description="读取并消费当前智能体的收件箱。")
    async def check_inbox() -> str:
        return as_json(runtime.team.check_inbox(agent_name))

    if not delegation:
        return

    @registry.register(name="task", description="委托隔离上下文的子任务，role 为 explorer 或 worker。")
    async def task(description: str, role: str = "worker", max_turns: int = 30) -> str:
        return await runtime.pool.dispatch(description, role=role, max_turns=max_turns)

    @registry.register(name="connect_mcp", description="连接项目已注册的 MCP 服务并加载其工具。")
    async def connect_mcp(name: str) -> str:
        return runtime.connect_mcp(name)

    @registry.register(name="spawn_teammate", description="启动后台队友，完成工作后轮询邮箱和任务看板。")
    async def spawn_teammate(name: str, role: str, prompt: str) -> str:
        if not re.fullmatch(r"[\w-]+", name) or name == "lead":
            raise ValueError("队友名称无效或与 lead 冲突")
        if len(runtime.team.active_names) >= 4:
            return "同时运行的队友数量已达到 4 个"
        child_registry = runtime.make_registry(agent_name=name, delegation=False)
        lifecycle = AgentLifecycle(
            name=name, role=role, bus=runtime.mailbox, llm_call=runtime.provider.create_message,
            task_board=runtime.tasks, workdir=runtime.workdir,
            idle_timeout=runtime.cfg.team_idle_timeout,
        )

        async def run_turn(messages, system, max_turns):
            return await runtime.run_child(messages, system, agent_name=name,
                                           registry=child_registry, max_iterations=max_turns)

        return runtime.team.spawn_async(name, lambda: lifecycle.run_async(prompt, run_turn=run_turn))

    @registry.register(name="list_teammates", description="列出当前仍在运行的队友。")
    async def list_teammates() -> str:
        return as_json(runtime.team.active_names)

    @registry.register(name="wait_teammate", description="等待指定队友完成并返回最终摘要。")
    async def wait_teammate(name: str) -> str:
        return await runtime.team.wait(name)

    @registry.register(name="shutdown_teammate", description="向队友发送优雅关闭请求。")
    async def shutdown_teammate(name: str) -> str:
        runtime.mailbox.send("lead", name, "请结束工作。", msg_type="shutdown_request")
        return f"已请求 {name} 关闭"
