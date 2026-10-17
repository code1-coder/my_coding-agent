"""智能体基础设施 —— 子智能体池、任务、团队、生命周期。"""

from coding_agent.agents.subagent import AgentPool, SubagentSpec
from coding_agent.agents.tasks import Task, TaskBoard
from coding_agent.agents.team import FileMailbox, MessageBus, TeamRegistry
from coding_agent.agents.lifecycle import (
    AgentLifecycle,
    LifecycleState,
    POLL_INTERVAL,
    IDLE_TIMEOUT,
    MAX_WORK_TURNS,
)

__all__ = [
    "AgentPool",
    "SubagentSpec",
    "Task",
    "TaskBoard",
    "MessageBus",
    "FileMailbox",
    "TeamRegistry",
    "AgentLifecycle",
    "LifecycleState",
    "POLL_INTERVAL",
    "IDLE_TIMEOUT",
    "MAX_WORK_TURNS",
]
