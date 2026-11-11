"""
任务系统 —— 文件持久化的任务图，支持 blockedBy 依赖。

存储布局::

    .tasks/
      task_1734123456_0042.json
      task_1734123500_0099.json
      ...

每个文件是 ``Task`` 的 JSON 表示::

    {
      "id": "task_1734123456_0042",
      "subject": "添加登录接口",
      "description": "POST /auth/login，使用 JWT",
      "status": "in_progress",
      "owner": "worker-01",
      "blockedBy": ["task_1734123000_0010"]
    }

状态流转: ``pending → in_progress → completed``（或 ``deleted``）。

用法::

    board = TaskBoard(workdir=Path.cwd())
    task = board.create("添加测试", "为 auth 模块编写 pytest 测试")
    board.claim(task.id, owner="agent")
    board.complete(task.id)  # 宣告下游被解锁的任务
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  常量
# ---------------------------------------------------------------------------

TASKS_DIR_NAME = ".tasks"

VALID_STATUSES = ("pending", "in_progress", "completed", "deleted")


# ---------------------------------------------------------------------------
#  Task 数据类
# ---------------------------------------------------------------------------


@dataclass
class Task:
    """一个带依赖追踪的工作单元。"""

    id: str
    subject: str
    description: str = ""
    status: str = "pending"          # 任务状态：待处理、进行中、已完成、已删除
    owner: str | None = None         # 智能体名称
    blockedBy: list[str] = field(default_factory=list)  # 依赖的任务 ID 列表
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None

    def to_dict(self) -> dict:
        """将任务字段转换为可写入 JSON 文件的字典。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Task:
        """从持久化字典恢复任务，忽略当前数据类未定义的字段。"""
        # 过滤未知键
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
#  任务看板（TaskBoard）
# ---------------------------------------------------------------------------


class TaskBoard:
    """基于文件系统的任务看板，支持依赖解析。

    读操作线程安全（每次操作读写独立文件）。
    对同一文件的并发写操作不安全（生产环境应使用文件锁）。
    """

    def __init__(self, workdir: Path | str | None = None) -> None:
        wd = Path(workdir) if workdir else Path.cwd()
        self._tasks_dir = wd / TASKS_DIR_NAME
        self._tasks_dir.mkdir(parents=True, exist_ok=True)

    # ── 文件 I/O ───────────────────────────────────────────

    def _task_path(self, task_id: str) -> Path:
        if not re.fullmatch(r"[\w-]+", task_id):
            raise ValueError("无效的任务 ID")
        return self._tasks_dir / f"{task_id}.json"

    def _generate_id(self) -> str:
        return f"task_{int(time.time())}_{uuid.uuid4().hex}"

    # ── 创建、读取、更新和删除 ───────────────────────────────────────────────

    def create(
        self,
        subject: str,
        description: str = "",
        blockedBy: list[str] | None = None,
    ) -> Task:
        """创建并持久化一个新的 ``pending`` 任务。"""
        task = Task(
            id=self._generate_id(),
            subject=subject,
            description=description,
            status="pending",
            blockedBy=blockedBy or [],
        )
        # Creation must never overwrite an existing task (including a completed
        # dependency). Exclusive creation also protects concurrent processes.
        while True:
            try:
                with self._task_path(task.id).open("x", encoding="utf-8") as stream:
                    json.dump(task.to_dict(), stream, indent=2)
                break
            except FileExistsError:
                task.id = self._generate_id()
        logger.debug("Task created: %s — %s", task.id, task.subject)
        return task

    def get(self, task_id: str) -> Task:
        """按 ID 加载单个任务。缺失时抛出 ``FileNotFoundError``。"""
        return Task.from_dict(json.loads(self._task_path(task_id).read_text()))

    def list_all(self) -> list[Task]:
        """返回所有任务，按创建时间排序。"""
        tasks: list[Task] = []
        for p in sorted(self._tasks_dir.glob("task_*.json")):
            try:
                tasks.append(Task.from_dict(json.loads(p.read_text())))
            except Exception:
                logger.warning("Skipping malformed task file: %s", p)
        return tasks

    def update(self, task: Task) -> None:
        """持久化 *task* 的变更。"""
        self._save(task)

    def delete(self, task_id: str) -> bool:
        """删除任务文件。存在并删除成功返回 True。"""
        path = self._task_path(task_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def _save(self, task: Task) -> None:
        """将任务完整写入对应的 UTF-8 JSON 文件，覆盖已有内容。"""
        self._task_path(task.id).write_text(
            json.dumps(task.to_dict(), indent=2), encoding="utf-8",
        )

    # ── 依赖解析 ──────────────────────────────────────────

    def can_start(self, task_id: str) -> bool:
        """所有 ``blockedBy`` 依赖均为 ``completed`` 时返回 True。

        缺失的依赖 → 视为阻塞（认为尚未创建）。
        """
        task = self.get(task_id)
        for dep_id in task.blockedBy:
            path = self._task_path(dep_id)
            if not path.exists():
                return False
            dep_task = Task.from_dict(json.loads(path.read_text()))
            if dep_task.status != "completed":
                return False
        return True

    def find_unblocked(self) -> list[str]:
        """返回依赖全部完成的 ``pending`` 任务 ID。"""
        return [
            t.id for t in self.list_all()
            if t.status == "pending" and self.can_start(t.id)
        ]

    def find_unclaimed(self) -> list[Task]:
        """返回可开始的、未认领的 ``pending`` 任务。"""
        return [
            t for t in self.list_all()
            if t.status == "pending" and t.owner is None and self.can_start(t.id)
        ]

    # ── 状态转换 ──────────────────────────────────────────

    def claim(self, task_id: str, owner: str = "agent") -> str:
        """认领一个 ``pending`` 任务 → ``in_progress``。

        返回人类可读的结果字符串。

        仅允许认领待处理、无人持有且依赖已完成的任务；成功时更新
        owner 和 status 并写回磁盘，失败时保留原状态并返回原因。
        """
        task = self.get(task_id)

        if task.status != "pending":
            return f"Task {task_id} is {task.status}, cannot claim"

        if task.owner:
            return f"Task {task_id} already owned by {task.owner}"

        if not self.can_start(task_id):
            deps = [
                d for d in task.blockedBy
                if not self._task_path(d).exists()
                or Task.from_dict(json.loads(self._task_path(d).read_text())).status != "completed"
            ]
            missing = [d for d in task.blockedBy if not self._task_path(d).exists()]
            parts: list[str] = []
            if deps:
                parts.append(f"blocked by: {deps}")
            if missing:
                parts.append(f"missing deps: {missing}")
            return "Cannot start — " + ", ".join(parts)

        task.owner = owner
        task.status = "in_progress"
        self._save(task)
        logger.info("Task claimed: %s → %s (owner: %s)", task_id, task.subject, owner)
        return f"Claimed {task.id} ({task.subject})"

    def complete(self, task_id: str) -> str:
        """将 ``in_progress`` 任务标记为 → ``completed``。

        报告刚刚被解锁的下游任务。
        """
        task = self.get(task_id)

        if task.status != "in_progress":
            return f"Task {task_id} is {task.status}, cannot complete"

        task.status = "completed"
        task.completed_at = time.time()
        self._save(task)

        # 找到新解锁的任务
        unblocked = [
            t for t in self.list_all()
            if t.status == "pending" and task_id in t.blockedBy and self.can_start(t.id)
        ]

        logger.info("Task completed: %s — %s", task_id, task.subject)
        msg = f"Completed {task.id} ({task.subject})"
        if unblocked:
            names = ", ".join(t.subject for t in unblocked)
            msg += f"\nUnblocked: {names}"
            logger.info("Unblocked tasks: %s", names)

        return msg

    @property
    def count(self) -> int:
        return len(list(self._tasks_dir.glob("task_*.json")))

    def __len__(self) -> int:
        return self.count
