"""网页控制台：HTTP、SSE 和 WebSocket 适配器，执行逻辑统一交给 AgentRuntime。"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from contextlib import aclosing, asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Literal

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, StrictStr

from coding_agent.agents.tasks import TaskBoard
from coding_agent.config import load_config
from coding_agent.hooks.bus import EventType
from coding_agent.memory import MemoryStore
from coding_agent.runtime import AgentRuntime, build_provider

logger = logging.getLogger(__name__)

sessions: dict[str, list[dict[str, Any]]] = {}
runtimes: dict[str, AgentRuntime] = {}
ws_connections: set[WebSocket] = set()
_session_locks: dict[str, asyncio.Lock] = {}
_workspace_locks: dict[str, asyncio.Lock] = {}
_subscribers: dict[str, set[asyncio.Queue]] = {}


@dataclass
class PendingApproval:
    session_id: str
    future: asyncio.Future


pending_approvals: dict[str, PendingApproval] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """服务关闭时停止队友、取消审批等待并关闭所有会话客户端。"""
    try:
        yield
    finally:
        for pending in list(pending_approvals.values()):
            pending.future.cancel()
        await asyncio.gather(*(runtime.aclose() for runtime in runtimes.values()),
                             return_exceptions=True)
        runtimes.clear()
        sessions.clear()
        pending_approvals.clear()
        _subscribers.clear()
        _session_locks.clear()
        _workspace_locks.clear()
        ws_connections.clear()


app = FastAPI(title="coding-agent", version="0.1.0", lifespan=lifespan)
_WEB_DIR = Path(__file__).resolve().parent.parent.parent.parent / "web"
if _WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_WEB_DIR)), name="static")


def _get_provider(cfg):
    """保留可替换的入口，实际模型选择使用公共工厂。"""
    return build_provider(cfg)


async def _broadcast_ws(payload: dict[str, Any]) -> None:
    """仅向同一会话广播，避免不同页面收到其他会话的工具或审批信息。"""
    dead = set()
    for ws in list(ws_connections):
        if ws.query_params.get("session_id", "default") != payload.get("session_id", "default"):
            continue
        try:
            await ws.send_json(payload)
        except Exception:
            dead.add(ws)
    ws_connections.difference_update(dead)


async def _publish(session_id: str, payload: dict[str, Any]) -> None:
    payload = {"session_id": session_id, **payload}
    for queue in tuple(_subscribers.get(session_id, ())):
        queue.put_nowait(payload)
    await _broadcast_ws(payload)


async def _get_runtime(session_id: str) -> AgentRuntime:
    """按会话复用运行层，使背景任务、Hook 状态和历史在请求间保持连续。"""
    lock = _session_locks.setdefault(session_id, asyncio.Lock())
    async with lock:
        cfg = load_config()
        current = runtimes.get(session_id)
        if current is not None:
            return current
        workspace = str(Path(cfg.workdir).resolve())
        workspace_lock = _workspace_locks.setdefault(workspace, asyncio.Lock())

        async def approve(tool_name, params, reason):
            approval_id = uuid.uuid4().hex
            future = asyncio.get_running_loop().create_future()
            pending_approvals[approval_id] = PendingApproval(session_id, future)
            try:
                await _publish(session_id, {
                    "type": "approval_required", "approval_id": approval_id,
                    "tool_name": tool_name, "input": params, "reason": reason,
                })
                return await asyncio.wait_for(future, timeout=120)
            except asyncio.TimeoutError:
                return False
            finally:
                pending_approvals.pop(approval_id, None)
                await _publish(session_id, {"type": "approval_finished", "approval_id": approval_id})

        runtime = AgentRuntime(cfg, provider_factory=_get_provider, session_id=session_id,
                               approval_handler=approve, workspace_lock=workspace_lock)

        async def notify_tool(event):
            await _broadcast_ws({
                "session_id": session_id, "type": "tool_use",
                "tool_id": event.data.get("tool_id", ""),
                "execution_id": event.data.get("execution_id", ""),
                "agent": event.data.get("agent", "lead"),
                "tool_name": event.data["tool_name"], "input": event.data.get("params", {}),
            })

        runtime.bus.subscribe(EventType.PRE_TOOL_USE, notify_tool, priority=90)

        async def notify_tool_done(event):
            await _broadcast_ws({
                "session_id": session_id, "type": "tool_done",
                "tool_id": event.data.get("tool_id", ""),
                "execution_id": event.data.get("execution_id", ""),
                "agent": event.data.get("agent", "lead"),
                "tool_name": event.data["tool_name"],
                "output": str(event.data.get("output", ""))[:4000],
                "is_error": bool(event.data.get("error")),
                "execution": event.data.get("execution", {}),
            })

        runtime.bus.subscribe(EventType.POST_TOOL_USE, notify_tool_done, priority=90)
        try:
            await runtime.start()
        except BaseException:
            await runtime.aclose()
            raise
        runtimes[session_id] = runtime
        sessions[session_id] = runtime.messages
        return runtime


@app.get("/health")
async def health():
    """检查服务是否存活，不调用模型。"""
    return {"status": "ok"}


@app.get("/tasks")
async def list_tasks():
    """复用 TaskBoard，并与工具执行使用相同的配置工作目录。"""
    cfg = load_config()
    return {"tasks": [task.to_dict() for task in TaskBoard(workdir=cfg.workdir).list_all()]}


class CreateTaskRequest(BaseModel):
    """网页手动创建任务的输入，限制长度并拒绝未知字段。"""

    model_config = ConfigDict(extra="forbid")
    subject: StrictStr = Field(min_length=1, max_length=200)
    description: StrictStr = Field(default="", max_length=10000)
    blockedBy: list[StrictStr] = Field(default_factory=list, max_length=100)


class TaskActionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["claim", "complete"]


def _task_board_and_lock() -> tuple[TaskBoard, asyncio.Lock]:
    """手动任务操作与智能体复用任务模块及工作区锁。"""
    workspace = str(Path(load_config().workdir).resolve())
    return TaskBoard(workdir=workspace), _workspace_locks.setdefault(workspace, asyncio.Lock())


@app.post("/tasks", status_code=201)
async def create_task(body: CreateTaskRequest):
    board, lock = _task_board_and_lock()
    subject = body.subject.strip()
    if not subject:
        raise HTTPException(422, "任务名称不能为空")
    async with lock:
        for dependency in body.blockedBy:
            try:
                task = board.get(dependency)
                if task.status == "deleted":
                    raise ValueError("依赖任务已删除")
            except (ValueError, FileNotFoundError):
                raise HTTPException(422, "依赖任务不存在或无效")
        task = board.create(subject, body.description.strip(), list(dict.fromkeys(body.blockedBy)))
    return {"task": task.to_dict()}


@app.patch("/tasks/{task_id}")
async def transition_task(task_id: str, body: TaskActionRequest):
    """通过 TaskBoard 的状态转换保留依赖检查，避免前端任意改写状态。"""
    board, lock = _task_board_and_lock()
    async with lock:
        try:
            task = board.get(task_id)
        except (ValueError, FileNotFoundError):
            raise HTTPException(404, "任务不存在")
        if body.action == "claim":
            if task.status != "pending" or task.owner or not board.can_start(task_id):
                raise HTTPException(409, "任务已被认领、状态已变化或仍有未完成的依赖")
            board.claim(task_id, owner="manual")
        else:
            if task.status != "in_progress":
                raise HTTPException(409, "只能完成正在处理的任务")
            board.complete(task_id)
        return {"task": board.get(task_id).to_dict()}


@app.get("/workspace")
async def workspace_snapshot(session_id: str = "default"):
    """读取看板需要的安全配置和模块快照；不创建模型客户端、不消费邮箱。"""
    cfg = load_config()
    workspace = Path(cfg.workdir).resolve()
    runtime = runtimes.get(session_id)
    running = bool(runtime and runtime.turn_lock.locked())
    team = [{"name": "lead", "role": "主智能体", "state": "working" if running else "idle"}]
    if runtime:
        team.extend({"name": name, "role": "后台队友", "state": "active"}
                    for name in runtime.team.active_names)
    # 仅显式列出界面字段，不能将含有 API 密钥的 Config 整体序列化。
    return {
        "project": workspace.name, "workdir": str(workspace), "model": cfg.model_id,
        "memories": MemoryStore(workdir=workspace).list_files(), "team": team,
        "running": running,
        "usage": {"input_tokens": runtime.usage.input_tokens if runtime else 0,
                  "output_tokens": runtime.usage.output_tokens if runtime else 0},
    }


@app.get("/")
async def index():
    index_path = _WEB_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(index_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>coding-agent</h1><p>找不到 web/index.html</p>")


@app.post("/approvals/{approval_id}")
async def resolve_approval(approval_id: str, request: Request):
    """只接受对应会话提交的布尔审批结果；过期或重复审批不会重新执行工具。"""
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(422, "请求体必须是 JSON 对象")
    pending = pending_approvals.get(approval_id)
    if pending is None or pending.future.done():
        raise HTTPException(404, "审批已结束或不存在")
    if body.get("session_id") != pending.session_id:
        raise HTTPException(403, "审批不属于当前会话")
    if type(body.get("approved")) is not bool:
        raise HTTPException(422, "approved 必须是布尔值")
    pending.future.set_result(body["approved"])
    return {"status": "ok"}


@app.post("/chat")
async def chat(request: Request):
    """将公共运行事件实时编码为 SSE，连接中断时取消该轮请求及相关等待。"""
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(422, "请求体必须是 JSON 对象")
    prompt = body.get("prompt", "")
    session_id = body.get("session_id", "default")
    if not isinstance(prompt, str) or not prompt.strip():
        return StreamingResponse(_single_event({"type": "error", "message": "Empty prompt"}),
                                 media_type="text/event-stream")
    if not isinstance(session_id, str) or not session_id or len(session_id) > 200:
        raise HTTPException(422, "session_id 必须是长度不超过 200 的非空字符串")

    async def event_stream() -> AsyncIterator[str]:
        queue: asyncio.Queue = asyncio.Queue()
        sentinel = object()
        subscribers = _subscribers.setdefault(session_id, set())
        subscribers.add(queue)

        async def produce():
            try:
                runtime = await _get_runtime(session_id)
                async with aclosing(runtime.run_turn(prompt)) as events:
                    async for event in events:
                        payload = {"type": event.type, **event.data}
                        if event.text or event.type == "text_reset":
                            payload["text"] = event.text
                        if event.tool_name:
                            payload.update(tool_name=event.tool_name, tool_id=event.tool_id,
                                           output=event.tool_output[:4000])
                        queue.put_nowait(payload)
            except Exception as exc:
                logger.exception("网页对话执行失败")
                queue.put_nowait({"type": "error", "message": str(exc)})
            finally:
                queue.put_nowait(sentinel)

        task = asyncio.create_task(produce())
        try:
            while True:
                payload = await queue.get()
                if payload is sentinel:
                    break
                yield _sse_event(payload)
        finally:
            subscribers.discard(queue)
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """为当前页面会话注册通知通道，支持 ping/pong 心跳。"""
    await ws.accept()
    ws_connections.add(ws)
    try:
        while True:
            if await ws.receive_text() == "ping":
                await ws.send_text("pong")
    except WebSocketDisconnect:
        pass
    finally:
        ws_connections.discard(ws)


def _sse_event(data: dict[str, Any]) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _single_event(data: dict[str, Any]) -> AsyncIterator[str]:
    yield _sse_event(data)
