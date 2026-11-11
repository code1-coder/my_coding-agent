"""网页入口与公共运行层的集成测试，包括 SSE、审批和会话清理。"""

import asyncio
import json
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.testclient import TestClient

import coding_agent.server.app as server
from coding_agent.config import Config
from coding_agent.memory import MemoryStore
from test_runtime import ScriptedProvider, text_response, tool_response


def decode_events(response):
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]


@pytest.fixture
def configured_server(tmp_path, monkeypatch):
    cfg = Config(workdir=str(tmp_path), team_idle_timeout=0.01)
    monkeypatch.setattr(server, "load_config", lambda: cfg)
    providers = []

    def factory(_):
        provider = ScriptedProvider(handler=lambda *args: text_response("网页回复"))
        providers.append(provider)
        return provider

    monkeypatch.setattr(server, "_get_provider", factory)
    return cfg, providers


def test_web_reuses_runtime_and_isolates_session_history(configured_server):
    cfg, providers = configured_server
    with TestClient(server.app) as client:
        for session_id, prompt in [("a", "第一轮"), ("a", "第二轮"), ("b", "独立对话")]:
            response = client.post("/chat", json={"prompt": prompt, "session_id": session_id})
            events = decode_events(response)
            assert events[0] == {"type": "text_delta", "text": "网页回复"}
            assert events[-1]["type"] == "done"
        assert len(providers) == 2
        assert len(server.sessions["a"]) == 4
        assert len(server.sessions["b"]) == 2
        assert server.runtimes["a"].usage.total_tokens > 0
    assert all(provider.closed for provider in providers)
    assert not server.runtimes


def test_web_delivers_result_after_real_file_write(configured_server, tmp_path, monkeypatch):
    provider = ScriptedProvider([
        tool_response("write_file", {"path": "result.txt", "content": "verified"}),
        text_response(""),
        text_response("已写入 result.txt，内容为 verified。"),
    ])
    monkeypatch.setattr(server, "_get_provider", lambda _: provider)
    with TestClient(server.app) as client:
        response = client.post("/chat", json={"prompt": "写文件并报告结果", "session_id": "delivery"})
        assert response.status_code == 200
        events = decode_events(response)
        visible = "".join(event.get("text", "") for event in events if event["type"] == "text_delta")
        assert visible == "已写入 result.txt，内容为 verified。"
        assert events[-1]["type"] == "done"
        assert events[-1]["text"] == visible
        assert sum(event["type"] == "tool_start" for event in events) == 1
        assert (tmp_path / "result.txt").read_text(encoding="utf-8") == "verified"
        assert server.sessions["delivery"][-1]["content"][0]["text"] == visible


def test_web_uses_skills_memory_context_hooks_and_task_board(configured_server, tmp_path, monkeypatch):
    cfg, providers = configured_server
    cfg.context_limit = 50
    cfg.mcp_servers = ["docs"]
    skill = tmp_path / "skills" / "review"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: review\ndescription: 审查\n---\n运行测试。", encoding="utf-8")
    memory = json.dumps([{"name": "assistant-name", "type": "feedback",
                          "description": "助手叫李华", "body": "助手的名字是李华。"}], ensure_ascii=False)
    provider = ScriptedProvider(
        [tool_response("create_task", {"subject": "网页创建任务"}), text_response("记住了")],
        maintenance=lambda prompt: "摘要" if "请摘要" in prompt else memory,
    )
    monkeypatch.setattr(server, "_get_provider", lambda _: provider)
    with TestClient(server.app) as client:
        response = client.post("/chat", json={"prompt": "记住你叫李华" + "。" * 200, "session_id": "web-test"})
        events = decode_events(response)
        assert any(e["type"] == "tool_start" for e in events)
        assert any(e["type"] == "tool_done" for e in events)
        assert any(e["type"] == "memory_saved" for e in events)
        assert "李华" in (tmp_path / ".memory" / "MEMORY.md").read_text(encoding="utf-8")
        assert client.get("/tasks").json()["tasks"][0]["subject"] == "网页创建任务"
        assert list((tmp_path / ".transcripts").glob("*.jsonl"))
        agent_call = next(c for c in provider.calls if c["tools"] is not None)
        assert "review" in agent_call["system"]
        assert any(t["name"] == "mcp__docs__search" for t in agent_call["tools"])


def test_web_reports_model_errors_and_cleans_up(configured_server, monkeypatch):
    provider = ScriptedProvider([RuntimeError("模拟模型请求失败")])
    monkeypatch.setattr(server, "_get_provider", lambda _: provider)
    with TestClient(server.app) as client:
        response = client.post("/chat", json={"prompt": "你好"})
        assert decode_events(response)[-1] == {"type": "error", "message": "模拟模型请求失败"}
    assert provider.closed


def test_health_empty_prompt_and_websocket_need_no_model(configured_server):
    _, providers = configured_server
    with TestClient(server.app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert "chat-area" in client.get("/").text
        assert decode_events(client.post("/chat", json={"prompt": " "}))[0]["type"] == "error"
        with client.websocket_connect("/ws?session_id=ping-test") as ws:
            ws.send_text("ping")
            assert ws.receive_text() == "pong"
    assert not providers


def test_dashboard_snapshot_uses_memory_and_session_team_without_opening_model(configured_server):
    cfg, providers = configured_server
    cfg.api_key = "dashboard-test-key-must-not-be-serialized"
    MemoryStore(workdir=cfg.workdir).write_file("中文偏好", "user", "使用中文", "注释与回复使用中文。")
    with TestClient(server.app) as client:
        response = client.get("/workspace?session_id=unused")
        snapshot = response.json()
        assert snapshot["project"]
        assert snapshot["memories"][0]["body"] == "注释与回复使用中文。"
        assert snapshot["team"] == [{"name": "lead", "role": "主智能体", "state": "idle"}]
        assert snapshot["usage"] == {"input_tokens": 0, "output_tokens": 0}
        assert cfg.api_key not in response.text
        assert not providers
        client.post("/chat", json={"prompt": "你好", "session_id": "active"})
        assert client.get("/workspace?session_id=active").json()["usage"]["input_tokens"] > 0
        assert client.get("/workspace?session_id=unused").json()["usage"]["input_tokens"] == 0
        assert len(providers) == 1


def test_dashboard_task_lifecycle_preserves_dependencies_and_shared_storage(configured_server):
    cfg, providers = configured_server
    from coding_agent.agents.tasks import TaskBoard

    with TestClient(server.app) as client:
        first = client.post("/tasks", json={"subject": " 梳理结构 ", "description": "中文描述"})
        assert first.status_code == 201
        first = first.json()["task"]
        assert TaskBoard(workdir=cfg.workdir).get(first["id"]).subject == "梳理结构"
        # 依赖真实的 TaskBoard 校验，前端不能越过依赖直接开始或完成任务。
        second = client.post("/tasks", json={"subject": "补充测试", "blockedBy": [first["id"]]}).json()["task"]
        second_url = "/tasks/" + second["id"]
        assert client.patch(second_url, json={"action": "claim"}).status_code == 409
        assert client.patch(second_url, json={"action": "complete"}).status_code == 409
        first_url = "/tasks/" + first["id"]
        claimed = client.patch(first_url, json={"action": "claim"}).json()["task"]
        assert claimed["owner"] == "manual" and claimed["status"] == "in_progress"
        assert client.patch(first_url, json={"action": "claim"}).status_code == 409
        assert client.patch(first_url, json={"action": "complete"}).json()["task"]["status"] == "completed"
        assert client.patch(second_url, json={"action": "claim"}).status_code == 200
        assert len(client.get("/tasks").json()["tasks"]) == 2
    assert not providers


@pytest.mark.parametrize("body", [
    {"subject": " "}, {"subject": "a" * 201}, {"subject": 123},
    {"subject": "测试", "status": "completed"},
    {"subject": "测试", "blockedBy": ["../../outside"]},
    {"subject": "测试", "blockedBy": ["task_missing"]},
])
def test_dashboard_rejects_invalid_task_input_without_writing(configured_server, body):
    with TestClient(server.app) as client:
        assert client.post("/tasks", json=body).status_code == 422
        assert client.get("/tasks").json()["tasks"] == []


def test_dashboard_missing_task_and_unknown_transition(configured_server):
    with TestClient(server.app) as client:
        assert client.patch("/tasks/task_missing", json={"action": "claim"}).status_code == 404
        assert client.patch("/tasks/task_missing", json={"action": "reopen"}).status_code == 422


@pytest.mark.asyncio
@pytest.mark.parametrize("approved", [True, False])
async def test_web_approval_round_trip_without_executing_shell(configured_server, monkeypatch, approved):
    provider = ScriptedProvider([
        tool_response("bash", {"command": "rm example.txt"}), text_response(),
    ])
    monkeypatch.setattr(server, "_get_provider", lambda _: provider)
    notifications = asyncio.Queue()
    original_publish = server._publish

    async def publish(session, payload):
        if payload["type"] == "approval_required":
            notifications.put_nowait(payload)
        await original_publish(session, payload)

    monkeypatch.setattr(server, "_publish", publish)
    async with server.lifespan(server.app):
        runtime = await server._get_runtime("owner")
        execute = AsyncMock(return_value="模拟命令已执行")
        # 只替换实际执行器，保留公共权限管线与审批流程，测试中不运行 shell。
        runtime.registry.register_handler({"name": "bash"}, execute)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://test") as client:
            request = asyncio.create_task(client.post("/chat", json={"prompt": "测试审批", "session_id": "owner"}))
            notification = await asyncio.wait_for(notifications.get(), 2)
            url = "/approvals/" + notification["approval_id"]
            assert (await client.post(url, json={"session_id": "other", "approved": True})).status_code == 403
            assert (await client.post(url, json={"session_id": "owner", "approved": "false"})).status_code == 422
            assert (await client.post(url, json={"session_id": "owner", "approved": approved})).status_code == 200
            response = await asyncio.wait_for(request, 2)
            assert decode_events(response)[-1]["type"] == "done"
            assert execute.await_count == int(approved)
            assert (await client.post(url, json={"session_id": "owner", "approved": True})).status_code == 404
        assert not server.pending_approvals


@pytest.mark.asyncio
async def test_websocket_tool_events_only_reach_their_session(configured_server):
    class Socket:
        def __init__(self, session_id):
            self.query_params = {"session_id": session_id}
            self.send_json = AsyncMock()

    first, other = Socket("one"), Socket("two")
    async with server.lifespan(server.app):
        server.ws_connections.update({first, other})
        await server._broadcast_ws({"session_id": "one", "type": "tool_use"})
        first.send_json.assert_awaited_once()
        other.send_json.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancelled_approval_cleans_up_and_next_turn_can_continue(configured_server, monkeypatch):
    provider = ScriptedProvider([
        tool_response("bash", {"command": "rm example.txt"}), text_response("恢复后完成"),
    ])
    monkeypatch.setattr(server, "_get_provider", lambda _: provider)
    waiting = asyncio.Event()
    original_publish = server._publish

    async def publish(session, payload):
        if payload["type"] == "approval_required":
            waiting.set()
        await original_publish(session, payload)

    monkeypatch.setattr(server, "_publish", publish)
    async with server.lifespan(server.app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://test") as client:
            request = asyncio.create_task(client.post("/chat", json={"prompt": "取消审批", "session_id": "cancel-test"}))
            await asyncio.wait_for(waiting.wait(), 2)
            request.cancel()
            await asyncio.gather(request, return_exceptions=True)
            assert not server.pending_approvals
            assert not server._subscribers["cancel-test"]
            assert server.sessions["cancel-test"][-1]["content"][0]["is_error"]
            response = await client.post("/chat", json={"prompt": "继续", "session_id": "cancel-test"})
            assert decode_events(response)[-1]["type"] == "done"
