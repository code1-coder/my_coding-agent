"""网页入口与公共运行层的集成测试，包括 SSE、审批和会话清理。"""

import asyncio
import json
import sys
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.testclient import TestClient

import coding_agent.server.app as server
from coding_agent.config import Config
from coding_agent.memory import MemoryStore
from coding_agent.providers.base import ModelRequestError, ProviderResponse
from test_code_tools import _process_running
from test_runtime import ScriptedProvider, single_role_analysis_provider, text_response, tool_response


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


@pytest.mark.skipif(sys.platform != "win32", reason="Windows Selector 事件循环不支持子进程")
def test_web_reports_shell_startup_failure_with_windows_selector_loop(configured_server, monkeypatch):
    provider = ScriptedProvider([
        tool_response("bash", {"command": "echo bash-ok"}), text_response("完成"),
    ])
    monkeypatch.setattr(server, "_get_provider", lambda _: provider)

    async def request_chat():
        async with server.lifespan(server.app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server.app), base_url="http://test",
            ) as client:
                return await client.post("/chat", json={"prompt": "检查 Shell", "session_id": "selector"})

    with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop) as runner:
        response = runner.run(request_chat())

    assert response.status_code == 200
    result = next(event for event in decode_events(response) if event["type"] == "tool_done")
    assert result["is_error"] is True
    assert result["execution"]["failure_stage"] == "startup"
    assert result["execution"]["termination_confirmed"] is None
    assert result["execution"]["exit_code"] is None
    assert result["execution"]["stdout"] == ""
    assert "SelectorEventLoop" in result["output"]
    assert "--reload" in result["output"]


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


def test_web_reports_model_errors_and_cleans_up(configured_server, monkeypatch, caplog):
    provider = ScriptedProvider([RuntimeError("模拟模型请求失败")])
    monkeypatch.setattr(server, "_get_provider", lambda _: provider)
    with TestClient(server.app) as client:
        response = client.post("/chat", json={"prompt": "你好"})
        assert decode_events(response)[-1] == {"type": "error", "message": "会话执行异常，请查看服务端日志。"}
        assert "RuntimeError: 模拟模型请求失败" in caplog.text
    assert provider.closed


def test_web_analysis_emits_checkpoints_and_partial_terminal(configured_server, monkeypatch):
    provider = ScriptedProvider([
        tool_response("analysis_checkpoint", {"action": "record", "dimension": "usage", "blocked": True,
                                             "finding": "没有读到入口文件。"}, "blocked"),
        tool_response("analysis_checkpoint", {"action": "finish"}, "finish"),
        text_response(),
    ])
    monkeypatch.setattr(server, "_get_provider", lambda _: provider)
    with TestClient(server.app) as client:
        assert client.post("/chat", json={"prompt": "检查", "mode": "invalid"}).status_code == 422
        events = decode_events(client.post("/chat", json={"prompt": "只解释入口", "mode": "analysis"}))
    assert any(event["type"] == "analysis_progress" and "没有读到入口文件" in event["text"] for event in events)
    assert events[-1]["type"] == "done"
    assert events[-1]["analysis_status"] == "partial"


@pytest.mark.parametrize("scope, status", [({}, "partial"), ({"analysis_dimensions": ["role"]}, "complete")])
def test_web_analysis_scope_is_set_by_request(configured_server, tmp_path, monkeypatch, scope, status):
    (tmp_path / "entry.py").write_text("def start(): pass\n", encoding="utf-8")
    monkeypatch.setattr(server, "_get_provider", lambda _: single_role_analysis_provider())
    prompt = "检查职责" if scope else "检查职责、调用关系、入口用法和生命周期"
    with TestClient(server.app) as client:
        response = client.post("/chat", json={"prompt": prompt, "mode": "analysis", **scope})
        assert response.status_code == 200
        events = decode_events(response)
    assert any(event["type"] == "tool_done" and event["tool_id"] == "plan" and event["is_error"] for event in events)
    assert events[-1]["type"] == "done" and events[-1]["analysis_status"] == status
    assert "提供入口函数" in events[-1]["text"]
    assert ("调用关系：未确认" in events[-1]["text"]) == (status == "partial")


def test_web_rejects_invalid_analysis_scope_before_creating_runtime(configured_server):
    _, providers = configured_server
    with TestClient(server.app) as client:
        for invalid in ([], ["role", "role"], ["unknown"], [True], "role", None):
            assert client.post("/chat", json={"prompt": "检查", "mode": "analysis",
                                             "analysis_dimensions": invalid}).status_code == 422
        assert client.post("/chat", json={"prompt": "检查", "analysis_dimensions": ["role"]}).status_code == 422
    assert not providers


def test_web_resume_rejects_other_sessions_and_consumed_ids(configured_server, monkeypatch):
    cfg, _ = configured_server
    cfg.model_max_attempts = 1
    provider = ScriptedProvider([ConnectionError("断开"), text_response("续接完成")])
    monkeypatch.setattr(server, "_get_provider", lambda _: provider)
    with TestClient(server.app) as client:
        failed = decode_events(client.post("/chat", json={"prompt": "继续检查", "session_id": "owner"}))[-1]
        recovery_id = failed["recovery_id"]
        assert client.post("/chat/retry", json={"session_id": "other", "recovery_id": recovery_id}).status_code == 404
        body = {"session_id": "owner", "recovery_id": recovery_id}
        resumed = decode_events(client.post("/chat/retry", json=body))
        assert resumed[-1]["type"] == "done" and resumed[-1]["text"] == "续接完成"
        assert client.post("/chat/retry", json=body).status_code == 409


def test_web_delivers_model_request_failure_without_success_done(configured_server, monkeypatch):
    provider = ScriptedProvider([ModelRequestError("模型服务暂不可用")])
    monkeypatch.setattr(server, "_get_provider", lambda _: provider)
    with TestClient(server.app) as client:
        events = decode_events(client.post("/chat", json={"prompt": "检查任务"}))

    assert events[-1]["type"] == "error"
    assert events[-1]["failed"] is True
    assert events[-1]["message"] == events[-1]["text"]
    assert "模型服务暂不可用" in events[-1]["message"]
    assert not any(event["type"] == "done" for event in events)
    assert provider.closed


def test_web_exposes_waiting_progress_and_model_timeout(configured_server, monkeypatch):
    cfg, _ = configured_server
    cfg.model_timeout = 0.15
    cfg.progress_interval = 0.02

    class StalledProvider(ScriptedProvider):
        async def create_message(self, *arguments, **options):
            await asyncio.Event().wait()

    provider = StalledProvider()
    monkeypatch.setattr(server, "_get_provider", lambda _: provider)
    with TestClient(server.app) as client:
        events = decode_events(client.post("/chat", json={"prompt": "检查等待", "session_id": "stall"}))
        assert any(event["type"] == "warning" and event["phase"] == "model" for event in events)
        assert events[-1]["type"] == "error"
        assert "已停止等待" in events[-1]["message"]
        assert not server.runtimes["stall"].turn_lock.locked()
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
@pytest.mark.parametrize("stop_scope", ["disconnect", "chat", "shell"])
async def test_cancelled_approval_cleans_up_and_next_turn_can_continue(configured_server, tmp_path, monkeypatch, stop_scope):
    script = tmp_path / "approval.py"
    script.write_text("from pathlib import Path\nPath('launched').touch()\n", encoding="utf-8")
    command = f'"{sys.executable}" "{script}" "rm example.txt"'
    provider = ScriptedProvider([
        tool_response("bash", {"command": command}), text_response("恢复后完成"), text_response("下一轮完成"),
    ])
    monkeypatch.setattr(server, "_get_provider", lambda _: provider)
    waiting = asyncio.Event()
    started = asyncio.Future()
    tool_started = asyncio.Future()
    approval_ids = []
    original_publish = server._publish

    async def publish(session, payload):
        if payload["type"] == "approval_required":
            approval_ids.append(payload["approval_id"])
            waiting.set()
        await original_publish(session, payload)

    monkeypatch.setattr(server, "_publish", publish)

    async def app_with_event_observer(scope, receive, send):
        async def observe(message):
            body = message.get("body", b"")
            if body.startswith(b"data: "):
                event = json.loads(body[6:])
                if event["type"] == "chat_started":
                    started.set_result(event["stop_id"])
                if event["type"] == "tool_start":
                    tool_started.set_result(event["execution_id"])
            await send(message)

        await server.app(scope, receive, observe)

    async with server.lifespan(server.app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app_with_event_observer), base_url="http://test",
        ) as client:
            request = asyncio.create_task(client.post("/chat", json={
                "prompt": "取消审批", "session_id": "cancel-test", "stop_confirmation": stop_scope != "disconnect",
            }))
            try:
                await asyncio.wait_for(waiting.wait(), 5)
                if stop_scope == "disconnect":
                    request.cancel()
                    await asyncio.gather(request, return_exceptions=True)
                else:
                    stop_id = await asyncio.wait_for(started, 5)
                    execution_id = await asyncio.wait_for(tool_started, 5)
                    url = f"/chat/{stop_id}"
                    if stop_scope == "shell":
                        url += f"/tools/{execution_id}"
                    response = await client.post(url + "/stop", json={"session_id": "cancel-test"})
                    assert response.status_code == 202
                    events = decode_events(await asyncio.wait_for(request, 5))
                    assert events[-1]["type"] == ("done" if stop_scope == "shell" else "chat_stopped")
                    assert any(event["type"] == "approval_finished" for event in events)
                    if stop_scope == "shell":
                        result = next(event for event in events if event["type"] == "tool_done")
                        assert result["execution_id"] == execution_id
                        assert result["execution"]["failure_stage"] == "cancelled_before_start"
                        assert result["execution"]["termination_confirmed"] is None
                        assert result["is_error"] is True
                assert not (tmp_path / "launched").exists()
                assert not server.pending_approvals
                assert not server._subscribers["cancel-test"]
                assert any(block.get("is_error") for message in server.sessions["cancel-test"]
                           if isinstance(message["content"], list) for block in message["content"])
                response = await client.post("/approvals/" + approval_ids[0], json={
                    "session_id": "cancel-test", "approved": True,
                })
                assert response.status_code == 404
                response = await client.post("/chat", json={"prompt": "继续", "session_id": "cancel-test"})
                assert decode_events(response)[-1]["type"] == "done"
            finally:
                if not request.done():
                    request.cancel()
                await asyncio.gather(request, return_exceptions=True)


@pytest.mark.skipif(sys.platform not in ("win32", "linux"), reason="需要受支持的 Shell 进程树管理")
async def test_web_streams_bounded_shell_output_before_completion(configured_server, tmp_path, monkeypatch):
    script = tmp_path / "preview.py"
    script.write_text(
        "import pathlib, sys, time\n"
        "sys.stdout.buffer.write(b'<preview>\\xe4'); sys.stdout.flush()\n"
        "print('diagnostic', file=sys.stderr, flush=True)\n"
        "while not pathlib.Path('release').exists(): time.sleep(0.01)\n"
        "sys.stdout.buffer.write(b'\\xb8\\xad\\n' + b'x' * 6000); sys.stdout.flush()\n",
        encoding="utf-8",
    )
    provider = ScriptedProvider([
        tool_response("bash", {"command": f'"{sys.executable}" "{script}"'}, "preview"),
        text_response("完成"),
    ])
    monkeypatch.setattr(server, "_get_provider", lambda _: provider)
    streamed = asyncio.Queue()

    async def app_with_event_observer(scope, receive, send):
        async def observe(message):
            body = message.get("body", b"")
            if body.startswith(b"data: "):
                streamed.put_nowait(json.loads(body[6:]))
            await send(message)

        await server.app(scope, receive, observe)

    async with server.lifespan(server.app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app_with_event_observer), base_url="http://test",
        ) as client:
            request = asyncio.create_task(client.post("/chat", json={
                "prompt": "运行测试命令", "session_id": "preview", "shell_output": True,
            }))
            try:
                channels = {}
                async with asyncio.timeout(10):
                    while len(channels) < 2:
                        event = await streamed.get()
                        assert event["type"] not in ("tool_done", "error", "done")
                        if event["type"] == "tool_start":
                            handle = event["execution_id"]
                        elif event["type"] == "tool_output_delta":
                            assert event["execution_id"] == handle
                            channels[event["stream"]] = event["output"]
                assert channels == {"stdout": "<preview>", "stderr": "diagnostic\n"}
                assert not request.done()
                (tmp_path / "release").touch()
                events = decode_events(await asyncio.wait_for(request, 10))
            finally:
                if not request.done():
                    request.cancel()
                await asyncio.gather(request, return_exceptions=True)

    done_index = next(index for index, event in enumerate(events) if event["type"] == "tool_done")
    chunks = [(index, event) for index, event in enumerate(events) if event["type"] == "tool_output_delta"]
    assert all(index < done_index and event["execution_id"] == handle for index, event in chunks)
    stdout = "".join(event["output"] for _, event in chunks if event["stream"] == "stdout")
    expected = "<preview>中\n" + "x" * 6000
    assert stdout == expected.encode()[:4000].decode()
    assert events[done_index]["execution"]["stdout"] == expected
    assert events[-1]["type"] == "done"


@pytest.mark.skipif(sys.platform not in ("win32", "linux"), reason="需要受支持的 Shell 进程树管理")
@pytest.mark.parametrize("confirmed", [
    True,
    pytest.param(False, marks=pytest.mark.skipif(sys.platform != "win32", reason="Windows Job 查询失败路径")),
])
@pytest.mark.parametrize("stop_scope", ["chat", "shell"])
async def test_web_stop_reports_real_process_tree_status(configured_server, tmp_path, monkeypatch, confirmed, stop_scope):
    if not confirmed:
        from coding_agent.tools.builtin import _WindowsProcessJob

        stopped_job = None
        stop_sent = asyncio.Event()
        finish_check = asyncio.Event()
        original_stop = _WindowsProcessJob.stop
        original_wait = _WindowsProcessJob.wait

        def stop(job):
            nonlocal stopped_job
            original_stop(job)
            if stopped_job is None:
                stopped_job = job
                stop_sent.set()

        async def unavailable(job):
            if job is stopped_job:
                await finish_check.wait()
                raise OSError("Job accounting unavailable")
            return await original_wait(job)

        monkeypatch.setattr(_WindowsProcessJob, "stop", stop)
        monkeypatch.setattr(_WindowsProcessJob, "wait", unavailable)

    script = tmp_path / "heartbeat.py"
    script.write_text(
        "import os, pathlib, subprocess, sys, time\n"
        "role = sys.argv[1]\n"
        "if role == 'parent': subprocess.Popen([sys.executable, __file__, 'child'])\n"
        "pathlib.Path(role + '.pid').write_text(str(os.getpid()))\n"
        "while role != 'peer' or not pathlib.Path('release-peer').exists():\n"
        "    with open(role + '.log', 'a') as stream: stream.write('tick\\n')\n"
        "    time.sleep(0.02)\n", encoding="utf-8",
    )
    command = f'"{sys.executable}" "{script}" parent'
    batch = tool_response("bash", {"command": command}, "target")
    if stop_scope == "shell":
        peer = tool_response("bash", {"command": f'"{sys.executable}" "{script}" peer'}, "peer")
        batch = ProviderResponse(content=[*batch.content, *peer.content], stop_reason="tool_use")
    provider = ScriptedProvider([batch, text_response("继续成功"), text_response("下一轮继续成功")])
    monkeypatch.setattr(server, "_get_provider", lambda _: provider)
    first_events = asyncio.Queue()

    async def app_with_event_observer(scope, receive, send):
        async def observe(message):
            body = message.get("body", b"")
            if scope["path"] == "/chat" and body.startswith(b"data: "):
                first_events.put_nowait(json.loads(body[6:]))
            await send(message)

        await server.app(scope, receive, observe)

    async with server.lifespan(server.app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app_with_event_observer), base_url="http://test",
        ) as client:
            request = asyncio.create_task(client.post("/chat", json={
                "prompt": "运行测试心跳", "session_id": "owner", "stop_confirmation": True,
            }))
            try:
                async with asyncio.timeout(8):
                    started = await first_events.get()
                    assert started["type"] == "chat_started"
                    while (tool := await first_events.get())["type"] != "tool_start":
                        pass
                    assert tool["tool_id"] == "target"
                    if stop_scope == "shell":
                        assert tool["stoppable"] is True
                        peer_started = await first_events.get()
                        assert peer_started["tool_id"] == "peer"
                        peer_log = tmp_path / "peer.log"
                        while not (peer_log.exists() and peer_log.stat().st_size):
                            await asyncio.sleep(0.02)
                    logs = [tmp_path / f"{role}.log" for role in ("parent", "child")]
                    while not all(log.exists() and log.stat().st_size for log in logs):
                        await asyncio.sleep(0.02)
                pids = [int(log.with_suffix(".pid").read_text()) for log in logs]
                assert all(_process_running(pid) for pid in pids)
                chat_url = "/chat/" + started["stop_id"]
                url = chat_url + "/stop"
                if stop_scope == "shell":
                    url = f'{chat_url}/tools/{tool["execution_id"]}/stop'
                    unknown = await client.post(f"{chat_url}/tools/unknown/stop", json={"session_id": "owner"})
                    assert unknown.status_code == 404
                assert (await client.post(url, json={"session_id": "other"})).status_code == 403
                assert all(_process_running(pid) for pid in pids)
                response = await client.post(url, json={"session_id": "owner"})
                assert response.status_code == 202
                if stop_scope == "shell":
                    assert response.json() == {"status": "stopping", "execution_id": tool["execution_id"]}
                if not confirmed:
                    await asyncio.wait_for(stop_sent.wait(), 5)
                    repeated = await client.post(url, json={"session_id": "owner"})
                    assert repeated.status_code == 202
                    assert repeated.json() == response.json()
                    finish_check.set()
                if stop_scope == "shell":
                    async with asyncio.timeout(8):
                        while (ended := await first_events.get())["type"] != "tool_done":
                            pass
                    assert not request.done()
                    assert _process_running(int(peer_log.with_suffix(".pid").read_text()))
                else:
                    assert response.json() == {"status": "stopping"}
                    events = decode_events(await asyncio.wait_for(request, 8))
                    assert events[-1]["type"] == "chat_stopped"
                    ended = next(event for event in events if event["type"] == "tool_done")
                    assert events[-1]["unconfirmed_execution_ids"] == ([] if confirmed else [ended["execution_id"]])
                assert ended["execution_id"] == tool["execution_id"]
                assert ended["execution"]["cancelled"] is confirmed, ended
                assert ended["execution"]["termination_confirmed"] is confirmed, ended
                assert not any(_process_running(pid) for pid in pids)
                snapshots = [log.read_bytes() for log in logs]
                await asyncio.sleep(0.2)
                assert snapshots == [log.read_bytes() for log in logs]
                assert (await client.post(url, json={"session_id": "owner"})).status_code == 404
                if stop_scope == "shell":
                    (tmp_path / "release-peer").touch()
                    events = decode_events(await asyncio.wait_for(request, 8))
                    done_tools = [event for event in events if event["type"] == "tool_done"]
                    assert [event["tool_id"] for event in done_tools] == ["target", "peer"]
                    assert done_tools[1]["execution_id"] == peer_started["execution_id"]
                    assert done_tools[1]["execution"]["exit_code"] == 0
                    assert events[-1]["type"] == "done"
                    assert events[-1]["text"] == "继续成功"
                stored = next(block for message in server.sessions["owner"] for block in message["content"]
                              if isinstance(block, dict) and block.get("tool_use_id") == tool["tool_id"])
                assert stored["tool_use_id"] == tool["tool_id"]
                assert stored["is_error"] is True
                assert json.loads(stored["content"])["termination_confirmed"] is confirmed
                response = await client.post("/chat", json={"prompt": "继续", "session_id": "owner"})
                assert decode_events(response)[-1]["text"] == ("下一轮继续成功" if stop_scope == "shell" else "继续成功")
            finally:
                if not confirmed:
                    finish_check.set()
                if not request.done():
                    request.cancel()
                await asyncio.gather(request, return_exceptions=True)
