"""命令行多轮对话与异步资源生命周期的回归测试。"""

import asyncio
import json
from copy import deepcopy

import pytest
from click.testing import CliRunner

from coding_agent.config import Config
from coding_agent.memory import MemoryStore
from coding_agent.providers.base import LLMProvider, ProviderResponse, StreamEvent
from coding_agent_cli import main


@pytest.mark.parametrize(("name", "parameters", "expected"), [
    ("read_file", {"path": "src/main.py"}, "> read_file src/main.py"),
    ("read_file", {"path": "中文 空格.py", "start_line": 12, "end_line": 20}, "> read_file 中文 空格.py:12-20"),
    ("read_file", {"path": "a.py", "start_line": 12, "limit": 2}, "> read_file a.py:12-13"),
    ("read_file", {"path": "a.py", "start_line": 12}, "> read_file a.py:12-"),
    ("write_file", {"path": "a.py", "content": "PRIVATE_BODY"}, "> write_file a.py"),
    ("edit_file", {"path": "a.py", "old_text": "PRIVATE_BODY", "new_text": "new"}, "> edit_file a.py"),
    ("code_outline", {"path": "a.py"}, "> code_outline a.py"),
    ("git_diff", {"path": "a.py", "staged": True}, "> git_diff a.py [暂存区]"),
    ("git_diff", {}, "> git_diff ."),
    ("git_diff", {"staged": True}, "> git_diff . [暂存区]"),
    ("glob", {"pattern": "**/*.py"}, "> glob **/*.py"),
    ("search_code", {"path": "src", "pattern": "run", "file_glob": "*.py"},
     '> search_code src  pattern="run"  files="*.py"'),
    ("git_status", {}, "> git_status"),
    ("unknown", {"token": "PRIVATE_BODY"}, "> unknown"),
    ("read_file", None, "> read_file"),
])
def test_tool_start_shows_target_without_file_content(capsys, name, parameters, expected):
    main._print_event(main.LoopEvent(type="tool_start", tool_name=name, data={"input": parameters}), color=False)
    output = capsys.readouterr().out
    assert output == "\n" + expected + "\n"
    assert "PRIVATE_BODY" not in output


def test_patch_display_lists_all_operations_without_body(capsys):
    patch = ("*** Begin Patch\n*** Add File: added.py\n+PRIVATE_BODY\n"
             "*** Update File: source.py\n@@\n-old\n+new\n"
             "*** Update File: old.py\n*** Move to: moved.py\n@@\n-old\n+new\n"
             "*** Delete File: deleted.py\n*** End Patch")
    main._print_event(main.LoopEvent(type="tool_start", tool_name="apply_patch", data={"input": {"patch": patch}}))
    output = capsys.readouterr().out
    assert output == ("\n> apply_patch\n  新增 added.py\n  修改 source.py\n"
                      "  移动 old.py → moved.py\n  删除 deleted.py\n")
    assert "PRIVATE_BODY" not in output


@pytest.mark.parametrize("patch", [None, 5, {}, "invalid", "*** Begin Patch\n*** Update File: file\n*** Move to: x\n*** End Patch"])
def test_invalid_patch_display_does_not_raise(capsys, patch):
    main._print_event(main.LoopEvent(type="tool_start", tool_name="apply_patch", data={"input": {"patch": patch}}))
    assert "补丁格式无效，无法提取文件列表" in capsys.readouterr().out


def test_tool_display_escapes_control_characters_and_preserves_path(capsys):
    path = "C:" + chr(92) + "项目" + chr(92) + "file\n\x1b[31m\r\t.txt"
    main._print_event(main.LoopEvent(type="tool_start", tool_name="read_file", data={"input": {"path": path}}), color=False)
    output = capsys.readouterr().out
    assert "C:" + chr(92) + "项目" + chr(92) + "file" in output
    assert r"\n\u001b[31m\r\t.txt" in output
    assert output.count("\n") == 2
    assert "\x1b" not in output


@pytest.mark.parametrize("is_error", [True, False])
def test_tool_failure_shows_path_and_escapes_error(capsys, is_error):
    main._print_event(main.LoopEvent(type="tool_done", tool_name="read_file", tool_output="Error: missing\nfile",
                                   data={"input": {"path": "missing.py"}, "is_error": is_error}))
    assert capsys.readouterr().err == "[工具失败] read_file missing.py: Error: missing\\nfile\n"


def test_tool_success_does_not_print_file_content(capsys):
    main._print_event(main.LoopEvent(type="tool_done", tool_name="read_file", tool_output="PRIVATE_BODY"))
    captured = capsys.readouterr()
    assert not captured.out and not captured.err


class LoopBoundProvider(LLMProvider):
    """模拟持有连接池的客户端，要求所有请求和关闭操作使用同一事件循环。"""

    def __init__(self, *, fail=False):
        super().__init__(model_id="mock")
        self.owner_loop = None
        self.history = []
        self.closed = False
        self.fail = fail
        self.systems = []
        self.memory_prompts = []
        self.memory_response = lambda prompt: "[]"

    async def create_message(self, messages, tools=None, system="", max_tokens=8000):
        self._check_loop()
        prompt = messages[0]["content"]
        self.memory_prompts.append(prompt)
        return ProviderResponse(
            content=[{"type": "text", "text": self.memory_response(prompt)}],
            stop_reason="end_turn",
        )

    def _check_loop(self):
        current = asyncio.get_running_loop()
        if self.owner_loop is None:
            self.owner_loop = current
        assert current is self.owner_loop, "对话跨轮次切换了事件循环"
        assert not self.owner_loop.is_closed()

    async def stream_message(self, messages, tools=None, system="", max_tokens=8000):
        self._check_loop()
        self.systems.append(system)
        self.history.append(deepcopy(messages))
        if self.fail:
            raise RuntimeError("模拟请求失败")
        text = "我叫李华。"
        yield StreamEvent(type="text_delta", text=text)
        yield StreamEvent(
            type="done",
            response=ProviderResponse(
                content=[{"type": "text", "text": text}], stop_reason="end_turn"
            ),
        )

    async def aclose(self):
        if self.owner_loop is not None:
            assert asyncio.get_running_loop() is self.owner_loop
        self.closed = True


@pytest.fixture
def provider(monkeypatch, tmp_path):
    fake = LoopBoundProvider()
    monkeypatch.setattr(
        main, "load_config", lambda _: Config(model_id="mock", workdir=str(tmp_path))
    )
    monkeypatch.setattr(main, "_build_provider", lambda _: fake)
    return fake


def test_chat_preserves_loop_and_history_across_turns(provider):
    result = CliRunner().invoke(
        main.cli, ["chat"], input="你记住你叫李华\n你叫什么？\nexit\n"
    )
    assert result.exit_code == 0, result.exception
    assert result.output.count("我叫李华。") == 2
    assert [msg["role"] for msg in provider.history[1]] == [
        "user", "assistant", "user"
    ]
    assert provider.history[1][0]["content"] == "你记住你叫李华"
    assert provider.history[1][1]["content"][0]["text"] == "我叫李华。"
    assert provider.closed
    assert provider.owner_loop.is_closed()


@pytest.mark.parametrize("user_input", ["q\n", "exit\n", ""])
def test_chat_closes_provider_on_quit_or_eof(provider, user_input):
    result = CliRunner().invoke(main.cli, ["chat"], input=user_input)
    assert result.exit_code == 0, result.exception
    assert provider.closed
    assert provider.history == []


def test_chat_closes_provider_on_request_failure(provider):
    provider.fail = True
    result = CliRunner().invoke(main.cli, ["chat"], input="你好\n")
    assert isinstance(result.exception, RuntimeError)
    assert str(result.exception) == "模拟请求失败"
    assert provider.closed
    assert provider.owner_loop.is_closed()


def test_chat_survives_exhausted_connection_failure(provider, monkeypatch):
    import httpx
    import openai
    from unittest.mock import AsyncMock

    monkeypatch.setattr("coding_agent.context.recovery.asyncio.sleep", AsyncMock())
    original = provider.stream_message
    attempts = []

    async def interrupted(*args, **kwargs):
        attempts.append(1)
        if len(attempts) <= 3:
            raise openai.APIConnectionError(request=httpx.Request("POST", "https://example.invalid"))
        async for event in original(*args, **kwargs):
            yield event

    monkeypatch.setattr(provider, "stream_message", interrupted)
    result = CliRunner().invoke(main.cli, ["chat"], input="你好\n再试一次\nexit\n")
    assert result.exit_code == 0, result.exception
    assert "3 次尝试" in result.output and "会话已保留" in result.output
    assert "我叫李华。" in result.output
    assert len(attempts) == 4 and provider.closed


@pytest.mark.parametrize("command", ["chat", "run"])
def test_cli_reports_partial_stream_failure_without_replay(provider, monkeypatch, command):
    import httpx
    import openai

    attempts = []

    async def interrupted(*args, **kwargs):
        attempts.append(1)
        yield StreamEvent(type="text_delta", text="已输出部分回复")
        raise openai.APIConnectionError(request=httpx.Request("POST", "https://example.invalid"))

    monkeypatch.setattr(provider, "stream_message", interrupted)
    arguments = ["chat"] if command == "chat" else ["run", "你好"]
    result = CliRunner().invoke(main.cli, arguments, input="你好\nexit\n")
    assert result.exit_code == (0 if command == "chat" else 1)
    assert result.output.count("已输出部分回复") == 1
    assert "未自动重放" in result.output and "Traceback" not in result.output
    assert len(attempts) == 1 and provider.closed


def test_chat_closes_provider_on_keyboard_interrupt(provider, monkeypatch):
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(main.click, "prompt", interrupt)
    result = CliRunner().invoke(main.cli, ["chat"])
    assert result.exit_code == 1
    assert provider.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["openai_compat", "anthropic"])
async def test_provider_closes_sdk_client(backend):
    """两个真实后端均应关闭各自的 SDK 客户端，无需联网。"""
    from coding_agent.providers import AnthropicProvider, OpenAICompatProvider

    provider_cls = {
        "openai_compat": OpenAICompatProvider,
        "anthropic": AnthropicProvider,
    }[backend]
    provider = provider_cls(api_key="test-key")
    try:
        assert not provider._client.is_closed()
    finally:
        await provider.aclose()
    assert provider._client.is_closed()


def _name_memory(name):
    return json.dumps([{
        "name": "assistant-name",
        "type": "feedback",
        "description": f"用户指定助手名为{name}",
        "body": f"用户要求助手记住自己叫{name}，今后使用这个名字。",
    }], ensure_ascii=False)


def test_chat_saves_memory_and_loads_it_in_new_session(provider, tmp_path, monkeypatch):
    provider.memory_response = lambda _: _name_memory("李华")
    first = CliRunner().invoke(main.cli, ["chat"], input="记住你叫李华\nexit\n")
    assert first.exit_code == 0, first.exception
    store = MemoryStore(workdir=tmp_path)
    assert "李华" in store.read_index()
    assert "李华" in store.read_file("assistant-name.md")
    assert "[记忆] 已保存 1 条" in first.output

    # 新客户端和新事件循环模拟退出后重新启动，而非复用内存中的历史。
    fresh = LoopBoundProvider()
    fresh.memory_response = lambda prompt: "[0]" if "JSON 整数数组" in prompt else "[]"
    monkeypatch.setattr(main, "_build_provider", lambda _: fresh)
    second = CliRunner().invoke(main.cli, ["chat"], input="你叫什么？\nexit\n")
    assert second.exit_code == 0, second.exception
    assert "李华" in fresh.systems[0]
    assert "<relevant_memories>" in fresh.systems[0]
    assert len(fresh.history[0]) == 1
    assert fresh.history[0][0]["content"] == "你叫什么？"
    assert store.file_count == 1
    assert fresh.closed


def test_chat_updates_memory_from_latest_turn_only(provider, tmp_path):
    def summarize(prompt):
        if "JSON 整数数组" in prompt:
            return "[0]"
        dialogue = prompt.split("对话:\n", 1)[1]
        return _name_memory("小明" if "改叫小明" in dialogue else "李华")

    provider.memory_response = summarize
    result = CliRunner().invoke(
        main.cli, ["chat"], input="记住你叫李华\n记住你现在改叫小明\nexit\n"
    )
    assert result.exit_code == 0, result.exception
    store = MemoryStore(workdir=tmp_path)
    assert store.file_count == 1
    assert "小明" in store.read_index()
    assert "李华" not in store.read_file("assistant-name.md")
    assert "李华" in provider.systems[1]
    extraction_prompts = [p for p in provider.memory_prompts if "已有记忆:" in p]
    latest_dialogue = extraction_prompts[-1].split("对话:\n", 1)[1]
    assert "改叫小明" in latest_dialogue
    assert "记住你叫李华" not in latest_dialogue


@pytest.mark.parametrize("response", ["[]", "这不是 JSON", '[null, 3, "bad", []]'])
def test_chat_handles_empty_or_invalid_memory_output(provider, tmp_path, response):
    provider.memory_response = lambda _: response
    result = CliRunner().invoke(main.cli, ["chat"], input="你好\nexit\n")
    assert result.exit_code == 0, result.exception
    assert MemoryStore(workdir=tmp_path).file_count == 0
    assert "[记忆] 已保存" not in result.output


def test_chat_reports_memory_write_failure(provider, monkeypatch):
    async def fail_write(self, messages):
        raise PermissionError("记忆目录只读")

    monkeypatch.setattr(MemoryStore, "extract", fail_write)
    result = CliRunner().invoke(main.cli, ["chat"], input="你好\nexit\n")
    assert result.exit_code == 0, result.exception
    assert "[记忆] 写入失败" in result.output
    assert provider.closed


def test_cli_run_also_persists_memory(provider, tmp_path):
    provider.memory_response = lambda _: _name_memory("李华")
    result = CliRunner().invoke(main.cli, ["run", "记住你叫李华"])
    assert result.exit_code == 0, result.exception
    assert "[记忆] 已保存 1 条" in result.output
    assert "李华" in MemoryStore(workdir=tmp_path).read_index()
    assert provider.closed


@pytest.mark.asyncio
async def test_terminal_input_can_be_cancelled_without_waiting_for_stdin():
    import threading

    release = threading.Event()
    task = asyncio.create_task(main._read_terminal(release.wait))
    await asyncio.sleep(0)
    task.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1)
    finally:
        release.set()
