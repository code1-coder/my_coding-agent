"""基于 Click 的命令行入口，复用与网页端相同的 AgentRuntime。"""

from __future__ import annotations

import asyncio
import json
import threading
from contextlib import aclosing
from typing import Any

import anthropic
import click
import httpx
import openai

from coding_agent.config import Config, load_config
from coding_agent.loop import LoopEvent
from coding_agent.providers.base import LLMProvider, ModelRequestError
from coding_agent.runtime import AgentRuntime, build_provider
from coding_agent.security.permission import create_terminal_pipeline
from coding_agent.tools.patch import parse_patch


REQUEST_ERRORS = (ModelRequestError, openai.APIError, anthropic.APIError,
                  httpx.TransportError, ConnectionError, TimeoutError)


def _build_provider(cfg: Config) -> LLMProvider:
    """通过公共工厂创建客户端，避免两端的模型配置逐渐产生差异。"""
    return build_provider(cfg)


async def _read_terminal(fn, *args, **kwargs):
    """异步读取终端；守护读取线程不会在取消时阻塞事件循环的关闭。"""
    loop = asyncio.get_running_loop()
    future = loop.create_future()

    def finish(value, error):
        if future.done():
            return
        if error is not None:
            future.set_exception(error)
        else:
            future.set_result(value)

    def read():
        try:
            value, error = fn(*args, **kwargs), None
        except BaseException as exc:
            value, error = None, exc
        try:
            loop.call_soon_threadsafe(finish, value, error)
        except RuntimeError:
            # CLI 已退出，读取线程无需再向已经关闭的循环回传结果。
            pass

    threading.Thread(target=read, daemon=True, name="coding-agent-input").start()
    return await future


def _create_runtime(cfg: Config) -> tuple[AgentRuntime, asyncio.Lock]:
    """连接终端审批，并串行读取用户输入，避免队友和主会话争抢 stdin。"""
    read_lock = asyncio.Lock()
    permissions = create_terminal_pipeline(workdir=cfg.workdir, deny_list=cfg.deny_list)
    async def approve(tool_name, params, reason):
        async with read_lock:
            click.echo(f"\n需要审批：{reason}\n工具：{tool_name}({params})")
            try:
                return await _read_terminal(click.confirm, "允许执行？", default=False)
            except (EOFError, click.Abort):
                return False

    permissions.approval_handler = approve
    return AgentRuntime(cfg, provider_factory=_build_provider, permissions=permissions), read_lock


def _terminal_text(value: Any) -> str:
    """转义换行和控制字符，保留中文、空格及正常路径分隔符。"""
    return "".join(character if character.isprintable() else
                   json.dumps(character, ensure_ascii=True)[1:-1] for character in str(value))


def _tool_summary(name: str, parameters: Any) -> list[str]:
    """只从已知工具参数提取摘要，避免打印文件正文。"""
    if not isinstance(parameters, dict):
        return []
    if name == "bash":
        command = parameters.get("command")
        return [_terminal_text(command)] if isinstance(command, str) else []
    if name == "apply_patch":
        try:
            operations = parse_patch(parameters.get("patch", ""))
        except (ValueError, TypeError, IndexError, AttributeError):
            return ["补丁格式无效，无法提取文件列表"]
        labels = {"Add": "新增", "Update": "修改", "Delete": "删除"}
        return [(f"移动 {_terminal_text(operation.path)} → {_terminal_text(operation.destination)}"
                 if operation.destination is not None else
                 f"{labels[operation.action]} {_terminal_text(operation.path)}")
                for operation in operations]
    if name in ("read_file", "write_file", "edit_file", "code_outline", "git_diff"):
        path = parameters.get("path")
        if not isinstance(path, str) or not path:
            summary = "." if name == "git_diff" else ""
        else:
            summary = _terminal_text(path)
        if not summary:
            return []
        if name == "read_file":
            start, end, limit = (parameters.get(key) for key in ("start_line", "end_line", "limit"))
            if start is not None or end is not None or limit is not None:
                first = start if start is not None else 1
                if type(first) is int and first > 0:
                    last = end if type(end) is int and end >= first else None
                    if type(limit) is int and limit > 0:
                        last = min(last, first + limit - 1) if last is not None else first + limit - 1
                    summary += f":{first}-{last}" if last is not None else f":{first}-"
        if name == "git_diff" and parameters.get("staged") is True:
            summary += " [暂存区]"
        return [summary]
    if name == "glob":
        return [_terminal_text(parameters["pattern"])] if isinstance(parameters.get("pattern"), str) else []
    if name == "search_code":
        parts = [_terminal_text(parameters.get("path") or ".")]
        for key, label in (("pattern", "pattern"), ("file_glob", "files")):
            if isinstance(parameters.get(key), str):
                parts.append(f"{label}=" + _terminal_text(json.dumps(parameters[key], ensure_ascii=False)))
        return ["  ".join(parts)]
    return []


def _print_event(event: LoopEvent, *, color: bool = True) -> None:
    """把公共运行事件渲染到终端，保持入口只负责交互和显示。"""
    if event.type == "text_delta":
        click.echo(event.text, nl=False)
    elif event.type == "text_reset":
        click.echo("\n[输出达到上限，正在重新生成]\n")
    elif event.type == "tool_start":
        label = f"> {_terminal_text(event.tool_name)}"
        details = _tool_summary(event.tool_name, event.data.get("input"))
        heading = click.style(label, fg="yellow") if color else label
        if event.tool_name == "apply_patch":
            click.echo("\n" + heading)
            for detail in details:
                click.echo("  " + detail)
        else:
            click.echo("\n" + heading + (" " + details[0] if details else ""))
    elif event.type == "tool_done" and (event.data.get("is_error") or event.tool_output.startswith("Error:")):
        label = f"[工具失败] {_terminal_text(event.tool_name)}"
        details = _tool_summary(event.tool_name, event.data.get("input"))
        if details:
            label += " " + "; ".join(details)
        execution = event.data.get("execution", {})
        if execution.get("exit_code") is not None:
            label += f" [退出码: {execution['exit_code']}]"
        if execution.get("duration_ms") is not None:
            label += f" [耗时: {execution['duration_ms']} ms]"
        if execution.get("failure_stage"):
            label += f" [阶段: {execution['failure_stage']}]"
        if execution.get("timed_out"):
            label += " [已超时]"
        if execution.get("truncated"):
            label += " [输出已截断]"
        error = _terminal_text(event.tool_output)
        if len(error) > 500:
            error = error[:500] + "…（错误信息已截断）"
        click.echo(label + ": " + error, err=True)
    elif event.type == "tool_done" and event.tool_name == "bash" and event.data.get("execution"):
        execution = event.data["execution"]
        click.echo(f"[bash 完成] 退出码: {execution['exit_code']}，耗时: {execution['duration_ms']} ms"
                   + ("，输出已截断" if execution.get("truncated") else ""))
    elif event.type == "memory_saved":
        click.echo(f"\n[记忆] 已保存 {event.data['count']} 条到 {event.data['path']}\n")
    elif event.type == "warning":
        click.echo("\n" + event.text, err=True)
    elif event.type == "error":
        click.echo("\n" + event.text + "\n本次失败，会话已保留，可继续输入。", err=True)


async def _run_agent_loop(prompt: str, cfg: Config, *, stream: bool = True, color: bool = True) -> str:
    """单次任务也经过完整运行流程，并在退出前释放队友和客户端。"""
    runtime, _ = _create_runtime(cfg)
    final = ""
    failure = None
    async with runtime:
        async with aclosing(runtime.run_turn(prompt, stream=stream)) as events:
            async for event in events:
                if event.type == "error":
                    failure = event.text
                    continue
                if stream or event.type != "text_delta":
                    _print_event(event, color=color)
                if event.type == "done":
                    final = event.text
    if failure:
        raise click.ClickException(failure)
    return final


@click.group()
@click.option("--model", default=None, help="覆盖模型标识。")
@click.option("--provider", "provider_", default=None, type=click.Choice(["anthropic", "openai_compat"]))
@click.option("--max-tokens", default=None, type=int, help="单次响应的最大令牌数。")
@click.option("--config", "config_path", default="config.yaml", help="YAML 配置文件路径。")
@click.pass_context
def cli(ctx, model, provider_, max_tokens, config_path):
    """coding-agent —— 模块化 AI 编码智能体。"""
    cfg = load_config(config_path)
    if model:
        cfg.model_id = model
    if provider_:
        cfg.provider = provider_
    if max_tokens:
        cfg.max_tokens = max_tokens
    ctx.ensure_object(dict)
    ctx.obj["cfg"] = cfg


@cli.command()
@click.pass_context
def chat(ctx):
    """启动交互式对话。"""
    asyncio.run(_chat_session(ctx.obj["cfg"]))


async def _chat_session(cfg: Config) -> None:
    """整个会话共用一个循环；等待终端输入时允许后台队友继续工作。"""
    runtime, read_lock = _create_runtime(cfg)
    async with runtime:
        click.echo(f"coding-agent chat  |  model: {cfg.model_id}  |  workdir: {cfg.workdir}")
        click.echo('输入 "q" 或 "exit" 退出。\n')
        while True:
            try:
                async with read_lock:
                    query = await _read_terminal(click.prompt, ">> ", prompt_suffix="")
            except (EOFError, click.Abort):
                click.echo()
                break
            if query.strip().lower() in ("q", "exit", ""):
                break
            click.echo()
            try:
                async with aclosing(runtime.run_turn(query)) as events:
                    async for event in events:
                        _print_event(event)
            except REQUEST_ERRORS as exc:
                click.echo(f"\n[请求失败] {exc}\n会话已保留，可继续输入或重试。", err=True)
            click.echo("\n")


@cli.command()
@click.argument("prompt")
@click.pass_context
def run(ctx, prompt):
    """执行单次提示词任务后退出。"""
    try:
        final = asyncio.run(_run_agent_loop(prompt, ctx.obj["cfg"], stream=True))
    except REQUEST_ERRORS as exc:
        raise click.ClickException(f"模型请求失败：{exc}") from exc
    if not final.endswith("\n"):
        click.echo()


if __name__ == "__main__":
    cli()
