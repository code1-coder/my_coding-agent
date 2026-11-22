"""基于 Click 的命令行入口，复用与网页端相同的 AgentRuntime。"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import AsyncGenerator
from contextlib import aclosing
from typing import Any

import anthropic
import click
import httpx
import openai

from coding_agent.config import Config, load_config
from coding_agent.loop import ANALYSIS_DIMENSION_IDS, LoopEvent, resolve_analysis_dimensions
from coding_agent.providers.base import LLMProvider, ModelRequestError
from coding_agent.runtime import AgentRuntime, RetryUnavailable, build_provider
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
    """渲染正文以外的运行事件。"""
    if event.type == "tool_start":
        label = f"> {_terminal_text(event.tool_name)}"
        details = _tool_summary(event.tool_name, event.data.get("input"))
        heading = click.style(label, fg="yellow") if color else label
        if event.tool_name == "apply_patch":
            click.echo("\n" + heading)
            for detail in details:
                click.echo("  " + detail)
        else:
            click.echo("\n" + heading + (" " + details[0] if details else ""))
    elif event.type == "tool_done" and (
        event.data.get("is_error") is True
        or ("is_error" not in event.data and event.tool_output.startswith("Error:"))
    ):
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
    elif event.type == "tool_done" and event.tool_name == "search_code":
        try:
            result = json.loads(event.tool_output)
        except json.JSONDecodeError:
            return
        if isinstance(result, dict) and result.get("fallback_reason") == "rg_unavailable":
            details = _tool_summary(event.tool_name, event.data.get("input"))
            label = "[搜索降级] " + (details[0] if details else ".")
            click.echo(label + "：" + _terminal_text(str(result.get("backend", "python")))
                       + ("，范围未完整搜索，请检查结果限制" if result.get("truncated") else ""))
    elif event.type == "memory_saved":
        click.echo(f"\n[记忆] 已保存 {event.data['count']} 条到 {event.data['path']}\n")
    elif event.type == "analysis_progress":
        summary = "\n".join(_terminal_text(line) for line in event.text.splitlines())
        click.echo(f"\n[分析检查点] 剩余读取批次：{event.data['remaining_batches']}\n{summary}")
    elif event.type == "warning":
        click.echo("\n" + event.text, err=True)
    elif event.type == "error":
        hint = ("本次失败，会话已保留；输入 /retry 可续接本次请求，已完成工具不会重放。"
                if event.data.get("recovery_id") else "本次失败，会话已保留，可继续输入。")
        delivery = "\n".join(_terminal_text(line) for line in event.text.split("\n"))
        click.echo("\n" + delivery + "\n" + hint, err=True)


async def _render_turn(operation: AsyncGenerator[LoopEvent, None], *, show_text: bool = True,
                       color: bool = True, fail_on_error: bool = False) -> str:
    """正文确认后再写入 stdout；工具、审批和进度提示仍即时可见。"""
    pending: list[str] = []
    committed_chars = 0
    final = ""

    def flush_text() -> None:
        nonlocal committed_chars
        if pending:
            text = "".join(pending)
            pending.clear()
            committed_chars += len(text)
            if text:
                click.echo(text, nl=False)

    async with aclosing(operation) as events:
        try:
            async for event in events:
                if event.type == "text_delta":
                    if show_text:
                        pending.append(event.text)
                elif event.type == "text_reset":
                    if show_text:
                        # reset 携带本轮完整正文，已确认并输出的前缀不能再打印一次。
                        pending[:] = [event.text[committed_chars:]]
                    if event.data.get("reason") != "response_reconciliation":
                        click.echo("\n[输出达到上限，正在重新生成]", err=True)
                else:
                    if event.type in ("tool_start", "memory_saved", "done", "error"):
                        flush_text()
                    if event.type == "done":
                        final = event.text
                    if event.type == "error" and fail_on_error:
                        raise click.ClickException(event.text)
                    _print_event(event, color=color)
        finally:
            # 流中断时保留已经收到、尚未被 reset 撤回的正文，原异常仍由入口处理。
            flush_text()
    return final


async def _run_agent_loop(prompt: str, cfg: Config, *, stream: bool = True, color: bool = True,
                          analysis: bool = False, analysis_dimensions: tuple[str, ...] | None = None) -> str:
    """单次任务也经过完整运行流程，并在退出前释放队友和客户端。"""
    runtime, _ = _create_runtime(cfg)
    async with runtime:
        return await _render_turn(runtime.run_turn(prompt, stream=stream, analysis=analysis,
                                                   analysis_dimensions=analysis_dimensions),
                                  show_text=stream, color=color, fail_on_error=True)


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
        click.echo('输入 /analyze <问题> 进行静态分析；输入 "q" 或 "exit" 退出。\n')
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
                words = query.strip().split(maxsplit=1)
                if query.strip() == "/retry":
                    operation = runtime.retry_turn()
                elif words[0] == "/analyze":
                    prompt = words[1] if len(words) == 2 else ""
                    dimensions = None
                    args = prompt.split(maxsplit=2)
                    if args and args[0] == "--dimensions":
                        if len(args) != 3:
                            click.echo("用法：/analyze --dimensions role,flow <问题>", err=True)
                            continue
                        try:
                            dimensions = resolve_analysis_dimensions(True, args[1].split(","))
                        except ValueError as exc:
                            click.echo(str(exc), err=True)
                            continue
                        prompt = args[2]
                    if not prompt:
                        click.echo("用法：/analyze [--dimensions role,flow] <问题>", err=True)
                        continue
                    operation = runtime.run_turn(prompt, analysis=True, analysis_dimensions=dimensions)
                else:
                    operation = runtime.run_turn(query)
                await _render_turn(operation)
            except (RetryUnavailable, PermissionError) as exc:
                click.echo(f"\n[无法续接] {exc}", err=True)
            except REQUEST_ERRORS as exc:
                click.echo(f"\n[请求失败] {exc}\n会话已保留，可继续输入或重试。", err=True)
            click.echo("\n")


@cli.command()
@click.argument("prompt")
@click.option("--analysis", is_flag=True, help="只做静态源码分析，不执行项目代码或修改文件。")
@click.option("--analysis-dimension", "dimensions", type=click.Choice(ANALYSIS_DIMENSION_IDS), multiple=True,
              help="指定必查维度，可重复使用；省略时检查全部维度。")
@click.pass_context
def run(ctx, prompt, analysis, dimensions):
    """执行单次提示词任务后退出。"""
    try:
        scope = resolve_analysis_dimensions(analysis, dimensions or None)
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="--analysis-dimension") from exc
    try:
        final = asyncio.run(_run_agent_loop(prompt, ctx.obj["cfg"], stream=True, analysis=analysis,
                                           analysis_dimensions=scope))
    except REQUEST_ERRORS as exc:
        raise click.ClickException(f"模型请求失败：{exc}") from exc
    if not final.endswith("\n"):
        click.echo()


if __name__ == "__main__":
    cli()
