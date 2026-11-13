"""
配置模块 —— YAML 默认值 + 环境变量覆盖。

加载顺序（后者优先）：
1. 数据类默认值
2. 工作目录下的 ``config.yaml``（如果存在）
3. 环境变量（``MODEL_ID``、``*_API_KEY`` 等）

用法::

    from coding_agent.config import Config, load_config

    cfg = load_config("config.yaml")
    print(cfg.model_id)
"""

from __future__ import annotations

import os
import math
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]


@dataclass
class Config:
    """编码智能体的运行时配置。"""

    model_id: str = "claude-sonnet-5"
    base_url: str | None = None
    api_key: str | None = None
    max_tokens: int = 8000
    system_prompt: str = (
        "You are a coding agent. Use tools to solve tasks. "
        "Do the work, then report the result, verification and any remaining limitations to the user."
    )
    provider: str = "auto"  # 模型后端类型：Anthropic、OpenAI 兼容接口或自动识别
    workdir: str = field(default_factory=lambda: str(Path.cwd()))
    context_limit: int = 50000
    max_messages: int = 50
    keep_recent: int = 3
    persist_threshold: int = 30000
    max_iterations: int = 50
    skills_dir: str = "skills"
    mcp_servers: list[str] = field(default_factory=list)
    deny_list: list[str] = field(default_factory=lambda: [
        "rm -rf /", "sudo", "shutdown", "reboot", "mkfs", "dd if=", "> /dev/sda",
    ])
    team_idle_timeout: float = 60.0
    fallback_model: str | None = None
    escalated_max_tokens: int = 64000
    model_timeout: float = 900.0
    model_connect_timeout: float = 20.0
    model_idle_timeout: float = 120.0
    final_answer_timeout: float = 30.0
    maintenance_timeout: float = 30.0
    progress_interval: float = 10.0


def load_config(path: str | Path = "config.yaml") -> Config:
    """从 YAML 文件加载配置，然后应用环境变量覆盖。

    参数：
        path: YAML 文件的路径。如果文件不存在则静默跳过。

    返回：
        填充好的 :class:`Config` 实例。
    """
    # 两种入口采用相同的 .env 规则，显式进程环境变量优先。
    load_dotenv(Path.cwd() / ".env", override=False)
    cfg = Config()

    # ── 1. YAML 文件 ────────────────────────────────────────
    try:
        resolved = Path(path).resolve()
    except (OSError, ValueError):
        resolved = None

    if resolved and resolved.exists():
        if yaml is None:
            import warnings
            warnings.warn("PyYAML not installed — skipping config.yaml")
        else:
            with open(resolved, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            _apply_dict(cfg, data)

    # ── 2. 环境变量 ─────────────────────────────────────────
    _apply_env_overrides(cfg)

    validate_config(cfg)
    return cfg


def validate_config(cfg: Config) -> None:
    """在创建客户端前检查运行参数，避免无效阈值裁掉当前用户输入。"""
    for name in ("max_tokens", "context_limit", "persist_threshold", "max_iterations", "escalated_max_tokens"):
        value = getattr(cfg, name)
        if type(value) is not int or value <= 0:
            raise ValueError(f"配置 {name} 必须是正整数")
    if type(cfg.max_messages) is not int or cfg.max_messages < 5:
        raise ValueError("配置 max_messages 必须是大于等于 5 的整数")
    if type(cfg.keep_recent) is not int or cfg.keep_recent < 0:
        raise ValueError("配置 keep_recent 必须是非负整数")
    if type(cfg.team_idle_timeout) not in (int, float) or cfg.team_idle_timeout < 0:
        raise ValueError("配置 team_idle_timeout 必须是非负数")
    for name in ("model_timeout", "model_connect_timeout", "model_idle_timeout",
                 "final_answer_timeout", "maintenance_timeout", "progress_interval"):
        value = getattr(cfg, name)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"配置 {name} 必须是有限正数（秒）")
    for name in ("mcp_servers", "deny_list"):
        value = getattr(cfg, name)
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ValueError(f"配置 {name} 必须是字符串列表")
    if cfg.provider not in ("auto", "anthropic", "openai_compat"):
        raise ValueError("配置 provider 必须是 auto、anthropic 或 openai_compat")


def _apply_dict(cfg: Config, data: dict) -> None:
    """把原始 YAML 值合并进配置（只认已知键，忽略 None 值）。"""
    known = set(Config.__dataclass_fields__)
    for key in known & set(data):
        val = data[key]
        if val is not None:
            setattr(cfg, key, val)


def _apply_env_overrides(cfg: Config) -> None:
    """把环境变量覆盖应用到 *cfg*。"""
    # 模型
    if v := os.getenv("MODEL_ID"):
        cfg.model_id = v

    # base_url —— 多个环境变量名共用
    for ev in ("ANTHROPIC_BASE_URL", "OPENAI_BASE_URL"):
        if v := os.getenv(ev):
            cfg.base_url = v
            break

    # api_key —— 取第一个已设置的值
    for ev in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY"):
        if v := os.getenv(ev):
            cfg.api_key = v
            break

    # 单次模型响应的最大令牌数（max_tokens）
    if v := os.getenv("MAX_TOKENS"):
        try:
            cfg.max_tokens = int(v)
        except ValueError:
            pass

    # 系统提示词
    if v := os.getenv("SYSTEM_PROMPT"):
        cfg.system_prompt = v

    # provider 提示
    if v := os.getenv("CODING_AGENT_PROVIDER"):
        cfg.provider = v

    # 工作目录
    if v := os.getenv("CODING_AGENT_WORKDIR"):
        cfg.workdir = v
