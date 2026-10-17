"""
记忆系统演示：观察跨会话记忆流水线的三个处理阶段。

本示例调用实际的 ``MemoryStore``（``src/coding_agent/memory/store.py``），
依次演示以下三个阶段：

    阶段一  select_and_load —— 收到用户输入后注入相关记忆
    阶段二  extract —— 对话轮次结束后将新记忆写入磁盘
    阶段三  consolidate —— 记忆数量达到阈值后合并并清理

支持两种运行模式：

1. 默认模式：使用结果确定的模拟模型，无需 API 密钥，可离线运行：

       python examples/memory_demo.py

2. 在线模式：调用已配置的真实模型，读取 config.yaml 和环境变量中的 API 密钥：

       python examples/memory_demo.py --live

如需保留生成的文件以便检查：

       python examples/memory_demo.py --keep --workdir ./demo_memory

``.memory/`` 目录会创建在 ``workdir`` 内部，即智能体使用的项目
根目录，与智能体实际保存记忆的位置一致。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

# Windows 控制台通常默认使用 GBK，此处强制使用 UTF-8，避免 Unicode 输出
# 中的制表符号和箭头引发 UnicodeEncodeError。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

# 从项目根目录运行时，确保可以导入 coding_agent 包
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coding_agent.memory.store import MemoryStore


# ═══════════════════════════════════════════════════════════════
#  输出结果确定的模拟模型
# ═══════════════════════════════════════════════════════════════

class MockSummarizer:
    """替代记忆处理三个阶段所需的模型调用。

    ``MemoryStore`` 调用 ``summarize(prompt)`` 并期望返回 JSON 字符串。
    此处根据提示词内容判断当前处理阶段，
    再返回相应的模拟 JSON 数据。换成真实模型服务即可在线运行，
    无需调整流水线中的其他部分。
    """

    # 从演示对话中提取记忆记录所使用的匹配规则
    # 各字段依次为：正则表达式、名称、类型、描述、正文
    EXTRACT_PATTERNS: list[tuple[re.Pattern, str, str, str, str]] = [
        (
            re.compile(r"Python\s*3\.11", re.I),
            "project-python-311", "project",
            "Project runs on Python 3.11 + FastAPI",
            "The project is built on Python 3.11 with FastAPI. "
            "**Why:** stated by the user in the first turn. "
            "**How to apply:** target 3.11-compatible code and use FastAPI idioms.",
        ),
        (
            re.compile(r"tabs over spaces", re.I),
            "user-prefers-tabs", "user",
            "User prefers tabs over spaces (4-width)",
            "The user said they prefer tabs over spaces, 4-space width. "
            "**Why:** personal preference. "
            "**How to apply:** use tabs for indentation in all generated code.",
        ),
        (
            re.compile(r"single Docker container", re.I),
            "deploy-single-container", "feedback",
            "Deploy as a single Docker container, no Kubernetes",
            "Deployment constraint: one Docker container, no Kubernetes. "
            "**Why:** user guidance in turn 2. "
            "**How to apply:** produce a single-stage Dockerfile; skip k8s manifests.",
        ),
        (
            re.compile(r"pip-tools", re.I),
            "pin-deps-pip-tools", "project",
            "Pin dependencies with pip-tools",
            "All dependencies are pinned using pip-tools. "
            "**Why:** project fact from turn 2. "
            "**How to apply:** use requirements.in / pip-compile workflow.",
        ),
    ]

    # ── 阶段二：提取记忆 ─────────────────────────────────

    def _answer_extract(self, prompt: str) -> str:
        """定位对话内容，并从中提取记忆。"""
        dialogue = prompt.split("Dialogue:", 1)[-1]
        found: list[dict[str, str]] = []
        for pattern, name, mem_type, desc, body in self.EXTRACT_PATTERNS:
            if pattern.search(dialogue) and not any(
                m["name"] == name for m in found
            ):
                found.append({
                    "name": name,
                    "type": mem_type,
                    "description": desc,
                    "body": body,
                })
        return json.dumps(found)

    # ── 阶段一：选择记忆 ──────────────────────────────────

    def _answer_select(self, prompt: str) -> str:
        """选择与近期对话存在共同关键词的记忆目录条目。"""
        # 提示词的结构为： "...Recent conversation:\n{recent}\n\nMemory catalog:\n{catalog}"
        body = prompt.split("Recent conversation:", 1)[-1]
        recent = body.split("Memory catalog:", 1)[0].lower()
        catalog_part = body.split("Memory catalog:", 1)[-1]

        # 截取单词前 6 个字符，使 preference 与 prefers 等词形能够匹配
        def keys(text: str) -> set[str]:
            return {w[:6] for w in re.findall(r"[a-z]+", text) if len(w) > 3}

        recent_keys = keys(recent)

        picked: list[int] = []
        for line in catalog_part.splitlines():
            line = line.strip()
            m = re.match(r"(\d+):\s*(.*)", line)
            if not m:
                continue
            idx, desc = int(m.group(1)), m.group(2).lower()
            # 使用共同关键词进行启发式匹配；真实模型可提供更准确的语义判断
            if recent_keys & keys(desc):
                picked.append(idx)
        return json.dumps(picked)

    # ── 阶段三：合并记忆 ──────────────────────────────

    def _answer_consolidate(self, prompt: str) -> str:
        """合并重复记忆，优先保留用户偏好，最多保留 5 条。

        遵循 ``store.consolidate`` 中实际使用的模型合并规则：
        1. 移除过期或临时信息，如本示例中的 ``scratch-fact-*``。
        2. 根据描述去重并合并重复记录。
        3. 优先保留重要的用户偏好。
        4. 总数不超过 ``max_memories``，本示例中为 5 条。
        """
        # 解析存储层写入提示词的内容块，格式为
        # 内容块格式：“## file\nname: ...\ntype: ...\ndescription: ...\nbody”。
        records: list[dict[str, str]] = []
        for block in re.split(r"^##\s", prompt, flags=re.M):
            lines = block.splitlines()
            name = typ = desc = None
            desc_line_idx = None
            for i, line in enumerate(lines):
                if line.startswith("name:") and name is None:
                    name = line.split(":", 1)[1].strip()
                elif line.startswith("type:") and typ is None:
                    typ = line.split(":", 1)[1].strip()
                elif line.startswith("description:"):
                    desc = line.split(":", 1)[1].strip()
                    desc_line_idx = i
            if name and desc is not None and desc_line_idx is not None:
                # 正文为 description 所在行之后的全部内容
                body = "\n".join(lines[desc_line_idx + 1:]).strip()
                records.append({
                    "name": name,
                    "type": typ or "user",
                    "description": desc,
                    "body": body,
                })

        # 规则一：优先移除临时或过期信息
        kept = [
            r for r in records
            if not r["name"].startswith("scratch-")
            and "scratch" not in r["description"].lower()
        ]

        # 规则三和四：按用户、反馈、项目、参考信息排序，最多保留 5 条
        order = {"user": 0, "feedback": 1, "project": 2, "reference": 3}
        seen: set[str] = set()
        merged: list[dict[str, str]] = []
        for rec in sorted(kept, key=lambda r: order.get(r["type"], 9)):
            key = rec["description"].lower()
            if key in seen:  # 规则二：合并重复记录
                continue
            seen.add(key)
            merged.append(rec)
            if len(merged) >= 5:
                break
        return json.dumps(merged)

    async def __call__(self, prompt: str) -> str:
        if "Extract user preferences" in prompt:
            return self._answer_extract(prompt)
        if "select the indices of memories" in prompt:
            return self._answer_select(prompt)
        if "Consolidate the following memory files" in prompt:
            return self._answer_consolidate(prompt)
        return "[]"


# ═══════════════════════════════════════════════════════════════
#  在线模式：将配置的模型服务封装为摘要回调
# ═══════════════════════════════════════════════════════════════

def build_live_summarizer(cfg) -> callable:
    """返回调用真实模型的异步摘要回调，接口为 ``async (prompt) -> str``。"""
    from coding_agent.providers import AnthropicProvider, OpenAICompatProvider

    provider: OpenAICompatProvider | AnthropicProvider
    if cfg.provider == "anthropic":
        provider = AnthropicProvider(
            model_id=cfg.model_id, api_key=cfg.api_key, base_url=cfg.base_url
        )
    else:
        provider = OpenAICompatProvider(
            model_id=cfg.model_id, api_key=cfg.api_key, base_url=cfg.base_url
        )

    async def summarize(prompt: str) -> str:
        resp = await provider.create_message(
            messages=[{"role": "user", "content": prompt}],
            system=(
                "You are the memory subsystem of a coding agent. "
                "Answer the prompt's instruction exactly — usually a JSON "
                "array of memory records or indices. No commentary."
            ),
            max_tokens=2000,
        )
        return resp.text

    return summarize


# ═══════════════════════════════════════════════════════════════
#  输出辅助函数
# ═══════════════════════════════════════════════════════════════

def show_memory_dir(workdir: Path) -> None:
    """打印 ``.memory/`` 目录树及其文件内容。"""
    mem_dir = workdir / ".memory"
    print(f"    └─ .memory/  ({len(list(mem_dir.glob('*.md')))} files)")
    for f in sorted(mem_dir.glob("*.md")):
        print(f"       ├─ {f.name}")
        for line in f.read_text(encoding="utf-8").splitlines():
            print(f"       │   {line}")
    print()


def show_injection(injected: str) -> None:
    """打印注入上下文的记忆内容块。"""
    if not injected:
        print("       (no relevant memories selected)\n")
        return
    for line in injected.splitlines():
        print(f"       {line}")
    print()


# ═══════════════════════════════════════════════════════════════
#  演示脚本
# ═══════════════════════════════════════════════════════════════

async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true",
                        help="call the real configured LLM instead of the mock")
    parser.add_argument("--workdir", default=None,
                        help="project root for the demo (default: temp dir)")
    parser.add_argument("--keep", action="store_true",
                        help="keep workdir (default: delete it on exit)")
    args = parser.parse_args()

    # ── 初始化工作目录 ─────────────────────────────────────
    if args.workdir:
        workdir = Path(args.workdir)
        if workdir.exists():
            shutil.rmtree(workdir)
        workdir.mkdir(parents=True)
    else:
        workdir = Path(tempfile.mkdtemp(prefix="memory-demo-"))

    print("=" * 70)
    print("Memory system demo — 3-stage pipeline (select / extract / consolidate)")
    print(f"workdir : {workdir}")
    print(f"mode    : {'LIVE (real LLM)' if args.live else 'deterministic mock'}")
    print("=" * 70)

    # ── 选择摘要生成器 ─────────────────────────────────
    if args.live:
        from coding_agent.config import load_config
        cfg = load_config("config.yaml")
        if not cfg.api_key:
            sys.exit("Error: --live requires an API key (ANTHROPIC_API_KEY / OPENAI_API_KEY / DEEPSEEK_API_KEY)")
        summarize = build_live_summarizer(cfg)
    else:
        summarize = MockSummarizer()

    # ── 使用实际的记忆存储和文件 ──────────────────────────────
    store = MemoryStore(workdir=workdir, summarize=summarize)

    # ---------------------------------------------------------
    # 第一轮：用户提供项目事实和个人偏好
    # ---------------------------------------------------------
    print("\n▶ TURN 1  (user: project facts + preference)")
    print("    user: \"This project uses Python 3.11 with FastAPI. "
          "Also, I prefer tabs over spaces, 4-space width.\"")
    print("    assistant: \"Got it — I'll keep that in mind.\"\n")

    turn1 = [
        {"role": "user", "content": (
            "This project uses Python 3.11 with FastAPI. "
            "Also, I prefer tabs over spaces, 4-space width."
        )},
        {"role": "assistant", "content": [{"type": "text", "text": "Got it — I'll keep that in mind."}]},
    ]
    n1 = await store.extract(turn1)          # 阶段二：轮次结束后提取记忆
    print(f"    extract() → wrote {n1} new memory file(s).  .memory/ now:\n")
    show_memory_dir(workdir)

    # ---------------------------------------------------------
    # 第二轮：用户补充部署约束
    # ---------------------------------------------------------
    print("▶ TURN 2  (user: deployment constraints)")
    print("    user: \"For deployment, use a single Docker container. "
          "No Kubernetes, keep it simple. Pin deps with pip-tools.\"")
    print("    assistant: \"Understood — one container, pip-tools.\"\n")

    turn2 = [
        {"role": "user", "content": (
            "For deployment, use a single Docker container. "
            "No Kubernetes, keep it simple. Pin deps with pip-tools."
        )},
        {"role": "assistant", "content": [{"type": "text", "text": "Understood — one container, pip-tools."}]},
    ]
    n2 = await store.extract(turn2)
    print(f"    extract() → wrote {n2} new memory file(s).  .memory/ now:\n")
    show_memory_dir(workdir)

    # ---------------------------------------------------------
    # 阶段一：开启新会话并重新注入记忆
    # ---------------------------------------------------------
    print("▶ STAGE 1 — select_and_load (a later, unrelated session)")
    print("    user: \"When you write the FastAPI router, remember my indentation preference.\"")
    print("    → memory injected into context:\n")

    new_session = [{"role": "user", "content": (
        "When you write the FastAPI router, remember my indentation preference."
    )}]
    injected = await store.select_and_load(new_session)
    show_injection(injected)

    # ---------------------------------------------------------
    # 阶段三：超过记忆数量阈值后执行合并
    # ---------------------------------------------------------
    print("▶ STAGE 3 — consolidate (threshold reached)")
    print(f"    current memory count: {store.file_count} "
          f"(threshold: {store.consolidate_threshold})\n")
    print("    seeding 8 more memories to cross the threshold...\n")

    for i in range(8):
        store.write_file(
            f"scratch-fact-{i}", "project",
            f"Scratch fact #{i}",
            f"Body of scratch fact {i}.",
        )
    print(f"    memory count now: {store.file_count} ≥ {store.consolidate_threshold} → consolidate():\n")
    final_count = await store.consolidate()
    print(f"    consolidate() → {final_count} memory file(s) after merge/prune.  .memory/ now:\n")
    show_memory_dir(workdir)

    # ---------------------------------------------------------
    # 演示收尾
    # ---------------------------------------------------------
    print("=" * 70)
    if args.keep and args.workdir:
        print(f"Generated files kept at: {workdir}\\.memory\\")
    else:
        shutil.rmtree(workdir, ignore_errors=True)
        print("Temp workspace cleaned up. Use --keep --workdir DIR to inspect files.")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
