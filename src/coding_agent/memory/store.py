"""
跨会话记忆系统 —— 三个阶段。

存储布局::

    .memory/
      MEMORY.md            ← 索引（每个记忆文件一行）
      user-pref-tabs.md    ← 单个记忆文件（YAML frontmatter + markdown）
      project-facts.md
      ...

每个记忆文件使用以下格式::

    ---
    name: user-preference-tabs
    description: >
      用户更喜欢用 tab 而非空格，宽度为 4。
    type: user
    ---

    用户说"我喜欢 tab"。所有生成的代码中都要使用 tab。
    **原因：** 个人偏好。**应用方式：** 写文件时使用 tab。

记忆处理的三个阶段：
    1. **select_relevant_memories**  — 每次用户输入后：将最近对话
       与记忆索引匹配 → 返回相关文件名。
    2. **extract_memories**          — 每轮对话结束后（stop_reason
       ≠ tool_use 时）：检查压缩前的消息快照，提取新的
       用户偏好 / 项目事实。
    3. **consolidate_memories**      — 记忆数量 ≥ 阈值（10）时：
       LLM 合并重复项，删除过时条目，目标 ≤ 30。

所有需要 LLM 的阶段都接受 ``summarize`` 回调
（``async (prompt: str) → str``）。如果省略，则阶段降级为
基于关键词匹配 / 空操作。
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Callable, Awaitable

try:
    import yaml as _yaml
except ImportError:
    _yaml = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  公开类型
# ---------------------------------------------------------------------------

SummarizeFn = Callable[[str], Awaitable[str]]

MEMORY_TYPES = ("user", "feedback", "project", "reference")

# ---------------------------------------------------------------------------
#  常量
# ---------------------------------------------------------------------------

MEMORY_DIR_NAME = ".memory"
MEMORY_INDEX_NAME = "MEMORY.md"
DEFAULT_CONSO_THRESHOLD = 10        # 文件数 ≥ 此值时触发合并
DEFAULT_SELECT_MAX_ITEMS = 5        # 每轮最多注入的记忆数
DEFAULT_MAX_MEMORIES_AFTER_CONSOLIDATION = 30


# ---------------------------------------------------------------------------
#  YAML frontmatter 辅助函数
# ---------------------------------------------------------------------------

def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """将 ``---\\n...\\n---\\n...`` 解析为 (meta, body)。"""
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    meta: dict[str, Any] = {}
    if _yaml is not None:
        try:
            parsed = _yaml.safe_load(parts[1])
            if isinstance(parsed, dict):
                meta = parsed
        except _yaml.YAMLError:
            # 回退到简单的 key: value 解析
            for line in parts[1].strip().splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip().strip('"').strip("'")
    else:
        for line in parts[1].strip().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip().strip('"').strip("'")
    return meta, parts[2].strip()


def _build_frontmatter(meta: dict[str, Any]) -> str:
    """_parse_frontmatter 的逆操作 —— YAML 转储。"""
    if _yaml is not None:
        return _yaml.dump(meta, default_flow_style=False, allow_unicode=True).strip()
    return "\n".join(f"{k}: {v}" for k, v in meta.items())


# ---------------------------------------------------------------------------
#  记忆存储（MemoryStore）
# ---------------------------------------------------------------------------

class MemoryStore:
    """基于文件系统的持久化记忆。

    用法::

        store = MemoryStore(workdir=Path.cwd(), summarize=my_llm_fn)

        # 阶段 1：用户输入后，注入相关记忆
        relevant = await store.select_and_load(messages)

        # 阶段 2：每轮结束后，提取新记忆
        await store.extract(pre_compress_snapshot)

        # 阶段 3：定期合并
        await store.consolidate()
    """

    def __init__(
        self,
        *,
        workdir: Path | str | None = None,
        summarize: SummarizeFn | None = None,
        consolidate_threshold: int = DEFAULT_CONSO_THRESHOLD,
        select_max_items: int = DEFAULT_SELECT_MAX_ITEMS,
        max_memories: int = DEFAULT_MAX_MEMORIES_AFTER_CONSOLIDATION,
    ) -> None:
        wd = Path(workdir) if workdir else Path.cwd()
        self._mem_dir = wd / MEMORY_DIR_NAME
        self._mem_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._mem_dir / MEMORY_INDEX_NAME
        self.summarize = summarize
        self.consolidate_threshold = consolidate_threshold
        self.select_max_items = select_max_items
        self.max_memories = max_memories

    # ── 底层 I/O ───────────────────────────────────────────

    def write_file(self, name: str, mem_type: str, description: str, body: str) -> Path:
        """写入或覆盖一个记忆文件。之后重建索引。"""
        # 名称可能由模型生成，统一限制为跨平台安全文件名，保护索引文件。
        slug = re.sub(r"[^\w-]+", "-", name.lower()).strip("-_")
        if not slug:
            raise ValueError("记忆名称不能为空")
        if slug.upper() in {"MEMORY", "CON", "PRN", "AUX", "NUL", *(
            f"{prefix}{n}" for prefix in ("COM", "LPT") for n in range(1, 10)
        )}:
            slug = "note-" + slug
        filename = f"{slug}.md"
        filepath = self._mem_dir / filename
        front = _build_frontmatter({
            "name": name,
            "description": description,
            "type": mem_type,
        })
        filepath.write_text(f"---\n{front}\n---\n\n{body}\n", encoding="utf-8")
        self._rebuild_index()
        return filepath

    def delete_file(self, filename: str) -> None:
        """按文件名删除一个记忆文件（如 ``user-pref-tabs.md``）。"""
        path = self._mem_dir / filename
        if path.exists():
            path.unlink()
        self._rebuild_index()

    def read_file(self, filename: str) -> str | None:
        """读取单个记忆文件的完整内容。"""
        path = self._mem_dir / filename
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    def list_files(self) -> list[dict[str, Any]]:
        """返回 ``[{filename, name, description, type, body}, ...]``。"""
        result: list[dict[str, Any]] = []
        for f in sorted(self._mem_dir.glob("*.md")):
            if f.name == MEMORY_INDEX_NAME:
                continue
            raw = f.read_text(encoding="utf-8")
            meta, body = _parse_frontmatter(raw)
            result.append({
                "filename": f.name,
                "name": meta.get("name", f.stem),
                "description": str(meta.get("description", "")),
                "type": str(meta.get("type", "user")),
                "body": body,
            })
        return result

    @property
    def file_count(self) -> int:
        """记忆文件数量（不含索引）。"""
        return sum(
            1 for f in self._mem_dir.glob("*.md")
            if f.name != MEMORY_INDEX_NAME
        )

    # ── 索引 ───────────────────────────────────────────────

    def read_index(self) -> str:
        """返回 MEMORY.md 的文本内容（缺失时返回空字符串）。"""
        if not self._index_path.exists():
            return ""
        return self._index_path.read_text(encoding="utf-8").strip()

    def _rebuild_index(self) -> None:
        """扫描所有 .md 文件并重建 MEMORY.md。"""
        files = self.list_files()
        lines: list[str] = []
        for f in files:
            desc = f["description"].replace("\n", " ")[:120]
            lines.append(f"- [{f['name']}]({f['filename']}) — {desc}")
        self._index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ── 阶段 1：选择 & 加载 ────────────────────────────────

    async def select_and_load(
        self,
        messages: list[dict[str, Any]],
    ) -> str:
        """返回用于注入上下文的相关记忆内容。

        1. 构建所有已存储记忆的目录（name + description）。
        2. 从 *messages* 中提取最近的用户文本（≤ 2000 字符）。
        3. 请求 LLM 选择相关索引 → 返回 JSON 数组。
        4. LLM 失败时回退到 name+description 的关键词匹配。
        5. 读取选中的文件并返回拼接后的内容。
        """
        files = self.list_files()
        if not files:
            return ""

        recent = _extract_recent_user_text(messages, max_chars=2000)
        if not recent.strip():
            return ""

        selected: list[str] = []

        if self.summarize is not None:
            # ── 基于 LLM 的选择 ─────────────────────────────
            catalog_lines = [
                f"{i}: {f['name']} — {f['description']}"
                for i, f in enumerate(files)
            ]
            catalog = "\n".join(catalog_lines)

            prompt = (
                "根据以下最近对话和记忆目录，"
                "选择明确相关的记忆索引。"
                "只返回一个 JSON 整数数组，例如 [0, 3]。"
                "如果没有相关记忆，返回 []。\n\n"
                f"最近对话:\n{recent}\n\n"
                f"记忆目录:\n{catalog}"
            )

            try:
                resp = await self.summarize(prompt)
                match = re.search(r"\[.*?\]", resp.strip(), re.DOTALL)
                if match:
                    indices = json.loads(match.group())
                    for idx in indices:
                        if isinstance(idx, int) and 0 <= idx < len(files):
                            selected.append(files[idx]["filename"])
                            if len(selected) >= self.select_max_items:
                                break
            except Exception:
                logger.debug("LLM 记忆选择失败，回退到关键词匹配")

        # ── 回退 / 补充：关键词匹配 ────────────────────────────
        if not selected:
            keywords = [
                w.lower() for w in recent.split()
                if len(w) > 3 and w.lower() not in ("this", "that", "with", "from", "what", "when", "have", "your", "about")
            ]
            for f in files:
                text = (f["name"] + " " + f["description"]).lower()
                if any(kw in text for kw in keywords):
                    selected.append(f["filename"])
                    if len(selected) >= self.select_max_items:
                        break

        if not selected:
            return ""

        # ── 加载文件 ──────────────────────────────────────
        parts: list[str] = ["<relevant_memories>"]
        for filename in selected:
            content = self.read_file(filename)
            if content:
                parts.append(content)
        parts.append("</relevant_memories>")
        return "\n\n".join(parts)

    # ── 阶段 2：提取新记忆 ─────────────────────────────────

    async def extract(self, messages: list[dict[str, Any]]) -> int:
        """扫描 *messages* 寻找新记忆。返回写入的数量。

        *messages* 应为 **压缩前** 的快照，以便 LLM
        看到完整、未裁剪的上下文。

        模型返回的有效记录会写入记忆文件并重建索引；未能解析响应或
        生成摘要失败时返回 0，不向对话消息列表追加内容。
        """
        if self.summarize is None:
            return 0

        dialogue = _extract_dialogue(messages, max_lines=10)

        if not dialogue.strip():
            return 0

        existing = self.list_files()
        existing_desc = (
            "\n".join(f"- {m['name']}: {m['description']}" for m in existing)
        ) if existing else "(无)"

        prompt = (
            '从此对话中提取用户偏好、约束、项目事实，以及用户明确要求记住的信息。\n'
            '用户指定的助手名称、身份或称呼也属于应保存的用户指导，使用 feedback 类型。\n'
            '只记录用户明确提供或确认的信息，不把助手自行编造的回复当作事实。\n'
            '返回一个 JSON 数组。每项: {"name": "...", "type": "...", "description": "...", "body": "..."}。\n'
            '- name: 简短的 kebab-case 标识符（如 "user-preference-tabs"）\n'
            "- type: 'user'（用户偏好）、'feedback'（用户指导）、"
            "'project'（项目事实）、'reference'（外部引用/URL）之一\n"
            '- description: 用于索引查找的一句话摘要\n'
            '- body: 完整的 markdown 详细内容\n'
            '如果新信息更新了已有记忆，复用其 name 并写入最新内容，避免保存互相矛盾的条目。\n'
            "如果没有新内容或已被已有记忆覆盖，返回 []。\n\n"
            f"已有记忆:\n{existing_desc}\n\n"
            f"对话:\n{dialogue[:4000]}"
        )

        try:
            resp = await self.summarize(prompt)
            match = re.search(r"\[.*\]", resp.strip(), re.DOTALL)
            if not match:
                return 0
            items = json.loads(match.group())
            if not isinstance(items, list) or not items:
                return 0
        except Exception:
            logger.exception("记忆提取失败")
            return 0

        count = 0
        for mem in items:
            if not isinstance(mem, dict):
                continue
            if any(not isinstance(mem.get(key), str) or not mem[key].strip()
                   for key in ("name", "description", "body")):
                continue
            name = str(mem.get("name", f"memory_{int(time.time())}"))
            mem_type = str(mem.get("type", "user"))
            desc = str(mem.get("description", ""))
            body = str(mem.get("body", ""))
            if desc and body:
                self.write_file(name, mem_type, desc, body)
                count += 1

        if count:
            logger.info("提取了 %d 条新记忆", count)
        return count

    # ── 阶段 3：合并 ────────────────────────────────────────

    async def consolidate(self) -> int:
        """记忆数 ≥ 阈值时合并 / 清理记忆。返回最终数量。"""
        files = self.list_files()
        if len(files) < self.consolidate_threshold:
            return len(files)

        if self.summarize is None:
            logger.warning(
                "记忆合并阈值已达到（%d ≥ %d）但未配置 "
                "summarize 回调 —— 跳过",
                len(files), self.consolidate_threshold,
            )
            return len(files)

        catalog = "\n\n".join(
            f"## {f['filename']}\nname: {f['name']}\n"
            f"type: {f['type']}\ndescription: {f['description']}\n"
            f"{f['body']}"
            for f in files
        )

        prompt = (
            "合并以下记忆文件。规则：\n"
            "1. 将重复的记忆合并为一条\n"
            "2. 删除过时/矛盾的记忆\n"
            f"3. 总数控制在 {self.max_memories} 条以内\n"
            "4. 优先保留重要的用户偏好\n"
            '返回一个 JSON 数组。每项: {"name":..., "type":..., "description":..., "body":...}.\n\n'
            + catalog[:16000]
        )

        try:
            resp = await self.summarize(prompt)
            match = re.search(r"\[.*\]", resp.strip(), re.DOTALL)
            if not match:
                return len(files)
            items = json.loads(match.group())
        except Exception:
            logger.exception("记忆合并失败")
            return len(files)

        # 先校验完整结果。空数组或格式错误时保留原有记忆，避免误删。
        if not isinstance(items, list) or not items or any(
            not isinstance(mem, dict) or any(
                not isinstance(mem.get(key), str) or not mem[key].strip()
                for key in ("name", "description", "body")
            ) for mem in items
        ):
            return len(files)

        # 新集合全部写入成功后，才移除已被替代的旧文件。
        written = set()
        for mem in items:
            name = str(mem.get("name", f"memory_{int(time.time())}"))
            mem_type = str(mem.get("type", "user"))
            desc = str(mem.get("description", ""))
            body = str(mem.get("body", ""))
            if desc and body:
                written.add(self.write_file(name, mem_type, desc, body).name)
        for old in files:
            if old["filename"] not in written:
                (self._mem_dir / old["filename"]).unlink(missing_ok=True)
        self._rebuild_index()

        final = self.file_count
        logger.info("合并记忆 %d → %d 条", len(files), final)
        return final


# ---------------------------------------------------------------------------
#  文本辅助函数
# ---------------------------------------------------------------------------

def _extract_recent_user_text(
    messages: list[dict[str, Any]], max_chars: int = 2000
) -> str:
    """收集最近的 user 消息文本（从最新到最旧），上限 *max_chars* 字符。"""
    parts: list[str] = []
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            text = " ".join(
                str(getattr(b, "text", b.get("text", "")))
                for b in content
                if (isinstance(b, dict) and b.get("type") == "text")
                   or getattr(b, "type", None) == "text"
            )
        elif isinstance(content, str):
            text = content
        else:
            text = str(content)

        if text.strip():
            parts.append(text.strip())
        if len(" ".join(reversed(parts))) >= max_chars:
            break

    return " ".join(reversed(parts))


def _extract_dialogue(
    messages: list[dict[str, Any]], max_lines: int = 10
) -> str:
    """提取最后 *max_lines* 轮对话，格式为 ``role: text``。"""
    lines: list[str] = []
    for msg in messages[-max_lines:]:
        role = msg.get("role", "?")
        content = msg.get("content", "")
        if isinstance(content, list):
            text = " ".join(
                b.get("text", "") if isinstance(b, dict)
                else getattr(b, "text", "")
                for b in content
                if (isinstance(b, dict) and b.get("type") == "text")
                   or getattr(b, "type", None) == "text"
            )
        elif isinstance(content, str):
            text = content
        else:
            text = str(content)
        if text.strip():
            lines.append(f"{role}: {text.strip()}")
    return "\n".join(lines)
