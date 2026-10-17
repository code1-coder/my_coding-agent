"""
技能加载系统 —— 两级按需加载。

第一级（轻量 —— 始终加载）
    启动时，``SkillLoader`` 扫描 ``skills/`` 目录寻找
    ``SKILL.md`` 文件。每个技能的 **名称** 和 **描述** 被注入
    系统提示词（约 100 tokens/技能），让 agent 了解哪些技能可用，
    而无需加载完整内容。

第二级（重量 —— 按需加载）
    agent 将 ``load_skill("code-review")`` 作为工具调用，
    返回完整的 SKILL.md 内容（约 2000 tokens/技能）。

目录布局::

    skills/
      code-review/
        SKILL.md
      my-skill/
        SKILL.md
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

try:
    import yaml as _yaml
except ImportError:
    _yaml = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Frontmatter 解析器
# ---------------------------------------------------------------------------

def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """从 markdown 正文中分离 YAML frontmatter ``---\\n...\\n---``。"""
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
            # 回退到简单的逐行解析
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


# ---------------------------------------------------------------------------
#  技能条目（SkillEntry）
# ---------------------------------------------------------------------------

class SkillEntry:
    """一个技能的元数据 + 完整内容。"""

    __slots__ = ("name", "description", "content", "directory")

    def __init__(
        self,
        name: str,
        description: str,
        content: str,
        directory: Path,
    ) -> None:
        self.name = name
        self.description = description
        self.content = content
        self.directory = directory

    def __repr__(self) -> str:
        return f"SkillEntry({self.name!r})"


# ---------------------------------------------------------------------------
#  技能加载器（SkillLoader）
# ---------------------------------------------------------------------------

class SkillLoader:
    """从目录树中扫描、列出和加载技能。

    用法::

        loader = SkillLoader(skills_dir="skills")
        loader.scan()                         # 发现所有 SKILL.md 文件

        # L1 —— 用于系统提示词的目录
        system_prompt += loader.build_catalog()

        # L2 —— 按需完整加载
        content = loader.load("code-review")  # → str 或 None
    """

    def __init__(self, skills_dir: str | Path = "skills") -> None:
        self._skills_dir = Path(skills_dir)
        self._registry: dict[str, SkillEntry] = {}

    # ── 扫描 ────────────────────────────────────────────────

    def scan(self) -> int:
        """（重新）扫描技能目录。返回找到的技能数量。

        幂等 —— 再次调用会刷新注册表。
        """
        self._registry.clear()

        if not self._skills_dir.exists():
            logger.debug("技能目录未找到: %s", self._skills_dir)
            return 0

        for child in sorted(self._skills_dir.iterdir()):
            if not child.is_dir():
                continue

            manifest = child / "SKILL.md"
            if not manifest.exists():
                continue

            try:
                raw = manifest.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                logger.warning("无法读取 %s: %s", manifest, exc)
                continue

            meta, _body = _parse_frontmatter(raw)

            name = meta.get("name", child.name)
            desc = meta.get(
                "description",
                # 回退：原始内容中的第一个非标题行
                raw.split("\n")[0].lstrip("#").strip(),
            )

            entry = SkillEntry(
                name=name,
                description=desc,
                content=raw,
                directory=child,
            )
            self._registry[name] = entry
            logger.debug("已注册技能: %s", name)

        return len(self._registry)

    # ── L1：目录（用于系统提示词）───────────────────────────

    def build_catalog(self) -> str:
        """返回技能名称 + 描述的 Markdown 列表。

        适合注入系统提示词::

            可用技能:
            - **code-review**: 审查代码中的 bug、风格和安全问题。
            - **pdf**: 从 PDF 文件中提取文本和元数据。
        """
        if not self._registry:
            return "(no skills found)"

        lines: list[str] = []
        for entry in self._registry.values():
            lines.append(f"- **{entry.name}**: {entry.description}")
        return "\n".join(lines)

    # ── L2：完整内容加载 ───────────────────────────────────

    def load(self, name: str) -> str | None:
        """返回 *name* 的完整 SKILL.md 内容，或 None。"""
        entry = self._registry.get(name)
        if entry is None:
            logger.warning("未找到技能: %s", name)
            return None
        return entry.content

    # ── 内省 ───────────────────────────────────────────────

    @property
    def names(self) -> list[str]:
        """已注册技能名称的排序列表。"""
        return sorted(self._registry)

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def __len__(self) -> int:
        return len(self._registry)

    def __iter__(self):
        return iter(self._registry.values())
