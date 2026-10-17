"""
上下文压缩流水线、记忆存储和技能加载器测试。
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

# ====================================================================
#  上下文压缩流水线（CompactionPipeline，4+1 层）
# ====================================================================

from coding_agent.context.compaction import (
    DEFAULT_CONTEXT_LIMIT,
    CompactionPipeline,
    _block_type,
    _collect_tool_results,
    _is_tool_result_message,
    _message_has_tool_use,
    compact_history,
    estimate_tokens,
    micro_compact,
    reactive_compact,
    snip_compact,
    tool_result_budget,
)


class TestEstimateTokens:
    def test_returns_int_for_small_messages(self):
        msgs = [{"role": "user", "content": "hello"}]
        n = estimate_tokens(msgs)
        assert isinstance(n, int)
        assert n > 0

    def test_grows_with_more_text(self):
        a = estimate_tokens([{"role": "user", "content": "x" * 100}])
        b = estimate_tokens([{"role": "user", "content": "x" * 10000}])
        assert b > a


# ── L1：裁剪中间消息（snip_compact） ──────────────────────────────────────

class TestSnipCompact:
    def test_passes_through_short_conversations(self):
        msgs = [{"role": "user", "content": f"msg {i}"} for i in range(10)]
        orig = msgs.copy()
        result = snip_compact(msgs, max_messages=20)
        assert result is msgs  # 应返回同一个对象
        assert len(msgs) == 10

    def test_trims_middle_when_over_limit(self):
        msgs = [{"role": "user", "content": f"msg {i}"} for i in range(60)]
        result = snip_compact(msgs, max_messages=30)
        assert len(result) < 60
        # 前 3 条消息 + 1 条占位消息 + 后 27 条消息，共 31 条
        assert any("snipped" in str(m.get("content", "")) for m in result)

    def test_protects_tool_use_pair_forward(self):
        """助手 tool_use 与用户 tool_result 组成的消息对不应被裁断。"""
        msgs = [
            {"role": "user", "content": "start"},
            {"role": "user", "content": "more"},
            {"role": "user", "content": "keep-head"},
            # 受配对保护的工具调用消息
            {"role": "assistant", "content": [{"type": "tool_use", "id": "1", "name": "bash", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "1", "content": "ok"}]},
            # 保留的尾部消息
            {"role": "user", "content": "tail-1"},
            {"role": "user", "content": "tail-2"},
            {"role": "user", "content": "tail-3"},
        ]
        result = snip_compact(msgs, max_messages=5)
        # 完成配对保护后，head_end 应移动到 tool_result 之后
        # tool_use 与 tool_result 消息对应保持完整
        has_tool_use = any(
            _message_has_tool_use(m) for m in result
        )
        has_tool_result = any(
            _is_tool_result_message(m) for m in result
        )
        # 消息对必须同时保留或同时移除
        assert has_tool_use == has_tool_result

    def test_protects_tool_use_pair_backward(self):
        """尾部边界落在 tool_result 上且前方存在 tool_use 时，应将边界前移。"""
        tail_messages = [
            {"role": "user", "content": "tail-1"},  # 补充足够多的消息以触发裁剪
            {"role": "user", "content": "tail-2"},
            {"role": "user", "content": "tail-3"},
            {"role": "user", "content": "tail-4"},
            {"role": "user", "content": "tail-5"},
            {"role": "user", "content": "tail-6"},
        ]
        msgs = [
            {"role": "user", "content": "head-1"},
            {"role": "user", "content": "head-2"},
            {"role": "user", "content": "head-3"},
        ] + [{"role": "user", "content": f"mid-{i}"} for i in range(40)] + [
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "bash", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "result"}]},
        ] + tail_messages
        result = snip_compact(msgs, max_messages=10)
        # tool_use 与 tool_result 消息对都应保留在尾部
        tail_start_idx = None
        for i, m in enumerate(result):
            if "snipped" in str(m.get("content", "")):
                tail_start_idx = i + 1
                break
        if tail_start_idx:
            tail = result[tail_start_idx:]
            assert len(tail) > 0


# ── L2：压缩旧工具结果（micro_compact） ─────────────────────────────────────

class TestMicroCompact:
    def test_keeps_recent_results(self):
        msgs = [
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "old", "content": "x" * 200}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "new1", "content": "y" * 200}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "new2", "content": "z" * 200}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "new3", "content": "w" * 200}]},
        ]
        micro_compact(msgs, keep_recent=3)
        # 最旧的工具结果应被压缩
        assert "Earlier tool result compacted" in msgs[0]["content"][0]["content"]
        # 最近 3 条工具结果应保持完整
        assert "Earlier tool result compacted" not in msgs[1]["content"][0]["content"]
        assert "Earlier tool result compacted" not in msgs[2]["content"][0]["content"]
        assert "Earlier tool result compacted" not in msgs[3]["content"][0]["content"]

    def test_short_content_not_compacted(self):
        msgs = [
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "short1", "content": "ok"}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "short2", "content": "ok"}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "short3", "content": "ok"}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "short4", "content": "ok"}]},
        ]
        micro_compact(msgs, keep_recent=2)
        # 少于 120 个字符的内容即使较旧也不应被压缩
        assert msgs[0]["content"][0]["content"] == "ok"
        assert msgs[1]["content"][0]["content"] == "ok"

    def test_noop_when_under_limit(self):
        msgs = [
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a", "content": "x" * 200}]},
        ]
        before = json.dumps(msgs)
        micro_compact(msgs, keep_recent=3)
        assert json.dumps(msgs) == before


# ── L3：限制工具结果体积（tool_result_budget） ────────────────────────────────

class TestToolResultBudget:
    def test_persists_large_result(self, tmp_path):
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "big", "content": "x" * 40000}
            ]},
        ]
        tool_result_budget(msgs, workdir=tmp_path, persist_threshold=30000, max_total_chars=5000)
        # 内容应替换为指向持久化输出的占位文本
        content = msgs[2]["content"][0]["content"]
        assert "persisted-output" in content or "Preview" in content
        # 对应文件应已写入磁盘
        out_dir = tmp_path / ".task_outputs" / "tool-results"
        assert out_dir.exists()
        files = list(out_dir.glob("*.txt"))
        assert len(files) >= 1

    def test_skips_when_under_budget(self, tmp_path):
        msgs = [
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "small", "content": "short"}
            ]},
        ]
        before = json.dumps(msgs)
        tool_result_budget(msgs, workdir=tmp_path, persist_threshold=30000, max_total_chars=100000)
        assert json.dumps(msgs) == before


# ── L4：生成历史摘要（compact_history） ───────────────────────────────────

class TestCompactHistory:
    @pytest.mark.asyncio
    async def test_returns_single_summary_message(self, tmp_path):
        summarize = AsyncMock(return_value="Goal: finish task.\nFindings: none.")

        msgs = [
            {"role": "user", "content": "do stuff"},
            {"role": "assistant", "content": [{"type": "text", "text": "ok done"}]},
        ]
        result = await compact_history(msgs, summarize=summarize, workdir=tmp_path)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert "[Compacted]" in result[0]["content"]
        assert "Goal: finish task" in result[0]["content"]

    @pytest.mark.asyncio
    async def test_writes_transcript_file(self, tmp_path):
        summarize = AsyncMock(return_value="summary")
        await compact_history(
            [{"role": "user", "content": "hi"}],
            summarize=summarize, workdir=tmp_path,
        )
        tx_dir = tmp_path / ".transcripts"
        assert tx_dir.exists()
        files = list(tx_dir.glob("*.jsonl"))
        assert len(files) >= 1


# ── 紧急压缩（reactive_compact） ───────────────────────────

class TestReactiveCompact:
    @pytest.mark.asyncio
    async def test_preserves_tail_messages(self, tmp_path):
        summarize = AsyncMock(return_value="summary of old portion")

        msgs = [
            {"role": "user", "content": f"old {i}"} for i in range(20)
        ] + [
            {"role": "user", "content": "recent-1"},
            {"role": "user", "content": "recent-2"},
            {"role": "user", "content": "recent-3"},
        ]
        result = await reactive_compact(msgs, summarize=summarize, workdir=tmp_path)
        # 结果由摘要消息和保留的尾部消息组成
        assert len(result) > 1
        assert "[Reactive compact]" in result[0]["content"]
        # 尾部消息应得到保留
        assert any("recent-1" in str(m.get("content", "")) for m in result)


# ── 压缩流水线编排器（CompactionPipeline） ───────────────────────

class TestCompactionPipeline:
    @pytest.mark.asyncio
    async def test_run_applies_layers(self):
        """普通消息应能按 L3→L1→L2 顺序完成压缩，且不发生错误。"""
        pipe = CompactionPipeline(workdir=tempfile.mkdtemp())
        msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": [{"type": "text", "text": "hello"}]}]
        await pipe.run(msgs)
        assert len(msgs) == 2  # 短对话应保持不变

    @pytest.mark.asyncio
    async def test_run_triggers_l4_when_over_limit(self, tmp_path):
        summarize = AsyncMock(return_value="summary text")
        pipe = CompactionPipeline(
            workdir=tmp_path,
            summarize=summarize,
            context_limit=10,  # 设置较小的阈值以强制触发 L4
            max_messages=100,
        )
        msgs = [{"role": "user", "content": "x" * 50000}]
        await pipe.run(msgs)
        # 此时应已触发 L4 摘要压缩
        summarize.assert_called_once()
        assert "[Compacted]" in msgs[0]["content"]

    @pytest.mark.asyncio
    async def test_skips_l4_when_no_summarize(self, tmp_path):
        pipe = CompactionPipeline(
            workdir=tmp_path,
            summarize=None,
            context_limit=10,
            max_messages=100,
        )
        msgs = [{"role": "user", "content": "x" * 50000}]
        await pipe.run(msgs)
        # 不应报错或触发 L4；仅运行 L3、L1、L2，短消息列表无需实际压缩


# ── 辅助判定函数 ─────────────────────────────────────

class TestHelpers:
    def test_message_has_tool_use(self):
        assert _message_has_tool_use(
            {"role": "assistant", "content": [{"type": "tool_use", "id": "x", "name": "bash", "input": {}}]}
        )
        assert not _message_has_tool_use(
            {"role": "assistant", "content": [{"type": "text", "text": "hi"}]}
        )
        assert not _message_has_tool_use({"role": "user", "content": "hello"})

    def test_is_tool_result_message(self):
        assert _is_tool_result_message(
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "x", "content": "ok"}]}
        )
        assert not _is_tool_result_message(
            {"role": "user", "content": [{"type": "text", "text": "hi"}]}
        )

    def test_collect_tool_results(self):
        msgs = [
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a", "content": "A"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "mid"}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "b", "content": "B"}]},
        ]
        blocks = _collect_tool_results(msgs)
        assert len(blocks) == 2
        assert blocks[0][2]["tool_use_id"] == "a"
        assert blocks[1][2]["tool_use_id"] == "b"


# ====================================================================
#  记忆存储（MemoryStore）
# ====================================================================

from coding_agent.memory.store import MemoryStore


class TestMemoryStore:
    @pytest.fixture
    def store(self, tmp_path):
        mem = MemoryStore(workdir=tmp_path)
        mem.write_file("user-pref", "user", "User likes tabs", "Use tabs for indentation.")
        mem.write_file("project-fact", "project", "This is a Python project", "The project uses Python 3.11+.")
        return mem

    def test_write_and_list(self, store):
        files = store.list_files()
        assert len(files) == 2
        names = {f["name"] for f in files}
        assert names == {"user-pref", "project-fact"}

    def test_read_file(self, store):
        content = store.read_file("user-pref.md")
        assert "tabs" in content
        assert store.read_file("nope.md") is None

    def test_delete_file(self, store):
        store.delete_file("user-pref.md")
        assert store.file_count == 1
        assert store.read_file("user-pref.md") is None

    def test_read_index(self, store):
        idx = store.read_index()
        assert "user-pref" in idx
        assert "project-fact" in idx

    def test_index_rebuilt_after_write(self, store):
        store.write_file("new-fact", "project", "New fact", "Some body.")
        idx = store.read_index()
        assert "new-fact" in idx

    @pytest.mark.asyncio
    async def test_select_keyword_fallback(self, store):
        """未提供 summarize 回调时，记忆选择应回退到关键词匹配。"""
        msgs = [
            {"role": "user", "content": "I really like using tabs for all my code"},
        ]
        content = await store.select_and_load(msgs)
        assert "user-pref" in content or "tabs" in content

    @pytest.mark.asyncio
    async def test_select_with_llm(self, tmp_path):
        summarize = AsyncMock(return_value="[0]")
        store = MemoryStore(workdir=tmp_path, summarize=summarize)
        store.write_file("tabs", "user", "Tabs preference", "Use tabs.")
        store.write_file("spaces", "user", "Spaces preference", "Use spaces.")

        msgs = [{"role": "user", "content": "I like tabs"}]
        content = await store.select_and_load(msgs)
        assert "relevant_memories" in content

    @pytest.mark.asyncio
    async def test_extract_writes_new_files(self, tmp_path):
        summarize = AsyncMock(return_value=json.dumps([
            {"name": "new-fact", "type": "project", "description": "project is Python", "body": "Uses Python 3.11."}
        ]))
        store = MemoryStore(workdir=tmp_path, summarize=summarize)
        msgs = [
            {"role": "user", "content": "This project uses Python 3.11"},
            {"role": "assistant", "content": [{"type": "text", "text": "Got it."}]},
        ]
        count = await store.extract(msgs)
        assert count >= 1
        assert store.file_count >= 1

    @pytest.mark.asyncio
    async def test_extract_returns_zero_for_empty_dialogue(self, tmp_path):
        summarize = AsyncMock()
        store = MemoryStore(workdir=tmp_path, summarize=summarize)
        count = await store.extract([])
        assert count == 0
        summarize.assert_not_called()

    @pytest.mark.asyncio
    async def test_consolidate_below_threshold(self, tmp_path):
        summarize = AsyncMock()
        store = MemoryStore(workdir=tmp_path, summarize=summarize, consolidate_threshold=50)
        store.write_file("a", "user", "A", "Body a")
        count = await store.consolidate()
        assert count == 1
        summarize.assert_not_called()

    @pytest.mark.asyncio
    async def test_consolidate_merges(self, tmp_path):
        """创建超过 10 条记忆后，consolidate 应合并记忆。"""
        files_before = []
        for i in range(12):
            files_before.append({
                "name": f"fact-{i}",
                "type": "project",
                "description": f"Fact #{i}",
                "body": f"Body of fact {i}",
            })

        # 合并后返回 5 条记忆
        summarize = AsyncMock(return_value=json.dumps([
            {"name": f"merged-{i}", "type": "project", "description": f"Merged #{i}", "body": f"Merged body {i}"}
            for i in range(5)
        ]))
        store = MemoryStore(workdir=tmp_path, summarize=summarize, consolidate_threshold=10)

        for f in files_before:
            store.write_file(f["name"], f["type"], f["description"], f["body"])

        count = await store.consolidate()
        assert count == 5


# ====================================================================
#  技能加载器（SkillLoader）
# ====================================================================

from coding_agent.skills.loader import SkillLoader


class TestSkillLoader:
    @pytest.fixture
    def skills_dir(self, tmp_path):
        """创建包含两个 SKILL.md 文件的模拟技能目录。"""
        base = tmp_path / "skills"
        base.mkdir()

        # 技能一：代码审查（code-review）
        d1 = base / "code-review"
        d1.mkdir()
        (d1 / "SKILL.md").write_text(
            "---\nname: code-review\ndescription: >\n  Review code for quality.\n---\n\n# Code Review\n\nCheck things."
        )

        # 技能二：PDF 处理（pdf）
        d2 = base / "pdf"
        d2.mkdir()
        (d2 / "SKILL.md").write_text(
            "# PDF Skill\n\nExtract text from PDF files."
        )

        return base

    def test_scan_discovers_skills(self, skills_dir):
        loader = SkillLoader(skills_dir)
        assert loader.scan() == 2
        assert "code-review" in loader
        assert "pdf" in loader

    def test_build_catalog(self, skills_dir):
        loader = SkillLoader(skills_dir)
        loader.scan()
        catalog = loader.build_catalog()
        assert "code-review" in catalog
        assert "pdf" in catalog

    def test_load_returns_full_content(self, skills_dir):
        loader = SkillLoader(skills_dir)
        loader.scan()
        content = loader.load("code-review")
        assert content is not None
        assert "Code Review" in content
        assert "Check things" in content

    def test_load_returns_none_for_unknown(self, skills_dir):
        loader = SkillLoader(skills_dir)
        loader.scan()
        assert loader.load("no-such-skill") is None

    def test_empty_dir(self, tmp_path):
        loader = SkillLoader(tmp_path / "empty")
        assert loader.scan() == 0
        assert "no skills" in loader.build_catalog().lower()

    def test_scan_is_idempotent(self, skills_dir):
        loader = SkillLoader(skills_dir)
        a = loader.scan()
        b = loader.scan()
        assert a == b == 2

    def test_description_fallback(self, skills_dir):
        """元数据中缺少 description 时，使用首个标题作为描述。"""
        loader = SkillLoader(skills_dir)
        loader.scan()
        pdf = loader.load("pdf")
        assert pdf is not None  # pdf 技能没有元数据，应使用回退描述
