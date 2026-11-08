"""Codex 风格的精确上下文补丁：先预检，再写入，写入异常时尝试回滚。"""

from __future__ import annotations

import os
import stat
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

from coding_agent.tools.base import Tool
from coding_agent.tools.builtin import _safe_path


_PATCH_LOCK = threading.RLock()


@dataclass
class PatchOperation:
    action: str
    path: str
    body: list[str]
    destination: str | None = None


def parse_patch(patch: str) -> list[PatchOperation]:
    if not isinstance(patch, str) or len(patch) > 1000000:
        raise ValueError("patch must be a string of at most 1000000 characters")
    lines = patch.splitlines()
    if len(lines) < 3 or lines[0] != "*** Begin Patch" or lines[-1] != "*** End Patch":
        raise ValueError("Expected *** Begin Patch and *** End Patch")
    operations = []
    position = 1
    while position < len(lines) - 1:
        header = lines[position]
        action = next((item for item in ("Add", "Update", "Delete")
                       if header.startswith(f"*** {item} File: ")), None)
        if action is None:
            raise ValueError(f"Invalid file header: {header}")
        path = header.split(": ", 1)[1]
        if not path.strip():
            raise ValueError("Patch path is empty")
        position += 1
        destination = None
        if action == "Update" and lines[position].startswith("*** Move to: "):
            destination = lines[position].removeprefix("*** Move to: ")
            position += 1
        body = []
        while position < len(lines) - 1:
            line = lines[position]
            if line.startswith(("*** Add File: ", "*** Update File: ", "*** Delete File: ")):
                break
            body.append(line)
            position += 1
        if action == "Delete" and body:
            raise ValueError("Delete File cannot contain a body")
        if action == "Add" and any(not line.startswith("+") for line in body):
            raise ValueError("Add File lines must start with +")
        if action == "Update" and not body:
            raise ValueError("Update File requires a hunk")
        operations.append(PatchOperation(action, path, body, destination))
        if len(operations) > 100:
            raise ValueError("At most 100 file operations per patch")
    return operations


def patch_path(workdir: Path, path: str) -> Path:
    relative = Path(path)
    parts = path.replace("\\", "/").split("/")
    if (not path.strip() or relative.is_absolute() or PureWindowsPath(path).drive
            or ".." in parts or ":" in path or any(part.casefold() == ".git" for part in parts)):
        raise ValueError(f"Patch requires a workspace-relative path outside .git: {path}")
    candidate = workdir / relative
    for ancestor in (candidate, *candidate.parents):
        if ancestor == workdir:
            break
        if ancestor.is_symlink() or (hasattr(ancestor, "is_junction") and ancestor.is_junction()):
            raise ValueError(f"Patch cannot follow symlinks or junctions: {path}")
    return _safe_path(workdir, path)


def _update_content(original: bytes, body: list[str]) -> bytes:
    bom = original.startswith(b"\xef\xbb\xbf")
    text = original.decode("utf-8-sig")
    newline = "\r\n" if "\r\n" in text else "\n"
    lines = text.splitlines()
    cursor = 0
    position = 0
    while position < len(body):
        header = body[position]
        if header != "@@" and not header.startswith("@@ "):
            raise ValueError("Update hunk must begin with @@ or @@ <exact anchor line>")
        if header != "@@":
            anchor = header[3:]
            anchors = [index for index in range(cursor, len(lines)) if lines[index] == anchor]
            if len(anchors) != 1:
                raise ValueError(f"Anchor must match exactly once: {anchor}")
            cursor = anchors[0] + 1
        position += 1
        before, after = [], []
        at_end = False
        changed = False
        while position < len(body) and not body[position].startswith("@@"):
            line = body[position]
            position += 1
            if line == "*** End of File":
                at_end = True
                if position < len(body) and not body[position].startswith("@@"):
                    raise ValueError("End of File must be the last line of a hunk")
                break
            if not line or line[0] not in " +-":
                raise ValueError(f"Invalid hunk line: {line}")
            if line[0] in " -":
                before.append(line[1:])
            if line[0] in " +":
                after.append(line[1:])
            changed = changed or line[0] in "+-"
        if not changed:
            raise ValueError("Hunk must add or remove a line")
        if before:
            candidates = [index for index in range(cursor, len(lines) - len(before) + 1)
                          if lines[index:index + len(before)] == before
                          and (not at_end or index + len(before) == len(lines))]
        else:
            if not at_end and header == "@@" and lines:
                raise ValueError("Insertion requires an anchor or *** End of File")
            candidates = [len(lines) if at_end else cursor]
        if len(candidates) != 1:
            raise ValueError("Patch context not found or ambiguous; read_file again and include more context")
        start = candidates[0]
        lines[start:start + len(before)] = after
        cursor = start + len(after)
    updated = newline.join(lines)
    if lines and (not original or text.endswith(("\n", "\r"))):
        updated += newline
    return (b"\xef\xbb\xbf" if bom else b"") + updated.encode("utf-8")


def _atomic_write(path: Path, content: bytes, mode: int | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".agent-patch-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(content)
        if mode is not None:
            os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class ApplyPatchTool(Tool):
    name = "apply_patch"
    description = ("应用精确上下文补丁。格式：*** Begin Patch / *** Add File: path（每行 +）/ "
                   "*** Update File: path（可选 *** Move to: path，@@ 后每行空格、- 或 +）/ "
                   "*** Delete File: path / *** End Patch。@@ <完整原行> 可定位；*** End of File 限定末尾。"
                   "路径相对工作区；上下文歧义则失败；先校验全部文件，保留现有换行符，拒绝覆盖新增/移动目标。")
    input_schema = {"type": "object", "properties": {"patch": {"type": "string"}},
                    "required": ["patch"]}

    def __init__(self, workdir: Path | str | None = None) -> None:
        self._workdir = Path(workdir or Path.cwd()).resolve()

    def execute(self, patch: str) -> str:
        with _PATCH_LOCK:
            return self._apply(patch)

    def _apply(self, patch: str) -> str:
        planned: dict[Path, bytes | None] = {}
        originals: dict[Path, bytes | None] = {}
        modes: dict[Path, int | None] = {}
        summaries = []
        for operation in parse_patch(patch):
            source = patch_path(self._workdir, operation.path)
            destination = patch_path(self._workdir, operation.destination) if operation.destination is not None else source
            for target in {source, destination}:
                if target in planned:
                    raise ValueError(f"Repeated patch target: {target.name}")
                if target.exists() and not target.is_file():
                    raise ValueError(f"Not a regular file: {target}")
                if target.exists() and target.stat().st_size > 2000000:
                    raise ValueError(f"Patch target exceeds 2 MB: {target.name}")
                if any(parent.exists() and not parent.is_dir() for parent in target.parents):
                    raise ValueError(f"Parent is not a directory: {target}")
                originals[target] = target.read_bytes() if target.exists() else None
                modes[target] = stat.S_IMODE(target.stat().st_mode) if target.exists() else None
            if operation.action == "Add":
                if originals[source] is not None:
                    raise ValueError(f"Add File already exists: {operation.path}")
                planned[source] = ("\n".join(line[1:] for line in operation.body) +
                                   ("\n" if operation.body else "")).encode("utf-8")
            else:
                if originals[source] is None:
                    raise FileNotFoundError(operation.path)
                if operation.action == "Delete":
                    planned[source] = None
                else:
                    if destination != source and originals[destination] is not None:
                        raise ValueError(f"Move destination already exists: {operation.destination}")
                    planned[destination] = _update_content(originals[source], operation.body)
                    modes[destination] = modes[source]
                    if destination != source:
                        planned[source] = None
            summaries.append(f"{operation.action}: {operation.path}" +
                             (f" -> {operation.destination}" if operation.destination else ""))
        if any(parent in planned for target in planned for parent in target.parents):
            raise ValueError("Patch targets overlap as file and parent directory")
        for target, previous in originals.items():
            patch_path(self._workdir, target.relative_to(self._workdir).as_posix())
            current = target.read_bytes() if target.exists() else None
            if current != previous:
                raise ValueError(f"File changed during patch preparation: {target.name}")
        applied = []
        try:
            for target, content in planned.items():
                if content is None:
                    target.unlink()
                else:
                    _atomic_write(target, content, modes[target])
                applied.append(target)
        except OSError as error:
            failures = []
            for target in reversed(applied):
                try:
                    if originals[target] is None:
                        target.unlink(missing_ok=True)
                    else:
                        _atomic_write(target, originals[target], modes[target])
                except OSError:
                    failures.append(str(target))
            if failures:
                raise RuntimeError(f"Patch failed: {error}; rollback failed for {failures}") from error
            raise RuntimeError(f"Patch write failed; file changes rolled back: {error}") from error
        return "Applied patch:\n" + "\n".join(summaries)
