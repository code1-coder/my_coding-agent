# 代码理解与编辑工具

CLI、网页、worker 和后台队友通过相同的 ToolRegistry 使用代码工具；explorer 增加搜索、提纲和 Git 查看，不提供 apply_patch。这些工具仍经过 Hook 和权限管线。

## 推荐工作流程

1. git_status、git_diff 查看已有改动。
2. search_code 找实现和调用位置，code_outline 查看 Python 符号。
3. read_file 按行读取相关代码。
4. apply_patch 修改，使用已有测试命令验证，再查看 git_diff。

同批工具会并发执行。存在依赖的修改、读取和测试应分成不同批次。

CLI 的 chat 和 run 会在工具开始时显示文件路径；分段读取同时显示行范围，例如 read_file src/main.py:20-60。apply_patch 逐项列出新增、修改、移动和删除的路径。search_code 显示搜索范围、pattern 和 files，glob 显示文件匹配模式。bash 显示实际执行命令，例如 > bash git status --short，保留引号、空格和中文。只显示参数摘要，不打印文件正文或替换内容；路径和命令中的换行与控制字符会转义。开始记录表示准备执行，工具返回失败时另行显示带目标路径或命令的简短错误。

## 工具参数

| 工具 | 示例参数 | 行为 |
|------|----------|------|
| search_code | pattern="run_turn", path="src", file_glob="*.py", context_lines=2, max_results=50 | 默认字面量；regex=true 启用正则，case_sensitive=false 忽略大小写；JSON 包含路径、1-based 行号、上下文和截断状态 |
| read_file | path="src/example.py", start_line=20, end_line=60 | 闭区间，分段默认带行号；limit 限制行数，show_line_numbers 控制行号；省略分段参数保留原有输出 |
| code_outline | path="src/example.py", max_symbols=200 | .py / .pyi 的 AST 提纲，含嵌套限定名、签名、装饰器、导入和起止行；不执行源码 |
| git_status | 无参数 | porcelain v1 状态及分支；显示未跟踪文件 |
| git_diff | path="src", staged=false, stat=false, context_lines=3 | 默认未暂存；staged=true 查看暂存，stat=true 查看统计；路径为字面量；不显示未跟踪文件内容 |
| apply_patch | patch="补丁文本" | 新增、精确上下文更新、移动及删除；JSON 参数需要正确编码换行 |

search_code 需要 PATH 中存在 ripgrep (rg)，Git 工具需要 Git。未安装时返回明确错误；无需新增 Python 依赖。搜索遵循 ripgrep 默认忽略规则，目录遍历不跟随符号链接。文件大小限制 2 MiB，匹配行最长展示 2000 字符；最多 1000 行结果，原始搜索输出最多读取 2 MB。Git 输出最多 50000 字节。命令超时 30 秒。结果截断时缩小路径或搜索范围。

## 补丁格式

采用 Codex 风格的显式文件操作及上下文行，不接受 Git unified diff 的行号头：

    *** Begin Patch
    *** Add File: src/helper.py
    +def helper():
    +    return 1
    *** Update File: src/main.py
    @@
     def main():
    -    return 0
    +    return 1
    *** Delete File: obsolete.py
    *** End Patch

更新文件头后可添加 *** Move to: new/path.py；@@ <完整原行> 将搜索位置移动到唯一匹配的锚点之后。每个 hunk 的空格前缀表示上下文，- 删除，+ 新增。*** End of File 限制匹配到文件末尾。上下文必须精确且唯一；重复代码需要提供更多上下文。纯新增 hunk 需要锚点或文件末尾标记。

全部文件先预检，任一上下文不匹配则不写入；已有新增/移动目标拒绝覆盖。同一补丁不接受重复目标或互相包含的文件目标。路径必须位于工作区内，拒绝 .git、上级路径、符号链接和可检测的 junction。保留现有 UTF-8 BOM、LF/CRLF 和末尾换行状态。单文件最多 2 MB，单补丁最多 100 万字符、100 个操作。

写文件采用同目录临时文件替换；写入出错时尝试恢复已修改文件，并报告回滚失败。多文件变更不是文件系统事务，进程崩溃或其他进程同时修改不受事务保证，回滚也可能留下新建的空目录。

## 工程参考

- [OpenAI Codex apply-patch](https://github.com/openai/codex/tree/main/codex-rs/apply-patch)：参考显式 Add/Update/Delete/Move 补丁协议、上下文定位和保留换行符的思路。本项目独立实现一个更严格的精确匹配子集，不承诺完整兼容。
- [ZCode](https://github.com/zai-org/ZCode)：参考其将 CLI 与运行时放入同一工程、工具与界面分层的组织方式；本项目继续复用现有 AgentRuntime，不引入其桌面或 Node.js 依赖。

当前 code_outline 提供语法结构，不做跨文件定义解析、引用查找或类型推断；这些能力可后续通过 LSP 扩展。
