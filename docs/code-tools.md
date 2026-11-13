# 代码理解与编辑工具

CLI、网页、worker 和后台队友通过相同的 ToolRegistry 使用代码工具；explorer 增加搜索、提纲和 Git 查看，不提供 apply_patch。这些工具仍经过 Hook 和权限管线。

## 推荐工作流程

1. git_status、git_diff 查看已有改动。
2. search_code 找实现和调用位置，code_outline 查看 Python 符号。
3. read_file 按行读取相关代码。
4. apply_patch 修改，使用已有测试命令验证，再查看 git_diff。

同批工具会并发执行。存在依赖的修改、读取和测试应分成不同批次。

每个工具完成后立即发送 tool_done，不再等待同批最慢的工具；写回模型的工具结果仍按调用顺序排列。取消或关闭本轮时会取消并回收批次中的异步任务，保留已完成结果，为未完成调用补充中断结果。同步线程工具仍受其自身取消能力限制，不把取消等待当作线程已停止。

等待工具、模型、上下文整理和记忆处理时，默认每 10 秒报告当前阶段；工具等待提示包含尚未完成的名称和调用 ID。config.yaml 可设置 progress_interval（秒）。model_timeout 默认 900 秒，限制一次模型请求及其重试、恢复的硬总预算；model_idle_timeout 默认 120 秒，限制单次模型尝试没有模型协议或 HTTP 响应活动的连续空闲时间；model_connect_timeout 默认 20 秒，限制连接建立时间。最终答复使用独立的 final_answer_timeout（默认 30 秒），不与主请求总预算混淆。Anthropic 和 OpenAI 兼容 Provider 关闭 SDK 内部重试，由 RecoveryManager 负责有限重试；HTTP 429/529 保留各自的限流和备用模型策略，连接中断及其他可重试服务端错误有次数上限。maintenance_timeout 默认 30 秒，限制单次内部摘要或记忆请求连同重试的等待时间。这些超时都必须是有限正数。等待提示区分无活动空闲期限与硬总预算；持续活动可以超过空闲期限，但不能突破总预算。取消会传播，不自动重放已经执行的工具。

每轮结束前应主动交付结果，说明实际完成事项、相关文件、实际验证与未完成事项。模型空回复、无有效工具调用或耗尽输出长度恢复时，运行层最多补充一次禁用工具的总结请求，30 秒内未能生成有效答复则显示明确提示。达到执行轮次上限时同样停止工具并补充总结，明确说明已达到上限，不能据此声称任务完成；这次总结请求不计入工具执行轮次。补充答复通过 text_delta 显示到 CLI 和网页，并保留在会话中；已有正常答复不会重复输出。此机制保证反馈可见，不替代对实际工作结果的核验。

CLI 的 chat 和 run 会在工具开始时显示文件路径；分段读取同时显示行范围，例如 read_file src/main.py:20-60。apply_patch 逐项列出新增、修改、移动和删除的路径。search_code 显示搜索范围、pattern 和 files，glob 显示文件匹配模式。bash 显示实际执行命令，例如 > bash git status --short，并在完成或失败时报告退出码、耗时、失败阶段、超时和截断状态，保留引号、空格和中文。只显示参数摘要，不打印文件正文或替换内容；路径和命令中的换行与控制字符会转义。开始记录表示准备执行，工具返回失败时另行显示带目标路径或命令的简短错误。

## 工具参数

| 工具 | 示例参数 | 行为 |
|------|----------|------|
| search_code | pattern="run_turn", path="src", file_glob="*.py", context_lines=2, max_results=50 | 默认字面量；regex=true 启用正则，case_sensitive=false 忽略大小写；JSON 包含路径、1-based 行号、上下文和截断状态 |
| read_file | path="src/example.py", start_line=20, end_line=60 | 闭区间，分段默认带行号；limit 限制行数，show_line_numbers 控制行号；省略分段参数保留原有输出 |
| code_outline | path="src/example.py", max_symbols=200 | .py / .pyi 的 AST 提纲，含嵌套限定名、签名、装饰器、导入和起止行；不执行源码 |
| git_status | 无参数 | porcelain v1 状态及分支；显示未跟踪文件 |
| git_diff | path="src", staged=false, stat=false, context_lines=3 | 默认未暂存；staged=true 查看暂存，stat=true 查看统计；路径为字面量；不显示未跟踪文件内容 |
| apply_patch | patch="补丁文本" | 新增、精确上下文更新、移动及删除；JSON 参数需要正确编码换行 |

search_code 需要 PATH 中存在 ripgrep (rg)，Git 工具需要 Git。未安装时返回明确错误；无需新增 Python 依赖。搜索遵循 ripgrep 默认忽略规则，目录遍历不跟随符号链接。文件大小限制 2 MiB，匹配行最长展示 2000 字符；最多 1000 行结果，原始搜索输出最多读取 2 MB。Git 输出最多 50000 字节。命令默认超时 120 秒；BashTool 可通过 timeout_seconds 调整超时时间。超时、启动失败、空命令和非零退出都会返回结构化失败结果，并保留 stdout、stderr、duration_ms、timed_out、cancelled、truncated 和 failure_stage 字段。结果截断时缩小路径或搜索范围。

BashTool 的异步入口直接管理子进程。取消时先终止进程树、回收输出和进程，再传播取消异常；工具结束事件中的 cancelled=true、termination_confirmed=true 表示终止已确认，failure_stage=cancelled。超时也使用相同清理流程。清理失败时 termination_confirmed=false、cancelled=false，failure_stage 为 cleanup 或 supervision，不将取消请求已收到当作进程已停止；启动前失败时 termination_confirmed=null。同步 execute 复用相同执行逻辑，异步调用方应使用 aexecute，避免阻塞事件循环。

Windows 将执行进程绑定到 Job Object 后才放行命令，并查询 Job 的 ActiveProcesses=0 确认退出。Linux 使用独立监督进程，在启动 Shell 前设置 child subreaper；取消时逐轮终止其直接子进程并接管、回收孤儿后代，包括 setsid 和双重 fork 后代。只有 waitpid 确认不存在任何待回收子进程，且监督进程退出，才通过独立控制通道确认终止；stdout/stderr EOF 和 Shell 主进程退出均不能作为进程树已退出的依据。正常运行时，Windows 和 Linux 均等待后代结束，因此关闭输出的后台任务仍受命令超时约束，并可继续取消。该机制用于管理普通命令后代，不构成对抗恶意命令或系统服务代启动的安全隔离。其他 Unix 平台当前缺少同等终止确认机制，会在启动前明确返回不支持，不能视为已经通过跨 Unix 验收。

在仓库根目录运行进程树验收命令：Windows 使用 .venv/Scripts/python.exe -m pytest -q -s tests/test_code_tools.py -k 'stops_real_process_tree or stops_detached_descendants'；Linux 使用 .venv/bin/python -m pytest -q -s tests/test_code_tools.py -k 'stops_real_process_tree or stops_detached_descendants'。用例启动真实父子进程并持续写入各自日志，经注册表取消、AgentLoop 取消及超时后检查 PID 已停止、日志不再变化，且下一条命令可正常执行；附加用例检查输出重定向、父进程先退出的后代及停止后的延迟修改。无需模型 API；该检查不等同于浏览器点击停止的端到端验证。本次未增加长任务句柄和流式输出接口。

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
