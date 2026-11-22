# 代码理解与编辑工具

CLI、网页、worker 和后台队友通过相同的 ToolRegistry 使用代码工具；explorer 增加搜索、提纲和 Git 查看，不提供 apply_patch。这些工具仍经过 Hook 和权限管线。

## 推荐工作流程

1. git_status、git_diff 查看已有改动。
2. search_code 找实现和调用位置，code_outline 查看 Python 符号。
3. read_file 按行读取相关代码。
4. apply_patch 修改，使用已有测试命令验证，再查看 git_diff。

同批工具会并发执行。存在依赖的修改、读取和测试应分成不同批次。

每个工具完成后立即发送 tool_done，不再等待同批最慢的工具；写回模型的工具结果仍按调用顺序排列。取消或关闭本轮时会取消并回收批次中的异步任务，保留已完成结果及工具提供的终止证据，为其余调用补充中断结果。同步线程工具仍受其自身取消能力限制，不把取消等待当作线程已停止。

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

read_file 可接收相对路径、当前宿主机绝对路径；WSL 环境也接受映射到 /mnt/<drive> 的 Windows 盘符路径。工作区外路径仍须逐次审批，授权只对本次读取和目标文件有效。目标盘符未挂载时明确拒绝，不进入审批等待。用户给出文件路径或要求查看文件时，系统提示要求优先调用 read_file，不使用与当前宿主机不匹配的 Shell 命令读取。

search_code 使用相同的宿主机与 WSL 路径解析；外部文件或目录逐次审批后可在本次调用中搜索。审批绑定等待用户之前解析的精确目标；文件授权不包含同级文件，目录搜索不越过批准范围，也不跟随符号链接或 junction。读取与搜索的授权互不通用，不授予写入、补丁、提纲或 Git 查看权限；拒绝、缺少审批处理器或取消均不会放宽边界。每次搜索独立持有范围，不改变会话工作区；完成或取消后授权失效。工作区内结果仍使用相对路径，外部结果使用绝对路径，便于后续逐文件审批读取。rg 缺失时仍遵循下述受限搜索规则，不因批准外部目录而启用无界遍历。

工具失败会作为错误结果传回模型；同一种工具连续失败 3 次后停止重复请求，直接交付最后错误和环境排查提示，避免模型在无效 Shell 命令上持续循环。成功执行该工具会清除对应失败计数。

search_code 优先使用 PATH 中的 ripgrep (rg)，项目通过 ripgrep-bin 将预编译 rg 安装到 Python 环境。Git 工具仍需要本机 Git。rg 搜索遵循其默认忽略规则，目录遍历不跟随符号链接。文件大小限制 2 MiB，匹配行最长展示 2000 字符；最多 1000 行结果，原始搜索输出最多读取 2 MB。Git 查看工具输出最多 50000 字节。命令默认超时 120 秒；BashTool 可通过 timeout_seconds 调整超时时间。超时、启动失败、空命令和非零退出都会返回结构化失败结果，并保留 stdout、stderr、duration_ms、timed_out、cancelled、truncated 和 failure_stage 字段。结果截断时缩小路径或搜索范围。

### 本地安装搜索依赖

Windows 在项目根目录执行，不需要 Docker：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\Activate.ps1
rg --version
```

只补装搜索程序可执行 `.\.venv\Scripts\python.exe -m pip install "ripgrep-bin>=15.2,<16"`；不激活环境时可通过 `.\.venv\Scripts\rg.exe --version` 检查安装。Linux/macOS 对应执行 `.venv/bin/python -m pip install -r requirements.txt`，然后 `source .venv/bin/activate`。服务启动进程必须继承虚拟环境的可执行文件 PATH；仅指定 Python 解释器不会自动修改子进程 PATH。

requirements.txt 与 pyproject.toml 同时声明 ripgrep-bin；前者包含 UTF-8 编码声明，避免 Windows pip 按 GBK 读取中文注释失败。ripgrep-bin 是分发预编译 rg 的第三方 Python 包，Git 仍使用系统安装版本。

### rg 缺失时的受限搜索

- rg 可用时保持原有字面量、正则和 glob 能力。缺失时，同一次调用自动使用 Git 的 ls-files 与 check-ignore --no-index 过滤文件，Python 执行字面量匹配，不发起额外模型请求。已跟踪但后来被忽略的文件同样被排除。
- Git 也不可用或目录不是 Git 工作树时，只支持显式指定的普通文件；目录请求明确报错，建议缩小到文件或安装 rg，不把失败包装成无匹配。
- 备用只支持 UTF-8 文本、普通文件名 glob（如 *.py）、上下文、忽略大小写与结果上限；regex=true、大括号扩展、路径 glob 和否定 glob 明确拒绝。忽略大小写使用 Unicode casefold，语义不保证与 rg 完全一致。
- 隐藏文件、.git、符号链接、可检测的 junction/reparse point 不被读取；跳过二进制、超过 2 MiB、非 UTF-8 文件，并在结果中标注不完整。权限拒绝和取消不会触发绕过限制的扫描。
- Git 目录搜索预检生效父级和范围内的 .ignore / .rgignore，发现无法实现的规则时明确拒绝；预检按 Git 规则裁剪忽略目录。不进入嵌套仓库或子模块，结果标注未完整覆盖。没有 Git 的单文件搜索明确声明未检查 Git 忽略规则。
- 每次备用扫描最多预检/处理 5000 个条目、读取 20 MiB；JSON 输出不超过 2 MB，最多 1000 行，每行展示 2000 字符。整体搜索、Git 调用和预检共用 30 秒预算，预检无法安全完成时明确失败。达到其他预算时返回已有结果及 truncated=true、truncation_reasons，不能据此认定全范围没有匹配。
- 后端以 backend=rg、python_git 或 python_file 标识；备用还返回 fallback_reason、warnings、skipped。状态字段位于正文前，保证已有界面预览先看到降级与不完整状态。CLI 成功降级时显示目标范围、后端和不完整提示，不打印文件正文。
- 工具实例初始化检测依赖，能力状态最多保留 30 秒；PATH/PATHEXT 改变立即重查。相对 PATH 项按工具工作目录解析。只缓存可执行能力，不缓存搜索结果；依赖恢复后可以重新使用 rg。
- 搜索异步入口取消时通知扫描线程，在文件、数据块及匹配循环之间停止，并回收本次受控 rg/Git 子进程后传播取消；这不改变 BashTool 的独立进程树监督约定。

验证时既运行相关回归，也在临时 Git 仓库中通过真实 PATH 隐藏 rg，直接调用 SearchCodeTool，核对路径、行号、忽略规则和 backend，再恢复 PATH 检查 rg 恢复。临时探针和生成内容不要加入仓库。

BashTool 的异步入口直接管理子进程。取消时先终止进程树、回收输出和进程，再传播取消异常。工具结束事件中的 cancelled=true、termination_confirmed=true 表示终止已确认，failure_stage=cancelled。超时也使用相同清理流程。超时清理期间收到取消请求时，清理结束后仍传播取消异常，并在结果中保留 timed_out=true。清理失败时 termination_confirmed=false、cancelled=false，failure_stage 为 cleanup 或 supervision，已收到的取消请求仍以取消异常向上传递。启动前失败时 termination_confirmed=null。同步 execute 复用相同执行逻辑，异步调用方应使用 aexecute，避免阻塞事件循环。

Windows 的 exit_code 以无符号 32 位整数保留原生退出状态，例如 0xC0000005 对应 3221225477。非零退出码仍标记为工具失败。

网页停止当前对话时保留 SSE 连接。界面先显示「正在停止」，再用同一 execution_id 的工具结束事件显示「已确认终止」或「终止未确认」。一次对话结束也可能包含终止未确认的工具。HTTP 客户端可在 /chat 或 /chat/retry 的请求体传入 stop_confirmation=true，从首个 chat_started 事件取得本次请求专用的 stop_id，再向 /chat/{stop_id}/stop 提交原 session_id。HTTP 202 只表示停止请求已接收，实际结果由原流中的 tool_done 和 chat_stopped 事件送达。

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
