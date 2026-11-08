# coding-agent

模块化 AI 编码智能体，提供网页控制台和命令行两种使用方式，支持 Anthropic 与 OpenAI 兼容模型接口。项目包含工具执行、权限检查、上下文压缩、跨会话记忆和子智能体协作等模块。

需要 **Python 3.11 或更高版本**。网页由 FastAPI 直接提供，无需单独安装前端依赖。

[快速开始](#快速开始) · [网页工作台](#网页工作台) · [命令行使用](#命令行使用) · [统一运行流程](#统一运行流程) · [PyCharm 运行配置](#pycharm-运行配置) · [配置说明](#配置说明) · [常见问题](#常见问题) · [Docker 运行](#docker-运行) · [上传 GitHub](#上传-github) · [开发与测试](#开发与测试)

## 快速开始

以下命令均在项目根目录执行，直接使用虚拟环境中的解释器，无需激活虚拟环境。

### 1. 获取项目并安装依赖

如果已经下载项目，直接进入包含 `pyproject.toml` 的目录，跳过克隆步骤。

```bash
git clone <仓库地址> coding-agent
cd coding-agent
```

**Windows PowerShell：**

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**macOS / Linux：**

```bash
python3 --version
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

`requirements.txt` 安装运行依赖和项目本身（`-e .`）。依赖版本范围与 `pyproject.toml` 一致，采用最低版本约束，不是精确版本锁定文件。

### 2. 配置模型

首次使用时，先复制配置模板（已有本地配置时跳过，不要覆盖）：

```powershell
# Windows PowerShell
Copy-Item config.example.yaml config.yaml
Copy-Item .env.example .env
```

```bash
# macOS / Linux
cp config.example.yaml config.yaml
cp .env.example .env
```

编辑本地 `config.yaml`，设置模型和 API 地址；密钥填入本地 `.env` 的对应变量，也可以按下文在终端设置。两份本地文件已被 Git 忽略，仓库只提交模板。以下为 OpenAI 兼容接口示例，模型标识和地址需要替换为服务商提供的实际值：

```yaml
model_id: your-model-id             # 服务商支持的模型标识
provider: openai_compat             # OpenAI 兼容接口
base_url: https://your-api-host/v1  # 服务商提供的 API 基础地址
api_key: null                       # 密钥通过环境变量传入
max_tokens: 8000                     # 单次响应的最大令牌数
```

在**即将启动服务的终端**中设置 API 密钥：

```powershell
# Windows PowerShell
$env:OPENAI_API_KEY = "你的 API 密钥"
```

```bash
# macOS / Linux
export OPENAI_API_KEY="你的 API 密钥"
```

其他后端的配置方式：

| 接口类型 | `provider` | 地址和密钥 |
|----------|------------|------------|
| OpenAI 及兼容服务 | `openai_compat` | 使用对应服务的模型和地址；密钥可通过 `OPENAI_API_KEY` 设置 |
| DeepSeek | `openai_compat` | 使用服务商提供的模型和地址；也可通过 `DEEPSEEK_API_KEY` 设置密钥 |
| Anthropic | `anthropic` | 设置对应模型；官方默认地址可用 `base_url: null`，密钥通过 `ANTHROPIC_API_KEY` 设置 |

网页服务可以在未配置密钥时启动，实际模型对话需要有效的模型配置和密钥。

### 3. 启动网页控制台

**Windows PowerShell：**

```powershell
.\.venv\Scripts\python.exe -m uvicorn coding_agent.server.app:app --host 127.0.0.1 --port 8080 --reload
```

**macOS / Linux：**

```bash
.venv/bin/python -m uvicorn coding_agent.server.app:app --host 127.0.0.1 --port 8080 --reload
```

看到 `Uvicorn running on http://127.0.0.1:8080` 后，打开 [网页控制台](http://127.0.0.1:8080)。Windows 上的 8000 端口可能被 Hyper-V、WSL 或其他系统组件列入排除范围；本项目默认使用 8080，避免 `WinError 10013`。

- [健康检查](http://127.0.0.1:8080/health)：正常返回 `{"status":"ok"}`，仅确认服务存活，不验证模型配置。
- [接口文档](http://127.0.0.1:8080/docs)：查看 HTTP 接口定义。

保持终端运行，按 `Ctrl+C` 停止服务。`--reload` 用于开发时自动重载代码。以后启动时，进入项目目录、设置所需环境变量，再执行本节命令即可，无需重新创建虚拟环境。

## 网页工作台

网页采用深色三栏布局：左侧切换任务、对话、记忆与团队，中间展示当前功能，右侧保留 AI 助手。窄屏下点击顶部的 AI 助手按钮展开对话，手机端的任务列按顺序纵向排列。

- **任务看板**：按待处理、进行中、已完成分栏，显示实际任务数量与完成进度。支持名称/描述搜索、我认领的任务及依赖筛选；点击卡片查看完整描述与前置任务。
- **手动管理**：点击“新建任务”填写名称、描述及可选的前置任务。详情中“认领并开始”将任务分配给自己，“完成任务”更新状态。前置任务未完成时不能认领。手动操作直接复用 `TaskBoard`，无需调用模型。
- **交给 AI**：任务详情中的“交给 AI”会将任务信息填入对话输入框，确认后发送。模型仍通过统一运行层、Hook、权限检查和已有任务工具执行。
- **AI 对话**：支持流式回答、代码块、复制回复和停止生成。按 Enter 发送，Shift + Enter 换行；快捷建议只填充输入框。左侧“AI 对话”可切换到宽幅对话页。
- **执行与审批**：右侧“执行记录”展示工具状态、参数与结果；敏感操作单独展示允许/拒绝按钮，120 秒未处理自动拒绝。停止生成会取消本轮请求及其等待中的审批。
- **记忆与团队**：记忆页面读取 `MemoryStore` 中的真实内容，支持搜索与展开；团队页面显示当前会话的主智能体与活跃队友，“安排协作”可填入协作请求。

任务与记忆每 5 秒同步一次，也可点击顶部刷新按钮。相同工作目录中的 CLI 与网页共享任务和记忆；团队和令牌用量按网页会话隔离。对话、执行记录保留在当前页面，刷新后会清空可见记录；服务进程内的模型上下文仍按原会话标识延续，持久化记忆单独保存在 `.memory/`。顶部连接状态仅代表页面与本地服务连接，不代表模型 API 已验证可用。

界面使用原生 HTML、CSS 和 JavaScript，Lucide 图标及许可证放在 `web/vendor/lucide/`，不依赖 CDN，也无需安装 Node.js。更新后重启网页服务并刷新页面即可。

## 命令行使用

命令行入口是 `coding_agent_cli.main`，需要指定 `chat` 或 `run` 子命令。完成依赖安装和模型配置后，可直接使用，无需先启动网页服务。

**Windows PowerShell：**

```powershell
# 交互式对话
.\.venv\Scripts\python.exe -m coding_agent_cli.main chat

# 执行单次任务
.\.venv\Scripts\python.exe -m coding_agent_cli.main run "列出项目中的 Python 文件"

# 查看帮助
.\.venv\Scripts\python.exe -m coding_agent_cli.main --help
```

macOS / Linux 将解释器路径替换为 `.venv/bin/python`。激活虚拟环境后，也可以使用安装时注册的 `coding-agent chat` 和 `coding-agent run "任务内容"`。

交互模式中输入 `q` 或 `exit` 退出。全局选项放在子命令之前，例如：

```powershell
.\.venv\Scripts\python.exe -m coding_agent_cli.main --model your-model-id --provider openai_compat chat
```

`chat` 每轮会读取相关的跨会话记忆，并在回答结束后通过模型提取新记忆。例如输入“记住你叫李华”或“记住以后用中文回答”，保存成功后会显示 `[记忆] 已保存 ...`。记忆保存在配置的 `workdir` 下的 `.memory/`，其中 `MEMORY.md` 是索引，其余 Markdown 文件记录具体内容。重新启动 `chat` 时使用同一个工作目录，即可继续读取这些记忆；仅在同一轮对话中口头答应“记住了”不能作为保存成功的依据。记忆选择和提取会额外调用配置的模型接口。

## 统一运行流程

代码工具现已支持 search_code、read_file 分段读取、apply_patch、git_status、git_diff 和 Python code_outline。参见[代码工具参数与补丁格式](docs/code-tools.md)。搜索需要 PATH 中的 ripgrep (rg)，Git 查看需要 Git；无需新增 Python 依赖。

CLI 的 `chat`、`run` 和网页 `/chat` 均通过 `AgentRuntime` 调用项目已有模块。入口负责终端交互或 HTTP/SSE 编码，实际处理流程保持一致：会话开始 → 提交用户输入 → 读取记忆、技能与队友回信 → 组装系统提示词 → 压缩上下文 → 调用模型 → 检查并执行工具 → 保存记忆 → 本轮结束。退出 CLI 或关闭网页服务时，会取消后台队友并关闭模型连接。

| 能力 | 已接入的模块与行为 |
|------|--------------------|
| Hook | `EventBus` 分发 9 类事件：会话开始/结束、用户输入、模型调用前/后、工具执行前/后、本轮停止和错误；同一会话共用 Hook 上下文 |
| 令牌统计与输出处理 | 默认注册 `TokenUsageHook` 和 `OutputTruncateHook`，统计模型返回的令牌用量，检查大体积输出；内部记忆和摘要调用也计入统计 |
| 上下文处理 | `CompactionPipeline` 在每次主模型调用前依次运行大输出持久化、消息裁剪、旧工具结果压缩，超限时调用摘要；工具调用与结果保持配对 |
| 错误恢复 | `RecoveryManager` 处理 429/529 退避、上下文过长后的紧急压缩、输出上限升级与续写；配置备用模型后，连续过载会切换到独立客户端 |
| 工具与权限 | `ToolRegistry` 统一注册和分发；`PermissionPipeline` 连接禁止列表、工作区和破坏性命令检查；工具执行前 Hook 可以阻止调用 |
| 跨会话记忆 | `MemoryStore` 在回答前加载相关记忆、结束后提取当前轮信息；达到阈值后合并，索引和正文写入 `.memory/` |
| 技能 | `SkillLoader` 扫描工作目录的 `skills/`，提示词中列出目录，模型通过 `load_skill` 按需读取完整内容 |
| 系统提示词 | `get_system_prompt` 根据真实工具、工作目录、记忆、技能和 MCP 状态组装并缓存提示词 |
| 任务看板 | `TaskBoard` 支持创建、查看、更新、认领、完成和删除任务；CLI、模型工具和网页 `/tasks` 使用同一工作目录 |
| 子智能体 | `AgentPool` 通过 `task` 工具委托隔离历史的 `explorer` 或 `worker`，最多 30 轮，同样经过上下文、Hook、权限和错误恢复流程；子任务不再递归委托 |
| 团队与生命周期 | `TeamRegistry`、`FileMailbox`、`AgentLifecycle` 支持后台队友、消息收发、空闲轮询、自动认领任务和优雅关闭；队友在应用原有事件循环运行 |
| MCP | `connect_mcp` 与 `assemble_tool_pool` 动态接入已注册服务，工具随后进入同一注册表、权限与 Hook 流程 |

CLI 中需要审批时会询问是否允许；网页中会出现“允许 / 拒绝”按钮，审批超时 120 秒默认拒绝。工具状态、记忆写入和错误信息通过 SSE/WebSocket 实时显示。网页每个页面使用独立会话标识，刷新后延续同一会话；网页对话历史保存在服务进程内，跨服务重启的信息由 `.memory/` 保留。相同工作目录中的任务和记忆共享，团队邮箱按会话隔离。

可以要求模型“创建任务并在完成后更新状态”“调用 explorer 查找配置文件”“启动 tester 队友运行测试”等。对应工具包括 `create_task`、`update_task`、`list_tasks`、`get_task`、`claim_task`、`complete_task`、`delete_task`、`task`、`spawn_teammate`、`list_teammates`、`wait_teammate`、`send_message`、`check_inbox` 和 `shutdown_teammate`。CLI 退出时后台队友随会话关闭；需要结果时应先等待队友完成。

当前 MCP 模块通过代码注册服务；内置 `docs`、`deploy` 是模拟服务，适合验证接入流程。配置 `mcp_servers` 只会加载这些已注册服务，不会自动建立任意远程 MCP 连接。

程序中可以注册自定义 Hook，CLI 和网页的默认运行层也使用相同的接口：

```python
from coding_agent import AgentRuntime, load_config
from coding_agent.hooks.bus import EventBus, EventType

bus = EventBus()

@bus.on(EventType.PRE_TOOL_USE, priority=20)
async def inspect_tool(event):
    print("准备执行：", event.data["tool_name"])
    # 返回 None 继续；返回其他值会阻止本次工具调用。
    return None

async def main():
    async with AgentRuntime(load_config(), event_bus=bus) as runtime:
        async for event in runtime.run_turn("列出当前项目的 Python 文件"):
            if event.type == "text_delta":
                print(event.text, end="", flush=True)
```

## PyCharm 运行配置

新建 Python 运行配置，将运行目标类型设为“模块名称”，根据用途填写：

| 用途 | 模块名称 | 运行参数 |
|------|----------|----------|
| 网页控制台 | `uvicorn` | `coding_agent.server.app:app --host 127.0.0.1 --port 8080 --reload` |
| 命令行对话 | `coding_agent_cli.main` | `chat` |
| 单次任务 | `coding_agent_cli.main` | `run "列出项目中的 Python 文件"` |

所有配置均使用项目虚拟环境中的解释器，工作目录设为项目根目录。API 密钥可以填写在运行配置的环境变量中。

直接运行 `src/coding_agent_cli/main.py` 而不提供子命令，会显示以 `Usage:` 开头的帮助信息，并以退出码 2 结束。命令行对话需要补充 `chat` 参数，网页控制台则使用上表中的 `uvicorn` 模块。

## 配置说明

配置读取顺序为：**代码默认值 → 当前目录的 `config.yaml` → 环境变量**，后读取的值优先。CLI 的 `--model`、`--provider` 和 `--max-tokens` 选项还会覆盖相应配置；`--config` 可以指定其他配置文件。

| 配置项 | 用途 | 环境变量 |
|--------|------|----------|
| `model_id` | 模型标识 | `MODEL_ID` |
| `provider` | 接口类型：`anthropic`、`openai_compat` 或 `auto` | `CODING_AGENT_PROVIDER` |
| `base_url` | API 基础地址 | `ANTHROPIC_BASE_URL`、`OPENAI_BASE_URL` |
| `api_key` | API 密钥 | `ANTHROPIC_API_KEY`、`OPENAI_API_KEY`、`DEEPSEEK_API_KEY` |
| `max_tokens` | 单次响应的最大令牌数 | `MAX_TOKENS` |
| `system_prompt` | 系统提示词 | `SYSTEM_PROMPT` |
| `workdir` | 工具使用的工作目录，默认是当前目录 | `CODING_AGENT_WORKDIR` |

以下运行选项可添加到同一份 `config.yaml`，CLI 与网页端均生效：

```yaml
context_limit: 50000        # 超过此估算令牌数后触发模型摘要
max_messages: 50            # 长历史裁剪阈值
keep_recent: 3             # 保留最近几条完整工具输出
persist_threshold: 30000   # 超过此字符数的工具输出写入 .task_outputs/
max_iterations: 50         # 每轮任务最多调用模型的次数
skills_dir: skills         # 相对 workdir 的技能目录
mcp_servers: []            # 启动时加载的已注册 MCP 服务，如 [docs]
team_idle_timeout: 60      # 队友空闲多少秒后退出
fallback_model: null       # 连续过载时使用的备用模型，需由当前服务商支持
escalated_max_tokens: 64000 # 输出截断后提高的上限，应按模型支持范围设置
deny_list:                 # bash 命令的硬阻止子串，可按需补充
  - "rm -rf /"
  - "sudo"
  - "shutdown"
  - "reboot"
  - "mkfs"
  - "dd if="
  - "> /dev/sda"
```

记忆选择、提取、合并以及超限摘要会额外调用模型接口。流式响应已经输出文本后若中途断开，会报告错误，避免自动重放造成重复内容。修改运行配置后，CLI 重新启动，网页服务也需重新启动，以重新创建会话运行层。

模型连接中断或超时且尚未显示文本时，运行层最多尝试 3 次（含首次请求），按指数退避重试；失败尝试中未完成的工具调用会丢弃。已显示部分回复后不会自动重放，也不会执行该不完整响应中的工具。交互式 chat 显示错误后保留会话，可继续输入；单次 run 显示简明错误并以非零状态退出。OpenAI 兼容接口若未发送 finish_reason 就结束流，也会作为不完整响应处理。

同类环境变量同时存在时，地址优先读取 `ANTHROPIC_BASE_URL`；密钥按表中的顺序采用第一个非空值。设置模型后端时应同时确认模型、地址和密钥属于同一服务。

`auto` 仅根据 `base_url` 是否包含 `anthropic` 选择后端，其余情况使用 OpenAI 兼容接口。使用 Anthropic 时可显式设置 `provider: anthropic`。

两种入口都会加载当前目录下的 `.env` 文件，显式设置的进程环境变量优先于 `.env`。两种入口都通过 `load_config` 合并 YAML 和环境配置。

`config.example.yaml` 按模型接口、输出恢复、上下文管理和任务协作分组，包含当前默认的命令禁止列表；`.env.example` 只启用空密钥变量，其余覆盖项均为注释，按需取消注释即可。切换服务商时应检查整套模型、接口类型、地址和密钥，避免同时留下多个服务商的非空配置。

## 常见问题

| 现象 | 检查与处理 |
|------|------------|
| 显示 `Usage: ... COMMAND [ARGS]...` 后退出 | CLI 缺少子命令，补充 `chat` 或 `run`；网页启动使用 `uvicorn` |
| 找不到 `coding_agent` 或 `uvicorn` | 在项目根目录使用同一个虚拟环境解释器重新运行 `-m pip install -r requirements.txt` |
| 提示端口被占用 | 检查是否已有服务运行，或将 `--port 8080` 改为其他未被占用的端口，并访问对应端口 |
| Windows 启动报 `WinError 10013` | 运行 `netsh interface ipv4 show excludedportrange protocol=tcp` 检查系统保留端口；本机的 8000 位于排除范围内，请使用项目默认的 8080，并访问对应端口 |
| 网页打开正常，但对话失败 | 检查模型标识、接口类型、地址和密钥；修改进程环境变量后重新启动服务 |
| 启动后找不到网页或配置 | 确认从完整项目目录运行，保留 `web/`，并将工作目录设为项目根目录 |
| 首次运行相关功能或测试时下载分词数据 | `tiktoken` 首次加载分词表需要联网，下载后使用本地缓存 |

## 接口说明

| 方法 | 路径 | 用途 |
|------|------|------|
| `GET` | `/` | 网页控制台 |
| `GET` | `/health` | 服务健康检查 |
| `GET` | `/tasks` | 任务列表 |
| `POST` | `/tasks` | 手动创建任务，字段为 `subject`、可选 `description` 和 `blockedBy` |
| `PATCH` | `/tasks/{task_id}` | 手动流转任务，`action` 为 `claim` 或 `complete`，保留依赖与状态检查 |
| `GET` | `/workspace?session_id=...` | 工作目录、模型名称、记忆、当前会话团队及令牌用量，不返回密钥、不调用模型 |
| `POST` | `/chat` | SSE 流式对话，请求包含 `prompt` 和可选的 `session_id` |
| `POST` | `/approvals/{approval_id}` | 提交 `session_id` 和布尔值 `approved`，处理工具审批 |
| `WebSocket` | `/ws` | 工具调用通知和心跳 |
| `GET` | `/docs` | 交互式 HTTP 接口文档 |

## 项目结构

| 路径 | 说明 |
|------|------|
| `src/coding_agent/loop.py` | 智能体主循环与工具并发执行 |
| `src/coding_agent/runtime.py` | CLI 与网页共用的模块组装、会话状态和资源生命周期 |
| `src/coding_agent/runtime_tools.py` | 技能、任务、子智能体、团队和 MCP 的工具适配 |
| `src/coding_agent/providers/` | 模型服务接口适配 |
| `src/coding_agent/tools/` | 内置工具、注册表和 MCP 集成 |
| `src/coding_agent/security/` | 权限检查 |
| `src/coding_agent/context/` | 上下文压缩和错误恢复 |
| `src/coding_agent/memory/` | 跨会话记忆的选择、提取和合并 |
| `src/coding_agent/agents/` | 子智能体池、任务依赖和团队通信 |
| `src/coding_agent/hooks/` | 事件总线及回调 |
| `src/coding_agent/skills/` | 技能目录扫描与内容加载 |
| `src/coding_agent/prompt.py` | 系统提示词组装和缓存 |
| `src/coding_agent/server/` | FastAPI 网页后端 |
| `src/coding_agent_cli/` | 命令行入口 |
| `web/` | 工作台页面、样式、交互与本地图标资源 |
| `skills/` | 项目技能说明 |
| `tests/`、`examples/` | 测试与演示脚本 |
| `config.example.yaml`、`.env.example` | 可提交的配置模板；复制为本地配置后使用 |
| `.github/workflows/tests.yml` | GitHub Actions 自动测试 |
| `docs/github-upload.md` | 首次上传 GitHub 的步骤与提交范围 |
| `pyproject.toml`、`requirements.txt` | 项目元数据与依赖声明 |

运行过程中，任务、记忆和团队消息等数据由相应模块保存到工作目录下的 `.tasks/`、`.memory/` 和 `.team/`；上下文压缩还可能生成 `.transcripts/` 和 `.task_outputs/`。

## Docker 运行

先按快速开始复制并填写 `config.yaml` 和 `.env`，然后在项目根目录执行：

```bash
docker compose up --build -d
docker compose logs -f
```

打开 [网页控制台](http://127.0.0.1:8000)。Compose 默认仅绑定本机地址；本地配置只读挂载，API 密钥通过环境变量传入，构建镜像时不会复制本地配置或密钥。任务、记忆、团队邮箱、会话记录与大输出分别使用独立命名卷。当前 Compose 不会自动迁移旧版 `coding_agent_data` 共用卷中的数据，已有部署应先备份并分别迁移所需内容。

修改源码或网页后重新执行 `docker compose up --build -d`；修改配置或环境变量后执行 `docker compose up -d --force-recreate`。执行 `docker compose down` 停止服务，命名卷中的数据会保留。

## 上传 GitHub

首次发布按 [GitHub 上传说明](docs/github-upload.md) 操作。`.gitignore` 已排除虚拟环境、IDE 设置、缓存、日志、本地密钥、配置和运行数据；`web/vendor/lucide/` 是网页必需资源，需要连同其许可证一起提交。

项目尚未声明整体开源许可证；发布公开仓库前，可根据自己的授权意愿添加 `LICENSE`。Lucide 图标的第三方许可见 [web/vendor/lucide/LICENSE](web/vendor/lucide/LICENSE)。

## 开发与测试

项目文本使用 UTF-8（无 BOM）、LF 换行，文件末尾保留一个换行，避免行尾空格。YAML 使用空格缩进；编辑 Markdown 时保留代码块内原本需要的缩进。

**Windows PowerShell：**

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

**macOS / Linux：**

```bash
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest tests/ -v
```

如需覆盖率报告，在同一虚拟环境中安装 `pytest-cov`，再使用该解释器运行 `-m pytest --cov=src tests/`。

记忆系统演示脚本位于 `examples/memory_demo.py`，默认使用模拟模型；添加 `--keep --workdir ./demo_memory` 可保留生成的文件，添加 `--live` 使用配置的真实模型服务。

```powershell
.\.venv\Scripts\python.exe examples/memory_demo.py
```

macOS / Linux 将解释器路径替换为 `.venv/bin/python`。
