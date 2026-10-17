# 上传到 GitHub

本项目按源码仓库组织，保留 `src/`、`web/`、`skills/`、`tests/` 和 `examples/`，以及项目根目录的说明、依赖声明和配置模板。

## 提交范围

| 内容 | 处理方式 |
|------|----------|
| Python 源码、测试、网页和技能 | 提交 |
| `web/vendor/lucide/` | 提交，包括第三方许可证；网页离线运行需要这些文件 |
| `config.example.yaml`、`.env.example` | 提交，不填写真实密钥 |
| `config.yaml`、`config.local.yaml`、`.env.*`、配置备份和私钥文件 | 忽略，仅保留在本机；`.env.example` 例外，需要提交 |
| `.venv/`、`__pycache__/`、`*.egg-info/` | 忽略，通过依赖安装重新生成 |
| `.idea/`、`.vscode/`、测试和覆盖率缓存 | 忽略 |
| `.memory/`、`.tasks/`、`.team/` | 忽略，属于本地运行数据 |
| `.transcripts/`、`.task_outputs/`、`.run/`、日志 | 忽略，可能包含会话内容和工具输出 |
| `examples/memory-demo-output/`、`demo_memory/` | 忽略，演示脚本可重新生成 |

忽略规则不会删除本机文件。通过 Git 提交或 GitHub Desktop 上传时规则会生效；不要把整个本地项目文件夹拖入网页上传，以免把忽略的本地文件一起公开。

`.gitignore` 只影响未跟踪文件；已经提交的文件不会因新增规则而自动移出仓库。依赖锁文件、源码和第三方网页资源仍应正常提交。

## 第一次提交

以下命令在项目根目录执行。尚未初始化仓库时先执行：

```bash
git init -b main
```

添加文件后检查提交清单：

```bash
git add .
git diff --cached --stat
git diff --cached --name-only
```

清单应包含源码、README 和配置模板，不应包含本地 `config.yaml`、`.env`、虚拟环境、运行记忆或日志。可以验证忽略规则：

```bash
git check-ignore -v config.yaml .env .venv/ .memory/ .run/
```

确认后提交：

```bash
git commit -m "chore: prepare project for GitHub"
```

如果 Git 提示缺少作者信息，在当前仓库设置自己的提交名称和邮箱，再执行提交：

```bash
git config user.name "你的提交名称"
git config user.email "你的 GitHub 邮箱或 noreply 邮箱"
```

## 创建远程仓库并推送

1. 在 GitHub 新建空仓库，不勾选自动创建 README、.gitignore 或许可证。
2. 将下面的地址替换为自己仓库的实际 HTTPS 地址。
3. 执行推送，并按 GitHub 的提示完成登录。

```bash
git remote add origin https://github.com/YOUR_USERNAME/coding-agent.git
git push -u origin main
```

若已经有 `origin`，先通过 `git remote -v` 检查地址，再根据需要使用 `git remote set-url origin <实际仓库地址>`。

## 上传后的使用

新克隆的项目不包含本地配置和历史运行数据。按 README 安装依赖，复制两份模板后填写模型与密钥，再启动服务。GitHub Actions 会在推送和拉取请求时运行 Windows / Linux、Python 3.11 / 3.12 的测试，无需配置模型 API 密钥。

当前项目没有整体开源许可证；如果希望授权他人使用、修改或分发，按自己的授权意愿另行选择并添加 `LICENSE`。第三方图标的现有许可证应保留。
