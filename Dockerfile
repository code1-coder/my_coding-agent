FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# 先复制源码再安装项目，依赖以 pyproject.toml 为准
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-cache-dir -e .

# 镜像仅包含公共配置模板，本地配置在运行时挂载
COPY web/ web/
COPY skills/ skills/
COPY config.example.yaml ./config.yaml

EXPOSE 8000

CMD ["uvicorn", "coding_agent.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
