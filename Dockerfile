# MONAI 3D 医学影像分割 - 多阶段构建

# ============================================
# Stage 1: Builder - 安装依赖
# ============================================
FROM python:3.10-slim as builder

WORKDIR /app

# 安装构建依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .
COPY requirements-api.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir --user -r requirements.txt
RUN pip install --no-cache-dir --user -r requirements-api.txt

# ============================================
# Stage 2: Runtime - 运行镜像
# ============================================
FROM python:3.10-slim

WORKDIR /app

# 安装运行时依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 从 builder 复制已安装的包
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# 复制应用代码
COPY . .

# 创建必要目录
RUN mkdir -p data/uploads results/predictions models logs

# 环境变量
ENV PYTHONUNBUFFERED=1
ENV MONAI_API_HOST=0.0.0.0
ENV MONAI_API_PORT=8000

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 启动命令
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
