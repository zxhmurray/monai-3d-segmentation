# Docker 部署指南

## 快速启动

### 1. 构建并启动所有服务

```bash
# 启动所有服务（CPU 模式）
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f api
```

### 2. 访问服务

- API 文档: http://localhost:8000/docs
- Web Dashboard: http://localhost:3000

## GPU 部署

### 前置要求

- NVIDIA Docker 安装 (nvidia-docker)
- NVIDIA GPU 驱动

### 启动 GPU 服务

```bash
# 使用 GPU 编排文件启动
docker-compose -f deploy/gpu-compose.yml up -d

# 查看 GPU 使用情况
docker-compose -f deploy/gpu-compose.yml logs api | grep GPU
```

## 环境变量配置

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `MONAI_API_HOST` | 0.0.0.0 | 服务监听地址 |
| `MONAI_API_PORT` | 8000 | 服务端口 |
| `MONAI_API_MODEL_DIR` | models | 模型目录 |
| `MONAI_API_DEFAULT_MODEL` | best_model.pt | 默认模型 |
| `MONAI_API_MODEL_DEVICE` | cuda | 设备 (cuda/mps/cpu) |

## 数据目录

通过 Docker Volume 挂载：

```
./models          -> /app/models          (模型文件)
./data/uploads    -> /app/data/uploads    (上传文件)
./results         -> /app/results         (预测结果)
./logs            -> /app/logs            (日志文件)
```

## 停止服务

```bash
# 停止所有服务
docker-compose down

# 停止并删除数据卷
docker-compose down -v
```

## 重新构建

```bash
# 重新构建镜像
docker-compose build --no-cache

# 拉取最新代码后重启
docker-compose up -d --build
```

## 生产环境建议

1. **使用 Nginx 反向代理**
2. **配置 HTTPS**
3. **使用 PostgreSQL 存储历史记录**
4. **配置日志收集 (ELK/Loki)**
5. **使用 Kubernetes 进行编排**
