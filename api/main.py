"""
FastAPI 主应用

MONAI 3D 医学影像分割推理服务
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

from .config import get_settings
from .models import HealthResponse, ModelInfo, ErrorResponse
from .inference import router as inference_router
from .model_manager import get_model_manager

settings = get_settings()

# 创建 FastAPI 应用
app = FastAPI(
    title="MONAI 3D Medical Image Segmentation API",
    description="3D 医学影像分割推理服务 API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(inference_router)


@app.get("/", response_class=FileResponse)
async def root():
    """根路径返回 API 文档"""
    return "api/index.html"


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    健康检查接口

    返回服务状态
    """
    return HealthResponse(
        status="healthy",
        version="1.0.0",
    )


@app.get("/models", response_model=list[str])
async def list_models():
    """
    列出可用模型
    """
    model_manager = get_model_manager()
    return model_manager.list_models()


@app.get("/models/{model_name}", response_model=ModelInfo)
async def get_model_info(model_name: str):
    """
    获取模型详细信息
    """
    model_manager = get_model_manager()
    info = model_manager.get_model_info(model_name)

    if info is None:
        raise HTTPException(status_code=404, detail=f"模型不存在: {model_name}")

    return ModelInfo(
        name=model_name,
        version=info.get("epoch", "unknown"),
        device=info.get("device", "unknown"),
        loaded=model_name in model_manager.get_loaded_models(),
        trained_epochs=info.get("epoch"),
        best_dice=info.get("best_metric"),
    )


@app.post("/models/{model_name}/load")
async def load_model(model_name: str):
    """
    预加载模型

    将模型加载到内存/显存
    """
    try:
        model_manager = get_model_manager()
        model = model_manager.load_model(model_name)
        return {"status": "success", "message": f"模型已加载: {model_name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models/{model_name}/unload")
async def unload_model(model_name: str):
    """
    卸载模型

    释放模型占用的内存/显存
    """
    try:
        model_manager = get_model_manager()
        model_manager.unload_model(model_name)
        return {"status": "success", "message": f"模型已卸载: {model_name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/loaded_models", response_model=list[str])
async def get_loaded_models():
    """
    获取已加载模型列表
    """
    model_manager = get_model_manager()
    return model_manager.get_loaded_models()


@app.exception_handler(ErrorResponse)
async def error_handler(request, exc: ErrorResponse):
    """错误响应处理"""
    return exc


def run_server():
    """启动服务器"""
    uvicorn.run(
        "api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )


if __name__ == "__main__":
    run_server()
