"""
API 服务配置

配置项：
- 服务器设置
- 模型路径
- 推理参数
"""

from typing import Optional
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """API 服务配置"""

    # 服务器
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    # 模型
    model_dir: str = "models"
    default_model: str = "best_model.pt"
    model_device: str = "cuda"  # cuda, mps, cpu

    # 推理参数
    roi_size: tuple = (128, 128, 128)
    overlap: float = 0.5
    threshold: float = 0.5
    sw_batch_size: int = 4

    # CORS
    cors_origins: list = ["*"]

    # 上传
    upload_dir: str = "data/uploads"
    max_upload_size: int = 500 * 1024 * 1024  # 500MB

    # 缓存
    enable_cache: bool = True
    cache_ttl: int = 3600  # 秒

    class Config:
        env_prefix = "MONAI_API_"
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    """获取缓存的配置单例"""
    return Settings()
