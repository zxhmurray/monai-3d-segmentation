"""
Pydantic 数据模型

定义 API 请求/响应的数据结构
"""

from typing import Optional, List
from pydantic import BaseModel, Field
from datetime import datetime


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = "healthy"
    version: str = "1.0.0"
    timestamp: datetime = Field(default_factory=datetime.now)


class ModelInfo(BaseModel):
    """模型信息"""
    name: str
    version: str
    device: str
    loaded: bool
    trained_epochs: Optional[int] = None
    best_dice: Optional[float] = None


class InferenceRequest(BaseModel):
    """推理请求"""
    model_name: Optional[str] = None
    threshold: Optional[float] = 0.5
    overlap: Optional[float] = 0.5
    return_prob: bool = False


class InferenceResult(BaseModel):
    """推理结果"""
    case_id: str
    status: str
    message: str
    prediction_path: Optional[str] = None
    dice_score: Optional[float] = None
    volume_cm3: Optional[float] = None
    voxel_count: Optional[int] = None
    processing_time: float


class BatchInferenceRequest(BaseModel):
    """批量推理请求"""
    model_name: Optional[str] = None
    file_paths: List[str]
    threshold: Optional[float] = 0.5
    overlap: Optional[float] = 0.5


class BatchInferenceResult(BaseModel):
    """批量推理结果"""
    total: int
    success: int
    failed: int
    results: List[InferenceResult]


class VolumeReport(BaseModel):
    """体积报告"""
    case_id: str
    voxel_count: int
    volume_mm3: float
    volume_cm3: float
    spacing: tuple


class ErrorResponse(BaseModel):
    """错误响应"""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
