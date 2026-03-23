"""
模型管理器

实现：
- 模型热加载/卸载
- 模型版本管理
- GPU 内存管理
"""

import os
import time
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ModelManager:
    """
    模型管理器

    负责模型的加载、缓存、卸载
    """

    def __init__(self, model_dir: str = "models", device: str = "cuda"):
        """
        Args:
            model_dir: 模型文件目录
            device: 运行设备 (cuda/mps/cpu)
        """
        self.model_dir = Path(model_dir)
        self.device = self._detect_device(device)
        self._models: Dict[str, tuple[nn.Module, Dict[str, Any]]] = {}
        self._model_info: Dict[str, Dict[str, Any]] = {}

    def _detect_device(self, device: str) -> str:
        """检测可用设备"""
        if device == "cuda" and torch.cuda.is_available():
            return "cuda"
        elif device == "mps" and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_checkpoint_info(self, checkpoint_path: str) -> Dict[str, Any]:
        """加载检查点信息"""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        return {
            "epoch": checkpoint.get("epoch", 0),
            "best_metric": checkpoint.get("best_metric", 0.0),
            "timestamp": datetime.now().isoformat(),
        }

    def load_model(self, model_name: str) -> nn.Module:
        """
        加载模型

        Args:
            model_name: 模型文件名

        Returns:
            加载的模型
        """
        if model_name in self._models:
            logger.info(f"模型已缓存: {model_name}")
            return self._models[model_name][0]

        model_path = self.model_dir / model_name
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        logger.info(f"加载模型: {model_path}")

        # 动态导入模型构建器
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))

        from src.model_builder import load_model

        model, checkpoint_info = load_model(str(model_path), device=self.device)
        model.eval()

        # 缓存模型
        self._models[model_name] = (model, checkpoint_info)
        self._model_info[model_name] = {
            "path": str(model_path),
            "loaded_at": datetime.now().isoformat(),
            "device": self.device,
            **checkpoint_info,
        }

        return model

    def unload_model(self, model_name: str):
        """
        卸载模型

        Args:
            model_name: 模型文件名
        """
        if model_name in self._models:
            del self._models[model_name]
            if model_name in self._model_info:
                del self._model_info[model_name]

            # 清理 GPU 内存
            if self.device == "cuda":
                torch.cuda.empty_cache()
            elif self.device == "mps":
                torch.mps.empty_cache()

            logger.info(f"模型已卸载: {model_name}")

    def get_model(self, model_name: str) -> nn.Module:
        """获取模型（如果未加载则自动加载）"""
        return self.load_model(model_name)

    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """获取模型信息"""
        if model_name not in self._model_info:
            model_path = self.model_dir / model_name
            if model_path.exists():
                # 加载检查点信息
                try:
                    info = self._load_checkpoint_info(str(model_path))
                    self._model_info[model_name] = {
                        "path": str(model_path),
                        "device": self.device,
                        **info,
                    }
                except Exception:
                    return None
            else:
                return None

        return self._model_info[model_name]

    def list_models(self) -> list[str]:
        """列出可用模型"""
        if not self.model_dir.exists():
            return []

        return [f.name for f in self.model_dir.glob("*.pt")]

    def get_loaded_models(self) -> list[str]:
        """获取已加载模型列表"""
        return list(self._models.keys())

    def gc(self):
        """垃圾回收，清理所有未使用的模型"""
        unloaded = []
        for name in list(self._models.keys()):
            self.unload_model(name)
            unloaded.append(name)

        return unloaded


# 全局单例
_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """获取模型管理器单例"""
    global _model_manager

    if _model_manager is None:
        from .config import get_settings
        settings = get_settings()
        _model_manager = ModelManager(
            model_dir=settings.model_dir,
            device=settings.model_device,
        )

    return _model_manager


def reset_model_manager():
    """重置模型管理器（用于测试）"""
    global _model_manager
    if _model_manager is not None:
        _model_manager.gc()
        _model_manager = None
