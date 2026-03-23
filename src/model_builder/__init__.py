"""
Model Builder 模块

提供 3D U-Net 网络架构的构建与配置

Submodules:
    config: 网络配置定义
    unet: 3D U-Net 模型实现
"""

from .config import (
    NetworkConfig,
    SMALL_MODEL_CONFIG,
    MEDIUM_MODEL_CONFIG,
    LARGE_MODEL_CONFIG,
    SPLEEN_SEGMENTATION_CONFIG,
    get_preset_config,
    estimate_model_memory,
)

from .unet import (
    Medical3DUNet,
    Medical3DUNetLite,
    Medical3DUNetLarge,
    create_3d_unet,
    load_model,
    save_model,
    create_model,
    MODEL_REGISTRY,
)

__all__ = [
    # Config
    "NetworkConfig",
    "SMALL_MODEL_CONFIG",
    "MEDIUM_MODEL_CONFIG",
    "LARGE_MODEL_CONFIG",
    "SPLEEN_SEGMENTATION_CONFIG",
    "get_preset_config",
    "estimate_model_memory",
    # UNet
    "Medical3DUNet",
    "Medical3DUNetLite",
    "Medical3DUNetLarge",
    "create_3d_unet",
    "load_model",
    "save_model",
    "create_model",
    "MODEL_REGISTRY",
]
