"""
MONAI Transforms 流水线定义

提供训练、验证、推理阶段的数据预处理与增强 Transform

依赖: monai, nibabel, numpy
"""

from typing import Optional, Sequence, Union, Tuple, Dict, Any
from pathlib import Path

import numpy as np
import torch

# MONAI Transform 相关导入
from monai.transforms import (
    # 基础 IO
    LoadImaged,
    EnsureChannelFirstd,

    # 空间变换
    Spacingd,
    Orientationd,
    Resized,

    # 强度变换
    ScaleIntensityRanged,
    NormalizeIntensityd,

    # 数据增强
    RandRotate90d,
    RandFlipd,
    RandAffined,
    RandCropByPosNegLabeld,

    # 转换
    ToTensord,

    # 后处理
    Activationsd,
    AsDiscreted,
    KeepLargestConnectedComponentd,
)

from monai.transforms.compose import Compose


# =============================================================================
# 通用配置
# =============================================================================

# 默认 Patch 尺寸配置（基于 GPU 显存）
DEFAULT_PATCH_SIZE = (128, 128, 128)
DEFAULT_SPACING = (1.0, 1.0, 1.0)  # 1mm³ 体素

# CT 窗宽窗位预设
CT_PRESETS = {
    "abdomen": {"a_min": -175, "a_max": 250, "b_min": 0.0, "b_max": 1.0},  # 腹脏器
    "brain": {"a_min": 0, "a_max": 80, "b_min": 0.0, "b_max": 1.0},        # 脑窗
    "lung": {"a_min": -1000, "a_max": 200, "b_min": 0.0, "b_max": 1.0},   # 肺窗
    "bone": {"a_min": -200, "a_max": 1000, "b_min": 0.0, "b_max": 1.0},   # 骨窗
}


# =============================================================================
# 训练数据 Transform Pipeline
# =============================================================================

def get_train_transforms(
    spatial_size: Union[Sequence[int], Tuple[int, int, int]] = (96, 96, 96),
    pixdim: Optional[Tuple[float, float, float]] = (1.0, 1.0, 1.0),
    ct_window: Optional[str] = "abdomen",
    pos_ratio: float = 0.5,
    num_samples: int = 1,
    image_keys: Tuple[str, str] = ("image", "label"),
    enable_augmentation: bool = True,
) -> Compose:
    """
    获取训练数据 Transform 流水线

    Args:
        spatial_size: Patch 裁剪尺寸
        pixdim: 重采样目标间距
        ct_window: CT 窗宽窗位预设 (None, "abdomen", "brain", "lung", "bone")
        pos_ratio: 正负样本裁剪比例
        num_samples: 每次采样数量
        image_keys: 图像和标签的 keys
        enable_augmentation: 是否启用数据增强

    Returns:
        MONAI Compose 对象
    """
    image_key, label_key = image_keys

    # CT 窗口设置
    if ct_window and ct_window in CT_PRESETS:
        ct_params = CT_PRESETS[ct_window]
    else:
        ct_params = {"a_min": 0, "a_max": 1, "b_min": 0.0, "b_max": 1.0}

    # 构建 Transform 列表
    train_transforms = [
        # 1. 加载 NIfTI
        LoadImaged(keys=[image_key, label_key], image_only=True),
        EnsureChannelFirstd(keys=[image_key, label_key]),

        # 2. CT 强度裁剪与归一化
        ScaleIntensityRanged(
            keys=[image_key],
            a_min=ct_params["a_min"],
            a_max=ct_params["a_max"],
            b_min=ct_params["b_min"],
            b_max=ct_params["b_max"],
            clip=True,
        ),
    ]

    # 5. 简单的随机裁剪（不使用复杂的正负标签裁剪）
    # 使用 Resized 来确保统一的输入尺寸
    if spatial_size:
        train_transforms.append(
            Resized(
                keys=[image_key, label_key],
                spatial_size=spatial_size,
                mode=["bilinear", "nearest"],
            )
        )

    # 6. 数据增强（可选）
    if enable_augmentation:
        train_transforms.extend([
            # 随机翻转
            RandFlipd(
                keys=[image_key, label_key],
                spatial_axis=[0],
                prob=0.5,
            ),
            RandFlipd(
                keys=[image_key, label_key],
                spatial_axis=[1],
                prob=0.5,
            ),
            # 随机旋转（小幅度的 3D 旋转）
            RandAffined(
                keys=[image_key, label_key],
                mode=["bilinear", "nearest"],
                prob=0.3,
                rotate_range=(15, 15, 15),
                scale_range=(0.1, 0.1, 0.1),
                translate_range=(5, 5, 5),
            ),
        ])

    # 7. 转换为 Tensor
    train_transforms.append(
        ToTensord(keys=[image_key, label_key])
    )

    return Compose(train_transforms)


# =============================================================================
# 验证数据 Transform Pipeline
# =============================================================================

def get_val_transforms(
    spatial_size: Optional[Union[Sequence[int], Tuple[int, int, int]]] = None,
    pixdim: Optional[Tuple[float, float, float]] = (1.0, 1.0, 1.0),
    ct_window: Optional[str] = "abdomen",
    image_keys: Tuple[str, str] = ("image", "label"),
) -> Compose:
    """
    获取验证数据 Transform 流水线（无数据增强）

    Args:
        spatial_size: 若指定，则将图像 resize 到此尺寸；否则保持原尺寸
        pixdim: 重采样目标间距
        ct_window: CT 窗宽窗位预设
        image_keys: 图像和标签的 keys

    Returns:
        MONAI Compose 对象
    """
    image_key, label_key = image_keys

    # CT 窗口设置
    if ct_window and ct_window in CT_PRESETS:
        ct_params = CT_PRESETS[ct_window]
    else:
        ct_params = {"a_min": 0, "a_max": 1, "b_min": 0.0, "b_max": 1.0}

    val_transforms = [
        LoadImaged(keys=[image_key, label_key], image_only=True),
        EnsureChannelFirstd(keys=[image_key, label_key]),
        ScaleIntensityRanged(
            keys=[image_key],
            a_min=ct_params["a_min"],
            a_max=ct_params["a_max"],
            b_min=ct_params["b_min"],
            b_max=ct_params["b_max"],
            clip=True,
        ),
    ]

    # 添加 Resized 确保统一尺寸
    if spatial_size:
        val_transforms.append(
            Resized(
                keys=[image_key, label_key],
                spatial_size=spatial_size,
                mode=["bilinear", "nearest"],
            )
        )

    val_transforms.append(ToTensord(keys=[image_key, label_key]))

    return Compose(val_transforms)


# =============================================================================
# 推理数据 Transform Pipeline
# =============================================================================

def get_inference_transforms(
    pixdim: Optional[Tuple[float, float, float]] = (1.0, 1.0, 1.0),
    ct_window: Optional[str] = "abdomen",
    image_key: str = "image",
) -> Compose:
    """
    获取推理数据 Transform 流水线（仅处理图像，不需要 label）

    Args:
        pixdim: 重采样目标间距
        ct_window: CT 窗宽窗位预设
        image_key: 图像的 key

    Returns:
        MONAI Compose 对象
    """
    # CT 窗口设置
    if ct_window and ct_window in CT_PRESETS:
        ct_params = CT_PRESETS[ct_window]
    else:
        ct_params = {"a_min": 0, "a_max": 1, "b_min": 0.0, "b_max": 1.0}

    inference_transforms = [
        LoadImaged(keys=[image_key], image_only=True),
        EnsureChannelFirstd(keys=[image_key]),
        ScaleIntensityRanged(
            keys=[image_key],
            a_min=ct_params["a_min"],
            a_max=ct_params["a_max"],
            b_min=ct_params["b_min"],
            b_max=ct_params["b_max"],
            clip=True,
        ),
        ToTensord(keys=[image_key]),
    ]

    return Compose(inference_transforms)


# =============================================================================
# 推理后处理 Transform Pipeline
# =============================================================================

def get_postprocess_transforms(
    threshold: float = 0.5,
    keep_largest: bool = True,
    apply_over_labels: Optional[Sequence[int]] = None,
) -> Compose:
    """
    获取推理后处理 Transform 流水线

    Args:
        threshold: 二值化阈值
        keep_largest: 是否保留最大连通域
        apply_over_labels: 应用于哪些标签（None 表示所有标签）

    Returns:
        MONAI Compose 对象
    """
    post_transforms = [
        # Sigmoid 激活
        Activationsd(keys="pred", sigmoid=True),

        # 阈值化
        AsDiscreted(keys="pred", threshold=threshold),

        # 保留最大连通域（去除孤立噪声）
    ]

    if keep_largest:
        post_transforms.append(
            KeepLargestConnectedComponentd(
                keys="pred",
                applied_over_labels=apply_over_labels or [1],
            )
        )

    return Compose(post_transforms)


# =============================================================================
# 辅助函数
# =============================================================================

def create_transforms_from_config(config: Dict[str, Any]) -> Tuple[Compose, Compose, Compose]:
    """
    根据配置字典创建训练、验证、推理 Transform

    Args:
        config: 配置字典

    Returns:
        (train_transforms, val_transforms, inference_transforms)
    """
    # 提取数据配置
    data_config = config.get("data", {})
    spatial_size = tuple(data_config.get("spatial_size", DEFAULT_PATCH_SIZE))
    pixdim = tuple(data_config.get("pixdim", DEFAULT_SPACING))
    ct_window = data_config.get("ct_window", "abdomen")
    pos_ratio = data_config.get("pos_ratio", 0.5)
    enable_augmentation = data_config.get("enable_augmentation", True)

    # 创建 Transform
    train_transforms = get_train_transforms(
        spatial_size=spatial_size,
        pixdim=pixdim,
        ct_window=ct_window,
        pos_ratio=pos_ratio,
        enable_augmentation=enable_augmentation,
    )

    val_transforms = get_val_transforms(
        spatial_size=None,  # 验证时保持原尺寸
        pixdim=pixdim,
        ct_window=ct_window,
    )

    inference_transforms = get_inference_transforms(
        pixdim=pixdim,
        ct_window=ct_window,
    )

    return train_transforms, val_transforms, inference_transforms


__all__ = [
    "DEFAULT_PATCH_SIZE",
    "DEFAULT_SPACING",
    "CT_PRESETS",
    "get_train_transforms",
    "get_val_transforms",
    "get_inference_transforms",
    "get_postprocess_transforms",
    "create_transforms_from_config",
]
