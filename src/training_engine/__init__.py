"""
Training Engine 模块

提供训练循环、验证、早停、检查点管理等功能

Submodules:
    trainer: 训练循环与验证逻辑
    loss: 损失函数定义
"""

from .trainer import (
    TrainingConfig,
    TrainingHistory,
    Trainer,
    train_with_config,
)

from .loss import (
    DiceLoss,
    DiceCELoss,
    FocalTverskyLoss,
    BoundaryLoss,
    create_loss,
    get_loss_by_config,
    LOSS_REGISTRY,
    STANDARD_LOSS_CONFIG,
    SMALL_TARGET_LOSS_CONFIG,
    BOUNDARY_AWARE_LOSS_CONFIG,
)

__all__ = [
    # Trainer
    "TrainingConfig",
    "TrainingHistory",
    "Trainer",
    "train_with_config",
    # Loss
    "DiceLoss",
    "DiceCELoss",
    "FocalTverskyLoss",
    "BoundaryLoss",
    "create_loss",
    "get_loss_by_config",
    "LOSS_REGISTRY",
    "STANDARD_LOSS_CONFIG",
    "SMALL_TARGET_LOSS_CONFIG",
    "BOUNDARY_AWARE_LOSS_CONFIG",
]
