"""
损失函数定义

提供医学影像分割常用的损失函数：
- Dice Loss
- DiceCELoss (Dice + Cross Entropy)
- Focal Loss
- Tversky Loss

依赖: torch, monai
"""

from typing import Optional, Union, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.losses import (
    DiceLoss,
    DiceCELoss,
    FocalLoss,
    TverskyLoss,
    GeneralizedDiceLoss,
    SSIMLoss,
)


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation

    适用于类别不平衡的分割任务
    """

    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = True,
        squared_pred: bool = False,
        reduction: str = "mean",
        smooth: float = 1e-5,
    ):
        """
        Args:
            include_background: 是否包含背景类
            to_onehot_y: 是否将 label 转换为 one-hot 编码
            sigmoid: 是否在预测上应用 sigmoid
            squared_pred: 是否使用平方项
            reduction: 损失聚合方式 ("mean", "sum", "none")
            smooth: 平滑项，防止除零
        """
        super().__init__()
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.squared_pred = squared_pred
        self.reduction = reduction
        self.smooth = smooth

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred: 预测值 (B, C, D, H, W)
            target: 真实标签 (B, 1, D, H, W) 或 (B, D, H, W)

        Returns:
            损失值
        """
        if self.sigmoid:
            pred = torch.sigmoid(pred)

        # 确保 target 是正确的形状
        # 如果 target 有通道维度但 pred 没有，或者反过来，需要对齐
        if pred.dim() == 5 and target.dim() == 4:
            # pred: (B, C, D, H, W), target: (B, D, H, W)
            target = target.unsqueeze(1)  # -> (B, 1, D, H, W)
        elif pred.dim() == 4 and target.dim() == 5:
            # pred: (B, D, H, W), target: (B, C, D, H, W)
            pred = pred.unsqueeze(1)  # -> (B, 1, D, H, W)

        # 移除多余的通道维度（如果有）
        if pred.shape[1] > 1 and target.shape[1] == 1:
            pred = pred[:, 1:2, ...]  # 取第二个通道（如果有）
        if target.shape[1] > 1 and pred.shape[1] == 1:
            target = target[:, 1:2, ...]

        # 计算 Dice Coefficient - 将 (B, C, D, H, W) 转为 (B, -1)
        pred = pred.contiguous().view(pred.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        # 确保 pred 和 target 形状一致
        min_len = min(pred.shape[1], target.shape[1])
        pred = pred[:, :min_len]
        target = target[:, :min_len]

        # 处理 mask 类型的目标
        if target.max() > 1:
            target = (target > 0.5).float()

        intersection = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        if self.reduction == "mean":
            return 1.0 - dice.mean()
        elif self.reduction == "sum":
            return (1.0 - dice).sum()
        else:
            return 1.0 - dice


class DiceCELoss(nn.Module):
    """
    Dice + Cross Entropy 混合损失

    结合 Dice Loss 和 CE Loss 的优点：
    - Dice Loss: 对类别不平衡鲁棒
    - CE Loss: 提供逐像素梯度信号

    Total_Loss = lambda_dice * Dice_Loss + lambda_ce * CE_Loss
    """

    def __init__(
        self,
        lambda_dice: float = 1.0,
        lambda_ce: float = 0.0,  # 默认关闭 CE Loss
        include_background: bool = True,
        sigmoid: bool = True,
        reduction: str = "mean",
    ):
        """
        Args:
            lambda_dice: Dice Loss 权重
            lambda_ce: CE Loss 权重（默认0，表示仅使用Dice Loss）
            include_background: 是否包含背景类
            sigmoid: 是否在预测上应用 sigmoid
            reduction: 损失聚合方式
        """
        super().__init__()
        self.lambda_dice = lambda_dice
        self.lambda_ce = lambda_ce
        self.dice_loss = DiceLoss(
            include_background=include_background,
            sigmoid=sigmoid,
            reduction=reduction,
        )
        if lambda_ce > 0:
            self.ce_loss = nn.CrossEntropyLoss(reduction=reduction)
        else:
            self.ce_loss = None

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred: 预测值 (B, C, D, H, W)
            target: 真实标签 (B, 1, D, H, W) 或 (B, D, H, W)

        Returns:
            总损失值
        """
        dice_loss = self.dice_loss(pred, target)

        if self.ce_loss is not None and self.lambda_ce > 0:
            # CE Loss 需要处理
            if pred.shape[1] > 1:
                # 多分类
                ce_input = pred.permute(0, 1, 3, 4, 2).contiguous()
                ce_input = ce_input.view(ce_input.shape[0], ce_input.shape[1], -1)
                ce_input = ce_input.permute(0, 2, 1)
                target_flat = target.view(target.shape[0], -1).long()
                if target_flat.max() >= pred.shape[1]:
                    target_flat = (target_flat > 0).long().squeeze(1)
            else:
                # 二分类
                target_flat = target.view(target.shape[0], -1).long()
                ce_input = pred.view(pred.shape[0], -1)

            ce_loss = self.ce_loss(ce_input, target_flat)
            return self.lambda_dice * dice_loss + self.lambda_ce * ce_loss

        return self.lambda_dice * dice_loss


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss

    适用于类别严重不平衡的分割任务，特别是小目标分割
    """

    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.3,
        gamma: float = 1.0,
        sigmoid: bool = True,
        reduction: str = "mean",
        smooth: float = 1e-5,
    ):
        """
        Args:
            alpha: FP 权重 (推荐值 0.3-0.7)
            beta: FN 权重
            gamma: Focal 参数 (gamma=1 时退化为 Tversky Loss)
            sigmoid: 是否应用 sigmoid
            reduction: 损失聚合方式
            smooth: 平滑项
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.sigmoid = sigmoid
        self.reduction = reduction
        self.smooth = smooth

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred: 预测概率 (B, C, D, H, W)
            target: 真实标签 (B, 1, D, H, W)

        Returns:
            损失值
        """
        if self.sigmoid:
            pred = torch.sigmoid(pred)

        pred = pred.contiguous().view(pred.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1).float()

        # True Positive, False Positive, False Negative
        tp = (pred * target).sum(dim=1)
        fp = (pred * (1 - target)).sum(dim=1)
        fn = ((1 - pred) * target).sum(dim=1)

        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        focal_tversky = torch.pow((1 - tversky), self.gamma)

        if self.reduction == "mean":
            return focal_tversky.mean()
        elif self.reduction == "sum":
            return focal_tversky.sum()
        else:
            return focal_tversky


class BoundaryLoss(nn.Module):
    """
    Boundary Loss

    补充 Dice Loss，专注于边界区域的学习
    """

    def __init__(
        self,
        sigmoid: bool = True,
        reduction: str = "mean",
    ):
        super().__init__()
        self.sigmoid = sigmoid
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """计算边界损失"""
        if self.sigmoid:
            pred = torch.sigmoid(pred)

        # 计算距离变换（简化版）
        # 完整实现需要 scipy.ndimage.distance_transform_edt
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)

        # 简化：使用预测值与标签的差异作为边界损失代理
        boundary = torch.abs(pred - target)

        if self.reduction == "mean":
            return boundary.mean()
        elif self.reduction == "sum":
            return boundary.sum()
        return boundary


# =============================================================================
# 损失函数工厂
# =============================================================================

LOSS_REGISTRY = {
    "dice": DiceLoss,
    "dice_ce": DiceCELoss,
    "focal_tversky": FocalTverskyLoss,
    "boundary": BoundaryLoss,
    "monai_dice": DiceLoss,
    "monai_dice_ce": DiceCELoss,
    "monai_focal": FocalLoss,
    "monai_tversky": TverskyLoss,
}


def create_loss(
    loss_name: str,
    **kwargs,
) -> nn.Module:
    """
    工厂函数：创建损失函数

    Args:
        loss_name: 损失函数名称
        **kwargs: 损失函数参数

    Returns:
        nn.Module 实例
    """
    loss_name = loss_name.lower()

    if loss_name not in LOSS_REGISTRY:
        raise ValueError(
            f"未知的损失函数: {loss_name}，可用: {list(LOSS_REGISTRY.keys())}"
        )

    loss_class = LOSS_REGISTRY[loss_name]
    return loss_class(**kwargs)


def get_loss_by_config(config: dict) -> nn.Module:
    """
    根据配置字典创建损失函数

    Args:
        config: 配置字典，格式:
            {
                "type": "dice_ce",
                "lambda_dice": 1.0,
                "lambda_ce": 1.0,
            }

    Returns:
        nn.Module 实例
    """
    loss_type = config.get("type", "dice_ce")
    loss_params = {k: v for k, v in config.items() if k != "type"}

    return create_loss(loss_type, **loss_params)


# =============================================================================
# 预定义损失配置
# =============================================================================

# 标准 DiceCELoss（推荐用于大多数分割任务）
STANDARD_LOSS_CONFIG = {
    "type": "dice_ce",
    "lambda_dice": 1.0,
    "lambda_ce": 1.0,
    "sigmoid": True,
}

# 适用于小目标分割的 Focal Tversky Loss
SMALL_TARGET_LOSS_CONFIG = {
    "type": "focal_tversky",
    "alpha": 0.3,
    "beta": 0.7,
    "gamma": 1.0,
}

# 边界增强损失
BOUNDARY_AWARE_LOSS_CONFIG = {
    "type": "dice_ce",
    "lambda_dice": 0.8,
    "lambda_ce": 1.0,
}


__all__ = [
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
