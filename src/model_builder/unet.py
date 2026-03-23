"""
3D U-Net 模型定义

基于 MONAI 框架构建 3D U-Net 分割网络

依赖: torch, monai
"""

from typing import Optional, Tuple, Union, Sequence

import torch
import torch.nn as nn

from monai.networks.nets import UNet, BasicUNet, DynUNet, AttentionUnet
from monai.networks.layers.factories import Act, Norm


class Medical3DUNet(nn.Module):
    """
    3D U-Net 分割网络

    封装 MONAI UNet，提供统一的接口

    Example:
        model = Medical3DUNet(
            in_channels=1,
            out_channels=1,
            channels=(32, 64, 128, 256),
        )
        output = model(input)  # input: (B, 1, D, H, W)
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        channels: Union[Sequence[int], Tuple[int, ...]] = (32, 64, 128, 256),
        strides: Union[Sequence[int], Tuple[int, ...]] = (2, 2, 2),
        num_res_units: int = 2,
        norm: str = "batch",
        act: str = "relu",
        dropout: float = 0.1,
        bias: bool = True,
    ):
        """
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数（分割类别数）
            channels: 编码器各层通道数
            strides: 下采样步长
            num_res_units: 残差单元数量
            norm: 归一化方法 ("batch", "instance", "layer")
            act: 激活函数 ("relu", "prelu", "leaky_relu")
            dropout: Dropout 比率
            bias: 是否使用偏置
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides

        # 构建 MONAI UNet
        self.model = UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
            norm=norm,
            act=act,
            dropout=dropout,
            bias=bias,
        )

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                # Kaiming 正态初始化（适合 ReLU 激活）
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm3d, nn.InstanceNorm3d)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入张量 (B, C, D, H, W)

        Returns:
            输出张量 (B, out_channels, D, H, W)
        """
        return self.model(x)

    def get_num_params(self) -> int:
        """获取模型参数总数"""
        return sum(p.numel() for p in self.parameters())

    def get_trainable_params(self) -> int:
        """获取可训练参数数"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self) -> dict:
        """返回模型摘要信息"""
        num_params = self.get_num_params()
        trainable_params = self.get_trainable_params()

        return {
            "name": "Medical3DUNet",
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "channels": self.channels,
            "strides": self.strides,
            "num_params": num_params,
            "trainable_params": trainable_params,
            "non_trainable_params": num_params - trainable_params,
        }


class Medical3DUNetLite(nn.Module):
    """
    轻量级 3D U-Net（适用于低显存环境）

    使用更少的通道数和残差单元
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
    ):
        super().__init__()
        self.model = UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(16, 32, 64, 128),  # 减少通道数
            strides=(2, 2, 2),
            num_res_units=1,
            norm="batch",
            act="relu",
            dropout=0.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Medical3DUNetLarge(nn.Module):
    """
    大型 3D U-Net（适用于高显存环境）

    使用更多通道数和残差单元
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
    ):
        super().__init__()
        self.model = UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(64, 128, 256, 512),  # 增加通道数
            strides=(2, 2, 2),
            num_res_units=3,
            norm="batch",
            act="relu",
            dropout=0.2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# =============================================================================
# 工厂函数
# =============================================================================

def create_3d_unet(
    model_size: str = "medium",
    in_channels: int = 1,
    out_channels: int = 1,
    **kwargs,
) -> nn.Module:
    """
    工厂函数：创建 3D U-Net 模型

    Args:
        model_size: 模型大小 ("tiny", "small", "medium", "large")
        in_channels: 输入通道数
        out_channels: 输出通道数
        **kwargs: 其他模型参数

    Returns:
        nn.Module 实例
    """
    presets = {
        "tiny": {
            "channels": (8, 16, 32, 64),
            "num_res_units": 1,
            "dropout": 0.0,
        },
        "small": {
            "channels": (16, 32, 64, 128),
            "num_res_units": 1,
            "dropout": 0.1,
        },
        "medium": {
            "channels": (32, 64, 128, 256),
            "num_res_units": 2,
            "dropout": 0.1,
        },
        "large": {
            "channels": (64, 128, 256, 512),
            "num_res_units": 3,
            "dropout": 0.2,
        },
    }

    if model_size.lower() not in presets:
        raise ValueError(f"未知的模型大小: {model_size}，可用: {list(presets.keys())}")

    config = presets[model_size.lower()]
    config.update(kwargs)

    model = Medical3DUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        **config,
    )

    return model


def load_model(
    checkpoint_path: str,
    model: Optional[nn.Module] = None,
    device: str = "cuda",
) -> Tuple[nn.Module, dict]:
    """
    加载模型检查点

    Args:
        checkpoint_path: 检查点路径
        model: 模型实例（若为 None，则创建新模型）
        device: 设备

    Returns:
        (model, checkpoint_info)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if model is None:
        # 从检查点恢复模型结构
        config = checkpoint.get("config", {})
        model = create_3d_unet(
            model_size=config.get("model_size", "medium"),
            in_channels=config.get("in_channels", 1),
            out_channels=config.get("out_channels", 1),
        )

    # 加载权重
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    info = {
        "epoch": checkpoint.get("epoch", None),
        "best_metric": checkpoint.get("best_metric", None),
        "config": checkpoint.get("config", {}),
    }

    return model, info


def save_model(
    model: nn.Module,
    checkpoint_path: str,
    epoch: Optional[int] = None,
    best_metric: Optional[float] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    config: Optional[dict] = None,
) -> None:
    """
    保存模型检查点

    Args:
        model: 模型实例
        checkpoint_path: 保存路径
        epoch: 当前 epoch
        best_metric: 最佳指标
        optimizer: 优化器实例
        config: 配置信息
    """
    import os
    from pathlib import Path

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
        "config": config or {},
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(checkpoint, checkpoint_path)


# =============================================================================
# 模型注册表
# =============================================================================

MODEL_REGISTRY = {
    "3d_unet_tiny": lambda: create_3d_unet("tiny"),
    "3d_unet_small": lambda: create_3d_unet("small"),
    "3d_unet_medium": lambda: create_3d_unet("medium"),
    "3d_unet_large": lambda: create_3d_unet("large"),
    "3d_unet_lite": Medical3DUNetLite,
    "3d_unet_large_full": Medical3DUNetLarge,
}


def create_model(model_name: str, **kwargs) -> nn.Module:
    """
    通过名称创建模型

    Args:
        model_name: 模型注册名称
        **kwargs: 模型参数

    Returns:
        nn.Module 实例
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"未知的模型名称: {model_name}，可用: {list(MODEL_REGISTRY.keys())}"
        )

    factory_fn = MODEL_REGISTRY[model_name]
    return factory_fn(**kwargs)


__all__ = [
    "Medical3DUNet",
    "Medical3DUNetLite",
    "Medical3DUNetLarge",
    "create_3d_unet",
    "load_model",
    "save_model",
    "create_model",
    "MODEL_REGISTRY",
]
