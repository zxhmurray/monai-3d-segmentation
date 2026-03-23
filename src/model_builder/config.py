"""
模型配置文件

定义 3D U-Net 网络架构的超参数

依赖: torch, monai
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Union, Dict, Any
from pathlib import Path
import json


@dataclass
class NetworkConfig:
    """3D U-Net 网络配置"""

    # 空间维度
    spatial_dims: int = 3

    # 输入输出通道数
    in_channels: int = 1
    out_channels: int = 1

    # 编码器各层通道数
    channels: Tuple[int, int, int, int] = (32, 64, 128, 256)

    # 下采样步长
    strides: Tuple[int, int, int] = (2, 2, 2)

    # 编码器/解码器层数（由 channels 长度决定）
    num_levels: int = 4

    # 每个 Block 的残差单元数
    num_res_units: int = 2

    # 归一化方法: "batch", "instance", "layer", "group"
    norm: str = "batch"

    # 激活函数: "relu", "prelu", "leaky_relu", "gelu"
    act: str = "relu"

    # Dropout 比率
    dropout: float = 0.1

    # 权重初始化
    weight_init: str = "kaiming"  # "kaiming", "xavier", "orthogonal"

    def __post_init__(self):
        # 自动计算 num_levels
        self.num_levels = len(self.channels)

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "spatial_dims": self.spatial_dims,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "channels": list(self.channels),
            "strides": list(self.strides),
            "num_levels": self.num_levels,
            "num_res_units": self.num_res_units,
            "norm": self.norm,
            "act": self.act,
            "dropout": self.dropout,
            "weight_init": self.weight_init,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NetworkConfig":
        """从字典创建"""
        # 处理 tuple 转换
        if isinstance(d.get("channels"), list):
            d["channels"] = tuple(d["channels"])
        if isinstance(d.get("strides"), list):
            d["strides"] = tuple(d["strides"])
        return cls(**d)

    def save(self, path: Union[Path, str]) -> None:
        """保存配置到 JSON 文件"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Union[Path, str]) -> "NetworkConfig":
        """从 JSON 文件加载配置"""
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))


# =============================================================================
# 预定义配置
# =============================================================================

# 小型模型（适用于 6GB GPU）
SMALL_MODEL_CONFIG = NetworkConfig(
    channels=(16, 32, 64, 128),
    strides=(2, 2, 2),
    num_res_units=1,
    dropout=0.1,
)

# 中型模型（适用于 8-12GB GPU）
MEDIUM_MODEL_CONFIG = NetworkConfig(
    channels=(32, 64, 128, 256),
    strides=(2, 2, 2),
    num_res_units=2,
    dropout=0.1,
)

# 大型模型（适用于 24GB+ GPU）
LARGE_MODEL_CONFIG = NetworkConfig(
    channels=(64, 128, 256, 512),
    strides=(2, 2, 2),
    num_res_units=3,
    dropout=0.2,
)

# 脾脏分割专用配置（基于经验调优）
SPLEEN_SEGMENTATION_CONFIG = NetworkConfig(
    in_channels=1,
    out_channels=1,
    channels=(32, 64, 128, 256),
    strides=(2, 2, 2),
    num_res_units=2,
    norm="batch",
    act="relu",
    dropout=0.1,
)


def get_preset_config(name: str) -> NetworkConfig:
    """获取预定义配置

    Args:
        name: 预设名称 ("small", "medium", "large", "spleen")

    Returns:
        NetworkConfig 实例
    """
    presets = {
        "small": SMALL_MODEL_CONFIG,
        "medium": MEDIUM_MODEL_CONFIG,
        "large": LARGE_MODEL_CONFIG,
        "spleen": SPLEEN_SEGMENTATION_CONFIG,
    }

    if name.lower() not in presets:
        raise ValueError(f"未知的预设配置: {name}，可用: {list(presets.keys())}")

    return presets[name.lower()]


# =============================================================================
# 显存估算
# =============================================================================

def estimate_model_memory(config: NetworkConfig, input_size: Tuple[int, int, int, int]) -> dict:
    """
    估算模型显存占用（粗略估计）

    Args:
        config: 网络配置
        input_size: 输入尺寸 (B, C, D, H, W)

    Returns:
        显存估算字典
    """
    batch_size = input_size[0]

    # 估算各层输出尺寸和参数量
    total_params = 0
    peak_activation_size = 0

    # 简化的 ResNet Block 参数量估算
    # 每个 Block: 2 * (in_channels * out_channels * kernel_size^3 * 2 + bn_params + out_channels * 2)
    # 假设 kernel_size = 3

    def count_block_params(in_ch, out_ch, num_units):
        # Conv3d params: in_ch * out_ch * 3^3 * 2 (for two convs)
        # BN params: 2 * out_ch * 2 (running_mean, running_var, weight, bias)
        base_params = in_ch * out_ch * 27 * 2 + out_ch * 4
        bn_params = out_ch * 4
        return base_params * num_units + bn_params * num_units

    # 编码器
    in_ch = config.in_channels
    current_size = input_size[1:]  # D, H, W
    sizes = [current_size]

    for i, out_ch in enumerate(config.channels):
        total_params += count_block_params(in_ch, out_ch, config.num_res_units)
        # 下采样后尺寸
        current_size = tuple(s // st for s, st in zip(current_size, (config.strides * config.spatial_dims)[:3]))
        sizes.append(current_size)
        in_ch = out_ch

    # Bottleneck
    total_params += count_block_params(in_ch, in_ch * 2, config.num_res_units)

    # 解码器
    decoder_in_ch = in_ch * 2
    for i, out_ch in enumerate(reversed(config.channels)):
        # Upconv + skip connection conv
        total_params += count_block_params(decoder_in_ch, out_ch, config.num_res_units)
        decoder_in_ch = out_ch

    # Final conv
    total_params += config.in_channels * config.out_channels * 27  # 1x1x1 conv

    # 参数量 -> 参数量 * 4 bytes (FP32)
    params_memory_mb = total_params * 4 / (1024 ** 2)

    # 激活值估算（简化）
    # 全分辨率层激活值
    peak_activation_size = batch_size * config.channels[0] * sizes[0][0] * sizes[0][1] * sizes[0][2]

    # 中间层激活值
    for size in sizes[1:]:
        peak_activation_size += batch_size * config.channels[-1] * size[0] * size[1] * size[2]

    activation_memory_mb = peak_activation_size * 4 / (1024 ** 2)

    # 梯度估算（约等于参数内存）
    gradient_memory_mb = params_memory_mb

    # 优化器状态（Adam: 2x 参数内存）
    optimizer_memory_mb = params_memory_mb * 2

    # 总计（不含推理时的中间激活）
    training_memory_mb = (
        params_memory_mb +          # 模型参数
        gradient_memory_mb +         # 梯度
        optimizer_memory_mb +        # 优化器状态
        activation_memory_mb * 0.3   # 部分激活值（简化估算）
    )

    inference_memory_mb = (
        params_memory_mb +
        activation_memory_mb
    )

    return {
        "params_mb": round(params_memory_mb, 2),
        "gradient_mb": round(gradient_memory_mb, 2),
        "optimizer_mb": round(optimizer_memory_mb, 2),
        "activation_mb": round(activation_memory_mb, 2),
        "training_total_mb": round(training_memory_mb, 2),
        "inference_total_mb": round(inference_memory_mb, 2),
        "estimated_params": total_params,
    }


# =============================================================================
# 类型别名
# =============================================================================


__all__ = [
    "NetworkConfig",
    "SMALL_MODEL_CONFIG",
    "MEDIUM_MODEL_CONFIG",
    "LARGE_MODEL_CONFIG",
    "SPLEEN_SEGMENTATION_CONFIG",
    "get_preset_config",
    "estimate_model_memory",
]
