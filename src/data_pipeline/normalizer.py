"""
医学影像强度归一化模块

提供：
- CT 窗宽窗位裁剪
- Min-Max 归一化
- Z-Score 标准化
- 自适应标准化

依赖: numpy, SimpleITK
"""

from typing import Tuple, Optional, Union, Callable
from enum import Enum

import numpy as np


class NormalizationMethod(Enum):
    """归一化方法枚举"""
    MIN_MAX = "min_max"
    Z_SCORE = "z_score"
    CT_WINDOW = "ct_window"
    ADAPTIVE = "adaptive"


class ImageNormalizer:
    """医学影像强度归一化器

    Example:
        # CT 图像归一化
        normalizer = CTWindowNormalizer(window_center=40, window_width=400)
        normalized = normalizer.normalize(ct_array)

        # Min-Max 归一化到 [0, 1]
        normalizer = MinMaxNormalizer()
        normalized = normalizer.normalize(image_array)
    """

    def __init__(self, method: str = "min_max", **kwargs):
        """
        Args:
            method: 归一化方法
            **kwargs: 方法特定参数
        """
        self.method = NormalizationMethod(method)
        self.params = kwargs

    def normalize(self, data: np.ndarray) -> np.ndarray:
        """执行归一化"""
        raise NotImplementedError("子类必须实现 normalize 方法")

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """反归一化（如果支持）"""
        raise NotImplementedError("子类可以选择实现 inverse_transform")


class MinMaxNormalizer(ImageNormalizer):
    """Min-Max 归一化

    将图像强度线性缩放到 [min_val, max_val] 范围
    """

    def __init__(
        self,
        min_val: float = 0.0,
        max_val: float = 1.0,
        per_channel: bool = False,
    ):
        """
        Args:
            min_val: 目标范围最小值
            max_val: 目标范围最大值
            per_channel: 是否对每个通道分别归一化
        """
        super().__init__("min_max")
        self.min_val = min_val
        self.max_val = max_val
        self.per_channel = per_channel

    def normalize(self, data: np.ndarray) -> np.ndarray:
        """执行 Min-Max 归一化"""
        data = data.astype(np.float32)

        if self.per_channel and len(data.shape) > 3:
            # 对多通道图像的每个通道分别归一化
            result = np.zeros_like(data)
            for c in range(data.shape[0]):
                result[c] = self._normalize_channel(data[c])
            return result
        else:
            return self._normalize_channel(data)

    def _normalize_channel(self, channel: np.ndarray) -> np.ndarray:
        """归一化单个通道"""
        channel_min = channel.min()
        channel_max = channel.max()

        if channel_max - channel_min < 1e-8:
            # 常数图像
            return np.full_like(channel, self.min_val)

        normalized = (channel - channel_min) / (channel_max - channel_min)
        normalized = normalized * (self.max_val - self.min_val) + self.min_val
        return normalized


class ZScoreNormalizer(ImageNormalizer):
    """Z-Score 标准化（均值方差归一化）

    将图像转换为零均值、单位方差的分布
    """

    def __init__(
        self,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        epsilon: float = 1e-8,
    ):
        """
        Args:
            mean: 预设均值（若为 None，则从数据计算）
            std: 预设标准差（若为 None，则从数据计算）
            epsilon: 防止除零的小常数
        """
        super().__init__("z_score")
        self.mean = mean
        self.std = std
        self.epsilon = epsilon

    def normalize(self, data: np.ndarray) -> np.ndarray:
        """执行 Z-Score 标准化"""
        data = data.astype(np.float32)

        if self.mean is None:
            mean = data.mean()
        else:
            mean = self.mean

        if self.std is None:
            std = data.std()
        else:
            std = self.std

        if std < self.epsilon:
            return data - mean

        normalized = (data - mean) / std
        return normalized


class CTWindowNormalizer(ImageNormalizer):
    """CT 窗宽窗位归一化

    根据 CT 图像的窗宽窗位（Window Center/Width）进行裁剪和归一化
    常用于腹脏器窗: WC=40, WW=400
             肺窗: WC=-600, WW=1600
             骨窗: WC=400, WW=1800
    """

    def __init__(
        self,
        window_center: float = 40,
        window_width: float = 400,
        output_range: Tuple[float, float] = (0, 1),
    ):
        """
        Args:
            window_center: 窗位 (Window Center / Level)
            window_width: 窗宽 (Window Width)
            output_range: 输出强度范围
        """
        super().__init__("ct_window")
        self.window_center = window_center
        self.window_width = window_width
        self.output_range = output_range

        # 计算窗位上下的强度值
        self.lower_bound = window_center - window_width / 2
        self.upper_bound = window_center + window_width / 2

    def normalize(self, data: np.ndarray) -> np.ndarray:
        """执行 CT 窗宽窗位归一化"""
        data = data.astype(np.float32)

        # 裁剪到窗口范围
        clipped = np.clip(data, self.lower_bound, self.upper_bound)

        # 归一化到输出范围
        normalized = (clipped - self.lower_bound) / self.window_width
        normalized = normalized * (self.output_range[1] - self.output_range[0]) + self.output_range[0]

        return normalized

    @staticmethod
    def get_preset(preset_name: str) -> "CTWindowNormalizer":
        """获取预定义的 CT 窗设置

        Args:
            preset_name: 预设名称
                - " abdomen": 腹脏器窗 WC=40, WW=400
                - "lung": 肺窗 WC=-600, WW=1600
                - "bone": 骨窗 WC=400, WW=1800
                - "brain": 脑窗 WC=40, WW=80
                - "liver": 肝脏窗 WC=60, WW=150

        Returns:
            配置好的 CTWindowNormalizer 实例
        """
        presets = {
            "abdomen": (40, 400),
            "lung": (-600, 1600),
            "bone": (400, 1800),
            "brain": (40, 80),
            "liver": (60, 150),
            "spleen": (40, 400),  # 脾脏与腹脏器类似
        }

        if preset_name.lower() not in presets:
            raise ValueError(f"未知的 CT 窗预设: {preset_name}，可用: {list(presets.keys())}")

        wc, ww = presets[preset_name.lower()]
        return CTWindowNormalizer(window_center=wc, window_width=ww)


class AdaptiveNormalizer(ImageNormalizer):
    """自适应归一化

    根据图像的局部/全局统计特性自适应选择归一化参数
    """

    def __init__(
        self,
        num_histogram_bins: int = 256,
        percentile: Tuple[float, float] = (1, 99),
    ):
        """
        Args:
            num_histogram_bins: 直方图 bin 数量
            percentile: 用于裁剪的百分位范围
        """
        super().__init__("adaptive")
        self.num_histogram_bins = num_histogram_bins
        self.percentile = percentile

    def normalize(self, data: np.ndarray) -> np.ndarray:
        """执行自适应归一化"""
        data = data.astype(np.float32)

        # 计算百分位裁剪范围
        lower = np.percentile(data, self.percentile[0])
        upper = np.percentile(data, self.percentile[1])

        # 裁剪并归一化
        clipped = np.clip(data, lower, upper)

        if upper - lower < 1e-8:
            return np.zeros_like(data)

        normalized = (clipped - lower) / (upper - lower)
        return normalized


class ClipNormalizer(ImageNormalizer):
    """简单的强度裁剪归一化

    裁剪超出范围的值为边界值，不进行缩放
    """

    def __init__(
        self,
        lower: Optional[float] = None,
        upper: Optional[float] = None,
    ):
        """
        Args:
            lower: 下界（若为 None，使用数据的 min）
            upper: 上界（若为 None，使用数据的 max）
        """
        super().__init__("clip")
        self.lower = lower
        self.upper = upper

    def normalize(self, data: np.ndarray) -> np.ndarray:
        """执行裁剪归一化"""
        data = data.astype(np.float32)

        lower = self.lower if self.lower is not None else data.min()
        upper = self.upper if self.upper is not None else data.max()

        return np.clip(data, lower, upper)


def create_normalizer(
    method: str,
    **kwargs
) -> ImageNormalizer:
    """工厂函数：创建归一化器

    Args:
        method: 归一化方法 ("min_max", "z_score", "ct_window", "adaptive", "clip")
        **kwargs: 归一化器参数

    Returns:
        ImageNormalizer 实例
    """
    normalizers = {
        "min_max": MinMaxNormalizer,
        "z_score": ZScoreNormalizer,
        "ct_window": CTWindowNormalizer,
        "adaptive": AdaptiveNormalizer,
        "clip": ClipNormalizer,
    }

    if method not in normalizers:
        raise ValueError(f"未知的归一化方法: {method}，可用: {list(normalizers.keys())}")

    return normalizers[method](**kwargs)


__all__ = [
    "NormalizationMethod",
    "ImageNormalizer",
    "MinMaxNormalizer",
    "ZScoreNormalizer",
    "CTWindowNormalizer",
    "AdaptiveNormalizer",
    "ClipNormalizer",
    "create_normalizer",
]
