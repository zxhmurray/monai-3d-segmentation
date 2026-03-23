"""
后处理模块

提供预测结果的后处理功能：
- 阈值化
- 连通域分析
- 形态学操作
- 体积过滤

依赖: torch, numpy, scipy, monai
"""

from typing import Optional, List, Tuple, Union, Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


# =============================================================================
# 后处理配置
# =============================================================================

@dataclass
class PostProcessConfig:
    """后处理配置"""
    threshold: float = 0.5
    min_volume: Optional[int] = None  # 最小体积（体素数）
    max_volume: Optional[int] = None  # 最大体积（体素数）
    keep_largest: bool = True
    fill_holes: bool = True
    smooth_boundary: bool = False
    kernel_size: int = 3  # 形态学操作核大小


# =============================================================================
# 后处理函数
# =============================================================================

def threshold_predictions(
    predictions: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.5,
) -> Union[np.ndarray, torch.Tensor]:
    """
    将预测概率阈值化为二值掩码

    Args:
        predictions: 预测概率 (B, C, D, H, W) 或 (D, H, W)
        threshold: 阈值

    Returns:
        二值掩码
    """
    if isinstance(predictions, torch.Tensor):
        return (predictions > threshold).float()
    else:
        return (predictions > threshold).astype(np.uint8)


def keep_largest_connected_component(
    mask: np.ndarray,
    structure: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    保留最大的连通域

    Args:
        mask: 二值掩码 (D, H, W)
        structure: 连通域结构（默认 26-连通）

    Returns:
        处理后的掩码
    """
    from scipy import ndimage

    if structure is None:
        structure = ndimage.generate_binary_structure(3, 3)  # 26-连通

    labeled, num_features = ndimage.label(mask, structure=structure)

    if num_features == 0:
        return mask

    # 找出最大连通域
    component_sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
    largest_component = np.argmax(component_sizes) + 1

    # 保留最大连通域
    result = np.zeros_like(mask)
    result[labeled == largest_component] = 1

    return result


def remove_small_components(
    mask: np.ndarray,
    min_size: int,
    structure: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    移除过小的连通域

    Args:
        mask: 二值掩码
        min_size: 最小体积（体素数）
        structure: 连通域结构

    Returns:
        处理后的掩码
    """
    from scipy import ndimage

    if structure is None:
        structure = ndimage.generate_binary_structure(3, 2)

    labeled, num_features = ndimage.label(mask, structure=structure)

    if num_features == 0:
        return mask

    # 计算每个连通域的大小
    component_sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))

    # 创建掩码：保留 >= min_size 的连通域
    mask_filtered = np.zeros_like(mask)
    for i, size in enumerate(component_sizes, start=1):
        if size >= min_size:
            mask_filtered[labeled == i] = 1

    return mask_filtered


def fill_holes_3d(
    mask: np.ndarray,
    structure: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    填充掩码中的孔洞

    Args:
        mask: 二值掩码
        structure: 结构元素

    Returns:
        填充后的掩码
    """
    from scipy import ndimage

    if structure is None:
        structure = ndimage.generate_binary_structure(3, 3)

    # 反转掩码，填充背景中的孔洞
    inverted = 1 - mask
    labeled, num_features = ndimage.label(inverted, structure=structure)

    # 计算每个背景连通域的大小
    component_sizes = ndimage.sum(inverted, labeled, range(1, num_features + 1))

    # 找出"洞"（被前景完全包围的背景连通域）
    # 简化处理：只填充完全在前景内部的小孔洞
    for i, size in enumerate(component_sizes, start=1):
        if size < mask.size * 0.01:  # 小于总体积 1% 的孔洞
            inverted[labeled == i] = 0

    return 1 - inverted


def smooth_boundary_3d(
    mask: np.ndarray,
    kernel_size: int = 3,
) -> np.ndarray:
    """
    平滑掩码边界

    Args:
        mask: 二值掩码
        kernel_size: 平滑核大小

    Returns:
        平滑后的掩码
    """
    from scipy import ndimage

    # 使用中值滤波平滑
    smoothed = ndimage.median_filter(mask.astype(np.float32), size=kernel_size)

    # 重新阈值化
    return (smoothed > 0.5).astype(np.uint8)


def morphological_closing(
    mask: np.ndarray,
    iterations: int = 1,
    structure: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    形态学闭运算（先膨胀后腐蚀）

    用于填充小孔洞和连接临近区域

    Args:
        mask: 二值掩码
        iterations: 迭代次数
        structure: 结构元素

    Returns:
        处理后的掩码
    """
    from scipy import ndimage

    if structure is None:
        structure = np.ones((3, 3, 3))

    # 膨胀
    dilated = ndimage.binary_dilation(mask, structure=structure, iterations=iterations)
    # 腐蚀
    closed = ndimage.binary_erosion(dilated, structure=structure, iterations=iterations)

    return closed.astype(np.uint8)


def morphological_opening(
    mask: np.ndarray,
    iterations: int = 1,
    structure: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    形态学开运算（先腐蚀后膨胀）

    用于去除小的噪声点

    Args:
        mask: 二值掩码
        iterations: 迭代次数
        structure: 结构元素

    Returns:
        处理后的掩码
    """
    from scipy import ndimage

    if structure is None:
        structure = np.ones((3, 3, 3))

    # 腐蚀
    eroded = ndimage.binary_erosion(mask, structure=structure, iterations=iterations)
    # 膨胀
    opened = ndimage.binary_dilation(eroded, structure=structure, iterations=iterations)

    return opened.astype(np.uint8)


# =============================================================================
# 后处理流水线
# =============================================================================

class PostProcessPipeline:
    """
    后处理流水线

    组合多个后处理步骤

    Example:
        pipeline = PostProcessPipeline(config)
        processed = pipeline.apply(prediction)
    """

    def __init__(
        self,
        config: Optional[PostProcessConfig] = None,
        threshold: float = 0.5,
        keep_largest: bool = True,
        min_volume: Optional[int] = None,
        max_volume: Optional[int] = None,
        fill_holes: bool = False,
        smooth_boundary: bool = False,
    ):
        """
        Args:
            config: 后处理配置
            threshold: 阈值
            keep_largest: 是否保留最大连通域
            min_volume: 最小体积（体素数）
            max_volume: 最大体积（体素数）
            fill_holes: 是否填充孔洞
            smooth_boundary: 是否平滑边界
        """
        if config is not None:
            self.threshold = config.threshold
            self.keep_largest = config.keep_largest
            self.min_volume = config.min_volume
            self.max_volume = config.max_volume
            self.fill_holes = config.fill_holes
            self.smooth_boundary = config.smooth_boundary
        else:
            self.threshold = threshold
            self.keep_largest = keep_largest
            self.min_volume = min_volume
            self.max_volume = max_volume
            self.fill_holes = fill_holes
            self.smooth_boundary = smooth_boundary

    def apply(
        self,
        prediction: Union[np.ndarray, torch.Tensor],
        return_numpy: bool = True,
    ) -> np.ndarray:
        """
        应用后处理流水线

        Args:
            prediction: 预测概率
            return_numpy: 是否返回 numpy 数组

        Returns:
            处理后的掩码
        """
        # 转为 numpy
        if isinstance(prediction, torch.Tensor):
            mask = prediction.detach().cpu().numpy()
            if mask.ndim == 5:
                mask = mask[0, 0]  # 取第一个样本的第一个通道
            elif mask.ndim == 4:
                mask = mask[0]
        else:
            mask = prediction.copy()

        # 1. 阈值化
        mask = (mask > self.threshold).astype(np.uint8)

        if mask.sum() == 0:
            # 全零掩码，跳过后处理
            return mask

        # 2. 填充孔洞
        if self.fill_holes:
            mask = fill_holes_3d(mask)

        # 3. 保留最大连通域
        if self.keep_largest:
            mask = keep_largest_connected_component(mask)

        # 4. 移除过小/过大的连通域
        if self.min_volume is not None:
            mask = remove_small_components(mask, min_size=self.min_volume)

        if self.max_volume is not None:
            # 暂时实现：保留最大 + 过滤最大
            labeled, num = ndimage.label(mask)
            if num > 0:
                sizes = ndimage.sum(mask, labeled, range(1, num + 1))
                for i, size in enumerate(sizes, start=1):
                    if size > self.max_volume:
                        mask[labeled == i] = 0

        # 5. 平滑边界
        if self.smooth_boundary:
            mask = smooth_boundary_3d(mask)

        return mask


# =============================================================================
# 便捷函数
# =============================================================================

def postprocess_predictions(
    predictions: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.5,
    keep_largest: bool = True,
    min_volume: Optional[int] = None,
    fill_holes: bool = False,
) -> np.ndarray:
    """
    便捷函数：后处理预测结果

    Args:
        predictions: 预测概率
        threshold: 阈值
        keep_largest: 是否保留最大连通域
        min_volume: 最小体积
        fill_holes: 是否填充孔洞

    Returns:
        处理后的掩码
    """
    pipeline = PostProcessPipeline(
        threshold=threshold,
        keep_largest=keep_largest,
        min_volume=min_volume,
        fill_holes=fill_holes,
    )
    return pipeline.apply(predictions)


__all__ = [
    "PostProcessConfig",
    "threshold_predictions",
    "keep_largest_connected_component",
    "remove_small_components",
    "fill_holes_3d",
    "smooth_boundary_3d",
    "morphological_closing",
    "morphological_opening",
    "PostProcessPipeline",
    "postprocess_predictions",
]
