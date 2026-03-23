"""
滑窗推理模块

提供：
- 滑窗推理（Sliding Window Inference）
- 预测结果融合

依赖: torch, monai, numpy
"""

from typing import Tuple, Optional, Union, List, Dict, Any
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.inferers import sliding_window_inference
from monai.data import decollate_batch


# =============================================================================
# 配置
# =============================================================================

def _get_default_device() -> str:
    """自动检测可用设备"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEFAULT_SWIN_CONFIG = {
    "roi_size": (128, 128, 128),
    "sw_batch_size": 4,
    "stride": (64, 64, 64),
    "overlap": 0.5,
    "blend_mode": "gaussian",
    "padding_mode": "constant",
    "cval": 0.0,
    "sw_device": _get_default_device(),
    "device": _get_default_device(),
}


# =============================================================================
# 推理器类
# =============================================================================

class SlidingWindowInferer:
    """
    滑窗推理器

    解决 3D 医疗影像显存占用大的问题，通过分块推理并融合结果

    Example:
        inferer = SlidingWindowInferer(model, config)
        prediction = inferer.inference(image_tensor)
    """

    def __init__(
        self,
        model: nn.Module,
        roi_size: Union[Tuple[int, int, int], List[int]] = (128, 128, 128),
        sw_batch_size: int = 4,
        overlap: float = 0.5,
        blend_mode: str = "gaussian",
        padding_mode: str = "constant",
        cval: float = 0.0,
        sw_device: str = "cuda",
        device: str = "cuda",
    ):
        """
        Args:
            model: 分割模型
            roi_size: 滑窗尺寸 (D, H, W)
            sw_batch_size: 每次推理的窗口数
            overlap: 重叠率 (0.0 ~ 1.0)
            blend_mode: 融合模式 ("gaussian", "constant")
            padding_mode: 边缘填充模式
            cval: 填充值
            sw_device: 滑窗推理设备
            device: 结果汇总设备
        """
        self.model = model
        self.roi_size = tuple(roi_size)
        self.sw_batch_size = sw_batch_size
        self.overlap = overlap
        self.blend_mode = blend_mode
        self.padding_mode = padding_mode
        self.cval = cval
        self.sw_device = sw_device
        self.device = device

    def inference(
        self,
        image: torch.Tensor,
        threshold: Optional[float] = None,
        return_prob: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        执行滑窗推理

        Args:
            image: 输入图像张量 (B, C, D, H, W) 或 (C, D, H, W)
            threshold: 可选的预测阈值
            return_prob: 是否返回概率图

        Returns:
            预测掩码 (或概率图 + 掩码)
        """
        if image.dim() == 4:
            # (C, D, H, W) -> (1, C, D, H, W)
            image = image.unsqueeze(0)

        # 滑窗推理
        prob_map = sliding_window_inference(
            inputs=image,
            roi_size=self.roi_size,
            sw_batch_size=self.sw_batch_size,
            predictor=self.model,
            overlap=self.overlap,
            blend_mode=self.blend_mode,
            padding_mode=self.padding_mode,
            cval=self.cval,
            device=self.sw_device,
        )

        # 移到指定设备
        prob_map = prob_map.to(self.device)

        if threshold is not None:
            mask = (prob_map > threshold).float()
            if return_prob:
                return prob_map, mask
            return mask

        if return_prob:
            return prob_map
        return (prob_map > 0.5).float()

    def inference_batch(
        self,
        images: torch.Tensor,
        threshold: Optional[float] = None,
        return_prob: bool = False,
    ) -> List[torch.Tensor]:
        """批量推理"""
        results = []
        for i in range(images.shape[0]):
            img = images[i]
            result = self.inference(img, threshold, return_prob)
            results.append(result)
        return results


class SimpleInferer:
    """
    简单推理器（适用于可以直接加载到显存的小图像）

    直接进行全图推理，不使用滑窗
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        sigmoid: bool = True,
    ):
        """
        Args:
            model: 分割模型
            device: 设备
            sigmoid: 是否在输出上应用 sigmoid
        """
        self.model = model
        self.device = device
        self.sigmoid = sigmoid

    @torch.no_grad()
    def inference(
        self,
        image: torch.Tensor,
        threshold: Optional[float] = None,
        return_prob: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        执行推理

        Args:
            image: 输入图像 (B, C, D, H, W) 或 (C, D, H, W)
            threshold: 预测阈值
            return_prob: 是否返回概率图

        Returns:
            预测掩码
        """
        if image.dim() == 4:
            image = image.unsqueeze(0)

        self.model.eval()
        image = image.to(self.device)

        output = self.model(image)

        if self.sigmoid:
            prob = torch.sigmoid(output)
        else:
            prob = output

        if threshold is not None:
            mask = (prob > threshold).float()
            if return_prob:
                return prob, mask
            return mask

        if return_prob:
            return prob
        return (prob > 0.5).float()

    def inference_batch(
        self,
        images: torch.Tensor,
        threshold: Optional[float] = None,
    ) -> torch.Tensor:
        """批量推理"""
        self.model.eval()
        images = images.to(self.device)

        with torch.no_grad():
            output = self.model(images)
            prob = torch.sigmoid(output) if self.sigmoid else output

        if threshold is not None:
            return (prob > threshold).float()
        return (prob > 0.5).float()


def create_inferer(
    model: nn.Module,
    config: Optional[Dict[str, Any]] = None,
    use_sliding_window: bool = True,
    device: str = None,
) -> Union[SlidingWindowInferer, SimpleInferer]:
    """
    工厂函数：创建推理器

    Args:
        model: 分割模型
        config: 推理配置
        use_sliding_window: 是否使用滑窗推理
        device: 设备 (None=自动检测, cuda, mps, cpu)

    Returns:
        Inferer 实例
    """
    # 自动检测设备
    if device is None:
        device = _get_default_device()

    if config is None:
        config = DEFAULT_SWIN_CONFIG

    if use_sliding_window:
        return SlidingWindowInferer(
            model=model,
            roi_size=config.get("roi_size", (128, 128, 128)),
            sw_batch_size=config.get("sw_batch_size", 4),
            overlap=config.get("overlap", 0.5),
            blend_mode=config.get("blend_mode", "gaussian"),
            padding_mode=config.get("padding_mode", "constant"),
            cval=config.get("cval", 0.0),
            sw_device=device,
            device=device,
        )
    else:
        return SimpleInferer(
            model=model,
            device=device,
            sigmoid=True,
        )


__all__ = [
    "DEFAULT_SWIN_CONFIG",
    "SlidingWindowInferer",
    "SimpleInferer",
    "create_inferer",
]
