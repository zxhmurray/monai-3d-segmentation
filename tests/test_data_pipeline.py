"""
数据流水线模块单元测试

Usage:
    python -m pytest tests/test_data_pipeline.py -v
"""

import pytest
import numpy as np
import torch
from pathlib import Path

# 尝试导入，如果失败则跳过
try:
    from src.data_pipeline.transforms import (
        get_train_transforms,
        get_val_transforms,
        get_inference_transforms,
    )
    from src.data_pipeline.normalizer import (
        MinMaxNormalizer,
        CTWindowNormalizer,
        ZScoreNormalizer,
    )
    HAS_MONAI = True
except ImportError:
    HAS_MONAI = False


@pytest.mark.skipif(not HAS_MONAI, reason="MONAI 未安装")
class TestTransforms:
    """Transform 测试"""

    def test_train_transforms_creation(self):
        """测试训练 Transform 创建"""
        transforms = get_train_transforms(
            spatial_size=(64, 64, 64),
            ct_window="abdomen",
            pos_ratio=0.5,
        )
        assert transforms is not None

    def test_val_transforms_creation(self):
        """测试验证 Transform 创建"""
        transforms = get_val_transforms(
            ct_window="abdomen",
        )
        assert transforms is not None

    def test_inference_transforms_creation(self):
        """测试推理 Transform 创建"""
        transforms = get_inference_transforms(
            ct_window="abdomen",
        )
        assert transforms is not None


class TestNormalizers:
    """归一化器测试"""

    def test_minmax_normalizer(self):
        """测试 MinMax 归一化"""
        normalizer = MinMaxNormalizer(min_val=0.0, max_val=1.0)

        data = np.array([0, 50, 100], dtype=np.float32)
        result = normalizer.normalize(data)

        assert result.min() == 0.0
        assert result.max() == 1.0

    def test_ct_window_normalizer(self):
        """测试 CT 窗宽窗位归一化"""
        normalizer = CTWindowNormalizer(
            window_center=40,
            window_width=400,
            output_range=(0, 1),
        )

        # 测试裁剪
        data = np.array([-200, 0, 50, 300], dtype=np.float32)
        result = normalizer.normalize(data)

        # -200 应该被裁剪到 0
        assert result[0] >= 0.0
        # 300 应该被裁剪到 1
        assert result[3] <= 1.0

    def test_ct_window_preset(self):
        """测试 CT 窗宽窗位预设"""
        normalizer = CTWindowNormalizer.get_preset("abdomen")
        assert normalizer.window_center == 40
        assert normalizer.window_width == 400


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
