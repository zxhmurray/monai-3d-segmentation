"""
评估模块单元测试

Usage:
    python -m pytest tests/test_evaluator.py -v
"""

import pytest
import numpy as np
import torch

try:
    from src.evaluator.postprocess import (
        threshold_predictions,
        keep_largest_connected_component,
        remove_small_components,
        PostProcessPipeline,
    )
    from src.evaluator.volume import compute_volume
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


@pytest.mark.skipif(not HAS_NUMPY, reason="NumPy 未安装")
class TestPostProcess:
    """后处理测试"""

    def test_threshold(self):
        """测试阈值化"""
        pred = np.array([0.1, 0.4, 0.6, 0.9])
        result = threshold_predictions(pred, threshold=0.5)
        expected = np.array([0, 0, 1, 1])
        np.testing.assert_array_equal(result, expected)

    def test_keep_largest_component(self):
        """测试保留最大连通域"""
        # 创建一个简单的测试掩码
        mask = np.zeros((10, 10, 10), dtype=np.uint8)
        # 大连通域
        mask[2:5, 2:5, 2:5] = 1
        # 小连通域
        mask[7:9, 7:9, 7:9] = 1

        result = keep_largest_connected_component(mask)
        # 应该保留大连通域
        assert result[2, 2, 2] == 1
        assert result[7, 7, 7] == 0

    def test_postprocess_pipeline(self):
        """测试后处理流水线"""
        pipeline = PostProcessPipeline(
            threshold=0.5,
            keep_largest=True,
            min_volume=10,
        )

        pred = np.random.rand(32, 32, 32)
        result = pipeline.apply(pred)
        assert result.dtype == np.uint8
        assert result.max() <= 1


@pytest.mark.skipif(not HAS_NUMPY, reason="NumPy 未安装")
class TestVolume:
    """体积计算测试"""

    def test_compute_volume_cube(self):
        """测试立方体体积计算"""
        # 创建一个 10x10x10 的立方体
        mask = np.zeros((10, 10, 10), dtype=np.uint8)
        mask[0:10, 0:10, 0:10] = 1

        # 体素间距 1mm
        spacing = (1.0, 1.0, 1.0)

        volume_mm3 = compute_volume(mask, spacing, "mm3")
        assert volume_mm3 == 1000.0  # 10 * 10 * 10 = 1000 mm³

        volume_cm3 = compute_volume(mask, spacing, "cm3")
        assert volume_cm3 == 1.0  # 1000 / 1000 = 1 cm³

    def test_compute_volume_with_spacing(self):
        """测试非均匀间距体积计算"""
        mask = np.zeros((10, 10, 10), dtype=np.uint8)
        mask[0:10, 0:10, 0:10] = 1

        # 体素间距 0.5 x 0.5 x 2.0 mm
        spacing = (0.5, 0.5, 2.0)

        volume_mm3 = compute_volume(mask, spacing, "mm3")
        expected = 10 * 10 * 10 * 0.5 * 0.5 * 2.0  # = 500 mm³
        assert volume_mm3 == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
