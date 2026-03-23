"""
模型模块单元测试

Usage:
    python -m pytest tests/test_model.py -v
"""

import pytest
import torch

# 尝试导入，如果失败则跳过
try:
    from src.model_builder import (
        create_3d_unet,
        Medical3DUNet,
        NetworkConfig,
    )
    from src.training_engine.loss import DiceCELoss, DiceLoss
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch 未安装")
class TestModel:
    """模型测试"""

    def test_create_small_model(self):
        """测试创建小型模型"""
        model = create_3d_unet("small", in_channels=1, out_channels=1)
        assert model is not None

        # 测试前向传播
        x = torch.randn(1, 1, 64, 64, 64)
        y = model(x)
        assert y.shape == x.shape

    def test_create_medium_model(self):
        """测试创建中型模型"""
        model = create_3d_unet("medium", in_channels=1, out_channels=1)

        x = torch.randn(1, 1, 64, 64, 64)
        y = model(x)
        assert y.shape == x.shape

    def test_model_summary(self):
        """测试模型摘要"""
        model = create_3d_unet("small", in_channels=1, out_channels=1)
        summary = model.summary()

        assert "num_params" in summary
        assert summary["in_channels"] == 1
        assert summary["out_channels"] == 1


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch 未安装")
class TestLoss:
    """损失函数测试"""

    def test_dice_loss(self):
        """测试 Dice Loss"""
        loss_fn = DiceLoss()

        # 创建预测和标签
        pred = torch.rand(1, 1, 8, 8, 8)
        target = (torch.rand(1, 1, 8, 8, 8) > 0.5).float()

        loss = loss_fn(pred, target)
        assert loss.item() >= 0.0
        assert loss.item() <= 1.0

    def test_dice_ce_loss(self):
        """测试 DiceCE Loss"""
        loss_fn = DiceCELoss(lambda_dice=1.0, lambda_ce=1.0)

        pred = torch.rand(1, 1, 8, 8, 8)
        target = (torch.rand(1, 1, 8, 8, 8) > 0.5).float()

        loss = loss_fn(pred, target)
        assert loss.item() >= 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
