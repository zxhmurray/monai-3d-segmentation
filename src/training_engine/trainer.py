"""
训练引擎模块

提供训练循环、验证循环、早停机制等核心训练功能

依赖: torch, monai
"""

import os
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from monai.data import decollate_batch
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference


# =============================================================================
# 配置数据类
# =============================================================================

@dataclass
class TrainingConfig:
    """训练配置"""
    max_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 1
    num_workers: int = 4

    # AMP 混合精度
    amp: bool = True
    gpu_ids: List[int] = field(default_factory=lambda: [0])

    # 验证
    val_interval: int = 5
    val_metric_name: str = "dice"

    # 早停
    early_stopping: bool = True
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 0.001

    # 梯度
    gradient_clip: Optional[float] = 1.0
    gradient_accumulation_steps: int = 1

    # 优化器
    optimizer: str = "AdamW"
    lr_scheduler: str = "CosineAnnealingLR"
    lr_scheduler_params: Dict[str, Any] = field(default_factory=dict)

    # 检查点
    checkpoint_dir: str = "models/checkpoints"
    save_checkpoint: bool = True
    save_best_only: bool = True

    # 日志
    log_dir: str = "results/logs"
    log_interval: int = 10


@dataclass
class TrainingHistory:
    """训练历史记录"""
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    val_metric: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    epoch_times: List[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "val_metric": self.val_metric,
            "learning_rates": self.learning_rates,
            "epoch_times": self.epoch_times,
        }

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TrainingHistory":
        with open(path, "r") as f:
            data = json.load(f)
        return cls(
            train_loss=data.get("train_loss", []),
            val_loss=data.get("val_loss", []),
            val_metric=data.get("val_metric", []),
            learning_rates=data.get("learning_rates", []),
            epoch_times=data.get("epoch_times", []),
        )


# =============================================================================
# 训练器类
# =============================================================================

class Trainer:
    """
    3D 医学影像分割训练器

    封装完整的训练循环、验证、早停、检查点保存等功能

    Example:
        trainer = Trainer(model, config)
        history = trainer.train(train_loader, val_loader)
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        loss_fn: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[Any] = None,
        device: str = "cuda",
    ):
        """
        Args:
            model: 分割模型
            config: 训练配置
            loss_fn: 损失函数
            optimizer: 优化器
            lr_scheduler: 学习率调度器
            device: 训练设备
        """
        self.config = config
        self.device = device

        # 模型
        self.model = model.to(device)

        # 损失函数
        if loss_fn is None:
            from .loss import DiceCELoss
            self.loss_fn = DiceCELoss().to(device)
        else:
            self.loss_fn = loss_fn

        # 优化器
        if optimizer is None:
            self.optimizer = self._create_optimizer()
        else:
            self.optimizer = optimizer

        # 学习率调度器
        if lr_scheduler is None:
            self.lr_scheduler = self._create_lr_scheduler()
        else:
            self.lr_scheduler = lr_scheduler

        # AMP GradScaler - 根据设备选择正确的后端
        if config.amp:
            if device == "cuda":
                self.scaler = GradScaler('cuda')
            elif device == "mps":
                self.scaler = GradScaler('mps')
            else:
                self.scaler = None
        else:
            self.scaler = None

        # 验证指标
        self.dice_metric = DiceMetric(include_background=True, reduction="mean")

        # 训练时的 Dice 指标
        self.train_dice_metric = DiceMetric(include_background=True, reduction="mean")

        # 早停
        self.best_metric = 0.0
        self.best_metric_epoch = 0
        self.early_stopping_counter = 0
        self.should_stop_early = False

        # 训练历史
        self.history = TrainingHistory()

        # 检查点目录
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 日志目录
        self.log_dir = Path(config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """创建优化器"""
        if self.config.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == "Adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == "SGD":
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.99,
            )
        else:
            raise ValueError(f"未知的优化器: {self.config.optimizer}")

        return optimizer

    def _create_lr_scheduler(self):
        """创建学习率调度器"""
        params = self.config.lr_scheduler_params

        if self.config.lr_scheduler == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=params.get("T_max", self.config.max_epochs),
                eta_min=params.get("eta_min", 1e-6),
            )
        elif self.config.lr_scheduler == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=params.get("step_size", 30),
                gamma=params.get("gamma", 0.1),
            )
        elif self.config.lr_scheduler == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                factor=params.get("factor", 0.5),
                patience=params.get("patience", 10),
            )
        else:
            scheduler = None

        return scheduler

    def _compute_dice(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        """计算 Dice 系数"""
        smooth = 1e-5
        intersection = (y_pred * y_true).sum()
        dice = (2.0 * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)
        return dice.item()

    def train_epoch(self, loader) -> Tuple[float, float, float]:
        """训练一个 epoch

        Returns:
            (avg_loss, avg_dice, lr)
        """
        self.model.train()
        total_loss = 0.0
        total_dice = 0.0
        num_batches = 0

        pbar = tqdm(loader, desc="Training", leave=False)
        for batch_idx, batch_data in enumerate(pbar):
            # 数据移至设备
            inputs = batch_data["image"].to(self.device)
            labels = batch_data["label"].to(self.device)

            # 混合精度前向
            if self.scaler is not None:
                with autocast(device_type=self.device):
                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, labels)

                # 梯度累积
                loss = loss / self.config.gradient_accumulation_steps

                self.scaler.scale(loss).backward()

                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    if self.config.gradient_clip:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.gradient_clip,
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()

                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    if self.config.gradient_clip:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.gradient_clip,
                        )
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1

            # 计算当前 batch 的 Dice
            outputs_sigmoid = torch.sigmoid(outputs)
            dice_val = self._compute_dice(outputs_sigmoid, labels)
            total_dice += dice_val

            pbar.set_postfix({
                "loss": f"{loss.item() * self.config.gradient_accumulation_steps:.4f}",
                "dice": f"{dice_val:.4f}"
            })

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_dice = total_dice / num_batches if num_batches > 0 else 0.0
        current_lr = self.optimizer.param_groups[0]["lr"]

        return avg_loss, avg_dice, current_lr

    @torch.no_grad()
    def validate(self, loader) -> Tuple[float, float]:
        """验证

        Returns:
            (avg_val_loss, avg_dice)
        """
        self.model.eval()
        total_val_loss = 0.0
        total_val_dice = 0.0
        num_val_batches = 0

        pbar = tqdm(loader, desc="Validation", leave=False)
        for batch_data in pbar:
            inputs = batch_data["image"].to(self.device)
            labels = batch_data["label"].to(self.device)

            # 滑窗推理（如果图像太大）
            # TODO: 根据输入尺寸决定是否使用滑窗
            outputs = self.model(inputs)

            loss = self.loss_fn(outputs, labels)
            total_val_loss += loss.item()
            num_val_batches += 1

            # 计算 Dice
            outputs_sigmoid = torch.sigmoid(outputs)
            batch_dice = self._compute_dice(outputs_sigmoid, labels)
            total_val_dice += batch_dice

            pbar.set_postfix({
                "val_loss": f"{loss.item():.4f}",
                "dice": f"{batch_dice:.4f}"
            })

        avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else 0.0
        avg_dice = total_val_dice / num_val_batches if num_val_batches > 0 else 0.0

        return avg_val_loss, avg_dice

    def train(
        self,
        train_loader,
        val_loader,
        max_epochs: Optional[int] = None,
    ) -> TrainingHistory:
        """
        执行完整的训练循环

        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            max_epochs: 最大训练 epoch 数

        Returns:
            TrainingHistory 对象
        """
        if max_epochs is None:
            max_epochs = self.config.max_epochs

        print(f"\n{'='*60}")
        print(f"开始训练")
        print(f"{'='*60}")
        print(f"最大 Epochs: {max_epochs}")
        print(f"设备: {self.device}")
        print(f"AMP: {'开启' if self.config.amp else '关闭'}")
        print(f"早停: {'开启' if self.config.early_stopping else '关闭'}")
        print(f"{'='*60}\n")

        for epoch in range(max_epochs):
            epoch_start_time = time.time()

            print(f"\nEpoch {epoch + 1}/{max_epochs}")
            print("-" * 40)

            # 训练
            train_loss, train_dice, current_lr = self.train_epoch(train_loader)

            # 验证（定期）
            do_val = (epoch + 1) % self.config.val_interval == 0

            if do_val:
                val_loss, val_dice = self.validate(val_loader)

                # 更新早停
                if self.config.early_stopping:
                    if val_dice > self.best_metric + self.config.early_stopping_min_delta:
                        self.best_metric = val_dice
                        self.best_metric_epoch = epoch + 1
                        self.early_stopping_counter = 0

                        # 保存最佳模型
                        if self.config.save_checkpoint and self.config.save_best_only:
                            self.save_checkpoint(
                                filename="best_model.pt",
                                epoch=epoch + 1,
                                metric=val_dice,
                            )
                    else:
                        self.early_stopping_counter += 1

                    if self.early_stopping_counter >= self.config.early_stopping_patience:
                        print(f"\n早停触发！连续 {self.early_stopping_counter} 个 epoch 无改善")
                        self.should_stop_early = True
            else:
                val_loss = 0.0
                val_dice = self.best_metric  # 使用上次最佳

            # 学习率调度
            if self.lr_scheduler is not None:
                if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step(val_dice)
                else:
                    self.lr_scheduler.step()

            # 记录历史
            epoch_time = time.time() - epoch_start_time
            self.history.train_loss.append(train_loss)
            self.history.val_loss.append(val_loss)
            self.history.val_metric.append(val_dice)
            self.history.learning_rates.append(current_lr)
            self.history.epoch_times.append(epoch_time)

            # 打印 epoch 摘要
            print(f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")
            if do_val:
                print(f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
            print(f"LR: {current_lr:.6f}, Time: {epoch_time:.1f}s")

            # 保存检查点（每个 epoch）
            if self.config.save_checkpoint and not self.config.save_best_only:
                if (epoch + 1) % 10 == 0:
                    self.save_checkpoint(
                        filename=f"checkpoint_epoch_{epoch+1}.pt",
                        epoch=epoch + 1,
                        metric=val_dice,
                    )

            # 早停检查
            if self.should_stop_early:
                break

        # 训练结束，保存历史
        self.history.save(str(self.log_dir / "training_history.json"))

        print(f"\n{'='*60}")
        print(f"训练完成！")
        print(f"最佳 Dice: {self.best_metric:.4f} (Epoch {self.best_metric_epoch})")
        print(f"{'='*60}\n")

        return self.history

    def save_checkpoint(
        self,
        filename: str,
        epoch: int,
        metric: float,
    ) -> str:
        """保存检查点"""
        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_metric": self.best_metric,
            "best_metric_epoch": self.best_metric_epoch,
            "config": {
                "max_epochs": self.config.max_epochs,
                "learning_rate": self.config.learning_rate,
            },
        }

        if self.lr_scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.lr_scheduler.state_dict()

        torch.save(checkpoint, checkpoint_path)
        print(f"检查点已保存: {checkpoint_path}")

        return str(checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str) -> dict:
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "scheduler_state_dict" in checkpoint and self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.best_metric = checkpoint.get("best_metric", 0.0)
        self.best_metric_epoch = checkpoint.get("best_metric_epoch", 0)

        print(f"检查点已加载: {checkpoint_path}")
        print(f"  - Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  - Best Metric: {self.best_metric:.4f}")

        return checkpoint


def train_with_config(
    model: nn.Module,
    train_loader,
    val_loader,
    config_dict: dict,
    device: str = "cuda",
) -> TrainingHistory:
    """
    根据配置字典执行训练的便捷函数

    Args:
        model: 分割模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        config_dict: 训练配置字典
        device: 设备

    Returns:
        TrainingHistory 对象
    """
    config = TrainingConfig(**config_dict)

    trainer = Trainer(
        model=model,
        config=config,
        device=device,
    )

    return trainer.train(train_loader, val_loader)


__all__ = [
    "TrainingConfig",
    "TrainingHistory",
    "Trainer",
    "train_with_config",
]
