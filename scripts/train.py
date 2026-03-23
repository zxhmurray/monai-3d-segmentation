"""
训练入口脚本

Usage:
    python scripts/train.py --config configs/train_config.yaml
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch

# 配置
from src.data_pipeline import (
    DataPipeline,
    DatasetConfig,
    DataLoaderConfig,
    get_train_transforms,
    get_val_transforms,
)
from src.model_builder import create_3d_unet, NetworkConfig
from src.training_engine import (
    Trainer,
    TrainingConfig,
    DiceCELoss,
)


def parse_args():
    parser = argparse.ArgumentParser(description="训练 3D 医学影像分割模型")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="训练配置文件路径"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="rawdata/MSD_Spleen",
        help="数据集目录"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models",
        help="输出目录"
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=None,
        help="最大训练 epoch 数"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="批次大小"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="学习率"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="训练设备 (auto/cuda/mps/cpu)，auto 会自动检测"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="恢复训练的检查点路径"
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """加载 YAML 配置文件"""
    import yaml

    config_path = Path(config_path)
    if not config_path.exists():
        print(f"配置文件不存在: {config_path}，使用默认配置")
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def main():
    args = parse_args()

    print("\n" + "="*60)
    print("MONAI 3D 医学影像分割 - 模型训练")
    print("="*60 + "\n")

    # 加载配置
    config = load_config(args.config)

    # 设备 - 自动检测 MPS/CUDA/CPU
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    elif args.device in ["cuda", "mps", "cpu"]:
        # 验证请求的设备是否可用
        if args.device == "cuda" and not torch.cuda.is_available():
            print(f"警告: CUDA 不可用，切换到 CPU")
            device = "cpu"
        elif args.device == "mps" and not torch.backends.mps.is_available():
            print(f"警告: MPS 不可用，切换到 CPU")
            device = "cpu"
        else:
            device = args.device
    else:
        device = "cpu"

    print(f"使用设备: {device}")

    # 数据集
    print(f"\n准备数据集: {args.data_dir}")
    dataset_config = DatasetConfig(
        data_dir=args.data_dir,
        train_split=0.8,
        val_split=0.2,
    )

    # 创建 Transform
    train_transforms = get_train_transforms(
        spatial_size=(128, 128, 128),
        pixdim=(1.0, 1.0, 1.0),
        ct_window="abdomen",
        pos_ratio=0.5,
        enable_augmentation=True,
    )

    val_transforms = get_val_transforms(
        pixdim=(1.0, 1.0, 1.0),
        ct_window="abdomen",
    )

    # 创建 Data Pipeline
    dataloader_config = DataLoaderConfig(
        batch_size=args.batch_size or config.get("batch_size", 1),
        num_workers=4,
        shuffle=True,
        pin_memory=True,
    )

    pipeline = DataPipeline(
        dataset_config=dataset_config,
        dataloader_config=dataloader_config,
        train_transforms=train_transforms,
        val_transforms=val_transforms,
    )

    print(f"训练样本数: {len(pipeline.train_list)}")
    print(f"验证样本数: {len(pipeline.val_list)}")

    # 创建 DataLoader
    train_loader = pipeline.get_train_loader()
    val_loader = pipeline.get_val_loader()

    # 模型
    print(f"\n创建 3D U-Net 模型...")
    model = create_3d_unet(
        model_size="medium",
        in_channels=1,
        out_channels=1,
    )

    # 损失函数
    loss_fn = DiceCELoss(lambda_dice=1.0, lambda_ce=1.0, sigmoid=True)

    # 训练配置
    train_config = TrainingConfig(
        max_epochs=args.max_epochs or config.get("max_epochs", 100),
        learning_rate=args.lr or config.get("learning_rate", 1e-4),
        weight_decay=1e-5,
        batch_size=args.batch_size or config.get("batch_size", 1),
        val_interval=5,
        early_stopping=True,
        early_stopping_patience=20,
        amp=True,
        checkpoint_dir=args.output_dir + "/checkpoints",
        log_dir="results/logs",
    )

    # 创建训练器
    trainer = Trainer(
        model=model,
        config=train_config,
        loss_fn=loss_fn,
        device=device,
    )

    # 恢复检查点
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # 开始训练
    print("\n开始训练...\n")
    history = trainer.train(train_loader, val_loader)

    print("\n" + "="*60)
    print("训练完成！")
    print(f"最佳 Dice: {trainer.best_metric:.4f}")
    print(f"检查点保存位置: {trainer.checkpoint_dir}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
