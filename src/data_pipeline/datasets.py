"""
数据集与 DataLoader 封装模块

提供：
- 医学影像数据集类
- 训练/验证/测试 DataLoader 创建
- 数据集路径管理

依赖: monai, nibabel, numpy, torch
"""

import os
import json
import warnings
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any, Union, Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from monai.data import (
    Dataset,
    CacheDataset,
    SmartCacheDataset,
    create_test_image_3d,
    set_track_meta,
)
from monai.data.utils import partition_dataset


# =============================================================================
# 数据集配置数据类
# =============================================================================

@dataclass
class DatasetConfig:
    """数据集配置"""
    name: str = "MSD_Spleen"
    data_dir: str = "rawdata/MSD_Spleen"
    images_dir: str = "images"
    labels_dir: str = "labels"
    cache_dir: Optional[str] = None
    train_split: float = 0.8
    val_split: float = 0.2
    seed: int = 42

    # 数据集划分文件路径
    train_list_path: Optional[str] = None
    val_list_path: Optional[str] = None
    test_list_path: Optional[str] = None


@dataclass
class DataLoaderConfig:
    """DataLoader 配置"""
    batch_size: int = 1
    num_workers: int = 4
    shuffle: bool = True
    pin_memory: bool = True
    drop_last: bool = False
    persistent_workers: bool = True


# =============================================================================
# 路径工具函数
# =============================================================================

def find_image_label_pairs(
    data_dir: Union[str, Path],
    images_subdir: str = "images",
    labels_subdir: str = "labels",
    image_suffix: str = ".nii.gz",
    label_suffix: str = ".nii.gz",
) -> List[Dict[str, str]]:
    """
    扫描数据目录，查找图像-标注配对

    Args:
        data_dir: 数据根目录
        images_subdir: 图像子目录名
        labels_subdir: 标注子目录名
        image_suffix: 图像文件后缀
        label_suffix: 标注文件后缀

    Returns:
        [{"image": path, "label": path, "name": case_id}, ...]
    """
    data_dir = Path(data_dir)
    images_dir = data_dir / images_subdir
    labels_dir = data_dir / labels_subdir

    if not images_dir.exists():
        raise FileNotFoundError(f"图像目录不存在: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"标注目录不存在: {labels_dir}")

    # 查找所有图像文件
    image_files = sorted(images_dir.glob(f"*{image_suffix}"))

    pairs = []
    for img_path in image_files:
        # 尝试构造对应的 label 路径
        # 常见命名模式: image_xxx.nii.gz -> label_xxx.nii.gz
        # 或: xxx_image.nii.gz -> xxx_label.nii.gz
        img_stem = img_path.stem.replace(".nii", "")  # 去掉 .nii 或 .nii.gz 的 .gz

        # 尝试多种命名模式
        possible_label_names = [
            img_stem.replace("image", "label"),
            img_stem.replace("img", "label"),
            img_stem.replace("_image", "_label"),
            img_stem + "_label",
        ]

        label_path = None
        for label_name in possible_label_names:
            candidate = labels_dir / f"{label_name}{label_suffix}"
            if candidate.exists():
                label_path = str(candidate)
                break

        if label_path is None:
            warnings.warn(f"未找到标注文件: {img_path.name}，跳过")
            continue

        # 提取 case_id
        case_id = img_stem.replace("image", "").replace("img", "").strip("_")

        pairs.append({
            "image": str(img_path),
            "label": str(label_path),
            "name": case_id or img_path.stem,
        })

    return pairs


def split_dataset(
    pairs: List[Dict[str, str]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    test_ratio: float = 0.0,
    shuffle: bool = True,
    seed: int = 42,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    """
    划分训练/验证/测试集

    Args:
        pairs: 图像-标注配对列表
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        shuffle: 是否打乱
        seed: 随机种子

    Returns:
        (train_pairs, val_pairs, test_pairs)
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(f"划分比例之和必须为 1.0，当前: {train_ratio + val_ratio + test_ratio}")

    n = len(pairs)
    indices = np.arange(n)

    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)

    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:] if test_ratio > 0 else []

    train_pairs = [pairs[i] for i in train_indices]
    val_pairs = [pairs[i] for i in val_indices]
    test_pairs = [pairs[i] for i in test_indices]

    return train_pairs, val_pairs, test_pairs


def save_split_list(
    pairs: List[Dict[str, str]],
    output_path: Union[str, Path],
) -> None:
    """保存划分列表到 JSON 文件"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)


def load_split_list(
    input_path: Union[str, Path],
) -> List[Dict[str, str]]:
    """从 JSON 文件加载划分列表"""
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# 数据集类
# =============================================================================

class MedicalImageDataset(Dataset):
    """医学影像数据集

    封装 MONAI Dataset，支持图像-标注配对加载

    Example:
        dataset = MedicalImageDataset(
            data_list=pair_list,
            transform=train_transforms,
        )
        sample = dataset[0]
    """

    def __init__(
        self,
        data_list: List[Dict[str, str]],
        transform: Optional[Any] = None,
        num_samples: Optional[int] = None,
    ):
        """
        Args:
            data_list: [{"image": path, "label": path}, ...] 列表
            transform: MONAI Transform 或 Compose
            num_samples: 限制样本数量（用于调试）
        """
        if num_samples is not None:
            data_list = data_list[:num_samples]

        super().__init__(data=data_list, transform=transform)


class MedicalImageCacheDataset(CacheDataset):
    """带缓存的医学影像数据集

    使用 CacheDataset 加速数据加载，适用于数据量较小的场景

    Example:
        dataset = MedicalImageCacheDataset(
            data_list=train_pairs,
            transform=train_transforms,
            cache_rate=0.5,  # 缓存 50% 的数据
        )
    """

    def __init__(
        self,
        data_list: List[Dict[str, str]],
        transform: Optional[Any] = None,
        cache_rate: float = 1.0,
        num_workers: int = 4,
        progress: bool = True,
    ):
        """
        Args:
            data_list: 数据列表
            transform: MONAI Transform
            cache_rate: 缓存比例 (0.0 ~ 1.0)
            num_workers: 数据加载线程数
            progress: 是否显示缓存进度
        """
        super().__init__(
            data=data_list,
            transform=transform,
            cache_rate=cache_rate,
            num_workers=num_workers,
            progress=progress,
        )


# =============================================================================
# DataLoader 创建
# =============================================================================

def create_train_dataloader(
    data_list: List[Dict[str, str]],
    transform: Any,
    batch_size: int = 1,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
    drop_last: bool = False,
    persistent_workers: bool = True,
    cache_rate: float = 0.0,  # 训练时默认不用缓存（数据增强需要每次重新计算）
) -> DataLoader:
    """
    创建训练 DataLoader

    Args:
        data_list: 训练数据列表
        transform: 训练 Transform
        其他参数同 DataLoaderConfig

    Returns:
        DataLoader 实例
    """
    if cache_rate > 0:
        dataset = MedicalImageCacheDataset(
            data_list=data_list,
            transform=transform,
            cache_rate=cache_rate,
            num_workers=num_workers,
        )
    else:
        dataset = MedicalImageDataset(
            data_list=data_list,
            transform=transform,
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers if num_workers > 0 else False,
    )

    return loader


def create_val_dataloader(
    data_list: List[Dict[str, str]],
    transform: Any,
    batch_size: int = 1,
    num_workers: int = 4,
    pin_memory: bool = True,
    cache_rate: float = 0.5,  # 验证时可以缓存
) -> DataLoader:
    """
    创建验证 DataLoader

    Args:
        data_list: 验证数据列表
        transform: 验证 Transform
        其他参数同 DataLoaderConfig

    Returns:
        DataLoader 实例
    """
    dataset = MedicalImageCacheDataset(
        data_list=data_list,
        transform=transform,
        cache_rate=cache_rate,
        num_workers=num_workers,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,  # 验证时不需要 shuffle
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=True if num_workers > 0 else False,
    )

    return loader


def create_inference_dataloader(
    data_list: List[Dict[str, str]],
    transform: Any,
    batch_size: int = 1,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """
    创建推理 DataLoader

    Args:
        data_list: 推理数据列表
        transform: 推理 Transform
        其他参数同 DataLoaderConfig

    Returns:
        DataLoader 实例
    """
    dataset = MedicalImageDataset(
        data_list=data_list,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return loader


# =============================================================================
# 数据流水线封装
# =============================================================================

class DataPipeline:
    """
    医学影像数据流水线封装类

    整合数据集划分、Transform 创建和 DataLoader 创建

    Example:
        pipeline = DataPipeline(config)
        train_loader = pipeline.get_train_loader()
        val_loader = pipeline.get_val_loader()
    """

    def __init__(
        self,
        dataset_config: Union[DatasetConfig, Dict[str, Any]],
        dataloader_config: Optional[Union[DataLoaderConfig, Dict[str, Any]]] = None,
        train_transforms: Optional[Any] = None,
        val_transforms: Optional[Any] = None,
        inference_transforms: Optional[Any] = None,
    ):
        """
        Args:
            dataset_config: 数据集配置
            dataloader_config: DataLoader 配置
            train_transforms: 训练 Transform
            val_transforms: 验证 Transform
            inference_transforms: 推理 Transform
        """
        # 处理配置字典
        if isinstance(dataset_config, dict):
            dataset_config = DatasetConfig(**dataset_config)
        if dataloader_config is None:
            dataloader_config = DataLoaderConfig()
        elif isinstance(dataloader_config, dict):
            dataloader_config = DataLoaderConfig(**dataloader_config)

        self.dataset_config = dataset_config
        self.dataloader_config = dataloader_config

        # 设置 Transform
        self._train_transforms = train_transforms
        self._val_transforms = val_transforms
        self._inference_transforms = inference_transforms

        # 数据列表
        self._train_list: List[Dict[str, str]] = []
        self._val_list: List[Dict[str, str]] = []
        self._test_list: List[Dict[str, str]] = []

        # DataLoader 缓存
        self._train_loader: Optional[DataLoader] = None
        self._val_loader: Optional[DataLoader] = None
        self._inference_loader: Optional[DataLoader] = None

        # 初始化
        self._prepare_data()

    def _prepare_data(self) -> None:
        """准备数据：扫描、划分、保存"""
        cfg = self.dataset_config

        # 检查是否已有划分文件
        list_dir = Path(cfg.data_dir)
        train_list_path = list_dir / "train_list.json"
        val_list_path = list_dir / "val_list.json"

        if train_list_path.exists() and val_list_path.exists():
            # 从文件加载
            self._train_list = load_split_list(train_list_path)
            self._val_list = load_split_list(val_list_path)
            print(f"从文件加载数据集划分: train={len(self._train_list)}, val={len(self._val_list)}")
        else:
            # 扫描并划分
            pairs = find_image_label_pairs(
                data_dir=cfg.data_dir,
                images_subdir=cfg.images_dir,
                labels_subdir=cfg.labels_dir,
            )
            print(f"扫描到 {len(pairs)} 个图像-标注配对")

            self._train_list, self._val_list, self._test_list = split_dataset(
                pairs=pairs,
                train_ratio=cfg.train_split,
                val_ratio=cfg.val_split,
                test_ratio=1.0 - cfg.train_split - cfg.val_split,
                seed=cfg.seed,
            )

            # 保存划分
            save_split_list(self._train_list, train_list_path)
            save_split_list(self._val_list, val_list_path)
            print(f"数据集划分完成: train={len(self._train_list)}, val={len(self._val_list)}")

    @property
    def train_list(self) -> List[Dict[str, str]]:
        return self._train_list

    @property
    def val_list(self) -> List[Dict[str, str]]:
        return self._val_list

    @property
    def test_list(self) -> List[Dict[str, str]]:
        return self._test_list

    def get_train_loader(
        self,
        transforms: Optional[Any] = None,
    ) -> DataLoader:
        """获取训练 DataLoader"""
        if transforms is None:
            transforms = self._train_transforms
        if transforms is None:
            raise ValueError("未提供训练 Transform，请设置 train_transforms 或传入 transforms 参数")

        dl_cfg = self.dataloader_config
        return create_train_dataloader(
            data_list=self._train_list,
            transform=transforms,
            batch_size=dl_cfg.batch_size,
            num_workers=dl_cfg.num_workers,
            shuffle=dl_cfg.shuffle,
            pin_memory=dl_cfg.pin_memory,
            drop_last=dl_cfg.drop_last,
            persistent_workers=dl_cfg.persistent_workers,
        )

    def get_val_loader(
        self,
        transforms: Optional[Any] = None,
    ) -> DataLoader:
        """获取验证 DataLoader"""
        if transforms is None:
            transforms = self._val_transforms
        if transforms is None:
            raise ValueError("未提供验证 Transform，请设置 val_transforms 或传入 transforms 参数")

        dl_cfg = self.dataloader_config
        return create_val_dataloader(
            data_list=self._val_list,
            transform=transforms,
            batch_size=dl_cfg.batch_size,
            num_workers=dl_cfg.num_workers,
            pin_memory=dl_cfg.pin_memory,
        )

    def get_inference_loader(
        self,
        data_list: Optional[List[Dict[str, str]]] = None,
        transforms: Optional[Any] = None,
    ) -> DataLoader:
        """获取推理 DataLoader"""
        if transforms is None:
            transforms = self._inference_transforms
        if transforms is None:
            raise ValueError("未提供推理 Transform，请设置 inference_transforms 或传入 transforms 参数")

        if data_list is None:
            # 默认使用验证集
            data_list = self._val_list

        dl_cfg = self.dataloader_config
        return create_inference_dataloader(
            data_list=data_list,
            transform=transforms,
            batch_size=dl_cfg.batch_size,
            num_workers=dl_cfg.num_workers,
            pin_memory=dl_cfg.pin_memory,
        )


__all__ = [
    "DatasetConfig",
    "DataLoaderConfig",
    "find_image_label_pairs",
    "split_dataset",
    "save_split_list",
    "load_split_list",
    "MedicalImageDataset",
    "MedicalImageCacheDataset",
    "create_train_dataloader",
    "create_val_dataloader",
    "create_inference_dataloader",
    "DataPipeline",
]
