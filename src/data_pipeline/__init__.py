"""
Data Pipeline 模块

提供医学影像数据的加载、预处理、增强等功能

Submodules:
    loaders: NIfTI 文件加载
    resample: 空间重采样与方向标准化
    normalizer: 强度归一化
    transforms: MONAI Transforms 流水线
    datasets: 数据集与 DataLoader 封装
"""

from .loaders import (
    NIFTIError,
    NIFTIFileNotFoundError,
    NIFTIFormatError,
    NIFTILoader,
    load_nifti_image,
    load_nifti_label,
    save_nifti_image,
)

from .resample import (
    ResampleError,
    OrientationError,
    STANDARD_ORIENTATIONS,
    MedicalImageResampler,
    MedicalImageOrienter,
    resample_to_spacing,
    normalize_orientation,
)

from .normalizer import (
    NormalizationMethod,
    ImageNormalizer,
    MinMaxNormalizer,
    ZScoreNormalizer,
    CTWindowNormalizer,
    AdaptiveNormalizer,
    ClipNormalizer,
    create_normalizer,
)

from .transforms import (
    DEFAULT_PATCH_SIZE,
    DEFAULT_SPACING,
    CT_PRESETS,
    get_train_transforms,
    get_val_transforms,
    get_inference_transforms,
    get_postprocess_transforms,
    create_transforms_from_config,
)

from .datasets import (
    DatasetConfig,
    DataLoaderConfig,
    find_image_label_pairs,
    split_dataset,
    save_split_list,
    load_split_list,
    MedicalImageDataset,
    MedicalImageCacheDataset,
    create_train_dataloader,
    create_val_dataloader,
    create_inference_dataloader,
    DataPipeline,
)

__all__ = [
    # Loaders
    "NIFTIError",
    "NIFTIFileNotFoundError",
    "NIFTIFormatError",
    "NIFTILoader",
    "load_nifti_image",
    "load_nifti_label",
    "save_nifti_image",
    # Resample
    "ResampleError",
    "OrientationError",
    "STANDARD_ORIENTATIONS",
    "MedicalImageResampler",
    "MedicalImageOrienter",
    "resample_to_spacing",
    "normalize_orientation",
    # Normalizer
    "NormalizationMethod",
    "ImageNormalizer",
    "MinMaxNormalizer",
    "ZScoreNormalizer",
    "CTWindowNormalizer",
    "AdaptiveNormalizer",
    "ClipNormalizer",
    "create_normalizer",
    # Transforms
    "DEFAULT_PATCH_SIZE",
    "DEFAULT_SPACING",
    "CT_PRESETS",
    "get_train_transforms",
    "get_val_transforms",
    "get_inference_transforms",
    "get_postprocess_transforms",
    "create_transforms_from_config",
    # Datasets
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
