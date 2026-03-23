"""
NIfTI 文件加载器模块

提供 NIfTI 格式的图像和标注加载功能，支持：
- 单文件加载
- 批量加载
- 元数据提取

依赖: nibabel, numpy
"""

import warnings
from pathlib import Path
from typing import Union, Tuple, Dict, List, Optional

import numpy as np
import nibabel as nib


class NIFTIError(Exception):
    """NIfTI 相关错误的基类异常"""
    pass


class NIFTIFileNotFoundError(NIFTIError):
    """文件未找到错误"""
    pass


class NIFTIFormatError(NIFTIError):
    """文件格式错误"""
    pass


class NIFTILoader:
    """NIfTI 文件加载器

    Example:
        loader = NIFTILoader()
        image, header = loader.load_image("path/to/image.nii.gz")
        label, header = loader.load_label("path/to/label.nii.gz")
    """

    SUPPORTED_EXTENSIONS = [".nii", ".nii.gz"]

    def __init__(self, strict: bool = False):
        """
        Args:
            strict: 是否启用严格模式（检查更多异常情况）
        """
        self.strict = strict

    def _validate_path(self, file_path: Union[str, Path]) -> Path:
        """验证文件路径"""
        file_path = Path(file_path)

        if not file_path.exists():
            raise NIFTIFileNotFoundError(f"文件不存在: {file_path}")

        if file_path.suffix not in self.SUPPORTED_EXTENSIONS:
            if self.strict:
                raise NIFTIFormatError(
                    f"不支持的文件格式: {file_path.suffix}，支持: {self.SUPPORTED_EXTENSIONS}"
                )
            else:
                warnings.warn(f"非标准文件扩展名: {file_path.suffix}")

        return file_path

    def load_image(
        self,
        file_path: Union[str, Path],
        image_only: bool = False
    ) -> Union[Tuple[np.ndarray, nib.Nifti1Image], np.ndarray]:
        """加载 NIfTI 图像

        Args:
            file_path: 图像文件路径
            image_only: 若为 True，仅返回 numpy 数组

        Returns:
            若 image_only=False: (data, nibabel_image)
            若 image_only=True: data
        """
        file_path = self._validate_path(file_path)

        try:
            img = nib.load(str(file_path))
        except Exception as e:
            raise NIFTIFormatError(f"加载图像失败: {file_path}, 错误: {e}")

        if image_only:
            return img.get_fdata()
        return img.get_fdata(), img

    def load_label(
        self,
        file_path: Union[str, Path],
        image_only: bool = False
    ) -> Union[Tuple[np.ndarray, nib.Nifti1Image], np.ndarray]:
        """加载 NIfTI 标注（标签）

        Args:
            file_path: 标注文件路径
            image_only: 若为 True，仅返回 numpy 数组

        Returns:
            若 image_only=False: (data, nibabel_image)
            若 image_only=True: data
        """
        return self.load_image(file_path, image_only)

    def load_pair(
        self,
        image_path: Union[str, Path],
        label_path: Union[str, Path]
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """同时加载图像和标注（配对加载）

        Args:
            image_path: 图像路径
            label_path: 标注路径

        Returns:
            (image_data, label_data, metadata_dict)
        """
        image_data, image_nib = self.load_image(image_path)
        label_data, label_nib = self.load_label(label_path)

        # 检查 shape 一致性
        if image_data.shape != label_data.shape:
            raise NIFTIFormatError(
                f"图像与标注 Shape 不匹配: "
                f"图像 {image_data.shape} vs 标注 {label_data.shape}"
            )

        metadata = {
            "image_path": str(image_path),
            "label_path": str(label_path),
            "shape": image_data.shape,
            "image_spacing": tuple(image_nib.header.get_zooms()),
            "label_spacing": tuple(label_nib.header.get_zooms()),
            "image_affine": image_nib.affine,
            "label_affine": label_nib.affine,
            "dtype": str(image_data.dtype),
        }

        return image_data, label_data, metadata

    def load_batch(
        self,
        file_paths: List[Union[str, Path]],
        image_only: bool = True
    ) -> List[np.ndarray]:
        """批量加载多个 NIfTI 文件

        Args:
            file_paths: 文件路径列表
            image_only: 若为 True，仅返回 numpy 数组

        Returns:
            numpy 数组列表
        """
        results = []
        for fp in file_paths:
            try:
                data = self.load_image(fp, image_only=image_only)
                results.append(data)
            except NIFTIError as e:
                warnings.warn(f"加载失败 {fp}: {e}")
                continue
        return results

    @staticmethod
    def get_metadata(file_path: Union[str, Path]) -> Dict:
        """获取 NIfTI 文件的元数据（不加载数据）

        Args:
            file_path: 文件路径

        Returns:
            包含元数据的字典
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise NIFTIFileNotFoundError(f"文件不存在: {file_path}")

        img = nib.load(str(file_path))
        header = img.header

        return {
            "filename": file_path.name,
            "path": str(file_path),
            "shape": img.shape,
            "spacing": tuple(header.get_zooms()),
            "dtype": img.get_data_dtype(),
            "affine": img.affine,
            "qform_code": header["qform_code"],
            "sform_code": header["sform_code"],
            "voxel_dims": tuple(header.get_zooms()[:3]),
        }


def load_nifti_image(
    file_path: Union[str, Path],
    dtype: Optional[np.dtype] = None
) -> Tuple[np.ndarray, nib.Nifti1Image]:
    """便捷函数：加载单个 NIfTI 图像

    Args:
        file_path: 文件路径
        dtype: 可选，指定输出数据类型

    Returns:
        (data_array, nibabel_image)
    """
    loader = NIFTILoader()
    data, img = loader.load_image(file_path)
    if dtype is not None:
        data = data.astype(dtype)
    return data, img


def load_nifti_label(
    file_path: Union[str, Path],
    dtype: Optional[np.dtype] = None
) -> Tuple[np.ndarray, nib.Nifti1Image]:
    """便捷函数：加载单个 NIfTI 标注

    Args:
        file_path: 文件路径
        dtype: 可选，指定输出数据类型

    Returns:
        (data_array, nibabel_image)
    """
    loader = NIFTILoader()
    data, img = loader.load_label(file_path)
    if dtype is not None:
        data = data.astype(dtype)
    return data, img


def save_nifti_image(
    data: np.ndarray,
    file_path: Union[str, Path],
    affine: Optional[np.ndarray] = None,
    header: Optional[nib.Nifti1Header] = None,
    compress: bool = True
) -> nib.Nifti1Image:
    """保存数据为 NIfTI 格式

    Args:
        data: numpy 数据数组
        file_path: 输出文件路径
        affine: 仿射变换矩阵（可选）
        header: NIfTI header（可选）
        compress: 是否压缩（.nii.gz）

    Returns:
        生成的 NIfTI 图像对象
    """
    if affine is None:
        affine = np.eye(4)

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    ext = ".nii.gz" if compress else ".nii"
    if not str(file_path).endswith(ext):
        file_path = file_path.with_suffix(ext)

    img = nib.Nifti1Image(data, affine, header)
    nib.save(img, str(file_path))
    return img


__all__ = [
    "NIFTIError",
    "NIFTIFileNotFoundError",
    "NIFTIFormatError",
    "NIFTILoader",
    "load_nifti_image",
    "load_nifti_label",
    "save_nifti_image",
]
