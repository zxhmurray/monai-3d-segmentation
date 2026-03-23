"""
医学影像重采样与方向标准化模块

提供：
- Spacing 重采样：将图像重采样到统一分辨率
- Orientation 归一化：将图像方向标准化到 RAS+ 等标准方向

依赖: SimpleITK, nibabel, numpy
"""

import warnings
from pathlib import Path
from typing import Union, Tuple, Optional, List

import numpy as np
import SimpleITK as sitk
import nibabel as nib


class ResampleError(Exception):
    """重采样相关错误"""
    pass


class OrientationError(Exception):
    """方向标准化相关错误"""
    pass


# 标准方向代码
STANDARD_ORIENTATIONS = {
    "RAS": ("R", "A", "S"),   # Right-Anterior-Superior (默认标准)
    "LPS": ("L", "P", "S"),   # Left-Posterior-Superior
    "AIR": ("A", "I", "R"),   # Anterior-Inferior-Right
}


class MedicalImageResampler:
    """医学影像重采样器

    使用 SimpleITK 进行高效的图像重采样

    Example:
        resampler = MedicalImageResampler(target_spacing=(1.0, 1.0, 1.0))
        resampled_img, new_spacing = resampler.resample(image_nifti)
    """

    def __init__(
        self,
        target_spacing: Optional[Tuple[float, float, float]] = (1.0, 1.0, 1.0),
        interpolator: str = "linear",
        default_pixel_value: float = 0.0,
    ):
        """
        Args:
            target_spacing: 目标体素间距 (x, y, z)，单位 mm
            interpolator: 插值方法 ("linear", "nearest", "cubic", "bspline")
            default_pixel_value: 重采样边界外的默认值
        """
        self.target_spacing = target_spacing
        self.interpolator = interpolator
        self.default_pixel_value = default_pixel_value

        self._interpolator_map = {
            "linear": sitk.sitkLinear,
            "nearest": sitk.sitkNearestNeighbor,
            "cubic": sitk.sitkCubic,
            "bspline": sitk.sitkBSpline,
        }

    def _get_interpolator(self, is_label: bool = False) -> int:
        """获取 SimpleITK 插值器"""
        if is_label:
            # 标签图像必须使用最近邻插值
            return sitk.sitkNearestNeighbor

        interp = self._interpolator_map.get(self.interpolator, sitk.sitkLinear)
        return interp

    def resample_nifti(
        self,
        nifti_image: nib.Nifti1Image,
        target_spacing: Optional[Tuple[float, float, float]] = None,
        is_label: bool = False,
    ) -> Tuple[nib.Nifti1Image, Tuple[float, float, float]]:
        """重采样 NIfTI 图像

        Args:
            nifti_image: nibabel 图像对象
            target_spacing: 目标间距（默认使用初始化时的值）
            is_label: 是否为标签图像（影响插值方法选择）

        Returns:
            (重采样后的 nibabel 图像, 新的 spacing)
        """
        if target_spacing is None:
            target_spacing = self.target_spacing

        if target_spacing is None:
            raise ResampleError("未指定 target_spacing")

        # 转换 nibabel -> SimpleITK
        sitk_image = self._nibabel_to_sitk(nifti_image)

        # 计算新的尺寸
        original_spacing = sitk_image.GetSpacing()
        original_size = sitk_image.GetSize()

        new_size = [
            int(round(osz * osp / tsp))
            for osz, osp, tsp in zip(original_size, original_spacing, target_spacing)
        ]

        # 创建重采样器
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(target_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(sitk_image.GetDirection())
        resampler.SetOutputOrigin(sitk_image.GetOrigin())
        resampler.SetTransform(sitk.Transform())
        resampler.SetDefaultPixelValue(self.default_pixel_value)
        resampler.SetInterpolator(self._get_interpolator(is_label))

        # 执行重采样
        resampled_sitk = resampler.Execute(sitk_image)

        # 转换回 nibabel
        resampled_nifti = self._sitk_to_nibabel(resampled_sitk, nifti_image.affine)

        new_spacing = resampled_sitk.GetSpacing()
        return resampled_nifti, new_spacing

    def resample_array(
        self,
        data: np.ndarray,
        original_spacing: Tuple[float, float, float],
        target_spacing: Optional[Tuple[float, float, float]] = None,
        origin: Optional[Tuple[float, float, float]] = None,
        direction: Optional[Tuple[float]] = None,
        is_label: bool = False,
    ) -> Tuple[np.ndarray, Tuple[float, float, float]]:
        """重采样 numpy 数组

        Args:
            data: 图像数据数组
            original_spacing: 原始体素间距
            target_spacing: 目标体素间距
            origin: 图像原点（可选）
            direction: 方向矩阵（可选）
            is_label: 是否为标签图像

        Returns:
            (重采样后的数组, 新的 spacing)
        """
        if target_spacing is None:
            target_spacing = self.target_spacing

        if target_spacing is None:
            raise ResampleError("未指定 target_spacing")

        # 创建 SimpleITK 图像
        sitk_image = sitk.GetImageFromArray(data)
        sitk_image.SetSpacing(original_spacing)

        if origin is not None:
            sitk_image.SetOrigin(origin)
        if direction is not None:
            sitk_image.SetDirection(direction)

        # 计算新尺寸
        original_size = sitk_image.GetSize()
        new_size = [
            int(round(osz * osp / tsp))
            for osz, osp, tsp in zip(original_size, original_spacing, target_spacing)
        ]

        # 重采样
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(target_spacing)
        resampler.SetSize(new_size)
        resampler.SetDefaultPixelValue(self.default_pixel_value)
        resampler.SetInterpolator(self._get_interpolator(is_label))

        resampled_sitk = resampler.Execute(sitk_image)

        # 转回 numpy
        resampled_array = sitk.GetArrayFromImage(resampled_sitk)
        new_spacing = resampled_sitk.GetSpacing()

        return resampled_array, new_spacing

    @staticmethod
    def _nibabel_to_sitk(nifti_image: nib.Nifti1Image) -> sitk.Image:
        """将 nibabel 图像转换为 SimpleITK 图像"""
        data = nifti_image.get_fdata()
        sitk_image = sitk.GetImageFromArray(data)

        # 设置元数据
        header = nifti_image.header
        sitk_image.SetSpacing(tuple(header.get_zooms()[:3]))
        sitk_image.SetOrigin(tuple(nifti_image.affine[:3, 3]))
        sitk_image.SetDirection(nifti_image.affine[:3, :3].flatten())

        return sitk_image

    @staticmethod
    def _sitk_to_nibabel(
        sitk_image: sitk.Image,
        original_affine: Optional[np.ndarray] = None
    ) -> nib.Nifti1Image:
        """将 SimpleITK 图像转换回 nibabel 图像"""
        data = sitk.GetArrayFromImage(sitk_image)

        # 构建 affine 矩阵
        if original_affine is not None:
            # 保持原始 affine 的方向信息，只更新 spacing
            affine = original_affine.copy()
            # 更新 spacing 信息到 affine
            spacing = sitk_image.GetSpacing()
            # 简化处理：重建一个标准的 affine
            spacing_matrix = np.diag(list(spacing) + [1.0])
            direction = sitk_image.GetDirection()
            direction_matrix = np.array(direction).reshape(3, 3)
            origin = sitk_image.GetOrigin()
            affine = direction_matrix @ spacing_matrix
            affine[:3, 3] = origin
        else:
            # 从 SimpleITK 元数据重建
            spacing = sitk_image.GetSpacing()
            origin = sitk_image.GetOrigin()
            direction = sitk_image.GetDirection()

            # 构建方向矩阵
            dir_matrix = np.array(direction).reshape(3, 3)
            spacing_matrix = np.diag(list(spacing) + [1.0])
            affine = dir_matrix @ spacing_matrix
            affine[:3, 3] = origin

        return nib.Nifti1Image(data, affine)


class MedicalImageOrienter:
    """医学影像方向标准化器

    将图像方向标准化到标准方向（默认 RAS+）

    Example:
        orienter = MedicalImageOrienter(target="RAS")
        oriented_img = orienter.orient_image(nifti_image)
    """

    def __init__(self, target: str = "RAS"):
        """
        Args:
            target: 目标方向代码 ("RAS", "LPS", "AIR")
        """
        if target not in STANDARD_ORIENTATIONS:
            raise OrientationError(f"不支持的方向代码: {target}，支持: {list(STANDARD_ORIENTATIONS.keys())}")
        self.target = target
        self.target_codes = STANDARD_ORIENTATIONS[target]

    def get_current_orientation(self, nifti_image: nib.Nifti1Image) -> Tuple[str, ...]:
        """获取当前图像的方向代码"""
        return nib.aff2axcodes(nifti_image.affine)

    def orient_image(self, nifti_image: nib.Nifti1Image) -> nib.Nifti1Image:
        """将图像方向标准化到目标方向

        Args:
            nifti_image: nibabel 图像对象

        Returns:
            方向标准化后的 nibabel 图像
        """
        current_codes = self.get_current_orientation(nifti_image)
        target_codes = self.target_codes

        if current_codes == target_codes:
            # 方向已经正确
            return nifti_image

        # 使用 nibabel 的 reorder_volumes
        try:
            oriented = nib.as_closest_canonical(nifti_image)
            # 如果还不是目标方向，做进一步转换
            if nib.aff2axcodes(oriented.affine) != target_codes:
                oriented = nib.apply_orientation(oriented, nib.orientations.axcodes2ornt(target_codes))
            return oriented
        except Exception as e:
            warnings.warn(f"方向标准化失败: {e}，返回原始图像")
            return nifti_image

    def orient_array(
        self,
        data: np.ndarray,
        current_codes: Tuple[str, str, str],
        target_codes: Optional[Tuple[str, str, str]] = None
    ) -> np.ndarray:
        """对 numpy 数组进行方向转换

        Args:
            data: 图像数组
            current_codes: 当前方向代码
            target_codes: 目标方向代码

        Returns:
            转换后的数组
        """
        if target_codes is None:
            target_codes = self.target_codes

        if current_codes == target_codes:
            return data

        # 计算需要的变换
        ornt_current = nib.orientations.axcodes2ornt(current_codes)
        ornt_target = nib.orientations.axcodes2ornt(target_codes)
        transform = nib.orientations.ornt_transform(ornt_current, ornt_target)

        # 应用变换
        data_oriented = nib.apply_orientation(data, transform)
        return data_oriented


def resample_to_spacing(
    image: np.ndarray,
    original_spacing: Tuple[float, float, float],
    target_spacing: Tuple[float, float, float],
    is_label: bool = False,
) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """便捷函数：重采样数组到目标间距

    Args:
        image: 图像数组
        original_spacing: 原始间距 (x, y, z)
        target_spacing: 目标间距 (x, y, z)
        is_label: 是否为标签图像

    Returns:
        (重采样后的数组, 新的间距)
    """
    resampler = MedicalImageResampler(target_spacing=target_spacing)
    return resampler.resample_array(
        image,
        original_spacing,
        target_spacing,
        is_label=is_label
    )


def normalize_orientation(
    nifti_image: nib.Nifti1Image,
    target: str = "RAS"
) -> nib.Nifti1Image:
    """便捷函数：标准化图像方向

    Args:
        nifti_image: nibabel 图像
        target: 目标方向 ("RAS", "LPS", "AIR")

    Returns:
        方向标准化后的图像
    """
    orienter = MedicalImageOrienter(target=target)
    return orienter.orient_image(nifti_image)


__all__ = [
    "ResampleError",
    "OrientationError",
    "STANDARD_ORIENTATIONS",
    "MedicalImageResampler",
    "MedicalImageOrienter",
    "resample_to_spacing",
    "normalize_orientation",
]
