"""
体积计算模块

提供基于预测掩码的脏器体积计算功能

依赖: numpy, nibabel
"""

from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path

import numpy as np
import nibabel as nib


# =============================================================================
# 体积计算
# =============================================================================

def compute_volume(
    mask: np.ndarray,
    spacing: Union[Tuple[float, float, float], np.ndarray],
    unit: str = "cm3",
) -> float:
    """
    计算掩码的体积

    Args:
        mask: 二值掩码 (D, H, W)
        spacing: 体素间距 (x, y, z) in mm
        unit: 体积单位 ("mm3", "cm3", "ml")

    Returns:
        体积值
    """
    # 体素个数
    voxel_count = np.sum(mask > 0)

    # 单个体素体积 (mm³)
    voxel_volume_mm3 = abs(spacing[0] * spacing[1] * spacing[2])

    # 总体积
    volume_mm3 = voxel_count * voxel_volume_mm3

    # 单位转换
    if unit == "mm3":
        return float(volume_mm3)
    elif unit == "cm3":
        return float(volume_mm3 / 1000.0)  # 1 cm³ = 1000 mm³
    elif unit == "ml":
        return float(volume_mm3 / 1000.0)  # 1 ml = 1 cm³
    else:
        raise ValueError(f"未知的体积单位: {unit}，可用: mm3, cm3, ml")


def compute_volume_from_nifti(
    nifti_path: Union[str, Path],
    label_value: int = 1,
    unit: str = "cm3",
) -> Dict[str, float]:
    """
    从 NIfTI 文件计算体积

    Args:
        nifti_path: NIfTI 文件路径
        label_value: 要计算的标签值
        unit: 体积单位

    Returns:
        包含体积信息的字典
    """
    nifti_path = Path(nifti_path)
    img = nib.load(str(nifti_path))

    data = img.get_fdata()
    spacing = img.header.get_zooms()[:3]

    # 创建二值掩码
    mask = (data == label_value).astype(np.uint8)

    # 计算体积
    volume = compute_volume(mask, spacing, unit)

    return {
        "filename": nifti_path.name,
        "path": str(nifti_path),
        "voxel_count": int(np.sum(mask)),
        "volume_mm3": float(volume) if unit != "mm3" else volume,
        "volume_cm3": float(volume) if unit == "cm3" else volume / 1000.0,
        "spacing": tuple(spacing),
        "unit": unit,
    }


def compute_multi_class_volumes(
    mask: np.ndarray,
    spacing: Union[Tuple[float, float, float], np.ndarray],
    class_labels: Optional[Dict[int, str]] = None,
    unit: str = "cm3",
) -> Dict[str, Dict[str, float]]:
    """
    计算多类别分割的各类别体积

    Args:
        mask: 多类别掩码 (D, H, W)，值为类别 ID
        spacing: 体素间距
        class_labels: 类别 ID 到名称的映射 {0: "background", 1: "spleen", 2: "tumor"}
        unit: 体积单位

    Returns:
        {class_name: {voxel_count, volume, ...}, ...}
    """
    if class_labels is None:
        unique_labels = np.unique(mask)
        class_labels = {int(l): f"class_{l}" for l in unique_labels}

    results = {}
    for label_id, label_name in class_labels.items():
        if label_id == 0:
            continue  # 跳过背景

        class_mask = (mask == label_id).astype(np.uint8)
        voxel_count = int(np.sum(class_mask))
        volume = compute_volume(class_mask, spacing, unit)

        results[label_name] = {
            "label_id": int(label_id),
            "voxel_count": voxel_count,
            f"volume_{unit}": volume,
        }

    return results


# =============================================================================
# 评估报告生成
# =============================================================================

def generate_dice_report(
    predictions: List[Union[str, Path, np.ndarray]],
    references: List[Union[str, Path, np.ndarray]],
    spacing: Optional[Union[Tuple[float, float, float], List[np.ndarray]]] = None,
    report_format: str = "dict",
) -> Dict:
    """
    生成 Dice 评估报告

    Args:
        predictions: 预测掩码列表
        references: 参考掩码列表
        spacing: 体素间距（单个或列表）
        report_format: 报告格式 ("dict", "dataframe")

    Returns:
        评估报告
    """
    from scipy import ndimage

    results = {
        "individual_dice": [],
        "mean_dice": 0.0,
        "std_dice": 0.0,
        "num_cases": len(predictions),
    }

    dice_scores = []

    for i, (pred, ref) in enumerate(zip(predictions, references)):
        # 加载为 numpy
        if isinstance(pred, (str, Path)):
            pred = nib.load(str(pred)).get_fdata()
        if isinstance(ref, (str, Path)):
            ref = nib.load(str(ref)).get_fdata()

        pred_binary = (pred > 0.5).astype(np.uint8)
        ref_binary = (ref > 0.5).astype(np.uint8)

        # 计算 Dice
        intersection = np.sum(pred_binary * ref_binary)
        union = np.sum(pred_binary) + np.sum(ref_binary)

        if union == 0:
            dice = 1.0 if np.sum(pred_binary) == 0 else 0.0
        else:
            dice = (2.0 * intersection) / union

        dice_scores.append(float(dice))

        results["individual_dice"].append({
            "case_id": i,
            "dice": dice,
            "intersection": int(intersection),
            "pred_voxel_count": int(np.sum(pred_binary)),
            "ref_voxel_count": int(np.sum(ref_binary)),
        })

    results["mean_dice"] = float(np.mean(dice_scores))
    results["std_dice"] = float(np.std(dice_scores))

    return results


def save_volume_report(
    volume_results: List[Dict],
    output_path: Union[str, Path],
    format: str = "csv",
) -> None:
    """
    保存体积报告

    Args:
        volume_results: 体积计算结果列表
        output_path: 输出路径
        format: 输出格式 ("csv", "json")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "csv":
        import csv

        # 获取所有可能的字段
        fieldnames = ["filename", "voxel_count", "volume_mm3", "volume_cm3", "unit"]

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in volume_results:
                row = {k: result.get(k, "N/A") for k in fieldnames}
                writer.writerow(row)

    elif format == "json":
        import json

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(volume_results, f, indent=2, ensure_ascii=False)

    else:
        raise ValueError(f"不支持的格式: {format}")

    print(f"体积报告已保存: {output_path}")


def save_dice_report(
    dice_results: Dict,
    output_path: Union[str, Path],
    format: str = "csv",
) -> None:
    """
    保存 Dice 评估报告

    Args:
        dice_results: Dice 计算结果
        output_path: 输出路径
        format: 输出格式 ("csv", "json")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "csv":
        import csv

        fieldnames = ["case_id", "dice", "intersection", "pred_voxel_count", "ref_voxel_count"]

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            # 写入摘要行
            writer.writerow({
                "case_id": "SUMMARY",
                "dice": f"mean={dice_results['mean_dice']:.4f}, std={dice_results['std_dice']:.4f}",
            })

            # 写入每个病例
            for row in dice_results["individual_dice"]:
                writer.writerow(row)

    elif format == "json":
        import json

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dice_results, f, indent=2, ensure_ascii=False)

    else:
        raise ValueError(f"不支持的格式: {format}")

    print(f"Dice 报告已保存: {output_path}")


# =============================================================================
# 综合评估
# =============================================================================

class VolumeAnalyzer:
    """
    体积分析器

    批量计算体积并生成报告
    """

    def __init__(
        self,
        spacing: Optional[Tuple[float, float, float]] = None,
        unit: str = "cm3",
    ):
        """
        Args:
            spacing: 体素间距（用于所有计算）
            unit: 体积单位
        """
        self.spacing = spacing
        self.unit = unit
        self.results: List[Dict] = []

    def analyze(
        self,
        mask: np.ndarray,
        case_name: str,
        spacing: Optional[Tuple[float, float, float]] = None,
    ) -> Dict:
        """
        分析单个掩码的体积

        Args:
            mask: 二值掩码
            case_name: 病例名称
            spacing: 体素间距（优先于此对象的 spacing）

        Returns:
            分析结果
        """
        spacing = spacing or self.spacing
        if spacing is None:
            raise ValueError("必须提供 spacing")

        volume = compute_volume(mask, spacing, self.unit)
        voxel_count = int(np.sum(mask > 0))

        result = {
            "case_name": case_name,
            "voxel_count": voxel_count,
            f"volume_{self.unit}": volume,
            "spacing": spacing,
        }

        self.results.append(result)
        return result

    def get_summary(self) -> Dict:
        """获取汇总统计"""
        if not self.results:
            return {}

        volumes = [r[f"volume_{self.unit}"] for r in self.results]

        return {
            "num_cases": len(self.results),
            f"mean_volume_{self.unit}": float(np.mean(volumes)),
            f"std_volume_{self.unit}": float(np.std(volumes)),
            f"min_volume_{self.unit}": float(np.min(volumes)),
            f"max_volume_{self.unit}": float(np.max(volumes)),
            f"median_volume_{self.unit}": float(np.median(volumes)),
        }

    def save_report(self, output_path: str, format: str = "csv"):
        """保存报告"""
        save_volume_report(self.results, output_path, format)


__all__ = [
    "compute_volume",
    "compute_volume_from_nifti",
    "compute_multi_class_volumes",
    "generate_dice_report",
    "save_volume_report",
    "save_dice_report",
    "VolumeAnalyzer",
]
