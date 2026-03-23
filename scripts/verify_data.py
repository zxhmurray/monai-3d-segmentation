"""
数据集校验脚本 - 验证 NIfTI 文件的完整性和一致性

Usage:
    python scripts/verify_data.py --data_dir rawdata/MSD_Spleen
    python scripts/verify_data.py --data_dir rawdata/MSD_Spleen --verbose
"""

import os
import sys
import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np


def load_nifti_safe(file_path: str) -> dict:
    """安全加载 NIfTI 文件，返回基本信息"""
    try:
        import nibabel as nib
        img = nib.load(str(file_path))
        data = img.get_fdata()
        header = img.header
        return {
            "success": True,
            "shape": data.shape,
            "spacing": tuple(header.get_zooms()),
            "dtype": str(data.dtype),
            "min": float(data.min()),
            "max": float(data.max()),
            "mean": float(data.mean()),
            "nonzero_count": int(np.count_nonzero(data)),
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def verify_patient_case(images_dir: Path, labels_dir: Path, patient_id: str, verbose: bool = False) -> dict:
    """验证单个病例的图像和标注配对"""
    image_files = list(images_dir.glob(f"{patient_id}*.nii.gz"))
    label_files = list(labels_dir.glob(f"{patient_id}*.nii.gz"))

    result = {
        "patient_id": patient_id,
        "has_images": len(image_files) > 0,
        "has_labels": len(label_files) > 0,
        "paired": False,
        "details": {},
    }

    if not result["has_images"]:
        if verbose:
            print(f"  [警告] 病例 {patient_id}: 未找到图像文件")
        return result

    if not result["has_labels"]:
        if verbose:
            print(f"  [警告] 病例 {patient_id}: 未找到标注文件")
        return result

    # 配对验证
    if len(image_files) != len(label_files):
        if verbose:
            print(f"  [警告] 病例 {patient_id}: 图像与标注数量不匹配 ({len(image_files)} vs {len(label_files)})")
        return result

    # 检查每个配对
    for img_file in image_files:
        # 查找对应的 label 文件
        label_file = labels_dir / img_file.name.replace("image", "label")
        if not label_file.exists():
            # 尝试其他命名模式
            label_file = labels_dir / img_file.name.replace("img", "label")

        if not label_file.exists():
            if verbose:
                print(f"  [警告] 病例 {patient_id}: 找不到标注文件 {img_file.name}")
            continue

        # 加载并比较
        img_info = load_nifti_safe(str(img_file))
        lbl_info = load_nifti_safe(str(label_file))

        if not img_info["success"]:
            if verbose:
                print(f"  [错误] 病例 {patient_id}: 图像加载失败 - {img_info.get('error')}")
            continue

        if not lbl_info["success"]:
            if verbose:
                print(f"  [错误] 病例 {patient_id}: 标注加载失败 - {lbl_info.get('error')}")
            continue

        # 检查 shape 一致性
        if img_info["shape"] != lbl_info["shape"]:
            if verbose:
                print(f"  [错误] 病例 {patient_id}: Shape 不匹配 - 图像 {img_info['shape']} vs 标注 {lbl_info['shape']}")
            continue

        result["paired"] = True
        result["details"][img_file.name] = {
            "image_shape": img_info["shape"],
            "image_spacing": img_info["spacing"],
            "label_shape": lbl_info["shape"],
            "label_spacing": lbl_info["spacing"],
            "label_unique_values": sorted(list(np.unique(
                nib_to_numpy_safe(str(label_file))
            ))) if lbl_info["success"] else [],
        }

    return result


def nib_to_numpy_safe(file_path: str) -> np.ndarray:
    """安全加载为 numpy 数组"""
    try:
        import nibabel as nib
        return nib.load(str(file_path)).get_fdata()
    except:
        return np.array([])


def verify_dataset(data_dir: str, verbose: bool = False) -> dict:
    """验证整个数据集"""
    data_dir = Path(data_dir)
    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"

    result = {
        "data_dir": str(data_dir),
        "exists": data_dir.exists(),
        "images_dir_exists": images_dir.exists(),
        "labels_dir_exists": labels_dir.exists(),
        "total_images": 0,
        "total_labels": 0,
        "paired_cases": 0,
        "unpaired_cases": [],
        "shape_stats": defaultdict(int),
        "spacing_stats": defaultdict(int),
        "errors": [],
    }

    if not data_dir.exists():
        result["errors"].append(f"数据集目录不存在: {data_dir}")
        return result

    if not images_dir.exists():
        result["errors"].append(f"图像目录不存在: {images_dir}")
        return result

    if not labels_dir.exists():
        result["errors"].append(f"标注目录不存在: {labels_dir}")
        return result

    # 获取所有患者 ID
    image_files = list(images_dir.glob("*.nii.gz"))
    label_files = list(labels_dir.glob("*.nii.gz"))

    result["total_images"] = len(image_files)
    result["total_labels"] = len(label_files)

    # 提取患者 ID（假设命名格式为 "patient_id_*.nii.gz"）
    patient_ids = set()
    for f in image_files:
        # 尝试从文件名提取 ID
        parts = f.stem.replace(".nii", "").replace("_image", "").replace("_img", "").split("_")
        if parts:
            patient_ids.add(parts[0])

    for pid in sorted(patient_ids):
        case_result = verify_patient_case(images_dir, labels_dir, pid, verbose)
        if case_result["paired"]:
            result["paired_cases"] += 1
            # 收集统计信息
            for file_name, details in case_result["details"].items():
                result["shape_stats"][details["image_shape"]] += 1
                result["spacing_stats"][details["image_spacing"]] += 1
        else:
            result["unpaired_cases"].append(pid)

    return result


def print_report(result: dict):
    """打印验证报告"""
    print(f"\n{'='*60}")
    print(f"数据集验证报告")
    print(f"{'='*60}\n")

    print(f"目录信息:")
    print(f"  - 数据集路径: {result['data_dir']}")
    print(f"  - 图像目录: {'存在' if result['images_dir_exists'] else '不存在'}")
    print(f"  - 标注目录: {'存在' if result['labels_dir_exists'] else '不存在'}")
    print()

    if result["errors"]:
        print(f"错误信息:")
        for err in result["errors"]:
            print(f"  - {err}")
        print()

    print(f"文件统计:")
    print(f"  - 图像文件总数: {result['total_images']}")
    print(f"  - 标注文件总数: {result['total_labels']}")
    print(f"  - 配对的病例数: {result['paired_cases']}")
    print(f"  - 未配对的病例: {len(result['unpaired_cases'])}")
    if result["unpaired_cases"]:
        print(f"    列表: {result['unpaired_cases'][:5]}{'...' if len(result['unpaired_cases']) > 5 else ''}")
    print()

    if result["shape_stats"]:
        print(f"Shape 分布 (图像尺寸):")
        for shape, count in sorted(result["shape_stats"].items(), key=lambda x: -x[1])[:5]:
            print(f"  - {shape}: {count} 个文件")
        print()

    if result["spacing_stats"]:
        print(f"Spacing 分布 (体素间距):")
        for spacing, count in sorted(result["spacing_stats"].items(), key=lambda x: -x[1])[:5]:
            print(f"  - {spacing}: {count} 个文件")
        print()

    # 总结
    is_valid = (
        result["exists"]
        and result["images_dir_exists"]
        and result["labels_dir_exists"]
        and result["paired_cases"] > 0
        and len(result["errors"]) == 0
    )

    print(f"{'='*60}")
    if is_valid:
        print(f"✓ 数据集验证通过 ({result['paired_cases']} 个有效病例)")
    else:
        print(f"✗ 数据集验证失败，请检查上述错误")
    print(f"{'='*60}\n")

    return is_valid


def save_report(result: dict, output_path: str):
    """保存验证报告到 JSON"""
    # JSON 不支持 defaultdict，需要转换
    result_save = {k: v for k, v in result.items() if k not in ["shape_stats", "spacing_stats"]}
    result_save["shape_stats"] = {str(k): v for k, v in result["shape_stats"].items()}
    result_save["spacing_stats"] = {str(k): v for k, v in result["spacing_stats"].items()}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_save, f, indent=2, ensure_ascii=False)
    print(f"验证报告已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="验证医学影像数据集")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="rawdata/MSD_Spleen",
        help="数据集目录路径"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="显示详细验证信息"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="验证报告输出路径 (JSON)"
    )
    args = parser.parse_args()

    print(f"正在验证数据集: {args.data_dir}")
    result = verify_dataset(args.data_dir, verbose=args.verbose)

    # 打印报告
    is_valid = print_report(result)

    # 保存报告
    if args.output:
        save_report(result, args.output)
    else:
        # 默认保存到 data_dir 下的 verify_report.json
        report_path = Path(args.data_dir) / "verify_report.json"
        save_report(result, str(report_path))

    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()
