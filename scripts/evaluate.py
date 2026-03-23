"""
评估入口脚本

对验证集/测试集进行评估，生成 Dice 报告和体积报告

Usage:
    python scripts/evaluate.py --model models/best_model.pt --data_dir rawdata/MSD_Spleen
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import nibabel as nib

from src.model_builder import load_model
from src.evaluator import (
    create_inferer,
    PostProcessPipeline,
    generate_dice_report,
    save_dice_report,
    save_volume_report,
    VolumeAnalyzer,
)


def parse_args():
    parser = argparse.ArgumentParser(description="3D 医学影像分割模型评估")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="模型检查点路径"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="数据集目录"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/reports",
        help="输出报告目录"
    )
    parser.add_argument(
        "--roi_size",
        type=int,
        nargs=3,
        default=[128, 128, 128],
        help="滑窗 ROI 尺寸"
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        help="滑窗重叠率"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="预测阈值"
    )
    parser.add_argument(
        "--postprocess",
        action="store_true",
        help="是否应用后处理"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="评估设备 (auto/cuda/mps/cpu)"
    )
    return parser.parse_args()


def find_image_label_pairs(data_dir: str) -> list:
    """查找图像-标注配对"""
    data_dir = Path(data_dir)
    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        # 尝试从列表文件加载
        train_list = data_dir / "train_list.json"
        val_list = data_dir / "val_list.json"

        if train_list.exists():
            import json
            with open(train_list) as f:
                return json.load(f)
        if val_list.exists():
            import json
            with open(val_list) as f:
                return json.load(f)

        raise FileNotFoundError(f"找不到图像或标注目录: {images_dir}, {labels_dir}")

    pairs = []
    for img_path in sorted(images_dir.glob("*.nii.gz")):
        label_path = labels_dir / img_path.name.replace("image", "label")
        if label_path.exists():
            pairs.append({
                "image": str(img_path),
                "label": str(label_path),
            })
    return pairs


def main():
    args = parse_args()

    print("\n" + "="*60)
    print("MONAI 3D 医学影像分割 - 模型评估")
    print("="*60 + "\n")

    # 设备 - 自动检测
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device if args.device in ["cuda", "mps", "cpu"] else "cpu"
    print(f"使用设备: {device}")

    # 加载模型
    print(f"加载模型: {args.model}")
    model, checkpoint_info = load_model(args.model, device=device)
    model.eval()
    print(f"模型加载完成，最佳 Dice: {checkpoint_info.get('best_metric', 'N/A')}")

    # 创建推理器
    inferer = create_inferer(
        model=model,
        config={
            "roi_size": tuple(args.roi_size),
            "overlap": args.overlap,
        },
        use_sliding_window=True,
        device=device,
    )

    # 创建后处理器
    postprocess_pipeline = None
    if args.postprocess:
        postprocess_pipeline = PostProcessPipeline(
            threshold=args.threshold,
            keep_largest=True,
            min_volume=500,
            fill_holes=True,
        )

    # 加载数据对
    pairs = find_image_label_pairs(args.data_dir)
    print(f"找到 {len(pairs)} 个图像-标注配对")

    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 评估
    print("\n开始评估...\n")

    dice_results = []
    volume_results = []
    volume_analyzer = VolumeAnalyzer(spacing=(1.0, 1.0, 1.0), unit="cm3")

    for i, pair in enumerate(pairs):
        print(f"[{i+1}/{len(pairs)}] 评估: {Path(pair['image']).name}")

        try:
            # 加载图像和标注
            img = nib.load(pair["image"])
            label_img = nib.load(pair["label"])

            image_data = img.get_fdata()
            label_data = label_img.get_fdata()
            spacing = img.header.get_zooms()[:3]

            # 转换为 tensor
            image_tensor = torch.from_numpy(image_data).float().unsqueeze(0).unsqueeze(0)
            image_tensor = image_tensor.to(device)

            # 推理
            with torch.no_grad():
                prob_map = inferer.inference(image_tensor, return_prob=True)

            # 转回 numpy
            prob_map = prob_map.cpu().numpy()[0, 0]

            # 后处理
            if postprocess_pipeline:
                pred_mask = postprocess_pipeline.apply(prob_map)
            else:
                pred_mask = (prob_map > args.threshold).astype(np.uint8)

            # 计算 Dice
            intersection = np.sum(pred_mask * (label_data > 0))
            pred_sum = np.sum(pred_mask)
            label_sum = np.sum(label_data > 0)
            union = pred_sum + label_sum

            if union == 0:
                dice = 1.0 if pred_sum == 0 else 0.0
            else:
                dice = (2.0 * intersection) / union

            dice_results.append({
                "case_id": i,
                "case_name": Path(pair["image"]).stem,
                "dice": float(dice),
                "pred_voxel_count": int(pred_sum),
                "ref_voxel_count": int(label_sum),
            })

            print(f"  - Dice: {dice:.4f}")

            # 体积计算
            pred_binary = (pred_mask > 0).astype(np.uint8)
            voxel_count = int(np.sum(pred_binary))
            volume_cm3 = voxel_count * abs(spacing[0] * spacing[1] * spacing[2]) / 1000.0

            volume_results.append({
                "case_name": Path(pair["image"]).stem,
                "voxel_count": voxel_count,
                "volume_cm3": float(volume_cm3),
                "spacing": tuple(spacing),
            })

        except Exception as e:
            print(f"  - 错误: {e}")
            continue

    # 生成报告
    print("\n" + "-"*40)
    print("生成评估报告...")

    # Dice 报告
    if dice_results:
        mean_dice = np.mean([r["dice"] for r in dice_results])
        std_dice = np.std([r["dice"] for r in dice_results])

        print(f"\nDice 评估结果:")
        print(f"  - 平均 Dice: {mean_dice:.4f} ± {std_dice:.4f}")
        print(f"  - 最高 Dice: {max(r['dice'] for r in dice_results):.4f}")
        print(f"  - 最低 Dice: {min(r['dice'] for r in dice_results):.4f}")

        # 保存 Dice 报告
        dice_report_path = output_dir / "dice_scores.csv"
        save_dice_report(
            {
                "individual_dice": dice_results,
                "mean_dice": mean_dice,
                "std_dice": std_dice,
                "num_cases": len(dice_results),
            },
            dice_report_path,
            format="csv"
        )

    # 体积报告
    if volume_results:
        volumes_cm3 = [r["volume_cm3"] for r in volume_results]
        mean_volume = np.mean(volumes_cm3)
        std_volume = np.std(volumes_cm3)

        print(f"\n体积统计 (cm³):")
        print(f"  - 平均体积: {mean_volume:.2f} ± {std_volume:.2f}")
        print(f"  - 最大体积: {max(volumes_cm3):.2f}")
        print(f"  - 最小体积: {min(volumes_cm3):.2f}")

        # 保存体积报告
        volume_report_path = output_dir / "volumes.csv"
        save_volume_report(volume_results, volume_report_path, format="csv")

    print("\n" + "="*60)
    print("评估完成！")
    print(f"报告保存位置: {output_dir}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
