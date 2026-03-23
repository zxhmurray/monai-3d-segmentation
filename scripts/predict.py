"""
推理入口脚本

Usage:
    python scripts/predict.py --model models/best_model.pt --input rawdata/MSD_Spleen/images
"""

import os
import sys
import argparse
from pathlib import Path
from glob import glob

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import nibabel as nib

from src.model_builder import load_model, create_3d_unet
from src.evaluator import (
    create_inferer,
    PostProcessPipeline,
    PostProcessConfig,
    save_volume_report,
)


def parse_args():
    parser = argparse.ArgumentParser(description="3D 医学影像推理")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="模型检查点路径"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入图像路径或目录"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/predictions",
        help="输出目录"
    )
    parser.add_argument(
        "--roi_size",
        type=int,
        nargs=3,
        default=[128, 128, 128],
        help="滑窗 ROI 尺寸 (D H W)"
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
        "--device",
        type=str,
        default="auto",
        help="推理设备 (auto/cuda/mps/cpu)"
    )
    parser.add_argument(
        "--postprocess",
        action="store_true",
        help="是否应用后处理"
    )
    return parser.parse_args()


def load_input_images(input_path: str) -> list:
    """加载输入图像"""
    input_path = Path(input_path)

    if input_path.is_file():
        return [str(input_path)]
    elif input_path.is_dir():
        # 查找目录下的所有 nii 文件
        nii_files = sorted(glob(str(input_path / "*.nii.gz")))
        if not nii_files:
            nii_files = sorted(glob(str(input_path / "**/*.nii.gz"), recursive=True))
        return nii_files
    else:
        raise FileNotFoundError(f"输入路径不存在: {input_path}")


def main():
    args = parse_args()

    print("\n" + "="*60)
    print("MONAI 3D 医学影像分割 - 推理")
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
    print(f"模型加载完成，训练 Epoch: {checkpoint_info.get('epoch', 'N/A')}")

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

    # 加载输入图像
    input_files = load_input_images(args.input)
    print(f"找到 {len(input_files)} 个图像文件")

    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 推理
    print("\n开始推理...\n")

    for i, file_path in enumerate(input_files):
        print(f"[{i+1}/{len(input_files)}] 处理: {Path(file_path).name}")

        try:
            # 加载图像
            img = nib.load(file_path)
            image_data = img.get_fdata()
            spacing = img.header.get_zooms()[:3]
            original_affine = img.affine

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

            # 计算体积
            voxel_count = int(np.sum(pred_mask))
            volume_mm3 = voxel_count * abs(spacing[0] * spacing[1] * spacing[2])
            volume_cm3 = volume_mm3 / 1000.0

            print(f"  - 预测体素数: {voxel_count}")
            print(f"  - 预测体积: {volume_cm3:.2f} cm³")

            # 保存预测结果
            pred_img = nib.Nifti1Image(
                pred_mask.astype(np.uint8),
                original_affine,
            )
            output_path = output_dir / f"{Path(file_path).stem}_pred.nii.gz"
            nib.save(pred_img, str(output_path))
            print(f"  - 保存至: {output_path}")

        except Exception as e:
            print(f"  - 错误: {e}")
            continue

    print("\n" + "="*60)
    print("推理完成！")
    print(f"结果保存位置: {output_dir}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
