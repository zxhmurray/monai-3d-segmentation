"""
生成合成测试数据集

用于在没有真实数据时测试训练流程

Usage:
    python scripts/generate_test_data.py --output_dir rawdata/MSD_Spleen --num_samples 10
"""

import os
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib


def create_synthetic_ct(size=(128, 128, 64)):
    """创建合成 CT 图像"""
    # 创建球形结构（代表脾脏）
    data = np.zeros(size, dtype=np.float32)

    # 添加背景噪声
    data = np.random.normal(0, 10, size)

    # 中心位置 (D, H, W)
    cd, ch, cw = size[0] // 2, size[1] // 2, size[2] // 2
    r = min(size) // 6

    # 创建球形脾脏
    for d in range(size[0]):
        for h in range(size[1]):
            for w in range(size[2]):
                dist = np.sqrt((d - cd)**2 + (h - ch)**2 + ((w - cw) * 2)**2)
                if dist < r:
                    data[d, h, w] = 50 + np.random.normal(0, 5)  # 脾脏密度

    # 添加一些随机小物体
    for _ in range(5):
        sd = np.random.randint(r, size[0] - r)
        sh = np.random.randint(r, size[1] - r)
        sw = np.random.randint(r, size[2] - r)
        sr = np.random.randint(r // 4, r // 2)
        for d in range(size[0]):
            for h in range(size[1]):
                for w in range(size[2]):
                    dist = np.sqrt((d - sd)**2 + (h - sh)**2 + ((w - sw) * 2)**2)
                    if dist < sr:
                        data[d, h, w] = 40 + np.random.normal(0, 3)

    return data


def create_synthetic_label(size=(128, 128, 64)):
    """创建合成标注（球形标签）"""
    label = np.zeros(size, dtype=np.uint8)

    cd, ch, cw = size[0] // 2, size[1] // 2, size[2] // 2
    r = min(size) // 6

    for d in range(size[0]):
        for h in range(size[1]):
            for w in range(size[2]):
                dist = np.sqrt((d - cd)**2 + (h - ch)**2 + ((w - cw) * 2)**2)
                if dist < r:
                    label[d, h, w] = 1

    return label


def generate_test_dataset(output_dir, num_samples=10, size=(128, 128, 64)):
    """生成测试数据集"""
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    print(f"生成 {num_samples} 个合成测试样本...")
    print(f"图像尺寸: {size}")
    print(f"输出目录: {output_dir}")

    for i in range(num_samples):
        # 创建合成数据
        image_data = create_synthetic_ct(size)
        label_data = create_synthetic_label(size)

        # 创建 NIfTI 图像
        # 仿射矩阵（单位矩阵 + spacing）
        affine = np.eye(4)
        affine[0, 0] = 1.0  # x spacing
        affine[1, 1] = 1.0  # y spacing
        affine[2, 2] = 2.0  # z spacing (更厚的层)

        image_nii = nib.Nifti1Image(image_data, affine)
        label_nii = nib.Nifti1Image(label_data.astype(np.float32), affine)

        # 保存
        image_path = images_dir / f"spleen_{i+1:03d}.nii.gz"
        label_path = labels_dir / f"spleen_{i+1:03d}.nii.gz"

        nib.save(image_nii, str(image_path))
        nib.save(label_nii, str(label_path))

        print(f"  [{i+1}/{num_samples}] {image_path.name}")

    print("\n测试数据集生成完成！")

    # 验证
    image_files = list(images_dir.glob("*.nii.gz"))
    label_files = list(labels_dir.glob("*.nii.gz"))
    print(f"\n验证:")
    print(f"  图像文件数: {len(image_files)}")
    print(f"  标注文件数: {len(label_files)}")

    # 加载一个样本验证
    if image_files:
        img = nib.load(str(image_files[0]))
        lbl = nib.load(str(label_files[0]))
        print(f"  样本 shape: {img.shape}")
        print(f"  样本 spacing: {img.header.get_zooms()}")
        print(f"  标注唯一值: {np.unique(lbl.get_fdata())}")

    return output_dir


def main():
    parser = argparse.ArgumentParser(description="生成合成测试数据集")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="rawdata/MSD_Spleen",
        help="输出目录"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="样本数量"
    )
    parser.add_argument(
        "--size",
        type=int,
        nargs=3,
        default=[64, 128, 128],
        help="图像尺寸 (D H W)"
    )
    args = parser.parse_args()

    generate_test_dataset(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        size=tuple(args.size)
    )


if __name__ == "__main__":
    main()
