"""
数据集下载脚本 - MSD Spleen 数据集

Medical Segmentation Decathlon (MSD) Task09_Spleen 数据集
下载地址: http://medicaldecathlon.com/

Usage:
    python scripts/download_data.py --output_dir rawdata
"""

import os
import sys
import json
import argparse
import urllib.request
import zipfile
import tarfile
from pathlib import Path
from tqdm import tqdm


# MSD Spleen 数据集信息
MSD_SPLEEN_INFO = {
    "name": "MSD_Spleen",
    "task_id": "Task09_Spleen",
    "url": "https://msd-for-monai.s3.us-east-2.amazonaws.com/Task09_Spleen.tar",
    "description": "Spleen CT Segmentation - Medical Segmentation Decathlon",
    "expected_images": 41,  # 训练集 41 个病例
    "expected_labels": 41,
}


class DownloadProgressBar(tqdm):
    """带进度条的下载器"""

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url: str, output_path: str, description: str = ""):
    """下载文件并显示进度条"""
    with DownloadProgressBar(
        unit="B", unit_scale=True, miniters=1, desc=description
    ) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def extract_archive(archive_path: str, output_dir: str) -> bool:
    """解压压缩包"""
    archive_path = Path(archive_path)
    output_dir = Path(output_dir)

    try:
        if archive_path.suffix == ".tar":
            with tarfile.open(archive_path, "r:*") as tar:
                tar.extractall(output_dir)
        elif archive_path.suffix == ".zip":
            with zipfile.ZipFile(archive_path, "r") as zip_ref:
                zip_ref.extractall(output_dir)
        else:
            print(f"Unsupported archive format: {archive_path.suffix}")
            return False
        return True
    except Exception as e:
        print(f"Extraction failed: {e}")
        return False


def verify_dataset(data_dir: Path) -> dict:
    """校验数据集完整性"""
    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"

    image_files = list(images_dir.glob("*.nii.gz")) if images_dir.exists() else []
    label_files = list(labels_dir.glob("*.nii.gz")) if labels_dir.exists() else []

    result = {
        "dataset": data_dir.name,
        "images_count": len(image_files),
        "labels_count": len(label_files),
        "complete": len(image_files) == len(label_files) == MSD_SPLEEN_INFO["expected_images"],
        "images_sample": [f.name for f in image_files[:3]] if image_files else [],
    }

    return result


def download_msd_spleen(output_dir: str) -> dict:
    """下载并解压 MSD Spleen 数据集"""
    output_dir = Path(output_dir)
    dataset_dir = output_dir / "MSD_Spleen"

    # 创建目录
    dataset_dir.mkdir(parents=True, exist_ok=True)
    archive_path = dataset_dir / "Task09_Spleen.tar"

    print(f"\n{'='*60}")
    print(f"下载 MSD Spleen 数据集")
    print(f"{'='*60}")
    print(f"数据集描述: {MSD_SPLEEN_INFO['description']}")
    print(f"预期图像数量: {MSD_SPLEEN_INFO['expected_images']}")
    print(f"下载地址: {MSD_SPLEEN_INFO['url']}")
    print(f"{'='*60}\n")

    # 检查是否已下载
    if verify_dataset(dataset_dir)["complete"]:
        print(f"数据集已完整存在，跳过下载: {dataset_dir}")
        return verify_dataset(dataset_dir)

    # 检查压缩包是否已存在
    if not archive_path.exists():
        print(f"正在下载数据集...")
        download_url(
            MSD_SPLEEN_INFO["url"],
            str(archive_path),
            description="Downloading MSD Spleen"
        )
    else:
        print(f"压缩包已存在: {archive_path}")

    # 解压
    print(f"\n正在解压数据集...")
    temp_extract_dir = output_dir / "temp_spleen"
    if extract_archive(str(archive_path), str(temp_extract_dir)):
        # 移动文件到正确位置
        extracted_dir = temp_extract_dir / "Task09_Spleen"
        if extracted_dir.exists():
            # 创建 images 和 labels 目录
            images_dir = dataset_dir / "images"
            labels_dir = dataset_dir / "labels"
            images_dir.mkdir(exist_ok=True)
            labels_dir.mkdir(exist_ok=True)

            # 移动文件
            for f in (extracted_dir / "images").glob("*"):
                f.rename(images_dir / f.name)
            for f in (extracted_dir / "labels").glob("*"):
                f.rename(labels_dir / f.name)

            # 清理临时目录
            temp_extract_dir.rmdir()
            print(f"解压完成！")
        else:
            print(f"警告: 解压后的目录结构不符合预期")
            print(f"解压内容: {list(temp_extract_dir.iterdir())}")
    else:
        print(f"解压失败！")
        return {"success": False}

    # 清理压缩包
    if archive_path.exists():
        archive_path.unlink()
        print(f"已清理压缩包: {archive_path}")

    # 验证
    result = verify_dataset(dataset_dir)
    result["success"] = result["complete"]
    return result


def generate_dataset_info(output_dir: str) -> dict:
    """生成数据集元信息"""
    dataset_dir = Path(output_dir) / "MSD_Spleen"

    # 扫描所有 nii.gz 文件
    all_files = list(dataset_dir.rglob("*.nii.gz"))

    # 统计基本信息
    info = {
        "dataset_name": "MSD_Spleen",
        "task_description": "Spleen CT Segmentation",
        "total_files": len(all_files),
        "images_dir": str(dataset_dir / "images"),
        "labels_dir": str(dataset_dir / "labels"),
        "files": [],
    }

    # 对每个文件记录基本信息
    try:
        import nibabel as nib
        for f in all_files[:5]:  # 只取前5个作为样本
            img = nib.load(str(f))
            info["files"].append({
                "filename": f.name,
                "path": str(f),
                "shape": list(img.shape),
                "spacing": list(img.header.get_zooms()),
            })
    except ImportError:
        print("提示: nibabel 未安装，跳过详细元信息扫描")
        for f in all_files:
            info["files"].append({
                "filename": f.name,
                "path": str(f),
            })

    return info


def save_dataset_info(info: dict, output_dir: str):
    """保存数据集元信息到 JSON"""
    info_path = Path(output_dir) / "dataset_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"数据集信息已保存: {info_path}")


def main():
    parser = argparse.ArgumentParser(description="下载 MSD Spleen 数据集")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="rawdata",
        help="数据集输出目录 (default: rawdata)"
    )
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="跳过下载，仅验证已存在的数据集"
    )
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"MONAI 3D 医学影像分割系统 - 数据集下载")
    print(f"{'='*60}\n")

    output_dir = args.output_dir

    if args.skip_download:
        # 仅验证
        dataset_dir = Path(output_dir) / "MSD_Spleen"
        if dataset_dir.exists():
            result = verify_dataset(dataset_dir)
            print(f"\n数据集验证结果:")
            print(f"  - 图像数量: {result['images_count']}")
            print(f"  - 标注数量: {result['labels_count']}")
            print(f"  - 数据完整: {'是' if result['complete'] else '否'}")
        else:
            print(f"数据集目录不存在: {dataset_dir}")
        return

    # 下载数据集
    result = download_msd_spleen(output_dir)

    if result.get("success") or result.get("complete"):
        print(f"\n{'='*60}")
        print(f"下载完成！")
        print(f"{'='*60}")
        print(f"图像数量: {result.get('images_count', 0)}")
        print(f"标注数量: {result.get('labels_count', 0)}")

        # 生成元信息
        info = generate_dataset_info(output_dir)
        save_dataset_info(info, output_dir)
        print(f"\n数据集已准备就绪，可以开始训练！")
    else:
        print(f"\n下载失败，请检查网络连接后重试。")
        sys.exit(1)


if __name__ == "__main__":
    main()
