"""
推理端点实现

提供图像推理接口
"""

import os
import time
import uuid
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import torch

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks

from .models import (
    InferenceRequest,
    InferenceResult,
    BatchInferenceRequest,
    BatchInferenceResult,
    VolumeReport,
    ErrorResponse,
)
from .config import get_settings
from .model_manager import get_model_manager

router = APIRouter(prefix="/inference", tags=["inference"])


def save_upload_file(upload_file: UploadFile, upload_dir: str) -> str:
    """保存上传文件"""
    settings = get_settings()
    os.makedirs(upload_dir, exist_ok=True)

    file_id = str(uuid.uuid4())
    file_ext = Path(upload_file.filename).suffix or ".nii.gz"
    saved_path = os.path.join(upload_dir, f"{file_id}{file_ext}")

    with open(saved_path, "wb") as f:
        content = upload_file.file.read()
        f.write(content)

    return saved_path


def compute_volume(mask: np.ndarray, spacing: tuple) -> dict:
    """计算体积"""
    voxel_count = int(np.sum(mask > 0))
    volume_mm3 = voxel_count * abs(spacing[0] * spacing[1] * spacing[2])
    volume_cm3 = volume_mm3 / 1000.0

    return {
        "voxel_count": voxel_count,
        "volume_mm3": float(volume_mm3),
        "volume_cm3": float(volume_cm3),
    }


@router.post("/predict", response_model=InferenceResult)
async def predict(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model_name: Optional[str] = None,
    threshold: float = 0.5,
    overlap: float = 0.5,
    return_prob: bool = False,
):
    """
    单图像推理接口

    上传 NIfTI 图像，返回预测结果
    """
    settings = get_settings()
    start_time = time.time()

    try:
        # 保存上传文件
        saved_path = save_upload_file(file, settings.upload_dir)
        case_id = Path(saved_path).stem

        # 获取模型
        model_manager = get_model_manager()
        model = model_manager.get_model(model_name or settings.default_model)

        # 加载图像
        img = nib.load(saved_path)
        image_data = img.get_fdata()
        spacing = img.header.get_zooms()[:3]

        # 转换为 tensor
        image_tensor = torch.from_numpy(image_data).float().unsqueeze(0).unsqueeze(0)
        image_tensor = image_tensor.to(model_manager.device)

        # 推理
        with torch.no_grad():
            from monai.inferers import sliding_window_inference

            prob_map = sliding_window_inference(
                inputs=image_tensor,
                roi_size=settings.roi_size,
                sw_batch_size=settings.sw_batch_size,
                predictor=model,
                overlap=overlap,
                blend_mode="gaussian",
                device=model_manager.device,
            )

        # 后处理
        prob_map = prob_map.cpu().numpy()[0, 0]
        pred_mask = (prob_map > threshold).astype(np.uint8)

        # 计算体积
        volume_info = compute_volume(pred_mask, spacing)

        # 保存预测结果
        output_dir = Path("results/predictions")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{case_id}_pred.nii.gz"

        pred_img = nib.Nifti1Image(pred_mask, img.affine)
        nib.save(pred_img, str(output_path))

        # 清理上传文件
        background_tasks.add_task(os.remove, saved_path)

        processing_time = time.time() - start_time

        return InferenceResult(
            case_id=case_id,
            status="success",
            message="推理完成",
            prediction_path=str(output_path),
            **volume_info,
            processing_time=round(processing_time, 2),
        )

    except Exception as e:
        processing_time = time.time() - start_time
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict_batch", response_model=BatchInferenceResult)
async def predict_batch(request: BatchInferenceRequest):
    """
    批量推理接口

    传入文件路径列表，返回所有预测结果
    """
    settings = get_settings()
    start_time = time.time()

    model_manager = get_model_manager()
    model = model_manager.get_model(request.model_name or settings.default_model)

    results = []
    success_count = 0
    failed_count = 0

    for file_path in request.file_paths:
        try:
            case_id = Path(file_path).stem

            # 加载图像
            img = nib.load(file_path)
            image_data = img.get_fdata()
            spacing = img.header.get_zooms()[:3]

            # 转换为 tensor
            image_tensor = torch.from_numpy(image_data).float().unsqueeze(0).unsqueeze(0)
            image_tensor = image_tensor.to(model_manager.device)

            # 推理
            with torch.no_grad():
                from monai.inferers import sliding_window_inference

                prob_map = sliding_window_inference(
                    inputs=image_tensor,
                    roi_size=settings.roi_size,
                    sw_batch_size=settings.sw_batch_size,
                    predictor=model,
                    overlap=request.overlap,
                    blend_mode="gaussian",
                    device=model_manager.device,
                )

            # 后处理
            prob_map = prob_map.cpu().numpy()[0, 0]
            pred_mask = (prob_map > request.threshold).astype(np.uint8)

            # 计算体积
            volume_info = compute_volume(pred_mask, spacing)

            # 保存预测结果
            output_dir = Path("results/predictions")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{case_id}_pred.nii.gz"

            pred_img = nib.Nifti1Image(pred_mask, img.affine)
            nib.save(pred_img, str(output_path))

            results.append(InferenceResult(
                case_id=case_id,
                status="success",
                message="推理完成",
                prediction_path=str(output_path),
                **volume_info,
                processing_time=0.0,
            ))
            success_count += 1

        except Exception as e:
            results.append(InferenceResult(
                case_id=Path(file_path).stem,
                status="failed",
                message=str(e),
                processing_time=0.0,
            ))
            failed_count += 1

    total_time = time.time() - start_time

    return BatchInferenceResult(
        total=len(request.file_paths),
        success=success_count,
        failed=failed_count,
        results=results,
    )


@router.get("/volume/{case_id}", response_model=VolumeReport)
async def get_volume(case_id: str):
    """
    获取指定病例的体积报告

    从预测结果中计算体积
    """
    prediction_path = Path("results/predictions") / f"{case_id}_pred.nii.gz"

    if not prediction_path.exists():
        raise HTTPException(status_code=404, detail=f"预测结果不存在: {case_id}")

    try:
        img = nib.load(str(prediction_path))
        mask = img.get_fdata()
        spacing = img.header.get_zooms()[:3]

        volume_info = compute_volume(mask, spacing)

        return VolumeReport(
            case_id=case_id,
            spacing=spacing,
            **volume_info,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
