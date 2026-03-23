"""
Evaluator 模块

提供推理、后处理、体积计算等评估功能

Submodules:
    inference: 滑窗推理
    postprocess: 后处理
    volume: 体积计算
"""

from .inference import (
    DEFAULT_SWIN_CONFIG,
    SlidingWindowInferer,
    SimpleInferer,
    create_inferer,
)

from .postprocess import (
    PostProcessConfig,
    threshold_predictions,
    keep_largest_connected_component,
    remove_small_components,
    fill_holes_3d,
    smooth_boundary_3d,
    morphological_closing,
    morphological_opening,
    PostProcessPipeline,
    postprocess_predictions,
)

from .volume import (
    compute_volume,
    compute_volume_from_nifti,
    compute_multi_class_volumes,
    generate_dice_report,
    save_volume_report,
    save_dice_report,
    VolumeAnalyzer,
)

__all__ = [
    # Inference
    "DEFAULT_SWIN_CONFIG",
    "SlidingWindowInferer",
    "SimpleInferer",
    "create_inferer",
    # Postprocess
    "PostProcessConfig",
    "threshold_predictions",
    "keep_largest_connected_component",
    "remove_small_components",
    "fill_holes_3d",
    "smooth_boundary_3d",
    "morphological_closing",
    "morphological_opening",
    "PostProcessPipeline",
    "postprocess_predictions",
    # Volume
    "compute_volume",
    "compute_volume_from_nifti",
    "compute_multi_class_volumes",
    "generate_dice_report",
    "save_volume_report",
    "save_dice_report",
    "VolumeAnalyzer",
]
