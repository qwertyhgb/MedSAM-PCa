"""模型定义。

包含：

* :mod:`src.models.medsam_encoder` —— MedSAM ViT-B 多层特征 encoder（Phase 0）
* :mod:`src.models.medsam_pca` —— 统一 model factory（Phase 1）
* 四个 decoder baseline：E0 linear / E1 simple pyramid / E2 UNETR / E3 multi-level FPN
"""
from .medsam_encoder import (
    DEFAULT_CHECKPOINT,
    DEFAULT_MEDSAM_REPO,
    VIT_B_BLOCK_INDICES,
    MedSAMMultiLevelEncoder,
    add_medsam_to_sys_path,
    build_medsam_multilevel_encoder,
    load_official_sam,
)
from .medsam_pca import (
    MODEL_FEATURE_KEYS,
    SUPPORTED_MODELS,
    MedSAMPCA,
    build_model,
)
from .multilevel_fpn import MultiLevelFPN
from .segmentation_head import LinearHead, SegmentationHead, TwoStageSegHead
from .simple_pyramid import SimplePyramid
from .unetr_decoder import UNETRStyleDecoder

__all__ = [
    "MedSAMMultiLevelEncoder",
    "build_medsam_multilevel_encoder",
    "load_official_sam",
    "add_medsam_to_sys_path",
    "VIT_B_BLOCK_INDICES",
    "DEFAULT_MEDSAM_REPO",
    "DEFAULT_CHECKPOINT",
    "MedSAMPCA",
    "build_model",
    "SUPPORTED_MODELS",
    "MODEL_FEATURE_KEYS",
    "LinearHead",
    "SegmentationHead",
    "TwoStageSegHead",
    "SimplePyramid",
    "UNETRStyleDecoder",
    "MultiLevelFPN",
]
