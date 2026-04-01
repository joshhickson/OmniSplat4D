"""
Configuration loader for OmniSplat4D.

Loads and validates the YAML config file using Pydantic v2 BaseModel.
Supports layered overrides: default.yaml < hardware profile YAML < env vars.

Usage:
    from omnisplat4d.core.config import load_config
    cfg = load_config("config/default.yaml",
                      profile="config/hardware_profiles/rtx3060_12gb.yaml")
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class IngestConfig(BaseModel):
    """Frame extraction and temporal sub-sampling parameters."""

    fps_target: float = Field(2.0, description="Target frames/sec after Laplacian keyframe selection")
    laplacian_threshold: float = Field(100.0, description="Variance threshold; frames below are blurry")
    max_image_size: int = Field(1024, description="Max edge dimension for SfM feature extraction")
    planar_fov_deg: float = Field(90.0, description="Horizontal FOV of each virtual planar camera (degrees)")
    planar_width: int = Field(1024, description="Output width of each planar crop in pixels")
    planar_height: int = Field(1024, description="Output height of each planar crop in pixels")
    num_cameras: int = Field(8, description="Number of overlapping virtual planar cameras around the sphere")


class SegmentConfig(BaseModel):
    """Masking cascade parameters (YOLOv8-nano + SAM 2.1 Tiny)."""

    yolo_model: str = Field("yolov8n.pt", description="Path or name of YOLOv8-nano weights")
    sam_checkpoint: str = Field("sam2.1_hiera_tiny.pt", description="Path to SAM 2.1 Tiny checkpoint")
    sam_config: str = Field("sam2.1_hiera_t.yaml", description="SAM 2.1 model config file")
    sam_max_vision_cache: int = Field(
        1, description="Strictly cap SAM vision feature cache entries to prevent OOM"
    )
    sam_storage_device: Literal["cpu", "cuda"] = Field(
        "cpu", description="Device for SAM temporal state (cpu = host RAM offload)"
    )
    sam_inference_device: Literal["cpu", "cuda"] = Field(
        "cpu", description="Device for SAM inference"
    )
    iou_reset_threshold: float = Field(
        0.5, description="IoU confidence below this triggers SAM tracking reset"
    )
    optical_flow_method: Literal["farneback"] = Field(
        "farneback", description="CPU optical flow algorithm for bounding-box warping"
    )


class ColmapConfig(BaseModel):
    """COLMAP reconstruction parameters."""

    route: Literal["stick", "drone"] = Field(
        "stick", description="Capture route: stick (operator masking) or drone (programmatic SfM)"
    )
    matcher: Literal["sequential", "vocab_tree"] = Field(
        "sequential",
        description="Feature matcher: sequential (stick) or vocab_tree (drone)",
    )
    sequential_overlap: int = Field(
        10, description="Temporal neighbor window for sequential matcher"
    )
    vocab_tree_neighbors: int = Field(
        20, description="Max NN images retrieved per query; capped to avoid 32GB RAM exhaustion"
    )
    vocab_tree_path: Optional[Path] = Field(None, description="Path to pre-built vocabulary tree")
    colmap_binary: str = Field("colmap", description="Path or name of the COLMAP executable")
    skip_feature_extraction: bool = Field(
        True,
        description=(
            "If True, bypass colmap feature_extractor entirely; "
            "cameras.txt and images.txt are injected programmatically by sfm/initializer.py"
        ),
    )


class StaticTrainConfig(BaseModel):
    """gsplat 3DGS training config for Phase 1 static background reconstruction."""

    sh_degree: int = Field(
        1, description="Spherical harmonics degree; 1 enforced for 12GB VRAM ceiling (vs 4× cost of SH3)"
    )
    packed: bool = Field(
        True, description="Enable packed tensor mode to avoid sparse gradient overhead"
    )
    densify_grad_threshold: float = Field(
        0.0004, description="Elevated densification threshold to cap primitive count on 12GB VRAM"
    )
    max_iterations: int = Field(30_000)
    learning_rate_positions: float = Field(1.6e-4)
    learning_rate_sh: float = Field(2.5e-3)
    learning_rate_opacity: float = Field(5e-2)
    learning_rate_scale: float = Field(5e-3)
    learning_rate_rotation: float = Field(1e-3)
    prune_interval: int = Field(1000, description="Iterations between opacity pruning passes")
    output_format: Literal["spz", "ply"] = Field("spz")


class DynamicTrainConfig(BaseModel):
    """SWinGS + MEGA 4DGS sliding-window training config for Phase 2 dynamic subjects."""

    window_size: int = Field(30, description="Temporal chunk size in frames (SWinGS window)")
    window_overlap: int = Field(5, description="Frame overlap between sequential windows for continuity")
    consistency_weight: float = Field(
        0.1, description="Lambda weighting the inter-window temporal consistency loss"
    )
    dc_color_only: bool = Field(
        True, description="Use MEGA DC+AC decomposition; disables SH arrays for 48× compression"
    )
    ac_mlp_hidden_dim: int = Field(64, description="Hidden layer width of the AC color predictor MLP")
    deformation_mlp_hidden_dim: int = Field(
        128, description="Hidden layer width of the temporal deformation MLP"
    )
    entropy_weight: float = Field(
        0.01, description="Coefficient for opacity entropy regularization loss"
    )
    opacity_prune_threshold: float = Field(
        0.005, description="Gaussians with opacity below this are pruned each prune_interval"
    )
    prune_interval: int = Field(1000)
    max_iterations_per_window: int = Field(10_000)
    checkpoint_fp16: bool = Field(
        True, description="Downcast canonical Gaussian state to FP16 before writing to disk"
    )


class CompositeConfig(BaseModel):
    """Compositing renderer parameters."""

    tile_size: int = Field(16, description="CUDA tile dimension for Radix sort (must be 16 for gsplat)")
    sort_key_bits: int = Field(
        64, description="Radix sort key width: high 32 bits = TileID, low 32 bits = float depth"
    )


class ExportConfig(BaseModel):
    """Export and packaging parameters."""

    formats: list[Literal["spz", "ogc_3dtiles", "unity_package"]] = Field(
        default_factory=lambda: ["spz"]
    )
    stream_chunk_size_mb: int = Field(50, description="Max SPZ chunk size for streaming manifests (MB)")
    onnx_opset: int = Field(17, description="ONNX opset version for MLP export")


class PipelineConfig(BaseModel):
    """Root config. All pipeline phase configs are nested here."""

    workspace_dir: Path = Field(Path("workspace"), description="Root directory for runtime artifacts")
    capture_route: Literal["stick", "drone"] = Field(
        "stick", description="Active capture route (synced into colmap.route)"
    )
    device: str = Field("cuda", description="PyTorch device string ('cuda' or 'cpu')")
    seed: int = Field(42)
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    segment: SegmentConfig = Field(default_factory=SegmentConfig)
    colmap: ColmapConfig = Field(default_factory=ColmapConfig)
    static_train: StaticTrainConfig = Field(default_factory=StaticTrainConfig)
    dynamic_train: DynamicTrainConfig = Field(default_factory=DynamicTrainConfig)
    composite: CompositeConfig = Field(default_factory=CompositeConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)

    @model_validator(mode="after")
    def _sync_route(self) -> "PipelineConfig":
        """Keep colmap.route in sync with top-level capture_route."""
        self.colmap.route = self.capture_route
        return self


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (override wins on conflicts)."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(
    default_path: str | Path,
    profile: str | Path | None = None,
) -> PipelineConfig:
    """
    Load and merge YAML config files into a validated PipelineConfig.

    Merge order (later wins):
        1. default.yaml
        2. hardware profile YAML (if provided)

    Args:
        default_path: Path to config/default.yaml.
        profile:      Optional path to a hardware profile override YAML.

    Returns:
        Fully validated PipelineConfig with workspace_dir resolved to an absolute path.

    Raises:
        FileNotFoundError: If default_path does not exist.
        pydantic.ValidationError: If any field is missing or out of range.
    """
    default_path = Path(default_path)
    if not default_path.exists():
        raise FileNotFoundError(f"Config file not found: {default_path}")

    with default_path.open() as f:
        data: dict = yaml.safe_load(f) or {}

    if profile is not None:
        profile_path = Path(profile)
        if not profile_path.exists():
            raise FileNotFoundError(f"Hardware profile not found: {profile_path}")
        with profile_path.open() as f:
            override: dict = yaml.safe_load(f) or {}
        data = _deep_merge(data, override)

    cfg = PipelineConfig.model_validate(data)
    # Always resolve workspace_dir to an absolute path
    cfg.workspace_dir = cfg.workspace_dir.resolve()
    return cfg
