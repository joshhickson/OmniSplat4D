"""
OmniSplat4D — Top-level pipeline DAG runner.

Executes the full pipeline in sequence. Each phase is a discrete step;
intermediate artifacts are written to workspace/ and consumed by the next.

Usage:
    python run_pipeline.py --config config/default.yaml
    python run_pipeline.py --config config/default.yaml \\
        --profile config/hardware_profiles/rtx3060_12gb.yaml
    python run_pipeline.py --config config/default.yaml --skip-phase1
    python run_pipeline.py --config config/default.yaml --dry-run

Phases:
    1. Ingest        Frame extraction + equirect→planar reprojection
    2. Segment       YOLOv8 + SAM 2.1 masking (Stick Route only)
    3. SfM           Programmatic COLMAP initialisation + feature matching + mapping
    4. Static Train  gsplat 3DGS background training
    5. Invert        Background annihilation for dynamic subject isolation
    6. Dynamic Train SWinGS 4DGS sliding-window training
    7. Export        SPZ + ONNX + 3D Tiles / Unity packaging

Pass --dry-run to validate config and workspace without executing any phase.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("omnisplat4d.runner")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OmniSplat4D — 4D volumetric pipeline runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/default.yaml"),
        help="Path to the pipeline config YAML (default: config/default.yaml)",
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=None,
        help="Optional hardware profile override YAML (e.g. config/hardware_profiles/rtx3060_12gb.yaml)",
    )
    parser.add_argument("--skip-phase1", action="store_true", help="Skip ingest + segment + SfM")
    parser.add_argument("--skip-phase2", action="store_true", help="Skip static 3DGS training")
    parser.add_argument("--skip-phase3", action="store_true", help="Skip dynamic 4DGS training")
    parser.add_argument("--skip-phase4", action="store_true", help="Skip export packaging")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and workspace structure without executing any phase",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable DEBUG-level logging"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # --- Load and validate config ---
    from omnisplat4d.core.config import load_config

    log.info("Loading config: %s", args.config)
    cfg = load_config(args.config, profile=args.profile)
    log.info("Workspace: %s", cfg.workspace_dir)
    log.info("Capture route: %s | Device: %s", cfg.capture_route, cfg.device)

    if args.dry_run:
        log.info("=== DRY RUN: config loaded and validated successfully ===")
        log.info("PipelineConfig:\n%s", cfg.model_dump_json(indent=2))
        sys.exit(0)

    # Ensure workspace exists
    cfg.workspace_dir.mkdir(parents=True, exist_ok=True)

    # --- Phase 1: Ingest + Segment + SfM ---
    if not args.skip_phase1:
        _run_phase1(cfg)
    else:
        log.info("Skipping Phase 1 (--skip-phase1)")

    # --- Phase 2: Static 3DGS Training ---
    if not args.skip_phase2:
        _run_phase2(cfg)
    else:
        log.info("Skipping Phase 2 (--skip-phase2)")

    # --- Phase 3: Dynamic 4DGS Training ---
    if not args.skip_phase3:
        _run_phase3(cfg)
    else:
        log.info("Skipping Phase 3 (--skip-phase3)")

    # --- Phase 4: Export ---
    if not args.skip_phase4:
        _run_phase4(cfg)
    else:
        log.info("Skipping Phase 4 (--skip-phase4)")

    log.info("Pipeline complete.")


def _run_phase1(cfg: object) -> None:
    """Phase 1: Frame extraction → projection → masking → COLMAP init + matching."""
    from omnisplat4d.core.config import PipelineConfig
    assert isinstance(cfg, PipelineConfig)

    log.info("=== Phase 1: Ingest ===")
    from omnisplat4d.ingest.extractor import extract_frames
    from omnisplat4d.ingest.projector import (
        build_rotation_matrices,
        build_virtual_cameras,
        project_all_frames,
    )

    video_dir = cfg.workspace_dir / "raw_video"
    videos = sorted(video_dir.glob("*.mp4"))
    if not videos:
        log.error("No .mp4 files found in %s", video_dir)
        return

    frames_dir = cfg.workspace_dir / "extracted_frames"
    # Extract equirectangular keyframes
    equirect_dir = frames_dir / "equirect"
    keyframes = extract_frames(videos[0], equirect_dir, cfg.ingest)
    log.info("Extracted %d keyframes", len(keyframes))

    # Project to virtual planar cameras
    cameras = build_virtual_cameras(cfg.ingest)
    rotation_matrices = build_rotation_matrices(cfg.ingest)
    frame_paths_per_camera = project_all_frames(
        equirect_dir, frames_dir, cameras, rotation_matrices
    )
    log.info("Projected frames to %d virtual cameras", len(cameras))

    # Stick Route: mask the operator
    if cfg.capture_route == "stick":
        log.info("=== Phase 1: Segment (Stick Route) ===")
        _run_masking(cfg, frame_paths_per_camera)

    # COLMAP initialisation
    log.info("=== Phase 1: SfM ===")
    from omnisplat4d.sfm.initializer import initialize_colmap_workspace
    from omnisplat4d.sfm.runner import run_full_reconstruction

    colmap_dir = cfg.workspace_dir / "colmap_data"
    initialize_colmap_workspace(
        colmap_dir, cameras, rotation_matrices, frame_paths_per_camera, cfg.colmap
    )
    run_full_reconstruction(colmap_dir, cfg.colmap)
    log.info("Phase 1 complete.")


def _run_masking(cfg: object, frame_paths_per_camera: list) -> None:
    """Run YOLOv8 + SAM 2.1 masking cascade on all planar frames."""
    from omnisplat4d.core.config import PipelineConfig
    from omnisplat4d.segment.detector import load_detector
    from omnisplat4d.segment.masker import load_sam

    assert isinstance(cfg, PipelineConfig)
    detector = load_detector(cfg.segment)
    predictor = load_sam(cfg.segment)
    # TODO: iterate frames, call detect_operator + mask_frame + apply_operator_mask
    log.info("Masking: detector and predictor loaded (implementation pending)")


def _run_phase2(cfg: object) -> None:
    """Phase 2: Static 3DGS training."""
    from omnisplat4d.core.config import PipelineConfig
    assert isinstance(cfg, PipelineConfig)
    log.info("=== Phase 2: Static 3DGS Training ===")
    from omnisplat4d.train.static_trainer import train_static

    colmap_dir = cfg.workspace_dir / "colmap_data" / "sparse" / "0"
    mask_dir = (
        cfg.workspace_dir / "semantic_masks" / "static_background_masks"
        if cfg.capture_route == "stick"
        else None
    )
    output_dir = cfg.workspace_dir / "splat_training_graphs" / "static_environment"
    train_static(colmap_dir, mask_dir, cfg.static_train, output_dir)
    log.info("Phase 2 complete.")


def _run_phase3(cfg: object) -> None:
    """Phase 3: Dynamic 4DGS sliding-window training."""
    from omnisplat4d.core.config import PipelineConfig
    assert isinstance(cfg, PipelineConfig)
    log.info("=== Phase 3: Dynamic 4DGS Training ===")
    from omnisplat4d.train.dynamic_trainer import build_frame_batches, train_dynamic

    subject_frames_dir = cfg.workspace_dir / "extracted_frames"
    output_dir = cfg.workspace_dir / "splat_training_graphs" / "dynamic_subjects" / "chunks"
    frame_batches = build_frame_batches(subject_frames_dir, cfg.dynamic_train, cfg.ingest.num_cameras)

    # Load static background checkpoint as canonical initialisation seed
    from omnisplat4d.export.spz_writer import read_spz
    static_spz = cfg.workspace_dir / "splat_training_graphs" / "static_environment" / "background_model.spz"
    canonical_init = read_spz(static_spz)
    train_dynamic(frame_batches, canonical_init, cfg.dynamic_train, output_dir)
    log.info("Phase 3 complete.")


def _run_phase4(cfg: object) -> None:
    """Phase 4: Export streaming assets."""
    from omnisplat4d.core.config import PipelineConfig
    assert isinstance(cfg, PipelineConfig)
    log.info("=== Phase 4: Export ===")
    # TODO: call tiles_packager and onnx_exporter based on cfg.export.formats
    log.info("Export phase: implementation pending")


if __name__ == "__main__":
    main()
