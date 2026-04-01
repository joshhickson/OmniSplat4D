"""
Create the workspace/ directory structure for OmniSplat4D.

Run this script once after cloning the repo and before starting any pipeline run.
It creates all required subdirectories under workspace/ so the pipeline can write
artifacts without checking for directory existence at every step.

Usage:
    python scripts/init_workspace.py
    python scripts/init_workspace.py --workspace /path/to/custom/workspace
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

NUM_CAMERAS = 8  # Must match IngestConfig.num_cameras default


def create_workspace(workspace_dir: Path) -> None:
    """Create all required workspace subdirectories."""
    dirs = [
        # Phase 1 — Ingest
        workspace_dir / "raw_video",
        # Phase 1 — Extracted planar frames (one per virtual camera)
        *[workspace_dir / "extracted_frames" / f"cam_{i:02d}" for i in range(NUM_CAMERAS)],
        # Phase 1/2 — Semantic masks
        workspace_dir / "semantic_masks" / "static_background_masks",
        workspace_dir / "semantic_masks" / "dynamic_subject_masks",
        # Phase 1 — COLMAP workspace
        workspace_dir / "colmap_data" / "sparse" / "0",
        workspace_dir / "colmap_data" / "dense",
        # Phase 1 — Static 3DGS training output
        workspace_dir / "splat_training_graphs" / "static_environment",
        # Phase 2 — Dynamic 4DGS training output (chunks created per-run)
        workspace_dir / "splat_training_graphs" / "dynamic_subjects" / "chunks",
        # Phase 4 — Export targets
        workspace_dir / "export_streaming_assets" / "webgl_3dtiles",
        workspace_dir / "export_streaming_assets" / "unity_package",
    ]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        log.info("  %s", d)

    log.info("\nWorkspace initialised at: %s", workspace_dir.resolve())
    log.info("Add a 360 video to workspace/raw_video/ and run:")
    log.info("  python run_pipeline.py --config config/default.yaml")


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialise OmniSplat4D workspace directory.")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("workspace"),
        help="Root workspace directory (default: workspace/ relative to CWD)",
    )
    args = parser.parse_args()
    create_workspace(args.workspace.resolve())


if __name__ == "__main__":
    main()
