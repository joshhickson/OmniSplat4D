"""
Phase 1 — SfM: COLMAP subprocess wrapper.

Invokes COLMAP for feature matching and sparse reconstruction via subprocess.
Never calls `colmap feature_extractor` — all camera/pose data is injected
programmatically by sfm/initializer.py before this runner is called.

Matcher selection:
    - Stick Route: sequential_matcher  (O(n) — exploits temporal adjacency)
    - Drone Route: vocab_tree_matcher  (max 20-30 NN; capped to fit 32GB DDR3 RAM)

The mapper step reads the sparse/0/ directory written by initializer.py and
produces a full sparse reconstruction in sparse/1/.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from omnisplat4d.core.config import ColmapConfig

log = logging.getLogger(__name__)


def _run_colmap(args: list[str], cfg: ColmapConfig) -> None:
    """
    Execute a COLMAP command as a subprocess.

    Args:
        args: Command arguments after the colmap binary name.
        cfg:  ColmapConfig (provides colmap_binary path).

    Raises:
        subprocess.CalledProcessError: If COLMAP exits with non-zero status.
        FileNotFoundError: If the COLMAP binary is not found on PATH.
    """
    cmd = [cfg.colmap_binary] + args
    log.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_feature_matching(workspace_dir: Path, cfg: ColmapConfig) -> None:
    """
    Run COLMAP feature matching against the pre-seeded workspace.

    Selects the matcher based on cfg.matcher:
        - "sequential": Uses sequential_matcher with cfg.sequential_overlap.
        - "vocab_tree": Uses vocab_tree_matcher with cfg.vocab_tree_neighbors
                        and requires cfg.vocab_tree_path to be set.

    IMPORTANT: This function NEVER calls `colmap feature_extractor`.
    Feature extraction is bypassed entirely; the workspace was seeded by
    sfm/initializer.py. If you see feature_extractor called anywhere in the
    pipeline, that is a bug.

    Args:
        workspace_dir: Root COLMAP workspace (contains sparse/0/).
        cfg:           ColmapConfig block.

    Raises:
        ValueError:  If cfg.matcher == "vocab_tree" but cfg.vocab_tree_path is None.
        AssertionError: If cfg.skip_feature_extraction is False.
    """
    assert cfg.skip_feature_extraction, (
        "run_feature_matching must only be called when skip_feature_extraction=True."
    )
    database_path = workspace_dir / "database.db"

    if cfg.matcher == "sequential":
        _run_colmap([
            "sequential_matcher",
            "--database_path", str(database_path),
            "--SequentialMatching.overlap", str(cfg.sequential_overlap),
        ], cfg)

    elif cfg.matcher == "vocab_tree":
        if cfg.vocab_tree_path is None:
            raise ValueError(
                "ColmapConfig.vocab_tree_path must be set when using vocab_tree matcher. "
                "Download a vocabulary tree from https://demuc.de/colmap/"
            )
        _run_colmap([
            "vocab_tree_matcher",
            "--database_path", str(database_path),
            "--VocabTreeMatching.vocab_tree_path", str(cfg.vocab_tree_path),
            "--VocabTreeMatching.num_nearest_neighbors", str(cfg.vocab_tree_neighbors),
        ], cfg)

    else:
        raise ValueError(f"Unknown matcher: {cfg.matcher!r}")


def run_mapper(workspace_dir: Path, cfg: ColmapConfig) -> Path:
    """
    Run COLMAP mapper to produce a sparse 3D reconstruction.

    Reads the pre-seeded sparse/0/ directory and the feature matches from
    the COLMAP database, producing a sparse reconstruction in sparse/1/.

    Args:
        workspace_dir: Root COLMAP workspace (contains database.db + sparse/0/).
        cfg:           ColmapConfig block.

    Returns:
        Path to the output sparse reconstruction directory (sparse/1/).
    """
    output_dir = workspace_dir / "sparse"
    output_dir.mkdir(parents=True, exist_ok=True)

    _run_colmap([
        "mapper",
        "--database_path", str(workspace_dir / "database.db"),
        "--image_path", str(workspace_dir / "images"),
        "--output_path", str(output_dir),
        "--input_path", str(output_dir / "0"),
    ], cfg)

    reconstruction_dir = output_dir / "1"
    if not reconstruction_dir.exists():
        # COLMAP names the first reconstruction 0 when starting from a seed
        reconstruction_dir = output_dir / "0"
    log.info("Sparse reconstruction written to %s", reconstruction_dir)
    return reconstruction_dir


def run_full_reconstruction(workspace_dir: Path, cfg: ColmapConfig) -> Path:
    """
    Orchestrate feature matching followed by sparse mapping.

    This is the primary entry point for the SfM phase. The workspace must
    already be initialised by sfm/initializer.initialize_colmap_workspace().

    Args:
        workspace_dir: Root COLMAP workspace.
        cfg:           ColmapConfig block.

    Returns:
        Path to the sparse reconstruction directory.
    """
    run_feature_matching(workspace_dir, cfg)
    return run_mapper(workspace_dir, cfg)
