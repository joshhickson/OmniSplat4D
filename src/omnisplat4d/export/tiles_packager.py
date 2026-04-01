"""
Phase 4 — Export: OGC 3D Tiles (WebGL) and Unity asset packaging.

Converts the trained .spz files and .onnx models into deployment-ready
asset bundles for the two primary runtime targets:

    1. WebGL / OGC 3D Tiles:
       - Static background converted to OGC 3D Tiles format for streaming
         via Niantic's SPZ web viewer.
       - Per-chunk .spz + .onnx pairs packaged with a JSON streaming manifest.

    2. Unity package:
       - .spz files and .onnx models bundled into a Unity AssetBundle.
       - The ONNX runtime (via Barracuda) evaluates deformation and color
         MLPs on GPU for real-time rendering in Unity scenes.

Both packagers are stubs — the actual conversion logic requires integrating
with the Niantic SPZ SDK and Unity Editor scripting respectively.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from omnisplat4d.core.config import ExportConfig

log = logging.getLogger(__name__)


def package_webgl_tiles(
    spz_path: Path,
    output_dir: Path,
    cfg: ExportConfig,
) -> None:
    """
    Package a static .spz file as an OGC 3D Tiles streaming asset.

    Creates a tileset.json manifest and splits the .spz into stream-sized
    chunks (cfg.stream_chunk_size_mb per chunk) for progressive loading.

    Args:
        spz_path:   Path to the static background .spz file.
        output_dir: Output directory for the 3D Tiles package.
        cfg:        ExportConfig (stream_chunk_size_mb).

    Side effects:
        Creates output_dir if it does not exist.
        Writes tileset.json and chunked .spz tile files.
    """
    raise NotImplementedError(
        "package_webgl_tiles: integrate with Niantic SPZ SDK for OGC 3D Tiles conversion."
    )


def package_unity(
    spz_paths: list[Path],
    onnx_paths: list[Path],
    output_dir: Path,
    cfg: ExportConfig,
) -> None:
    """
    Bundle .spz files and .onnx models into a Unity streaming asset package.

    Generates a package manifest JSON and copies assets into a structured
    directory ready for Unity AssetBundle import. ONNX inference in Unity
    uses the Barracuda runtime.

    Args:
        spz_paths:   List of .spz file paths (static + dynamic chunk canonicals).
        onnx_paths:  List of .onnx file paths (deformation + color predictors).
        output_dir:  Output directory for the Unity package.
        cfg:         ExportConfig.

    Side effects:
        Creates output_dir if it does not exist.
        Writes a package_manifest.json and copies all asset files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": 1,
        "spz_files": [str(p) for p in spz_paths],
        "onnx_files": [str(p) for p in onnx_paths],
    }
    manifest_path = output_dir / "package_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info("Wrote Unity package manifest to %s", manifest_path)
    raise NotImplementedError(
        "package_unity: implement AssetBundle creation via Unity Editor scripting or CLI."
    )
