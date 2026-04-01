"""
Phase 4 — Export: .spz binary serialisation.

Writes GaussianCheckpoint data to the .spz format used by Niantic's SPZ
viewer and compatible WebGL/Unity streaming pipelines. All floating-point
data is downcast to FP16 before writing to minimise file size and streaming
bandwidth.

.spz field layout per Gaussian (FP16, packed):
    position:  (3,) → 6 bytes
    rotation:  (4,) quaternion → 8 bytes
    scale:     (3,) log-scale → 6 bytes
    opacity:   (1,) → 2 bytes
    dc_color:  (3,) → 6 bytes
    Total per Gaussian: 28 bytes

Note: The .spz format spec is maintained by Niantic at
      https://github.com/nianticlabs/spz. This writer implements a simplified
      compatible subset; full spec compliance should be verified against the
      reference reader before shipping streaming assets.
"""

from __future__ import annotations

import logging
import struct
from pathlib import Path

import numpy as np

from omnisplat4d.core.types import GaussianCheckpoint

log = logging.getLogger(__name__)

SPZ_MAGIC = b"SPZ1"  # 4-byte magic header
SPZ_VERSION = 1


def write_spz(checkpoint: GaussianCheckpoint, output_path: Path) -> None:
    """
    Serialise a GaussianCheckpoint to a .spz binary file.

    Downcasts all float32 arrays to float16 before writing. The output
    is a compact binary file suitable for streaming and WebGL/Unity loading.

    File layout:
        [0:4]   magic bytes "SPZ1"
        [4:8]   version (uint32 LE)
        [8:12]  num_gaussians (uint32 LE)
        [12:]   packed Gaussian data (positions, rotations, scales,
                opacities, dc_colors — all FP16, interleaved per Gaussian)

    Args:
        checkpoint:   GaussianCheckpoint to serialise.
        output_path:  Destination .spz file path.

    Side effects:
        Creates parent directories if they do not exist.
        Overwrites output_path if it exists.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(checkpoint.positions)

    pos = checkpoint.positions.astype(np.float16)
    rot = checkpoint.rotations.astype(np.float16)
    sca = checkpoint.scales.astype(np.float16)
    opa = checkpoint.opacities.astype(np.float16)
    col = checkpoint.dc_colors.astype(np.float16)

    with output_path.open("wb") as f:
        f.write(SPZ_MAGIC)
        f.write(struct.pack("<I", SPZ_VERSION))
        f.write(struct.pack("<I", n))
        for i in range(n):
            f.write(pos[i].tobytes())   # 6 bytes
            f.write(rot[i].tobytes())   # 8 bytes
            f.write(sca[i].tobytes())   # 6 bytes
            f.write(opa[i].tobytes())   # 2 bytes
            f.write(col[i].tobytes())   # 6 bytes

    size_mb = output_path.stat().st_size / 1024**2
    log.info("Wrote %d Gaussians to %s (%.1f MB)", n, output_path, size_mb)


def read_spz(input_path: Path) -> GaussianCheckpoint:
    """
    Load a GaussianCheckpoint from a .spz binary file.

    Reads the file written by write_spz() and returns arrays in float16
    (matching the on-disk representation). Callers that need float32 for
    GPU training should upcast after loading.

    Args:
        input_path: Path to the .spz file.

    Returns:
        GaussianCheckpoint with float16 numpy arrays.

    Raises:
        ValueError: If the magic bytes or version do not match.
        FileNotFoundError: If input_path does not exist.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"SPZ file not found: {input_path}")

    with input_path.open("rb") as f:
        magic = f.read(4)
        if magic != SPZ_MAGIC:
            raise ValueError(f"Invalid SPZ magic: {magic!r} (expected {SPZ_MAGIC!r})")
        version = struct.unpack("<I", f.read(4))[0]
        if version != SPZ_VERSION:
            raise ValueError(f"Unsupported SPZ version: {version}")
        n = struct.unpack("<I", f.read(4))[0]

        positions = np.zeros((n, 3), dtype=np.float16)
        rotations = np.zeros((n, 4), dtype=np.float16)
        scales = np.zeros((n, 3), dtype=np.float16)
        opacities = np.zeros((n, 1), dtype=np.float16)
        dc_colors = np.zeros((n, 3), dtype=np.float16)

        for i in range(n):
            positions[i] = np.frombuffer(f.read(6), dtype=np.float16)
            rotations[i] = np.frombuffer(f.read(8), dtype=np.float16)
            scales[i] = np.frombuffer(f.read(6), dtype=np.float16)
            opacities[i] = np.frombuffer(f.read(2), dtype=np.float16)
            dc_colors[i] = np.frombuffer(f.read(6), dtype=np.float16)

    log.info("Loaded %d Gaussians from %s", n, input_path)
    return GaussianCheckpoint(
        positions=positions,
        rotations=rotations,
        scales=scales,
        opacities=opacities,
        dc_colors=dc_colors,
        source_path=input_path,
    )
