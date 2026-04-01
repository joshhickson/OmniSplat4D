"""
VRAM and system RAM guard utilities for the RTX 3060 12GB ceiling.

All modules that allocate GPU tensors in a loop must call flush_cuda_cache()
at the end of each iteration. This module is the single authoritative location
for that discipline — never call gc.collect() / torch.cuda.empty_cache()
directly in pipeline code; call flush_cuda_cache() instead.

This module has zero imports from other omnisplat4d sub-packages to prevent
circular imports.
"""

from __future__ import annotations

import gc
import logging
from contextlib import contextmanager
from typing import Generator

log = logging.getLogger(__name__)

VRAM_CEILING_BYTES: int = 12 * 1024**3  # 12 GB hard limit for RTX 3060


def get_vram_used_bytes() -> int:
    """
    Return current CUDA allocated memory in bytes.

    Returns 0 if no CUDA device is available or PyTorch is not installed.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.memory_allocated()
    except ImportError:
        pass
    return 0


def get_vram_free_bytes() -> int:
    """
    Return approximate free VRAM in bytes based on the ceiling constant.

    This is a conservative estimate: ceiling minus currently allocated.
    Does not account for fragmentation or non-PyTorch CUDA allocations.
    """
    return max(0, VRAM_CEILING_BYTES - get_vram_used_bytes())


def flush_cuda_cache() -> None:
    """
    Aggressively free GPU memory.

    Executes gc.collect() then torch.cuda.empty_cache() in sequence.

    MUST be called:
        - After every SAM 2.1 frame inference in segment/masker.py
        - After every temporal window in train/dynamic_trainer.py
        - After densification steps in train/static_trainer.py

    Without this, PyTorch's garbage collector does not release CUDA memory
    blocks back to the allocator, causing OOM on 12GB hardware within minutes.
    """
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            log.debug(
                "CUDA cache flushed. Allocated: %.1f MB",
                get_vram_used_bytes() / 1024**2,
            )
    except ImportError:
        pass


@contextmanager
def vram_guard(required_bytes: int, label: str = "") -> Generator[None, None, None]:
    """
    Context manager that checks VRAM headroom before a block and flushes on exit.

    Args:
        required_bytes: Estimated VRAM needed for the operation.
        label:          Human-readable label for log messages.

    Raises:
        MemoryError: If free VRAM < required_bytes before entering the block.

    Example:
        with vram_guard(required_bytes=2 * 1024**3, label="SAM2 inference"):
            mask = predictor.predict(...)
    """
    free = get_vram_free_bytes()
    tag = f"[{label}] " if label else ""
    if free < required_bytes:
        raise MemoryError(
            f"{tag}Insufficient VRAM: need {required_bytes / 1024**2:.0f} MB, "
            f"free ~{free / 1024**2:.0f} MB"
        )
    log.debug("%sEntering VRAM-guarded block (need %.0f MB)", tag, required_bytes / 1024**2)
    try:
        yield
    finally:
        flush_cuda_cache()
        log.debug("%sExited VRAM-guarded block. Allocated: %.1f MB", tag, get_vram_used_bytes() / 1024**2)
