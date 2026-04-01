"""
Core utilities shared across all pipeline phases.

Modules:
    config  — Pydantic config loader (load_config, PipelineConfig)
    memory  — VRAM guard utilities (flush_cuda_cache, vram_guard)
    types   — Shared dataclasses (CameraIntrinsics, GaussianCheckpoint, …)
"""
