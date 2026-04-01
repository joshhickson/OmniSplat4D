# Session: Initial repo architecture seeding — 2026-04-01

## What was done

Built the full Python package scaffold for OmniSplat4D from scratch, based on the two research documents in `docs/research/`.

Implemented 39 files across 8 sub-packages:

- **`core/`**: Pydantic v2 `PipelineConfig` with all phase sub-configs, `flush_cuda_cache()` / `vram_guard()` VRAM utilities, shared dataclasses (`CameraIntrinsics`, `CameraPose`, `FrameBatch`, `GaussianCheckpoint`)
- **`ingest/`**: FFmpeg subprocess frame extractor with Laplacian sharpness filter, equirectangular→planar projector with pre-computed remap grids and `BORDER_WRAP`
- **`segment/`**: YOLOv8-nano detector, SAM 2.1 Tiny masker (CPU offloading enforced), multi-view spherical bbox handoff tracker, background annihilation inverter
- **`sfm/`**: Programmatic COLMAP workspace initialiser (writes `cameras.txt` / `images.txt` / `points3D.txt` directly), COLMAP subprocess runner (sequential + vocab_tree matchers only — never `feature_extractor`)
- **`train/`**: `DeformationMLP` and `ACColorMLP` (both ONNX-traceable fixed-size MLPs), entropy-constrained opacity pruner. `static_trainer.py` and `dynamic_trainer.py` are stubs.
- **`composite/`**: `renderer.py` stub with `concat_gaussians()` implemented; `radix_sort_by_depth()` and `render_frame()` are stubs.
- **`export/`**: SPZ binary reader/writer (FP16), ONNX MLP exporter. `tiles_packager.py` is a stub.
- **Config**: `default.yaml`, `rtx3060_12gb.yaml`, `a100_80gb.yaml` hardware profiles
- **Tests**: 30+ tests for projector math, COLMAP format validation, pruner logic — all run without GPU
- **Infrastructure**: `pyproject.toml` (hatchling, dep groups), `.gitignore`, `run_pipeline.py` DAG runner, `scripts/init_workspace.py`

## Key decisions made (and why)

- **`src/` layout** over flat: subprocess-spawned training scripts need `omnisplat4d` importable from any working directory; `pip install -e .` guarantees this
- **`workspace/` outside `src/`**: runtime artifacts are gigabytes of binary data; must be gitignored and separate from the package tree
- **`flush_cuda_cache()` wrapper in `core/memory.py`**: centralises the `gc.collect()` + `torch.cuda.empty_cache()` pattern so it's never skipped accidentally
- **`sam_storage_device="cpu"` as the Pydantic default**: wrong default here would silently OOM every run on the 12GB target
- **ONNX-traceable constraint on MLPs**: documented explicitly in module docstrings because it's easy to accidentally break with a conditional in `forward()`

## Files changed

All files created new. See `git log` for the full list.

## What's left / known issues

**Stubs to implement (in sprint order):**
1. `train/static_trainer.py` — gsplat 3DGS training loop
2. `train/dynamic_trainer.py` — SWinGS sliding-window 4DGS loop
3. `composite/renderer.py` — unified CUDA depth-sort + alpha blend
4. `export/tiles_packager.py` — OGC 3D Tiles + Unity asset bundle

**Known gaps:**
- The SPZ binary format implemented in `spz_writer.py` is a simplified subset. Full compliance with the Niantic SPZ spec should be verified against their reference reader before shipping streaming assets.
- `run_pipeline.py` Phase 4 (`_run_phase4`) is a placeholder — needs wiring once the export stubs are implemented.
- No end-to-end integration test exists yet. Add one once `static_trainer.py` is implemented.
