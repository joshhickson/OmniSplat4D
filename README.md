# OmniSplat4D

End-to-end pipeline for generating free-viewpoint 4D volumetric environments from 360° equirectangular video.

The pipeline decomposes raw 360° footage into a static 3D background (3DGS) and dynamic 4D keyframable subjects (4DGS), then composites them in real-time via unified CUDA depth-sorting. Output is a set of streamable `.spz` geometry files and `.onnx` neural network binaries that render in WebGL or Unity without any cloud dependency at playback time.

---

## A Note to the CorridorKey Community

If you found this from the CorridorKey Discord or GitHub — this section is written for you.

[CorridorKey](https://github.com/nikopueringer/CorridorKey.git) is doing something genuinely important: building a production-quality, open-source green screen solution that anyone can run without a render farm. What happened in the week after its release — the community driving VRAM down from 23GB to 8GB in a single day, the DaVinci Resolve plugin appearing on day five, the Nuke integration, the volunteer GPU cloud — is exactly what open source is supposed to look like. We've been watching it closely.

OmniSplat4D is being built from the opposite direction. Where CorridorKey is a 2D color separation tool that produces extraordinary alpha mattes, OmniSplat4D is a 3D geometric reconstruction pipeline that produces explicit scene geometry — a static background as a 3D Gaussian Splat, and a dynamic subject as a 4D Gaussian Splat with a continuous deformation field. Both projects run on consumer GPUs. Both are fully open source. And we believe they are directly complementary in ways that address some of the specific gaps the CorridorKey community has already identified.

**Three concrete intersections we are building toward:**

**1. Geometry-derived hint masks — no BiRefNet required, no green screen required.**
CorridorKey's inference model accepts a coarse alpha hint and refines it into a sub-pixel accurate linear matte. Right now that hint comes from BiRefNet, GVM, or a manual paint — all of which are learned priors guessing at subject boundaries from color and texture alone. OmniSplat4D's Phase 1 trains an explicit 3D Gaussian reconstruction of the static background. Once that reconstruction exists, we can render a synthetic background at any camera pose and subtract it from the original footage — pixels that deviate from the render are the dynamic subject. That difference map is a hint derived from actual scene geometry, not a learned guess. It requires no green screen, no special lighting, and no additional model. It feeds directly into CorridorKey's existing hint input slot.

**2. Temporal alpha stabilization from 4D Gaussian opacity fields.**
The stress test in Niko's recent video identified temporal flicker at silhouette edges as a real concern. This is an inherent limitation of per-frame neural inference: each frame is processed independently, so alpha values at semi-transparent boundaries have no inter-frame constraint. OmniSplat4D's Phase 2 fits a 4D Gaussian representation to the dynamic subject with a temporal consistency loss enforced across sliding window boundaries. The Gaussian rasterizer produces per-pixel accumulated opacity as a geometric output — not inferred from color, but derived from the 3D structure of the primitives. Re-deriving alpha from the 4D Gaussian field instead of CorridorKey's per-frame estimate would produce a temporally smooth alpha channel by geometric construction. The two outputs are complementary: CorridorKey's continuous linear alpha at edges, stabilized over time by the 4DGS opacity field.

**3. Depth-correct compositing at the Gaussian level.**
The final compositing step in CorridorKey's pipeline is 2D layer stacking — foreground plate over background plate. OmniSplat4D's Phase 3 merges static background Gaussians and dynamic subject Gaussians into a single primitive set before rasterization, then radix-sorts them by physical depth. The result is geometrically correct occlusion, parallax, and free viewpoint rendering without rotoscope, without Z-buffer, and without a render farm. If the destination background is a 3DGS and the foreground subject is a 4DGS, the composite is computed from 3D geometry — not from pixel layers.

**The long-term vision, stated plainly:** a workflow where OmniSplat4D reconstructs the scene geometry and CorridorKey handles the color unmixing, and together they produce a composite that is depth-correct, temporally stable, and renderable from any viewpoint on a single consumer GPU — with no cloud dependency at playback time and no green screen required at capture time.

We have written a detailed research document covering the feasibility analysis, the specific structural intersections, and the open questions that still need to be answered: [`docs/research/long_term_vision_corridorkey_integration.md`](docs/research/long_term_vision_corridorkey_integration.md). That document is honest about what is implemented, what is stubbed, and what is still research. Read it if you want the full picture.

This project is being developed openly. If you are working on CorridorKey and see something here worth building on, we want to hear from you.

---

## Hardware Philosophy

| Target | Spec |
|---|---|
| Primary prototyping | Intel i7-4790k · 32GB DDR3 · Nvidia RTX 3060 12GB |
| Cloud scaling (Phase 4) | Lambda Labs A100 / H100 80GB |

Every architectural decision in this codebase is a direct consequence of the **12GB VRAM ceiling**. The five strategies that make this work:

1. **Programmatic COLMAP injection** — cameras.txt / images.txt seeded directly from pre-calculated K-matrices and scipy quaternions; `feature_extractor` is never called, eliminating SIFT memory overhead
2. **SH Degree 1 only** — Spherical Harmonics Degree 3 balloons the backward-pass payload by ~8–10×, guaranteeing OOM on 12GB for scenes with >1M Gaussians; Degree 1 gives 75% reduction with negligible outdoor PSNR impact
3. **Packed tensors** (`packed=True` in gsplat) — contiguous memory allocation avoids sparse gradient overhead during backpropagation
4. **SWinGS sliding-window chunking** — 4DGS training partitioned into 30-frame windows; backpropagation graph never spans the full video, keeping tensor allocations static
5. **MEGA DC+AC color decomposition** — replaces 144-float SH arrays per Gaussian with 3-float DC base + shared 3-layer AC MLP → 48× VRAM compression for dynamic scene color

---

## Pipeline Overview

The pipeline is a directed acyclic graph (DAG) with four phases. Each phase writes standardised artifacts to `workspace/` that the next phase consumes.

### Phase 1 — Static Background (3DGS)

**Goal:** Reconstruct the static environment as a compressed `.spz` Gaussian splat.

| Step | What happens |
|---|---|
| Ingest | FFmpeg decodes the equirectangular `.mp4`; Laplacian variance filter keeps only sharp keyframes (target: 1–3 fps from 30 fps source) |
| Stick Route masking | YOLOv8-nano detects the operator bounding box; SAM 2.1 Tiny generates pixel-perfect binary masks. SAM runs on **CPU** with `max_vision_features_cache_size=1` and `flush_cuda_cache()` after every frame — non-negotiable on 12GB |
| Projection | Equirectangular frames reprojected into **8 overlapping planar virtual cameras** via pre-computed `cv2.remap()` grids with `BORDER_WRAP` (no polar distortion, no seam artifacts) |
| COLMAP init | `cameras.txt` and `images.txt` written directly from analytical K-matrices and rotation-derived quaternions. Translation is always `[0,0,0]` — all 8 virtual cameras share one optical centre |
| Feature matching | Sequential Matcher (Stick Route, O(n)) or Vocabulary Tree Matcher (Drone Route, ≤30 NN to fit 32GB DDR3). `--SiftExtraction.max_image_size 1024` |
| 3DGS training | gsplat backend: `sh_degree=1`, `packed=True`, `densify_grad_threshold=0.0004`, 16×16 tile frustum culling |
| Output | `workspace/splat_training_graphs/static_environment/background_model.spz` (FP16) |

### Phase 2 — Dynamic Subject (4DGS)

**Goal:** For each moving subject, produce one temporal chunk directory per 30-frame window.

| Step | What happens |
|---|---|
| Inverted masking | SAM 2.1 masks inverted: background blacked out via Hadamard product, dynamic subject isolated against pure void |
| Multi-view tracking | When subject crosses a virtual camera boundary: inverse-project 2D bbox → unit sphere → re-project into adjacent frustum using known `R_i` matrices. Farneback optical flow (CPU) warps YOLOv8 bbox between frames. If IoU drops below threshold, `predictor.reset_tracking_data()` + fresh YOLOv8 detection |
| SWinGS training | 30-frame windows, 5-frame overlap. Overlap region applies temporal consistency loss against the frozen prior window to prevent jitter at boundaries. `flush_cuda_cache()` between every window |
| MEGA color | DC component: 3 floats per Gaussian (base RGB). AC component: shared 3-layer MLP predicts view- and time-dependent color variation. 48× smaller than SH Degree 3 arrays |
| Entropy pruning | Opacity entropy penalty added to loss coerces opacities toward binary states (0 or 1). Periodic hard pruning every 1000 iters removes Gaussians below opacity threshold |
| Output per chunk | `canonical_base.spz` · `deformation_field.onnx` · `ac_color_predictor.onnx` |

### Phase 3 — Compositing

**Goal:** Real-time depth-correct occlusion between static and dynamic primitives.

Naïve two-pass framebuffer compositing breaks occlusion when a dynamic subject moves behind static foreground geometry. The solution: merge everything **before** rasterisation.

1. Deformation MLP evaluates temporal offsets `Δ(x,y,z,t)` for frame `t`; applied to canonical dynamic positions
2. Static + deformed dynamic Gaussian arrays concatenated in GPU VRAM: `G_composite = [G_static | G_dynamic(t)]`
3. 64-bit Radix sort key per primitive: high 32 bits = Tile ID, low 32 bits = float depth. `cub::DeviceRadixSort` sorts the combined array in parallel
4. Front-to-back transmittance accumulation: dense static geometry saturates opacity rapidly; dynamic primitives physically behind static foreground are automatically occluded

Both primitive sets share the same world coordinate frame (seeded from the same COLMAP trajectory), so concatenation is geometrically valid with no transformation.

### Phase 4 — Export & Streaming

| Asset | Format | Loaded by |
|---|---|---|
| Static environment | `.spz` (FP16) or OGC 3D Tiles (LoD) | Once at init |
| Canonical chunk geometry | `.spz` (FP16) | Async, per temporal window |
| Temporal deformation | `deformation_field.onnx` | Edge inference, per frame |
| View/time color variation | `ac_color_predictor.onnx` | Edge inference, per frame |

**Runtime loop:** client passes normalised `t` → ONNX outputs `Δ(x,y,z,t)` → GPU shader applies offsets to canonical SPZ → Radix sort → render.

**WebGL:** Niantic SPZ viewer / three.js compute shader  
**Unity XR:** Barracuda inference library executes ONNX models natively

---

## Repository Structure

```
OmniSplat4D/
├── src/omnisplat4d/
│   ├── core/
│   │   ├── config.py          # Pydantic PipelineConfig — all phase sub-configs and defaults
│   │   ├── memory.py          # flush_cuda_cache(), vram_guard() — call these, not torch.cuda.empty_cache() directly
│   │   └── types.py           # Shared dataclasses: CameraIntrinsics, CameraPose, FrameBatch, GaussianCheckpoint
│   ├── ingest/
│   │   ├── extractor.py       # FFmpeg subprocess + Laplacian keyframe filter
│   │   └── projector.py       # Equirect→8 planar cameras (remap grids, project_all_frames)
│   ├── segment/
│   │   ├── detector.py        # YOLOv8-nano bounding box detection
│   │   ├── masker.py          # SAM 2.1 Tiny — CPU offloading enforced
│   │   ├── tracker.py         # Spherical bbox handoff + optical flow warping
│   │   └── inverter.py        # Background annihilation (Phase 2) + operator masking (Phase 1)
│   ├── sfm/
│   │   ├── initializer.py     # write_cameras_txt(), write_images_txt(), build_poses_from_rotations()
│   │   └── runner.py          # COLMAP subprocess — sequential_matcher / vocab_tree_matcher ONLY
│   ├── train/
│   │   ├── static_trainer.py  # [STUB] gsplat 3DGS training loop
│   │   ├── dynamic_trainer.py # [STUB] SWinGS sliding-window 4DGS training loop
│   │   ├── deformation.py     # DeformationMLP — ONNX-traceable, fixed architecture
│   │   ├── color_mlp.py       # ACColorMLP — ONNX-traceable, fixed architecture
│   │   └── pruner.py          # entropy_regularization_loss(), prune_low_opacity(), densification_mask()
│   ├── composite/
│   │   └── renderer.py        # [STUB] concat_gaussians(), radix_sort_by_depth(), render_frame()
│   └── export/
│       ├── spz_writer.py      # write_spz() / read_spz() — FP16 binary format
│       ├── onnx_exporter.py   # export_deformation_mlp(), export_color_mlp()
│       └── tiles_packager.py  # [STUB] OGC 3D Tiles + Unity asset bundle
├── config/
│   ├── default.yaml           # Full pipeline config matching PipelineConfig schema
│   └── hardware_profiles/
│       ├── rtx3060_12gb.yaml  # RTX 3060 overrides (enforces all VRAM constraints)
│       └── a100_80gb.yaml     # Cloud scaling overrides (unlocks SH3, larger windows)
├── workspace/                 # GITIGNORED — all runtime artifacts live here
│   ├── raw_video/
│   ├── extracted_frames/      # cam_00/ … cam_07/
│   ├── semantic_masks/        # static_background_masks/ + dynamic_subject_masks/
│   ├── colmap_data/           # sparse/0/, database.db
│   ├── splat_training_graphs/ # static_environment/ + dynamic_subjects/chunks/
│   └── export_streaming_assets/
├── scripts/
│   └── init_workspace.py      # Creates all workspace/ subdirs
├── tests/
│   ├── test_projector.py      # Equirect→planar math (no GPU needed)
│   ├── test_initializer.py    # cameras.txt / images.txt COLMAP format validation
│   └── test_pruner.py         # Entropy loss + opacity pruning correctness
├── docs/
│   ├── ARCHITECTURE.md        # Pipeline overview (keep current as stubs are implemented)
│   ├── ROADMAP.md             # Phase milestones
│   ├── research/
│   │   ├── phase1_3DGS_static.md    # Full research basis for Phase 1 decisions
│   │   └── phase2_4DGS_dynamic.md  # Full research basis for Phase 2/3/4 decisions
│   └── sessions/              # Session logs (see Documentation Conventions below)
├── run_pipeline.py            # Top-level DAG runner
└── pyproject.toml             # hatchling build; dep groups: [segment], [ingest], [dev]
```

`[STUB]` = `raise NotImplementedError` — see Implementation Status below.

---

## Implementation Status

### Done

| Module | What's implemented |
|---|---|
| `core/config.py` | Full Pydantic v2 `PipelineConfig` with all phase sub-configs; `load_config()` with YAML merging and absolute path resolution |
| `core/memory.py` | `flush_cuda_cache()`, `vram_guard()` context manager, `VRAM_CEILING_BYTES` |
| `core/types.py` | `CameraIntrinsics`, `CameraPose`, `FrameBatch`, `GaussianCheckpoint` |
| `ingest/extractor.py` | FFmpeg subprocess frame extraction, `laplacian_variance()`, `iter_ffmpeg_frames()` generator |
| `ingest/projector.py` | `build_virtual_cameras()`, `build_rotation_matrices()`, `build_remap_grids()`, `project_all_frames()` |
| `segment/detector.py` | `load_detector()`, `detect_operator()` |
| `segment/masker.py` | `load_sam()` (CPU offloading enforced), `mask_frame()` (flushes after every call), `reset_tracker()`, `compute_iou()` |
| `segment/tracker.py` | `project_bbox_to_sphere()`, `reproject_sphere_to_camera()`, `handoff_tracking()`, `warp_bbox_optical_flow()` |
| `segment/inverter.py` | `invert_mask()`, `apply_black_background()`, `apply_operator_mask()` |
| `sfm/initializer.py` | `write_cameras_txt()`, `write_images_txt()`, `write_points3d_txt()`, `build_poses_from_rotations()`, `initialize_colmap_workspace()` |
| `sfm/runner.py` | `run_feature_matching()` (sequential + vocab_tree), `run_mapper()`, `run_full_reconstruction()` |
| `train/deformation.py` | `DeformationMLP` — 3-layer fixed MLP, ONNX-traceable |
| `train/color_mlp.py` | `ACColorMLP` — 3-layer fixed MLP, ONNX-traceable |
| `train/pruner.py` | `entropy_regularization_loss()`, `prune_low_opacity()`, `densification_mask()` |
| `export/spz_writer.py` | `write_spz()` + `read_spz()` — FP16 binary round-trip |
| `export/onnx_exporter.py` | `export_deformation_mlp()`, `export_color_mlp()` |
| `run_pipeline.py` | Phase orchestration, `--dry-run`, `--skip-phase1/2/3/4`, `--profile` |
| `scripts/init_workspace.py` | All `workspace/` subdirs created programmatically |
| `tests/` | 30+ tests across projector math, COLMAP format, pruner — all run without GPU |

### Stubs — Next to Implement

| Module | Function | Sprint | Research reference |
|---|---|---|---|
| `train/static_trainer.py` | `train_static()` | Sprint 3 | `docs/research/phase1_3DGS_static.md` §4.2 |
| `train/dynamic_trainer.py` | `train_dynamic()`, `_train_single_window()`, `build_frame_batches()` | Sprint 4 | `docs/research/phase2_4DGS_dynamic.md` §VRAM-Agnostic 4DGS Training |
| `composite/renderer.py` | `radix_sort_by_depth()`, `render_frame()` | Sprint 5 | `docs/research/phase2_4DGS_dynamic.md` §Compositing Graph |
| `export/tiles_packager.py` | `package_webgl_tiles()`, `package_unity()` | Sprint 5 | `docs/research/phase2_4DGS_dynamic.md` §Streaming Integration |

---

## Setup & Installation

**Prerequisites (install before this package):**
- CUDA-capable GPU with appropriate driver
- PyTorch ≥ 2.2 with CUDA support — install from [pytorch.org](https://pytorch.org)
- gsplat ≥ 1.0 — `pip install gsplat`
- COLMAP — install system binary or build from source; ensure `colmap` is on `PATH`

```bash
git clone https://github.com/joshhickson/OmniSplat4D.git
cd OmniSplat4D

# Install package + all optional groups
pip install -e ".[segment,ingest,dev]"

# Create workspace directory structure
python scripts/init_workspace.py

# Verify config loads (no GPU required)
python run_pipeline.py --config config/default.yaml --dry-run
```

`[segment]` installs `ultralytics` (YOLOv8) and `sam-2`.  
`[ingest]` installs `ffmpeg-python`.  
`[dev]` installs `pytest`, `ruff`, `mypy`.

---

## Usage

```bash
# Drop your equirectangular .mp4 into workspace/raw_video/, then:

# Full pipeline — RTX 3060 profile
python run_pipeline.py \
    --config config/default.yaml \
    --profile config/hardware_profiles/rtx3060_12gb.yaml

# Full pipeline — cloud A100 profile
python run_pipeline.py \
    --config config/default.yaml \
    --profile config/hardware_profiles/a100_80gb.yaml

# Resume from Phase 2 (Phase 1 artifacts already in workspace/)
python run_pipeline.py --config config/default.yaml --skip-phase1

# Dry run — validate config without executing anything
python run_pipeline.py --config config/default.yaml --dry-run

# Run tests (no GPU required)
pytest tests/ -v
```

---

## Agent Onboarding Guide

> This section is written for coding agents arriving at this repository with no prior conversation history.

### The single most important invariant

`sfm/runner.py` **must never call `colmap feature_extractor`**. This is intentional, not a bug. Camera intrinsics and poses are injected programmatically by `sfm/initializer.py` using pre-calculated K-matrices and scipy-derived quaternions. Calling `feature_extractor` would override this with COLMAP's own (inferior, drift-prone) estimates and invalidate the entire downstream coordinate frame.

### Five non-negotiable VRAM rules (RTX 3060 profile)

Violating any of these will OOM the GPU, typically within the first few minutes of a run.

1. **`sh_degree=1` in `StaticTrainConfig`** — Degree 3 uses 4× more color parameters. The backward pass amplifies this 8–10×. Never increase this for the `rtx3060_12gb.yaml` profile.

2. **`sam_storage_device="cpu"` in `SegmentConfig`** — SAM 2.1's temporal state must live in host RAM. Setting this to `"cuda"` will exhaust VRAM during the masking loop before training even starts. The `a100_80gb.yaml` profile may set this to `"cuda"`.

3. **Call `flush_cuda_cache()` from `core/memory.py` after every SAM frame and every 4DGS temporal window** — PyTorch's garbage collector does not automatically release CUDA memory blocks. Skipping this causes silent accumulation that crashes within 50–100 frames. Never call `torch.cuda.empty_cache()` directly — always use the wrapper in `core/memory.py`.

4. **`DeformationMLP` and `ACColorMLP` `forward()` must remain ONNX-traceable** — no Python `if/else` on tensor shapes, no variable-length loops, no dynamic tensor dimensions. These models are exported to `.onnx` for client-side inference; any non-traceable operation will break `export/onnx_exporter.py`.

5. **`workspace_dir` is always an absolute path** — `load_config()` resolves it to absolute before returning. Every module writes artifacts via `cfg.workspace_dir / ...`. Never construct file paths relative to CWD in pipeline code.

### Where to find things

| Question | Where to look |
|---|---|
| What config fields exist and what are their defaults? | `src/omnisplat4d/core/config.py` |
| What types are passed between pipeline stages? | `src/omnisplat4d/core/types.py` |
| How does VRAM management work? | `src/omnisplat4d/core/memory.py` |
| In what order do phases execute and how are they wired? | `run_pipeline.py` |
| Why was SH Degree 1 chosen? Why gsplat over Splatfacto? Why 8 cameras? | `docs/research/phase1_3DGS_static.md` |
| Why sliding windows? How does the temporal consistency loss work? Why DC+AC? | `docs/research/phase2_4DGS_dynamic.md` |

### What to implement next (Sprint order)

1. **Sprint 3 — `train/static_trainer.py`**: Implement `train_static()`. Use gsplat's rasterisation API. Apply `sh_degree=1`, `packed=True`, `densify_grad_threshold=0.0004`. Call `flush_cuda_cache()` after each densification step. Log VRAM usage at DEBUG level. See `docs/research/phase1_3DGS_static.md` §4.2 for every hyperparameter's rationale and memory calculation.

2. **Sprint 4 — `train/dynamic_trainer.py`**: Implement `train_dynamic()`, `_train_single_window()`, and `build_frame_batches()`. Outer loop over 30-frame windows with 5-frame overlap. Freeze the prior window's model as a rendering reference for the overlap region. Loss = L1 + SSIM + entropy + temporal consistency. Serialize each window to `canonical_base.spz` + MLP weights. `flush_cuda_cache()` between every window. See `docs/research/phase2_4DGS_dynamic.md` §VRAM-Agnostic 4DGS Training.

3. **Sprint 5 — `composite/renderer.py`**: Implement `radix_sort_by_depth()` using gsplat's internal sort utilities (64-bit key: Tile ID | depth). Implement `render_frame()` with `concat_gaussians()` → unified rasterisation. See `docs/research/phase2_4DGS_dynamic.md` §Compositing Graph.

4. **Sprint 5 — `export/tiles_packager.py`**: Integrate Niantic SPZ SDK for OGC 3D Tiles and Unity Barracuda asset bundle packaging. See `docs/research/phase2_4DGS_dynamic.md` §Streaming Integration.

---

## Documentation & Session Log Conventions

### Doc file roles

| File | Role | Mutability |
|---|---|---|
| `README.md` | Project overview + agent onboarding + implementation status | Update when stubs are implemented or new phases added |
| `docs/ARCHITECTURE.md` | Technical pipeline details | Update when core design decisions change |
| `docs/ROADMAP.md` | Phase milestones and hardware targets | Update when a phase completes or scope changes |
| `docs/research/phase1_3DGS_static.md` | Full research basis for Phase 1 | **Read-only** — authored research document |
| `docs/research/phase2_4DGS_dynamic.md` | Full research basis for Phase 2/3/4 | **Read-only** — authored research document |
| `docs/sessions/` | Session logs | Append-only — create a new file per session |

### Session log format

Every coding session (human or agent) that implements a stub, adds a new module, or makes a non-trivial architectural decision should create a log at:

```
docs/sessions/YYYY-MM-DD_<short-topic>.md
```

Template:

```markdown
# Session: <topic> — YYYY-MM-DD

## What was done

## Key decisions made (and why)

## Files changed

## What's left / known issues
```

**When to write a log:** implementing a stub, adding a module, changing config defaults, resolving an architectural ambiguity, fixing a bug that reveals a design issue.

**When not to write a log:** typo fixes, formatting, trivial one-liners.

**Logs are append-only.** Never edit a past session log. Open a new file for follow-up work.

### What NOT to put in session logs

- In-progress or ephemeral state — use code TODOs instead
- Information already in `git log` or readable directly from the code
- Config values — those live in `config/` and `core/config.py`

### Keeping this README current

When a `[STUB]` module is fully implemented:
1. Move it from the "Stubs" table to the "Done" table in **Implementation Status**
2. Remove `[STUB]` from its entry in the **Repository Structure** tree
3. Add a one-line entry to `docs/ARCHITECTURE.md` under the relevant phase section
4. Create a session log in `docs/sessions/`

---

## References

Full technical rationale for every implementation decision is documented in the research files:

- [`docs/research/phase1_3DGS_static.md`](docs/research/phase1_3DGS_static.md) — Operator masking cascade, equirect-to-planar mathematics, COLMAP optimisations, gsplat VRAM hyperparameters, Grendel distributed training
- [`docs/research/phase2_4DGS_dynamic.md`](docs/research/phase2_4DGS_dynamic.md) — Dynamic subject isolation, SWinGS temporal windowing, MEGA colour decomposition, entropy-constrained pruning, unified CUDA depth-sorting, SPZ/ONNX streaming architecture
- [`docs/research/long_term_vision_corridorkey_integration.md`](docs/research/long_term_vision_corridorkey_integration.md) — Long-term speculative vision: OmniSplat4D as a 3D geometric foundation for production compositing pipelines; integration analysis with CorridorKey; open research questions. **Not a sprint deliverable.**
- [`docs/OPEN_QUESTIONS.md`](docs/OPEN_QUESTIONS.md) — Valid technical critiques, unresolved architectural questions, and gaps between stated goals and current implementation. Updated as scrutiny arrives.
