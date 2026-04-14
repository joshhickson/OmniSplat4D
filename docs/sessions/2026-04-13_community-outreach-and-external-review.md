# Session: Community outreach and external review — 2026-04-13

## What was done

This session did not implement any stubs. The work was entirely documentation, community strategy, and processing external technical scrutiny. The next agent should move directly to Sprint 3 implementation.

### CorridorKey integration vision

- Discovered CorridorKey (https://github.com/nikopueringer/CorridorKey.git) — an open-source neural green screen alpha unmixing tool by Niko Pueringer (Corridor Digital). The project gained 8,000 GitHub stars and 5,000 Discord members within one week of a major update video.
- Analyzed the bidirectional integration opportunity: OmniSplat4D's 3DGS background renders can replace BiRefNet/GVM as a geometry-derived hint generator for CorridorKey (no green screen required); CorridorKey's linear alpha output can replace OmniSplat4D's binary SAM masks as training data for Phase 2 4DGS.
- Wrote `docs/research/long_term_vision_corridorkey_integration.md` — a full speculative vision document covering four integration components (hint generation, temporal alpha stabilization, Gaussian-level depth-correct compositing, physical relighting), feasibility analysis against current stubs, and a consolidated open research questions table. **This is not a sprint deliverable.** Completing Sprints 3–5 is a prerequisite for any integration work.
- Added a "Note to the CorridorKey Community" preface to `README.md` — written specifically for the CorridorKey Discord/GitHub audience, naming three concrete technical intersections relevant to their known pain points (hint generation, temporal flicker, depth-correct compositing).
- Added `vendor/CorridorKey` as a git submodule for ongoing reference.

### Community engagement (CorridorKey Discord)

Josh posted in the CorridorKey Discord and received responses from several community members. Key interactions:

- **yo odd / Omegaindebt** — both independently working on YOLO + SAM cascades on VRAM budgets. OmniSplat4D's `segment/` stack is directly relevant to their work. A reply was drafted and sent pointing them to the repo.
- **Rolaand Jayz** — built AMD-VE (https://github.com/Rolaand-Jayz/AMD-V.E.-), an AMD-first C++20 video enhancement app using MiGraphX as the primary inference backend. Added as a submodule, analyzed, then removed. Reference notes kept at `notes/amd_ve_reference.md`. No direct crossover with OmniSplat4D's current CUDA/PyTorch stack; relevant if AMD hardware support is ever a target.
- **texasgreentea** — game engine VFX/XR practitioner with roomscale VR documentary production background. Read the README thoroughly and provided the most substantive external technical critique received so far (see below).

### External technical review — texasgreentea

A technically credible external reviewer with professional XR production experience read the README and raised four valid critiques, all of which are now documented in `docs/OPEN_QUESTIONS.md`:

1. **Single-viewpoint SH problem** — limited angular coverage of subjects in 360° stick route footage means SH reconstruction of subject faces is not achievable. SH Degree 1 was chosen for VRAM reasons; the angular coverage gap exists regardless.
2. **Behind-subject occlusion artifacts** — Gaussians behind occluders are undertrained. In roomscale VR this is a visible artifact class. VFX supervisors will attack it.
3. **CorridorKey integration bootstrap dependency** — the proposed hint generation pipeline has a circular dependency (need background model to generate hint; need hint to train background model). Documented as open research question in vision doc but not acknowledged in README.
4. **Free-viewport vs. compositing/roto use case tension** — the README leads with "free-viewpoint 4D volumetric environments" but the more defensible near-term use case is probably better roto and depth maps for real-time comp of 360 content.

Reviewer also provided:
- Validation that **stationary 360 viewing is achievable** once goals are met, except where specular cues are off from limited viewpoints.
- The **concrete quality benchmark**: beat photogrammetry + manual cleanup + stereo billboard video cards for hero subjects. Splats win on the "billboard breaks at the feet" problem — but only if the artifact profile (z-popping, stray Gaussians, off speculars) is cleaner.
- **Z-popping** identified as a likely dailies-level blocker even before more subtle issues. VFX supervisor tolerance is essentially zero for eye-catching blemishes.
- **Temporal cleanup at scale** — single-frame splat cleanup labor already matches messy photogrammetry. 4DGS multiplies that by thousands of frames. Temporal coherence (SWinGS) is the mechanism to reduce per-frame cleanup debt, but it's unvalidated at production scale.
- **Lightfield context** — prior equirectangular-to-lightfield pipelines stalled on manual cleanup labor and prohibitive data footprint. Shelved when NeRFs and splats arrived. Flat-to-splat on 360 footage remains unsolved by most of the field.

### New files created this session

| File | Purpose |
|---|---|
| `docs/research/long_term_vision_corridorkey_integration.md` | Full CorridorKey integration vision — speculative, not a sprint deliverable |
| `docs/OPEN_QUESTIONS.md` | Live document: valid critiques, unresolved questions, implementation gaps |
| `notes/amd_ve_reference.md` | Reference note on AMD-VE repo for future AMD/ROCm/Vulkan consideration |

### README changes

- Added "A Note to the CorridorKey Community" section after the intro paragraphs
- Added `docs/OPEN_QUESTIONS.md` and long-term vision doc to References section
- Removed hardware spec from the one-line description (minor cleanup, commit `7b3a8d1`)

---

## Key decisions made (and why)

- **Did not update the README's free-viewport claim** — the external critique is valid but the claim is aspirational and correctly reflects the long-term goal. The tension is documented in `OPEN_QUESTIONS.md` rather than removed from the README. Removing it would misrepresent the project's direction.
- **Kept CorridorKey as a submodule** — ongoing reference for integration work. AMD-VE submodule was removed because it has no near-term crossover; notes file retained.
- **OPEN_QUESTIONS.md is not a bug tracker** — entries are removed when resolved by measurement, implementation, or deliberate scope decision, not when they become uncomfortable. This distinction is stated explicitly in the file.

---

## Files changed

```
README.md                                                  — CorridorKey preface + references
docs/research/long_term_vision_corridorkey_integration.md  — new
docs/OPEN_QUESTIONS.md                                     — new
notes/amd_ve_reference.md                                  — new
.gitmodules                                                — CorridorKey submodule entry
vendor/CorridorKey                                         — submodule (AMD-VE added and removed same session)
```

---

## What's left / next steps

### Immediate: Sprint 3 — `train/static_trainer.py`

This is the next implementation task. Nothing else should be worked on until `train_static()` is implemented. The stubs for Sprint 3 are:

**File:** `src/omnisplat4d/train/static_trainer.py`  
**Function:** `train_static(colmap_dir, mask_dir, cfg, output_dir) -> GaussianCheckpoint`

Implementation requirements (all documented in `docs/research/phase1_3DGS_static.md`):
- Use gsplat's rasterisation API
- `sh_degree=1`, `packed=True`, `densify_grad_threshold=0.0004` — read from `StaticTrainConfig`, never hardcoded
- Call `flush_cuda_cache()` from `core/memory.py` after every densification step — never `torch.cuda.empty_cache()` directly
- Log VRAM usage at DEBUG level every 1000 iterations
- Write checkpoint to `output_dir` every 5000 iterations
- Write final `background_model.spz` via `export/spz_writer.py`
- Return a `GaussianCheckpoint` (FP16 if `cfg.output_format == "spz"`)
- `mask_dir` is optional — `None` for Drone Route (no operator masking)

**Do not call `colmap feature_extractor`** anywhere in this function or any function it calls. This is the single most important invariant in the codebase. COLMAP poses are injected programmatically by `sfm/initializer.py` and must not be overridden.

### After Sprint 3: Sprint 4 — `train/dynamic_trainer.py`

Three functions: `train_dynamic()`, `_train_single_window()`, `build_frame_batches()`. Full spec in `docs/research/phase2_4DGS_dynamic.md` §VRAM-Agnostic 4DGS Training and in the function docstrings in `dynamic_trainer.py`.

### After Sprint 4: Sprint 5 — `composite/renderer.py`

`radix_sort_by_depth()` and `render_frame()`. Once `render_frame()` exists, two thin wrappers become trivial to add: `render_background_at_pose()` and `extract_alpha_matte()` — both needed for the CorridorKey integration path.

### Ongoing: `docs/OPEN_QUESTIONS.md`

Add entries as new technical critiques arrive. Update or close entries as the pipeline produces real output and questions become empirically answerable. The z-popping and temporal-coherence-at-scale questions in particular cannot be answered until Sprint 5 is complete and real footage has been processed.

---

## Context for the incoming agent

- The repo owner (Josh) is not a VFX professional — he is learning the domain while building. Explain decisions clearly, do not assume prior compositing or XR production knowledge.
- The project is being developed in public and is being watched by the CorridorKey community. Code quality and architectural honesty matter for credibility.
- `docs/OPEN_QUESTIONS.md` is the honest state-of-play document. Read it before making any claims about what the pipeline produces.
- All five VRAM invariants in the README Agent Onboarding section are non-negotiable for the RTX 3060 profile. Violating any of them will OOM the GPU.
- The CorridorKey integration is a long-term vision, not a current sprint. Do not implement any CorridorKey-facing code until `render_frame()` exists.
