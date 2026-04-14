# Open Questions and Known Limitations

This document accumulates valid technical critiques, unresolved architectural questions, and honest gaps in the current design. It is updated as scrutiny arrives — from external reviewers, from implementation surprises, and from self-assessment. Entries are not removed when they become uncomfortable; they are removed when they are resolved or definitively answered.

---

## Implementation State (as of April 2026)

Before anything else: the following core functions are `NotImplementedError` stubs. The pipeline produces no output on real footage yet.

| Function | File | Sprint |
|---|---|---|
| `train_static()` | `train/static_trainer.py` | Sprint 3 |
| `train_dynamic()` / `_train_single_window()` / `build_frame_batches()` | `train/dynamic_trainer.py` | Sprint 4 |
| `radix_sort_by_depth()` | `composite/renderer.py` | Sprint 5 |
| `render_frame()` | `composite/renderer.py` | Sprint 5 |

All claims about what the pipeline produces are architectural claims, not empirical ones. None of the output quality questions below can be answered until real footage has been processed.

---

## Valid Technical Critiques

### 1. Single-viewpoint spherical harmonics on subjects

**Source:** External review (texasgreentea, game engine VFX/XR background)

**The problem:** If a subject moves through the scene while generally facing the same direction — which is common in stick route footage — the 360° camera accumulates very limited angular coverage of that subject's face and front-facing surfaces. Spherical Harmonics reconstruction requires multi-view angular diversity to be meaningful. You cannot extrapolate the dark side of a face you never captured. Models that attempt semantic extrapolation of unobserved geometry typically degrade within a few parallax degrees.

**What the codebase says:** SH Degree 1 was chosen for VRAM reasons (`sh_degree=1` enforced in `StaticTrainConfig`, rationale in `docs/research/phase1_3DGS_static.md`). The README states "negligible outdoor PSNR impact" — this applies to the static background, which is observed from many angles. The angular coverage problem for dynamic subjects is a separate concern that the current design does not address.

**Current status:** Unresolved. The pipeline should not claim accurate SH reconstruction of subjects with limited angular coverage. The compositing use cases that don't require free-viewport rotation of subject surfaces (e.g. depth maps, roto masks, locked-camera effects) are less affected by this limitation than full 3D-ification of subjects for HMD viewing.

---

### 2. Behind-subject occlusion artifacts in free-viewport viewing

**Source:** External review (texasgreentea)

**The problem:** Gaussian primitives behind an occluder receive no photometric supervision during training — those camera rays never reach them. In a locked or limited-parallax viewing context this doesn't matter. In roomscale VR where a user steps sideways, the occluded regions behind subjects become visible and are either missing geometry or filled with floaters. Most production pipelines would not greenlight this artifact class for roomscale XR.

**What the codebase says:** The README describes "free-viewpoint 4D volumetric environments" and Unity XR as a target. This claim is aspirational. The pipeline makes no special provision for supervising occluded geometry, and the entropy pruning in `train/pruner.py` will aggressively remove low-opacity Gaussians including any that might otherwise partially fill occluded regions.

**Current status:** Unresolved. The severity is unknown until real footage is processed and viewed in a roomscale context. The limited-parallax use case (seated HMD, narrow 6DOF volume, or locked-camera WebGL) is more defensible near-term than full roomscale. The "better roto and depth maps for real-time comp" use case is unaffected by this limitation entirely.

---

### 3. The CorridorKey integration bootstrap problem

**Source:** Documented in `docs/research/long_term_vision_corridorkey_integration.md` Section 4.3; reinforced by external review

**The problem:** The proposed long-term integration uses OmniSplat4D's Phase 1 3DGS background render to generate a geometry-derived hint mask for CorridorKey, replacing BiRefNet/GVM. But Phase 1 background training itself currently depends on SAM-based masks to exclude the dynamic subject. If the goal is to replace SAM with CorridorKey's higher-quality alpha output, a circular dependency arises:

- You need the trained 3DGS background to generate the geometry-derived hint
- You need the hint to cleanly exclude the subject from 3DGS background training

**Proposed resolution (not yet implemented):** Two-pass iterative approach — coarse SAM mask → initial 3DGS → geometry-derived hint → CorridorKey alpha → refined 3DGS retrain. Whether the refined 3DGS meaningfully improves on the initial reconstruction is an empirical question.

**Current status:** Open research question. The README's description of the hint generation pipeline implies a clean one-pass flow that does not currently exist.

---

### 4. Free-viewport claim vs. compositing depth/roto claim

**Source:** External review (texasgreentea)

**The problem:** The README's first sentence describes "free-viewpoint 4D volumetric environments." The external reviewer correctly identified that the more defensible near-term use case is probably "better roto and depth maps for real-time compositing of 360 content" — interactive particle effects, fog, depth-correct occlusion in a locked viewing context. These are different products with different quality bars. The free-viewport claim requires solving the SH coverage problem (Issue 1) and the occlusion artifact problem (Issue 2). The compositing depth/roto claim does not.

**Current status:** The README has not been updated to reflect this distinction. Both use cases remain stated as goals. Priority ordering between them should be established once the training pipeline produces real output and the artifact profile is known.

---

## Pre-Existing Open Questions (from vision document)

These are documented in `docs/research/long_term_vision_corridorkey_integration.md` Section 9 and reproduced here for visibility.

| Question | Where it bites |
|---|---|
| How large is the photometric gap between 3DGS background render and raw footage? | Hint generation quality |
| Does CorridorKey's per-camera inference produce spatially consistent alpha across OmniSplat4D's 8 virtual camera boundaries? | Phase 2 training data |
| Does the entropy regularization loss prevent accurate modeling of semi-transparent silhouette regions (hair, motion blur)? | Alpha quality at edges |
| What is the optimal blend between CorridorKey neural alpha and 4DGS geometric alpha? | Compositing fidelity |
| How is metric scale registered between an externally-sourced background 3DGS and the subject 4DGS? | Cross-scene compositing |
| What is the real-time performance of the unified Gaussian radix sort over a combined background+subject set on RTX 3060? | Phase 3 feasibility |

---

### 5. Z-popping at the VFX supervisor bar

**Source:** External review (texasgreentea, roomscale VR documentary production background)

**The problem:** Z-popping — the shimmery tile-order instability when Gaussian sort keys flip between frames as the camera moves — is a known artifact of tile-based Gaussian rasterization. OmniSplat4D's unified radix sort (Phase 3) addresses depth-correct occlusion but does not eliminate z-popping at Gaussian boundaries. Based on production experience running dailies past professional VFX supervisors and producers, this artifact class alone is likely unacceptable. The tolerance level is described as: essentially no eye-catching blemishes. Engineers tend to set a much lower bar than supervisors will accept.

**Current status:** Uncharacterized. The severity of z-popping in OmniSplat4D's output is unknown until `render_frame()` is implemented and real footage is processed. The entropy pruning in `train/pruner.py` reduces stray Gaussians but does not target z-popping specifically.

---

### 6. Temporal coherence at production scale (thousands of frames)

**Source:** External review (texasgreentea)

**The problem:** Single-frame splats already require manual cleanup comparable to messy photogrammetry. For 4DGS, that cleanup multiplies by frame count — potentially thousands of frames. This is unworkable as a manual process. The SWinGS sliding-window approach enforces temporal consistency across 30-frame windows with 5-frame overlap, but whether this produces acceptable temporal stability across a full 4DGS sequence of 1,000+ frames without accumulated drift or artifact propagation is entirely unvalidated. A model that is good at automated cleanup and doesn't have to redo most work each frame is described as the key enabler for practical 4DGS production use.

**Current status:** Unvalidated. `dynamic_trainer.py` is a stub. The assumption that 5-frame overlap windows prevent drift across a full sequence has not been tested on any footage.

---

### 7. The concrete quality benchmark to clear

**Source:** External review (texasgreentea)

**The benchmark:** Splats need to beat the 10-year-old best-in-class affordable approach for XR production: photogrammetry scenery with manual cleanup, custom shader work, and stereo billboard video cards for hero subjects. Stereo billboard hero subjects break immersion when the audience pays attention to their feet. Splats eliminate the billboard breakage by construction — but only if they don't introduce heavier artifacting (z-popping, stray Gaussians, off speculars) that supervisors attack next.

**What this means for the pipeline:** The quality improvement over billboards at the feet is real and meaningful. But it only counts if the artifact profile from the splat pipeline is cleaner than what billboard subjects introduce at seam edges. The stationary 360 viewing case is validated as achievable by an experienced XR practitioner. The roomscale case remains the harder and more valuable target.

**Current status:** No output yet to evaluate against this benchmark.

---

## Questions That Require External Expertise

These cannot be resolved from inside the codebase. They require either production pipeline experience or empirical testing on real hardware.

- **What does "acceptable" parallax look like in a game engine VFX/XR context?** The quality bar for a compositing depth-map use case vs. a roomscale viewer use case vs. a flat WebGL viewer use case are different. External input from practitioners in each context needed.

- **What viewing modality is actually the target?** Full 6DOF roomscale, seated limited-parallax HMD, locked-camera 360 WebGL, or a screened flat-panel free-viewport experience each have different tolerances for the artifact classes this pipeline will produce. Note: external reviewer with roomscale VR documentary production background states roomscale is the only form that will retain the XR "wow factor" for XR 2.0.

- **What happened to equirectangular-to-lightfield conversion pipelines?** Partial answer received: too many blemishes requiring manual cleanup, prohibitive data footprint, effectively shelved when NeRFs and splats arrived. The industry consensus is that splats are the easier path to the same goal. Flat-to-splat on legacy video and 360 footage remains unsolved by most of the field.

---

## What This Document Is Not

This is not a list of bugs. It is a list of unresolved questions about design choices, unvalidated assumptions in stated goals, and gaps between what the README claims and what the codebase can currently demonstrate. When a question is resolved — by measurement, by implementation, or by a deliberate scope decision — it moves out of this document and into the relevant session log or research file.
