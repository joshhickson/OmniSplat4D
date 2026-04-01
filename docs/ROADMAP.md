# Execution Roadmap

### Phase 1: Local Prototyping (VRAM Constraints)
* **Target Hardware:** Intel i7-4790k, 32GB DDR3 RAM, Nvidia RTX 3060 12GB.
* **Input Data:** Samsung SM-C200 (Gear 360) test sequences.
* **Objectives:** * Implement the equirectangular-to-planar projection script.
  * Integrate the YOLOv8 + SAM 2.1 CPU-offloaded masking cascade.
  * Validate programmatic COLMAP initialization.
  * Successfully train a static 3DGS environment under the 12GB threshold.

### Phase 2: Dynamic Subject Isolation & Temporal Chunking
* **Target Hardware:** Local RTX 3060 12GB.
* **Objectives:**
  * Implement the inverted masking workflow to extract moving subjects across the 8 planar views.
  * Engineer the 30-frame temporal sliding window for 4DGS training.
  * Validate VRAM flush efficiency between chunk serializations.
  * Export the canonical `.spz` and temporal `.onnx` pair.

### Phase 3: Compositing & Viewer Integration
* **Objectives:**
  * Develop the combined render graph for WebGL and/or Unity.
  * Ensure accurate real-time depth sorting between the two independent models sharing the global coordinate space.

### Phase 4: Cloud Scaling (High-Fidelity)
* **Target Hardware:** Lambda Labs instances (A100 or H100).
* **Input Data:** 8K 360-video captures.
* **Objectives:**
  * Refactor hardware profiling constraints.
  * Unlock Spherical Harmonics (SH) Degree 3.
  * Increase dense point budgets and expand temporal sliding window limits.