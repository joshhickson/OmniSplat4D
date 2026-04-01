# Pipeline Architecture

This pipeline circumvents hardware limitations by rigorously decoupling the scene into static and dynamic passes.

## 1. Preprocessing: Geometric Projection & Semantic Masking
Standard Structure-from-Motion (SfM) solvers fail on non-linear spherical data.
* **Planar Reprojection:** The continuous 360 video is sliced into 8 overlapping rectilinear virtual cameras. 
* **The Stick Route (Operator Masking):** A YOLOv8-nano + SAM 2.1 Tiny cascade identifies the human operator and selfie stick. These models are forced to execute via CPU offloading, preserving VRAM for the subsequent training phase. The operator is masked out, creating clean plates for the static background solver.

## 2. Phase 1: Static Background Generation (3DGS)
The static environment is reconstructed without the interference of moving subjects.
* **Programmatic SfM:** Feature extraction is entirely bypassed. Pre-calculated K-matrices and quaternions for the 8 virtual cameras are injected directly into `cameras.txt` and `images.txt`, forcing COLMAP to accept the trajectory without computational overhead.
* **VRAM Throttling:** The `gsplat` backend is clamped to Spherical Harmonics (SH) Degree 1, utilizing packed tensor representations. The densification gradient threshold is elevated to `0.0004` to aggressively cull unnecessary points and maintain the 12GB VRAM ceiling.
* **Output:** A highly compressed `.spz` file representing the mathematically pristine static scene.

## 3. Phase 2: Dynamic Subject Generation (4DGS)
Moving subjects are isolated and trained as independent temporal entities.
* **Inverted Masking:** The SAM 2.1 masks from preprocessing are inverted. The static background is blacked out entirely, leaving only the dynamic subject against a void.
* **Temporal Chunking:** To prevent immediate OOM failures on consumer GPUs, the pipeline processes the video in 30-frame sequential chunks. The optimizer learns the deformation grid for the current chunk, serializes the data to local disk, flushes the VRAM cache entirely, and loads the subsequent chunk.
* **Output:** A base `.spz` representing the subject's canonical shape, and an `.onnx` neural network file containing the temporal deformation fields.

## 4. Real-Time Compositing Graph
The decoupled assets must be visually re-integrated.
* **Shared Coordinate Space:** Because the static background and the isolated dynamic subject were tracked against the identical virtual camera trajectory in Phase 1, their global coordinate spaces are perfectly aligned.
* **Depth Sorting:** Both the `.spz` (static) and `.onnx` (dynamic) splats are loaded into the VRAM of the renderer simultaneously. A hardware-accelerated radix sort evaluates the exact physical depth of every point prior to rasterization, ensuring the dynamic subject correctly occludes (and is occluded by) static environmental geometry without rendering artifacts.