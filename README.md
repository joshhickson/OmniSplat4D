# 4D Composite Gaussian Splatting from 360 Video (VRAM-Agnostic)

This repository provides an end-to-end pipeline for generating free-viewpoint 4D volumetric environments from equirectangular 360-degree video. It decomposes raw footage into static 3D background environments and dynamic 4D keyframable subjects, compositing them in real-time.

The architecture is explicitly designed to run on mid-range consumer hardware, bypassing the massive memory bottlenecks typically associated with 4D Gaussian Splatting (4DGS). 

## Hardware Philosophy
Standard 4D radiance field training requires clusters of A100s. This pipeline is engineered to operate strictly within a **12GB VRAM** ceiling (e.g., Nvidia RTX 3060) and **32GB system RAM**. It achieves this through programmatic camera injection, aggressive model quantization, out-of-core temporal chunking, and cascaded AI masking. It scales seamlessly to 80GB tensor core instances for high-fidelity 8K output.

## Core Capabilities
* **Dual Capture Paradigms:** Supports both handheld 360 cameras (The Stick Route - requiring automated operator masking) and aerial 360 cameras (The Drone Route - utilizing programmatic SfM initialization).
* **Equirectangular Ingestion:** Bypasses standard photogrammetry polar distortion failures by mathematically reprojecting spherical video into 8 overlapping planar virtual cameras.
* **VRAM-Agnostic 4DGS:** Trains dynamic subjects using a 30-frame temporal sliding window, serializing deformation fields to disk to prevent Out-Of-Memory (OOM) crashes.
* **Real-Time Compositing:** Glues the decoupled static `.spz` splats and dynamic `.onnx` deformation models back together in a shared coordinate space for WebGL or Unity rendering.

## Repository Structure
```text
├── docs/
│   ├── ARCHITECTURE.md                 # Core pipeline logic and memory management
│   ├── ROADMAP.md                      # Execution phases and scaling targets
│   ├── research/
│   │   ├── phase1_3DGS_static.md       # Research report: Static base generation
│   │   └── phase2_4DGS_dynamic.md      # Research report: Temporal chunking & compositing
├── src/
│   ├── ingest/                         # Equirectangular to cubemap/planar reprojection
│   ├── segment/                        # YOLOv8 + SAM 2.1 dynamic subject isolation
│   ├── sfm/                            # Programmatic COLMAP initialization scripts
│   ├── train_static/                   # 3DGS backend optimized for SH1 and VRAM capping
│   ├── train_dynamic/                  # Temporal windowing and 4D deformation logic
│   └── composite/                      # Real-time depth-sorting render graph
└── requirements.txt