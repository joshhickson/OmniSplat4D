# AMD-VE — Reference Note

**Repo:** https://github.com/Rolaand-Jayz/AMD-V.E.-.git  
**Submodule:** `vendor/AMD-VE`  
**Author:** Rolaand Jayz (CorridorKey Discord)  
**Status as of note:** v0.1.0-beta.1, Linux-only, primary verified stack is Arch Linux + RX 7900 GRE

---

## What it is

C++20 AI video enhancement pipeline (denoise, upscale, compression restoration) built AMD-first. Primary inference backend is MiGraphX — AMD's graph compiler inside the ROCm stack. Falls back through ROCm/HIP ONNX Runtime → Vulkan Compute → NCNN Vulkan → FFmpeg filter in that order. Has both a CLI (`ave`) and an optional Qt GUI (`ave_gui`).

## Why it's potentially useful for OmniSplat4D

OmniSplat4D is currently CUDA/PyTorch-first and targets NVIDIA hardware (RTX 3060 12GB prototype, A100/H100 cloud). AMD hardware is explicitly out of scope for the current sprint roadmap. However:

**1. ROCm inference path reference.** If OmniSplat4D ever adds AMD hardware support — either as a community contribution or a Phase 4+ target — AMD-VE is one of the only public consumer apps that has actually wired MiGraphX into a real media pipeline end-to-end. The MiGraphX backend in this repo (model compilation, `.mxr` artifact caching, runtime fingerprinting, problem cache management) is unusually detailed for a public project. It is a reference implementation, not a toy.

**2. Vulkan Compute fallback pattern.** AMD-VE's Vulkan Compute backend is hardware-agnostic — Vulkan runs on both NVIDIA and AMD. If OmniSplat4D ever needs a non-CUDA rasterization path (e.g. for the compositing renderer on hardware where gsplat's CUDA kernels are unavailable), the Vulkan Compute backend structure here is worth reviewing.

**3. Portable bundle + packaging toolchain.** AMD-VE has a working packaging pipeline for portable Linux bundles that bundle a custom MiGraphX runtime. If OmniSplat4D ever ships a standalone binary (rather than a Python package install), the CMake/packaging approach here is a concrete example to reference.

## What it is not

Not relevant to the current sprint plan (Sprints 3–5 are CUDA gsplat work). Not a Python project. Not cross-platform yet. The CorridorKey AMD optimization work Rolaand mentioned in the Discord (PRs 4707-4710 in the AMD repo) appears to be separate from this video enhancement project.

## When to revisit

- If a community contributor asks about AMD/ROCm support for OmniSplat4D
- If gsplat adds a Vulkan backend and we evaluate adopting it
- If Phase 4 cloud scaling expands to AMD Instinct hardware
