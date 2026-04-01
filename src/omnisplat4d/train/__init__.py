"""
Train sub-package — Gaussian Splatting training for Phase 1 and Phase 2.

Modules:
    static_trainer   — gsplat 3DGS Phase 1 static background training
    dynamic_trainer  — SWinGS sliding-window 4DGS Phase 2 dynamic subject training
    deformation      — Temporal deformation MLP (ONNX-traceable)
    color_mlp        — MEGA AC view/time color predictor MLP (ONNX-traceable)
    pruner           — Entropy-constrained opacity pruning utilities
"""
