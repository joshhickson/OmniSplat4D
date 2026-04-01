"""
SfM sub-package — Structure-from-Motion via programmatic COLMAP seeding.

Modules:
    initializer — Writes cameras.txt, images.txt, points3D.txt directly,
                  bypassing COLMAP's feature_extractor entirely.
    runner      — COLMAP subprocess wrapper for feature matching and mapping.
"""
