# -*- coding: utf-8 -*-
"""
DB-GS: Depth-Guided Budget-Aware Training for Memory-Efficient 3D Gaussian Splatting.

四个模块与官方 gaussian-splatting 的集成关系:
  Module 1 (深度先验初始化)      -> dbgs/depth_init.py   + 替换 scene.create_from_pcd 点云来源
  Module 2 (深度-不确定性感知稠密化) -> dbgs/densify_prune.py (gaussians.densify_* 的 gate)
  Module 3 (预算感知剪枝)         -> dbgs/densify_prune.py (gaussians.densify_and_prune 的 budget)
  Module 4 (深度一致性损失)       -> 官方 train.py 已内置 (-d + depth_l1_weight)，直接复用

本包不依赖 GPU 即可 import；深度估计(DVA-V2)与训练需 torch。
"""

from .depth_estimator import DepthPriorEstimator
from .depth_init import backproject_depth_to_pcd, build_depth_prior_ply
from .densify_prune import (
    depth_consistency_gate,
    budget_prune_mask,
    DepthAwareDensifier,
)

__all__ = [
    "DepthPriorEstimator",
    "backproject_depth_to_pcd",
    "build_depth_prior_ply",
    "depth_consistency_gate",
    "budget_prune_mask",
    "DepthAwareDensifier",
]
