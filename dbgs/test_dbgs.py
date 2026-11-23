# -*- coding: utf-8 -*-
"""DB-GS 模块逻辑单元测试 (无需 GPU, 纯 numpy/torch 验证正确性)。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch


def test_backproject():
    from dbgs.depth_init import backproject_depth_to_pcd
    # 相机朝向 +z 的正前方平面, 距离 z0
    z0 = 5.0
    K = np.array([[100., 0., 50.], [0., 100., 50.], [0., 0., 1.]])
    R = np.eye(3)
    T = np.zeros(3)
    depth = np.full((10, 10), z0, dtype=np.float32)
    pts = backproject_depth_to_pcd(depth, K, R, T, stride=2)
    assert pts.shape[1] == 3 and pts.shape[0] == 25, pts.shape
    # 所有点 z 应 == z0, x/y 相对中心对称
    assert np.allclose(pts[:, 2], z0, atol=1e-4), pts[:3]
    print("[OK] backproject: 25 pts, z 全部 =", z0)


def test_backproject_rotation():
    from dbgs.depth_init import backproject_depth_to_pcd
    # 相机绕 y 轴旋转 90°, 相机系 +z 转到世界 -x
    z0 = 3.0
    K = np.array([[100., 0., 50.], [0., 100., 50.], [0., 0., 1.]])
    th = np.pi / 2
    R = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
    T = np.zeros(3)
    depth = np.full((10, 10), z0, dtype=np.float32)
    pts = backproject_depth_to_pcd(depth, K, R, T, stride=5)
    # 世界坐标 x 应 == -z0
    assert np.allclose(pts[:, 0], -z0, atol=1e-3), pts[:3]
    print("[OK] backproject rotation: 世界 x 全部 =", -z0)


def test_budget_prune():
    from dbgs.densify_prune import budget_prune_mask
    N = 100
    xyz = torch.randn(N, 3)
    opacity = torch.rand(N)
    scaling = torch.rand(N, 3) * 0.1
    grads = torch.rand(N)
    budget = 40
    mask = budget_prune_mask(xyz, opacity, scaling, grads, budget, keep_ratio_soft=1.0)
    assert mask.shape == (N,)
    keep = (~mask).sum().item()
    assert keep <= budget, f"keep={keep} > budget={budget}"
    # 高分点应被保留
    high_idx = torch.topk(grads * opacity, 5).indices
    assert not mask[high_idx].any(), "高分点被误剪"
    print(f"[OK] budget_prune: N={N} budget={budget} -> keep={keep}")


def test_depth_gate():
    from dbgs.densify_prune import depth_consistency_gate
    N = 10
    sel = torch.ones(N, dtype=torch.bool)
    rendered = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    mono = torch.tensor([1.0, 1.05, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    dmask = torch.ones(N, dtype=torch.bool)
    grad = torch.tensor([0.01, 0.01, 0.01, 1.0, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
    gate = depth_consistency_gate(sel, rendered, mono, dmask, grad, tau_depth=0.3, tau_grad_floor=0.1)
    # idx1 深度一致(残差0.05<0.3)放行; idx2 残差0.5 但梯度低 -> 剪; idx3 高显著度放行
    assert gate[1] and not gate[2] and gate[3], gate
    print("[OK] depth_gate: 深度一致/高显著度放行, 不一致低梯度剪")


if __name__ == "__main__":
    test_backproject()
    test_backproject_rotation()
    test_budget_prune()
    test_depth_gate()
    print("\nALL DB-GS UNIT TESTS PASSED")
