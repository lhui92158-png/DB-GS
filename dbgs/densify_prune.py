# -*- coding: utf-8 -*-
"""
DB-GS Module 2 + 3: 深度-不确定性感知稠密化 与 预算感知剪枝。

Module 2 (depth_consistency_gate):
  在 gaussians.densify_and_clone / densify_and_split 的 selected_pts_mask 上再乘一个
  深度一致性门控: 只有当"该高斯的渲染深度 vs 单目先验深度"一致(残差小)或该像素
  显著度高(梯度大)时才允许增点, 从而抑制平坦/纹理弱/深度不确定区域的过度稠密化。

Module 3 (budget_prune_mask):
  硬性高斯数量上限 budget。超预算时按重要性分数排序剪枝, 保证训练峰值显存有上界。
  重要性 = opacity * (norm(scaling) 的反相关) * (累计梯度), 可换。

这两个模块都是"插入 gaussian_model.py 的一小段", 不重写训练流程, 便于消融实验
(可单独开/关 Module 2 或 Module 3)。
"""

import torch


# ---------------------------------------------------------------------------
# Module 2: 深度一致性门控
# ---------------------------------------------------------------------------
def depth_consistency_gate(
    selected_mask,
    rendered_depth,
    mono_depth,
    depth_mask,
    grad_norm,
    tau_depth,
    tau_grad_floor,
):
    """返回一个与 selected_mask 同 shape 的 bool gate, 用于进一步筛选增点候选。

    Args:
        selected_mask: (N,) bool, densify 原始候选 (梯度阈值已过)。
        rendered_depth: (N,) float, 渲染得到的视点空间深度。
        mono_depth: (N,) float, 对齐后的单目先验深度 (相机系 z)。
        depth_mask: (N,) bool, 单目深度有效掩码。
        grad_norm: (N,) float, 每点的视图空间梯度 norm (去噪前的原始累计值)。
        tau_depth: 深度一致性阈值 (相对残差)。残差 = |rendered-mono| / mono。
        tau_grad_floor: 梯度下限; 梯度高于此值视为"高显著度", 无条件放行。

    规则: 放行 if (深度一致 OR 高显著度) AND 原候选。
         深度一致 = depth_mask & (rel_residual < tau_depth)。
    """
    rel_res = torch.abs(rendered_depth - mono_depth) / torch.clamp(mono_depth, min=1e-6)
    consistent = depth_mask & (rel_res < tau_depth)
    salient = grad_norm > tau_grad_floor
    # 深度缺失(无 mask)时退回纯梯度驱动(不阻碍)
    no_depth = ~depth_mask
    return selected_mask & (consistent | salient | no_depth)


# ---------------------------------------------------------------------------
# Module 3: 预算感知剪枝
# ---------------------------------------------------------------------------
def budget_prune_mask(score, budget, keep_ratio_soft=0.9):
    """返回需要被剪掉的点的 bool mask, 使存活点数 <= budget。

    重要性分数由调用方提供 (score)。DB-GS 用视图空间梯度 norm 作为分数
    (高梯度 = 欠重建 = 重要 = 保留; 低梯度 = 已收敛/冗余 = 剪掉),
    这与 Taming 3DGS / Gaussians on a Diet 的 score-based 剪枝同源。

    Args:
        score: (N,) 或 (N,1), 重要性分数 (越大越重要)。
        budget: 硬性高斯数上限。
        keep_ratio_soft: 软阈值系数——当点数超过 budget*keep_ratio_soft 时才触发剪枝,
                         避免在接近预算时频繁抖剪。
    Returns:
        prune_mask: (N,) bool, True=剪掉。
    """
    n = score.shape[0]
    score = score.reshape(-1).float()
    soft_budget = int(budget * keep_ratio_soft)
    if n <= soft_budget:
        return torch.zeros(n, dtype=torch.bool, device=score.device)

    # 剪掉分数最低的 (n - soft_budget) 个
    k = n - soft_budget
    prune_idx = torch.topk(score, k, largest=False).indices
    prune_mask = torch.zeros(n, dtype=torch.bool, device=score.device)
    prune_mask[prune_idx] = True
    return prune_mask


# ---------------------------------------------------------------------------
# Module 3b: Lagrangian 对偶软剪枝 (方向1: 可微显存约束优化)
# ---------------------------------------------------------------------------
def dual_prune_step(score, budget, lambda_val, lr):
    """Lagrangian 对偶软剪枝一步: 剪除 normalized_score < lambda_val 的点, 并更新 lambda。

    理论 (见 理论分析.md §5): 保留判据 s_i >= lambda * c, 其中 lambda 是 Lagrange 乘子,
    s_i 是重要性分数 (边际质量损失代理)。这里把分数归一化到 [0,1] 后, lambda 也归一化到
    [0,1] (常数 c 被吸收), 阈值即为 lambda。对偶上升:
        lambda <- clip(lambda + lr * (|G|/budget - 1), 0, 1)
    当 |G| > budget 时 lambda 增大(剪更多), |G| < budget 时 lambda 减小(剪更少),
    收敛到满足互补松弛的 lambda*。

    Args:
        score: (N,) 或 (N,1), 重要性分数 (越大越重要, 与 budget_prune_mask 一致)。
        budget: 目标点数。
        lambda_val: 当前 lambda (float, [0,1])。
        lr: 对偶步长 eta。
    Returns:
        (prune_mask, new_lambda): prune_mask (N,) bool; new_lambda float。
    """
    n = score.shape[0]
    score = score.reshape(-1).float()

    # 归一化到 [0,1]（score >= 0，故只需除 max）
    smax = score.max()
    score_norm = (score / smax) if smax > 0 else torch.zeros_like(score)

    # 软剪枝: 剪 normalized_score < lambda
    prune_mask = score_norm < float(lambda_val)

    # 对偶上升: 用剪枝后的点数 n_after 而非剪枝前 n 计算违规,
    # 否则 lambda 会系统性偏高(过度剪枝→欠预算)。见 理论分析.md §5 修正。
    n_after = n - int(prune_mask.sum().item())
    new_lambda = float(lambda_val) + lr * (n_after / budget - 1.0)
    new_lambda = max(0.0, min(1.0, new_lambda))
    return prune_mask, new_lambda


# ---------------------------------------------------------------------------
# 便捷封装: 便于在 gaussian_model.py 里一行接入
# ---------------------------------------------------------------------------
class DepthAwareDensifier:
    """把 Module 2/3 的阈值与开关聚合成一个可注入对象, 方便消融。

    用法 (在 gaussian_model.py 的 densify_and_prune 内):
        dad = DepthAwareDensifier(budget=1_500_000, enabled_depth=True, enabled_budget=True)
        # ... 在 clone/split 前:
        sel = dad.gate_clone_split(sel, rendered_d, mono_d, d_mask, grad_norm)
        # ... 在 prune 阶段末尾:
        extra_prune = dad.budget_prune(gaussians.get_xyz, gaussians.get_opacity,
                                       gaussians.get_scaling, gaussians.xyz_gradient_accum)
    """
    def __init__(self, budget=None, tau_depth=0.3, tau_grad_floor=1e-4,
                 enabled_depth=True, enabled_budget=True, keep_ratio_soft=0.9):
        self.budget = budget
        self.tau_depth = tau_depth
        self.tau_grad_floor = tau_grad_floor
        self.enabled_depth = enabled_depth
        self.enabled_budget = enabled_budget
        self.keep_ratio_soft = keep_ratio_soft

    def gate_clone_split(self, selected_mask, rendered_depth, mono_depth, depth_mask, grad_norm):
        if not self.enabled_depth or rendered_depth is None or mono_depth is None:
            return selected_mask
        return depth_consistency_gate(selected_mask, rendered_depth, mono_depth,
                                      depth_mask, grad_norm, self.tau_depth, self.tau_grad_floor)

    def budget_prune(self, score, budget=None):
        if not self.enabled_budget or (budget or self.budget) is None:
            return None
        return budget_prune_mask(score, budget or self.budget, self.keep_ratio_soft)


# ---------------------------------------------------------------------------
# Module 2 辅助: 计算每个高斯的"深度边沿强度"
# ---------------------------------------------------------------------------
def compute_depth_edge(invdepthmap, depth_mask, viewspace_points, visibility_filter,
                       W, H, FoVx, FoVy):
    """返回 (N,) 边沿强度张量: 非可见高斯=1.0(不门控), 可见高斯=该像素单目深度梯度强度[0,1]。

    Args:
        invdepthmap: (1,H,W) torch, 对齐后的单目逆深度 (viewpoint_cam.invdepthmap)。
        depth_mask:  (1,H,W) torch bool, 深度有效掩码 (viewpoint_cam.depth_mask)。
        viewspace_points: (N_vis,3) torch, 可见高斯在相机系坐标 (x右,y上,z前)。
        visibility_filter: (N,) bool, 全部高斯 -> 可见。
        W,H: 图像宽高; FoVx,FoVy: 相机视场角。
    """
    import math
    dev = viewspace_points.device
    invD = invdepthmap.to(dev).squeeze(0)  # (H,W)
    mask = depth_mask.to(dev).squeeze(0)   # (H,W)
    # 单目逆深度空间梯度 = 边沿强度 (平坦区≈0, 边界/棱角≈大)
    gy, gx = torch.gradient(invD)
    edge = torch.sqrt(gx * gx + gy * gy) * mask.float()
    emax = edge.max()
    if emax > 0:
        edge = edge / emax

    # 注意: visibility_filter 是 (radii>0).nonzero() = (N_vis,1) 的索引张量, 非布尔掩码
    vis_idx = visibility_filter.reshape(-1)          # (N_vis,) 可见高斯索引
    vis_pts = viewspace_points[vis_idx]              # (N_vis,3) 相机系坐标

    # 相机系 -> 像素 (u=cx+fx*x/z, v=cy-fy*y/z)
    fx = W / (2.0 * math.tan(FoVx / 2.0))
    fy = H / (2.0 * math.tan(FoVy / 2.0))
    z = vis_pts[:, 2].clamp(min=1e-6)
    u = (W / 2.0 + fx * vis_pts[:, 0] / z).long().clamp(0, W - 1)
    v = (H / 2.0 - fy * vis_pts[:, 1] / z).long().clamp(0, H - 1)

    N = viewspace_points.shape[0]
    edge_weight = torch.ones(N, device=dev, dtype=torch.float32)
    edge_weight[vis_idx] = edge[v, u]                # (N_vis,) = (N_vis,)
    return edge_weight


def compute_image_edge(image, viewspace_points, visibility_filter, W, H, FoVx, FoVy):
    """图像梯度边沿强度（改进3 对照实验）：与 compute_depth_edge 完全相同的几何，
    唯一区别是边沿源 = RGB 灰度图的图像梯度（外观纹理边沿），而非单目深度梯度（几何边沿）。
    用于消融验证「深度先验 vs 图像梯度」哪个才是门控的有效信号。

    Args:
        image: (H,W) 或 (H,W,3) 或 (3,H,W) 或 (1,H,W) torch, GT/渲染图 (float, 任意范围)。
        其余参数同 compute_depth_edge。
    """
    import math
    dev = viewspace_points.device
    img = image.to(dev)
    if img.dim() == 3:
        if img.shape[0] == 3:          # (3,H,W)
            gray = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
        elif img.shape[2] == 3:        # (H,W,3)
            gray = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
        else:                          # (1,H,W)
            gray = img.squeeze(0)
    else:                              # (H,W)
        gray = img
    gy, gx = torch.gradient(gray)
    edge = torch.sqrt(gx * gx + gy * gy)
    emax = edge.max()
    if emax > 0:
        edge = edge / emax

    vis_idx = visibility_filter.reshape(-1)          # (N_vis,) 索引张量
    vis_pts = viewspace_points[vis_idx]              # (N_vis,3) 相机系
    fx = W / (2.0 * math.tan(FoVx / 2.0))
    fy = H / (2.0 * math.tan(FoVy / 2.0))
    z = vis_pts[:, 2].clamp(min=1e-6)
    u = (W / 2.0 + fx * vis_pts[:, 0] / z).long().clamp(0, W - 1)
    v = (H / 2.0 - fy * vis_pts[:, 1] / z).long().clamp(0, H - 1)

    N = viewspace_points.shape[0]
    edge_weight = torch.ones(N, device=dev, dtype=torch.float32)
    edge_weight[vis_idx] = edge[v, u]
    return edge_weight
