# -*- coding: utf-8 -*-
"""
DB-GS Module 1 前置: 单目深度先验估计 (Depth-Anything-V2)。

为每张训练图像生成单目深度图 (相对深度, 尺度模糊), 并计算与 SfM 尺度对齐的
depth_params.json (仿射: inv_depth_aligned = scale * inv_depth_mono + offset),
与官方 gaussian-splatting 的深度管线 (scene/dataset_readers.py, scene/cameras.py) 完全兼容。

输出结构:
    <source_path>/depths/<image_name>.png           # 16-bit 深度图 (归一化)
    <source_path>/sparse/0/depth_params.json         # {image_name: {"scale":.., "offset":..}}
"""

import json
import os

import numpy as np


def load_depth_anything_v2(encoder="vits", device="cuda", repo_root=None):
    """加载 DVA-V2。encoder: vits(24.8M)/vitb(97.5M)/vitl(335M)。
    4GB 显存用 vits; 云端(T4/A100)可用 vitb/vitl。
    """
    if repo_root is None:
        repo_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Depth-Anything-V2")
    import sys
    sys.path.insert(0, repo_root)
    from depth_anything_v2.dpt import DepthAnythingV2

    configs = {
        "vits": {"encoder": "vits", "features": 64,  "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    }
    cfg = configs[encoder]
    model = DepthAnythingV2(**cfg)
    ckpt = os.path.join(repo_root, "checkpoints", f"depth_anything_v2_{encoder}.pth")
    state = __import__("torch").load(ckpt, map_location="cpu")
    model.load_state_dict(state)
    model = model.to(device).eval()
    return model


class DepthPriorEstimator:
    """批量生成深度图 + 计算尺度对齐参数。"""

    def __init__(self, model, device="cuda"):
        import torch
        self.model = model
        self.device = device
        self.torch = torch

    @__import__("torch").no_grad()
    def infer(self, image_bgr, input_size=518):
        """image_bgr: (H,W,3) uint8。返回 (H,W) float32 相对深度 (越大越远)。"""
        depth = self.model.infer_image(image_bgr, input_size=input_size)
        return np.asarray(depth, dtype=np.float32)

    # ------------------------------------------------------------------
    # 尺度对齐 (仿射拟合: 1/z_sfm ~= scale * (1/d_mono) + offset)
    # ------------------------------------------------------------------
    def fit_scale_offset(self, inv_depth_mono, sfm_points_cam_z, mask=None):
        """最小二乘拟合仿射。inv_depth_mono:(H,W) 单目逆深度; sfm_points_cam_z: 与
        图像像素对应的 SfM 点在相机系 z 坐标(>=0), 长度==有效像素数。
        返回 (scale, offset)。失败返回 (1.0, 0.0) 表示无法对齐(退化为单位尺度)。
        """
        y = 1.0 / np.maximum(sfm_points_cam_z, 1e-6)  # 目标: SfM 逆深度
        x = inv_depth_mono.reshape(-1) if mask is None else inv_depth_mono[mask]
        if x.size < 50 or not np.all(np.isfinite(x)):
            return 1.0, 0.0
        A = np.stack([x, np.ones_like(x)], axis=1)
        try:
            (scale, offset), *_ = np.linalg.lstsq(A, y, rcond=None)
        except np.linalg.LinAlgError:
            return 1.0, 0.0
        if not np.isfinite(scale) or not np.isfinite(offset) or scale <= 0:
            return 1.0, 0.0
        return float(scale), float(offset)

    # ------------------------------------------------------------------
    # 生成并落盘
    # ------------------------------------------------------------------
    def build_depth_dataset(self, image_paths, out_depth_dir, depth_params_path,
                            input_size=518, scale_align_fn=None):
        """遍历图像生成深度图, 写入 out_depth_dir; scale_align_fn(image_bgr, depth)
        可选地返回 (scale, offset) 用于尺度对齐 (否则 scale=1/offset=0)。
        返回 depth_params dict。
        """
        import cv2
        os.makedirs(out_depth_dir, exist_ok=True)
        depth_params = {}
        for imp in image_paths:
            name = os.path.splitext(os.path.basename(imp))[0]
            img = cv2.imread(imp, cv2.IMREAD_COLOR)
            if img is None:
                continue
            d = self.infer(img, input_size=input_size)
            d = np.maximum(d, 1e-6)
            # 归一化到 [0,1] 存 16-bit png
            d_norm = (d - d.min()) / (d.max() - d.min() + 1e-8)
            cv2.imwrite(os.path.join(out_depth_dir, f"{name}.png"),
                        (d_norm * 65535).astype(np.uint16))

            scale, offset = 1.0, 0.0
            if scale_align_fn is not None:
                scale, offset = scale_align_fn(img, d)
            depth_params[name] = {"scale": scale, "offset": offset}

        os.makedirs(os.path.dirname(depth_params_path), exist_ok=True)
        with open(depth_params_path, "w") as f:
            json.dump(depth_params, f, indent=2)
        return depth_params
