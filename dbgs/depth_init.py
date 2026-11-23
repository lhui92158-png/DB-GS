# -*- coding: utf-8 -*-
"""
DB-GS Module 1: 深度先验初始化 (Depth-Prior Initialization)。

把单目深度图反投影为稠密 3D 点云, 用作高斯初始化, 替代/增强 SfM 稀疏点云。
动机: SfM 点云稀疏(每场景仅 ~1k-10k 点), 高斯需从稀疏点靠梯度稠密化爬坡, 既慢又易
过密化; 深度反投影给出结构感知的稠密点云(~每图 10k-100k 点), 初始化更接近真实
表面 -> 更少迭代 + 更少高斯即可收敛 -> 训练时显存/时间双降 (DB-GS 核心卖点)。

约定 (与官方 scene/cameras.py 一致):
  - 逆深度对齐: inv_depth_aligned = scale * inv_depth_mono + offset
  - 相机外参: w2c = [R|T], P_cam = R @ P_world + T  =>  P_world = R^T @ (P_cam - T)
  - 内参 K: [u,v,1]^T = K @ [X_c/Z_c, Y_c/Z_c, 1]^T
"""

import numpy as np


def backproject_depth_to_pcd(depth_aligned, K, R, T, stride=1, min_depth=1e-3, max_depth=None,
                             return_uv=False):
    """反投影对齐后的深度图到世界坐标点云。

    Args:
        depth_aligned: (H, W) float32, 对齐后的**深度**(相机系 z, 场景尺度)。越小越近。
        K: (3,3) 内参矩阵 (像素单位)。
        R, T: 相机外参 w2c=[R|T], R:(3,3), T:(3,1) 或 (3,)。P_cam = R@P_world + T。
        stride: 像素采样步长 (1=全密度, 2=1/4, 4=1/16)。
        min_depth/max_depth: 深度截断。
        return_uv: True 时额外返回有效点的像素坐标 (uu, vv), 用于颜色对齐。
    Returns:
        pts: (N, 3) float32 世界坐标点云; 若 return_uv 则返回 (pts, uu, vv)。
    """
    H, W = depth_aligned.shape[:2]
    u = np.arange(0, W, stride)
    v = np.arange(0, H, stride)
    uu, vv = np.meshgrid(u, v)  # (H', W')
    z = depth_aligned[vv, uu].astype(np.float64)

    valid = np.isfinite(z) & (z > min_depth)
    if max_depth is not None:
        valid &= (z < max_depth)
    uu, vv, z = uu[valid], vv[valid], z[valid]

    # 相机系坐标: [Xc, Yc, Zc] = Zc * K^-1 @ [u, v, 1]
    Kinv = np.linalg.inv(K.astype(np.float64))
    ones = np.ones_like(z)
    cam = np.stack([uu, vv, ones], axis=0)  # (3, N)
    cam = Kinv @ cam                          # 归一化相机坐标
    cam = cam * z[None, :]                    # (3, N) 相机系

    # 世界坐标: 3DGS 的 R = qvec2rotmat(qvec)^T (camera-to-world),
    # 故 P_cam = R^T @ P_world + T  =>  P_world = R @ (P_cam - T)
    R = np.asarray(R, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64).reshape(3, 1)
    world = R @ (cam - T)
    pts = world.T.astype(np.float32)  # (N, 3)
    if return_uv:
        return pts, uu, vv
    return pts


def build_depth_prior_ply(cam_infos, depth_folder, depth_params, out_ply,
                          stride=2, max_points_per_view=200_000, subsample=None):
    """聚合所有训练相机的深度反投影点云, 写一个 .ply 供 create_from_pcd 使用。

    Args:
        cam_infos: list[CameraInfo], 需含 image_name / image_path / width / height /
                   FovX/FovY/R/T, 以及 depth_params 对齐参数 (或传入 depth_params)。
        depth_folder: 深度图目录 (<name>.png)。
        depth_params: {image_name: {"scale":..,"offset":..}}。
        out_ply: 输出 ply 路径。
        stride: 反投影采样步长。
        max_points_per_view: 每视角点数上限 (防爆内存)。
        subsample: 全局点数上限 (None=不限制)。超出时均匀降采样。
    Returns:
        (points, colors) numpy 数组; 同时写 ply。
    """
    import cv2
    import os

    all_pts, all_cols = [], []
    Kinv_cache = {}
    for cam in cam_infos:
        name = os.path.splitext(os.path.basename(cam.image_name))[0]
        dp = depth_params.get(name, {"scale": 1.0, "offset": 0.0})
        dpath = os.path.join(depth_folder, f"{name}.png")
        if not os.path.exists(dpath):
            continue
        d16 = cv2.imread(dpath, cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535.0  # 归一化
        inv_d_mono = 1.0 / np.maximum(d16, 1e-6)
        inv_d_aligned = dp["scale"] * inv_d_mono + dp["offset"]
        depth_aligned = 1.0 / np.maximum(inv_d_aligned, 1e-8)  # 相机系 z

        # 从 Fov 重建 K (官方 CameraInfo 用 FoV, 无显式 K; 见 utils/graphics_utils.getProjectionMatrix)
        W, H = cam.width, cam.height
        if name not in Kinv_cache:
            fx = (W / 2.0) / np.tan(cam.FovX / 2.0)
            fy = (H / 2.0) / np.tan(cam.FovY / 2.0)
            K = np.array([[fx, 0, W / 2.0], [0, fy, H / 2.0], [0, 0, 1.0]], dtype=np.float64)
            Kinv_cache[name] = K
        K = Kinv_cache[name]

        # 深度图需 resize 到原分辨率 (若深度图与图像同尺寸则跳过)
        if depth_aligned.shape[:2] != (H, W):
            depth_aligned = cv2.resize(depth_aligned, (W, H), interpolation=cv2.INTER_LINEAR)

        pts, uu, vv = backproject_depth_to_pcd(depth_aligned, K, cam.R, cam.T,
                                               stride=stride, return_uv=True)
        if pts.shape[0] > max_points_per_view:
            idx = np.random.choice(pts.shape[0], max_points_per_view, replace=False)
            pts, uu, vv = pts[idx], uu[idx], vv[idx]
        all_pts.append(pts)

        img = cv2.imread(cam.image_path, cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.resize(img, (W, H)) if img.shape[:2] != (H, W) else img
            # 与反投影有效点逐像素对齐的颜色
            cols = img[vv, uu]  # (N, 3) BGR
            cols = cols[:, ::-1].astype(np.float32) / 255.0  # BGR -> RGB
        else:
            cols = np.full((pts.shape[0], 3), 0.5, dtype=np.float32)
        all_cols.append(cols)

    points = np.concatenate(all_pts, axis=0)
    colors = np.concatenate(all_cols, axis=0)
    if subsample is not None and points.shape[0] > subsample:
        idx = np.random.choice(points.shape[0], subsample, replace=False)
        points, colors = points[idx], colors[idx]

    _write_ply(out_ply, points, colors)
    return points, colors


def _write_ply(path, xyz, rgb):
    """写标准顶点 ply (与 SfM points3D 同格式, 供 fetchPly 读取)。"""
    import struct
    n = xyz.shape[0]
    with open(path, "wb") as f:
        f.write(b"ply\nformat binary_little_endian 1.0\n")
        f.write(f"element vertex {n}\n".encode())
        f.write(b"property float x\nproperty float y\nproperty float z\n")
        f.write(b"property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for i in range(n):
            f.write(struct.pack("<fffBBB", xyz[i, 0], xyz[i, 1], xyz[i, 2],
                                int(np.clip(rgb[i, 0] * 255, 0, 255)),
                                int(np.clip(rgb[i, 1] * 255, 0, 255)),
                                int(np.clip(rgb[i, 2] * 255, 0, 255))))
