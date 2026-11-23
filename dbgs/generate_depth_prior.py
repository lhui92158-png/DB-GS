# -*- coding: utf-8 -*-
"""
DB-GS Module 1 CLI: 生成深度先验初始化点云 (带尺度对齐)。

单遍流程 (不经过 16-bit 深度图 roundtrip, 直接拟合尺度):
  1. 加载场景 (Colmap / Blender)。
  2. 加载 Depth-Anything-V2 (默认 vits, 4GB 可用)。
  3. 对每张训练图:
     a. DVA-V2 推理单目相对深度 d_mono (越大越远)。
     b. 把 SfM 稀疏点投影到该相机, 得到像素坐标与相机系 z。
     c. 最小二乘拟合逆深度仿射: 1/z_sfm = scale * (1/d_mono) + offset。
     d. 用对齐后的深度反投影到世界系, 采样图像颜色。
  4. 聚合所有视角点云 -> dbgs_init.ply, 供 train.py --dbgs_init_ply 使用。

用法:
    python dbgs/generate_depth_prior.py -s <scene> -o <scene>/dbgs_init.ply --encoder vits
"""

import argparse
import os
import sys

import numpy as np


def build_K_from_fov(W, H, fovx, fovy):
    fx = (W / 2.0) / np.tan(fovx / 2.0)
    fy = (H / 2.0) / np.tan(fovy / 2.0)
    return np.array([[fx, 0, W / 2.0], [0, fy, H / 2.0], [0, 0, 1.0]], dtype=np.float64)


def project_sfm_to_camera(sfm_pts, K, R, T, W, H):
    """把世界系 SfM 点投影到相机, 返回 (uu, vv, z_cam) 有效像素。"""
    R = np.asarray(R, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64).reshape(3, 1)
    # 注意: 3DGS 的 CameraInfo.R = qvec2rotmat(qvec)^T (即 camera-to-world)。
    # COLMAP 标准变换 x_cam = R_w2c @ X + tvec, 而 R_w2c = R^T, 故 x_cam = R.T @ X + T。
    cam = R.T @ sfm_pts.T + T  # (3, N)
    z = cam[2]
    valid = z > 1e-6
    cam = cam[:, valid]
    z = z[valid]
    if cam.shape[1] == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float64)
    px = K @ cam  # (3, M)
    u = px[0] / px[2]
    v = px[1] / px[2]
    ok = (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
    return np.round(u[ok]).astype(np.int64), np.round(v[ok]).astype(np.int64), z[ok]


def fit_affine_depth(depth_mono_at_pts, z_sfm):
    """拟合 z_sfm ≈ scale * depth_mono + offset (最小二乘, 排除离群)。返回 (scale, offset)。"""
    x = depth_mono_at_pts
    y = z_sfm
    finite = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x, y = x[finite], y[finite]
    if x.size < 50:
        return 1.0, 0.0
    # 裁剪离群 (2.5% - 97.5% 分位), 避免天空/远背景/零值污染拟合
    xl, xh = np.percentile(x, [2.5, 97.5])
    yl, yh = np.percentile(y, [2.5, 97.5])
    mask = (x >= xl) & (x <= xh) & (y >= yl) & (y <= yh)
    x, y = x[mask], y[mask]
    if x.size < 50:
        return 1.0, 0.0
    A = np.stack([x, np.ones_like(x)], axis=1)
    try:
        (scale, offset), *_ = np.linalg.lstsq(A, y, rcond=None)
    except np.linalg.LinAlgError:
        return 1.0, 0.0
    if not np.isfinite(scale) or not np.isfinite(offset) or scale <= 0:
        return 1.0, 0.0
    return float(scale), float(offset)


def backproject(depth_aligned, K, R, T, stride=2):
    """反投影对齐深度图 (相机系 z) 到世界系点云。返回 (pts, uu, vv)。"""
    H, W = depth_aligned.shape[:2]
    u = np.arange(0, W, stride)
    v = np.arange(0, H, stride)
    uu, vv = np.meshgrid(u, v)
    z = depth_aligned[vv, uu].astype(np.float64)
    valid = np.isfinite(z) & (z > 1e-3)
    uu, vv, z = uu[valid], vv[valid], z[valid]
    Kinv = np.linalg.inv(K.astype(np.float64))
    ones = np.ones_like(z)
    cam = np.stack([uu, vv, ones], axis=0)
    cam = Kinv @ cam
    cam = cam * z[None, :]
    R = np.asarray(R, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64).reshape(3, 1)
    # 注意: R 是 camera-to-world (3DGS 存储转置), 故 world = R @ (cam - T)
    world = R @ (cam - T)
    return world.T.astype(np.float32), uu, vv


def write_ply(path, xyz, rgb):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--source_path", required=True)
    ap.add_argument("--images", default="images")
    ap.add_argument("-o", "--out_ply", default=None)
    ap.add_argument("--encoder", default="vits", choices=["vits", "vitb", "vitl"])
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--input_size", type=int, default=518)
    ap.add_argument("--max_points_per_view", type=int, default=200_000)
    ap.add_argument("--subsample", type=int, default=None)
    args = ap.parse_args()

    repo = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gaussian-splatting")
    sys.path.insert(0, repo)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    import torch
    import cv2
    from scene.dataset_readers import sceneLoadTypeCallbacks
    from dbgs.depth_estimator import load_depth_anything_v2

    print(f"[DB-GS] 加载场景: {args.source_path}")
    if os.path.exists(os.path.join(args.source_path, "sparse")):
        scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, "", False, False)
    elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
        scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, False, "", False)
    else:
        raise SystemExit("无法识别场景类型")

    cam_infos = list(scene_info.train_cameras) + list(scene_info.test_cameras)
    print(f"[DB-GS] 相机数: {len(cam_infos)}")

    sfm_pts = None
    if scene_info.point_cloud is not None and len(scene_info.point_cloud.points) > 0:
        sfm_pts = np.asarray(scene_info.point_cloud.points, dtype=np.float64)
        print(f"[DB-GS] SfM 稀疏点: {sfm_pts.shape[0]} (用于尺度对齐)")

    model = load_depth_anything_v2(args.encoder, device=args.device)
    model.eval()

    all_pts, all_cols = [], []
    aligned_stats = []
    for ci, cam in enumerate(cam_infos):
        img = cv2.imread(cam.image_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"  [skip] 读图失败 {cam.image_name}")
            continue
        H, W = img.shape[:2]

        with torch.no_grad():
            d_mono = model.infer_image(img, input_size=args.input_size)  # (H,W) float32
        d_mono = np.asarray(d_mono, dtype=np.float32)
        if d_mono.shape[:2] != (H, W):
            d_mono = cv2.resize(d_mono, (W, H), interpolation=cv2.INTER_LINEAR)
        # DVA-V2 原始输出是 disparity(越大越近)，取倒数得深度(越大越远，与 z_sfm 同向)
        disp = np.maximum(d_mono, 1e-3)
        depth_mono = 1.0 / disp
        valid_disp = d_mono > 1e-2  # disparity 过小 = 天空/远背景/不可靠，掩掉

        K = build_K_from_fov(W, H, cam.FovX, cam.FovY)

        # 尺度对齐: 拟合 z_sfm ≈ scale * depth_mono + offset
        scale, offset = 1.0, 0.0
        if sfm_pts is not None:
            uu, vv, z_sfm = project_sfm_to_camera(sfm_pts, K, cam.R, cam.T, W, H)
            if uu.size >= 50:
                depth_at_pts = depth_mono[vv, uu]
                scale, offset = fit_affine_depth(depth_at_pts, z_sfm)
        depth_aligned = scale * depth_mono + offset
        depth_aligned[~valid_disp] = np.nan  # 掩掉不可靠区域
        aligned_stats.append((scale, offset))

        pts, puu, pvv = backproject(depth_aligned, K, cam.R, cam.T, stride=args.stride)
        if pts.shape[0] > args.max_points_per_view:
            idx = np.random.choice(pts.shape[0], args.max_points_per_view, replace=False)
            pts, puu, pvv = pts[idx], puu[idx], pvv[idx]
        all_pts.append(pts)

        cols = img[pvv, puu].astype(np.float32) / 255.0  # BGR
        cols = cols[:, ::-1]  # -> RGB
        all_cols.append(cols)
        if (ci + 1) % 25 == 0:
            print(f"  [{ci+1}/{len(cam_infos)}] scale={scale:.4f} offset={offset:.4f} pts={pts.shape[0]}")

    points = np.concatenate(all_pts, axis=0)
    colors = np.concatenate(all_cols, axis=0)
    if args.subsample is not None and points.shape[0] > args.subsample:
        idx = np.random.choice(points.shape[0], args.subsample, replace=False)
        points, colors = points[idx], colors[idx]

    out_ply = args.out_ply or os.path.join(args.source_path, "dbgs_init.ply")
    write_ply(out_ply, points, colors)
    scales = np.array([s for s, _ in aligned_stats])
    print(f"[DB-GS] 完成: {points.shape[0]} 点 -> {out_ply}")
    print(f"[DB-GS] scale 中位={np.median(scales):.4f} (min={scales.min():.4f} max={scales.max():.4f})")
    print(f"[DB-GS] 训练时加 --dbgs_init_ply {out_ply}")


if __name__ == "__main__":
    main()
