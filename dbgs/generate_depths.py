# -*- coding: utf-8 -*-
"""生成官方 3DGS 兼容的单目深度文件: depths/<name>.png (16-bit disparity) + depth_params.json。

DVA-V2 输出 disparity (逆深度, 越大越近)。官方管线 (utils/make_depth_scale.py) 约定:
  - depths/<name>.png 存 16-bit disparity, loadCam 读入后 / 2**16 归一化到 [0,1)。
  - depth_params.json = {name: {scale, offset}}, 对齐 metric 逆深度 1/z ≈ scale*disp + offset。
本脚本用 SfM 稀疏点做 median/MAD 鲁棒仿射对齐 (与官方 make_depth_scale.py 一致)。
"""
import os, sys, json, argparse
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gaussian-splatting"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def build_K_from_fov(W, H, fovx, fovy):
    fx = (W / 2.0) / np.tan(fovx / 2.0)
    fy = (H / 2.0) / np.tan(fovy / 2.0)
    return np.array([[fx, 0, W / 2.0], [0, fy, H / 2.0], [0, 0, 1.0]], dtype=np.float64)


def project_sfm_to_camera(sfm_pts, K, R, T, W, H):
    R = np.asarray(R, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64).reshape(3, 1)
    cam = R.T @ sfm_pts.T + T  # 3DGS 约定: R = w2c^T, 故 x_cam = R.T @ X + T
    z = cam[2]
    valid = z > 1e-6
    cam, z = cam[:, valid], z[valid]
    if cam.shape[1] == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float64)
    px = K @ cam
    u, v = px[0] / px[2], px[1] / px[2]
    ok = (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
    return np.round(u[ok]).astype(np.int64), np.round(v[ok]).astype(np.int64), z[ok]


def fit_inv_depth_median(disp_at_pts, z_sfm):
    """拟合 metric 逆深度 1/z ≈ scale * disp + offset (median/MAD 鲁棒, 与官方一致)。"""
    inv_z = 1.0 / np.maximum(z_sfm, 1e-6)
    finite = np.isfinite(disp_at_pts) & np.isfinite(inv_z) & (disp_at_pts > 0) & (inv_z > 0)
    disp_at_pts, inv_z = disp_at_pts[finite], inv_z[finite]
    if disp_at_pts.size < 50:
        return 1.0, 0.0
    t_colmap = np.median(inv_z)
    s_colmap = np.mean(np.abs(inv_z - t_colmap))
    t_mono = np.median(disp_at_pts)
    s_mono = np.mean(np.abs(disp_at_pts - t_mono))
    if s_mono < 1e-12:
        return 1.0, 0.0
    scale = s_colmap / s_mono
    offset = t_colmap - t_mono * scale
    if not np.isfinite(scale) or not np.isfinite(offset) or scale <= 0:
        return 1.0, 0.0
    return float(scale), float(offset)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--source_path", required=True)
    ap.add_argument("--images", default="images")
    ap.add_argument("--encoder", default="vits", choices=["vits", "vitb", "vitl"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--input_size", type=int, default=518)
    ap.add_argument("--depths_dir", default="depths")
    args = ap.parse_args()

    import torch
    from scene.dataset_readers import sceneLoadTypeCallbacks
    from dbgs.depth_estimator import load_depth_anything_v2

    print(f"[DB-GS] 加载场景: {args.source_path}")
    if os.path.exists(os.path.join(args.source_path, "sparse")):
        scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, "", False, False)
    else:
        raise SystemExit("需要 COLMAP 场景 (sparse/)")

    cam_infos = list(scene_info.train_cameras) + list(scene_info.test_cameras)
    sfm_pts = None
    if scene_info.point_cloud is not None and len(scene_info.point_cloud.points) > 0:
        sfm_pts = np.asarray(scene_info.point_cloud.points, dtype=np.float64)

    model = load_depth_anything_v2(args.encoder, device=args.device)
    model.eval()

    depths_dir = os.path.join(args.source_path, args.depths_dir)
    os.makedirs(depths_dir, exist_ok=True)
    depth_params = {}

    for ci, cam in enumerate(cam_infos):
        img = cv2.imread(cam.image_path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        H, W = img.shape[:2]
        with torch.no_grad():
            d_mono = model.infer_image(img, input_size=args.input_size)
        d_mono = np.asarray(d_mono, dtype=np.float32)
        if d_mono.shape[:2] != (H, W):
            d_mono = cv2.resize(d_mono, (W, H), interpolation=cv2.INTER_LINEAR)
        disp = np.maximum(d_mono, 0.0)

        # 归一化 disparity 到 [0,1] 用 99.5 分位, 保证 16-bit 量化精度
        pmax = np.percentile(disp[disp > 0], 99.5) if (disp > 0).sum() > 0 else 1.0
        pmax = max(pmax, 1e-6)
        disp_norm = np.clip(disp / pmax, 0.0, 1.0)

        # 保存 16-bit png
        name = os.path.splitext(cam.image_name)[0]
        png16 = (disp_norm * 65535.0).astype(np.uint16)
        cv2.imwrite(os.path.join(depths_dir, f"{name}.png"), png16)

        # 仿射对齐: 1/z ≈ scale * disp_norm + offset
        scale, offset = 1.0, 0.0
        if sfm_pts is not None:
            K = build_K_from_fov(W, H, cam.FovX, cam.FovY)
            uu, vv, z_sfm = project_sfm_to_camera(sfm_pts, K, cam.R, cam.T, W, H)
            if uu.size >= 50:
                scale, offset = fit_inv_depth_median(disp_norm[vv, uu], z_sfm)
        depth_params[name] = {"scale": scale, "offset": offset}
        if (ci + 1) % 25 == 0:
            print(f"  [{ci+1}/{len(cam_infos)}] scale={scale:.4f} offset={offset:.4f}")

    with open(os.path.join(args.source_path, "sparse", "0", "depth_params.json"), "w") as f:
        json.dump(depth_params, f, indent=2)
    print(f"[DB-GS] 完成: {len(depth_params)} 张深度 -> {depths_dir}")


if __name__ == "__main__":
    main()
