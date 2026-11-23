"""改进3(图像梯度门控对照) + 改进4(深度正则化) 训练→渲染→PSNR 全流程。

用法: python run_improvements.py
GPU 空闲时运行 (Pareto 扫描完成后)。
"""
import os, sys, subprocess, json, glob
import numpy as np

GS = r"D:\DSH_Project\AI_Projects\gaussian-splatting"
PY = r"D:\DSH_Project\AI_Projects\venv_gs\Scripts\python.exe"
SCENE = r"D:\DSH_Project\AI_Projects\data\smoke_truck"
DATA_DEV = "cpu"

CONFIGS = [
    # (输出名, 额外参数列表)
    ("tandt_truck_img_edge", ["--dbgs_image_edge"]),
    ("tandt_truck_depth_reg", ["-d", "depths"]),
]

def run(cmd):
    r = subprocess.run(cmd, cwd=GS, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.returncode, r.stdout, r.stderr

def npoints(ply):
    if not os.path.exists(ply):
        return None
    # 读 ply 头部 element vertex
    with open(ply, "rb") as f:
        head = f.read(200000).decode("latin1")
    for line in head.split("\n"):
        if line.startswith("element vertex "):
            return int(line.split()[-1])
    return None

def psnr_dir(renders, gt):
    rs = sorted(glob.glob(os.path.join(renders, "*.png")))
    gs = sorted(glob.glob(os.path.join(gt, "*.png")))
    if not rs or len(rs) != len(gs):
        return None
    import cv2
    vals = []
    for rp, gp in zip(rs, gs):
        a = cv2.imread(rp).astype(np.float64)
        b = cv2.imread(gp).astype(np.float64)
        mse = np.mean((a - b) ** 2)
        if mse < 1e-10:
            vals.append(100.0)
        else:
            vals.append(10 * np.log10(255.0 ** 2 / mse))
    return float(np.mean(vals))

for name, extra in CONFIGS:
    m = os.path.join(r"D:\DSH_Project\AI_Projects\output", name)
    print(f"=== TRAIN {name} ===", flush=True)
    cmd = [PY, "train.py", "-s", SCENE, "-m", m, "-r", "1", "--data_device", DATA_DEV,
           "--iterations", "3000", "--disable_viewer", "--test_iterations", "-1"] + extra
    rc, so, se = run(cmd)
    print(f"train rc={rc}", flush=True)
    if rc != 0:
        print(se[-3000:], flush=True)
        print(f"RESULT {name} TRAIN_FAIL", flush=True)
        continue
    # render
    print(f"=== RENDER {name} ===", flush=True)
    rc, so, se = run([PY, "render.py", "-m", m, "--iteration", "3000"])
    print(f"render rc={rc}", flush=True)
    if rc != 0:
        print(se[-3000:], flush=True)
        print(f"RESULT {name} RENDER_FAIL", flush=True)
        continue
    ply = os.path.join(m, "point_cloud", "iteration_3000", "point_cloud.ply")
    np_ = npoints(ply)
    p = psnr_dir(os.path.join(m, "train", "ours_3000", "renders"),
                 os.path.join(m, "train", "ours_3000", "gt"))
    print(f"RESULT {name} npts={np_} PSNR={p:.3f}" if p is not None else f"RESULT {name} npts={np_} PSNR=None", flush=True)

print("IMPROVEMENTS_DONE", flush=True)
