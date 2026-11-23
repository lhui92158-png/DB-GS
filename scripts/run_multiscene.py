#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""多场景复验：baseline vs 模块3(预算剪枝)，train -> render -> PSNR + 高斯数。"""
import subprocess, os, sys, glob

PY = r"D:\DSH_Project\AI_Projects\venv_gs\Scripts\python.exe"
GS = r"D:\DSH_Project\AI_Projects\gaussian-splatting"
ROOT = r"D:\DSH_Project\AI_Projects"

# (数据相对路径, 输出名, 分辨率 -r)
SCENES = [
    ("data/tandt_train",    "tandt_train",    1),
    ("data/smoke_truck",    "tandt_truck",    1),
    ("data/db_drjohnson",   "db_drjohnson",   1),
    ("data/db_playroom",    "db_playroom",    1),
    ("data/mip360_bicycle", "mip360_bicycle", 4),
    ("data/mip360_bonsai",  "mip360_bonsai",  4),
    ("data/mip360_counter", "mip360_counter", 4),
    ("data/mip360_garden",  "mip360_garden",  4),
    ("data/mip360_kitchen", "mip360_kitchen", 4),
    ("data/mip360_room",    "mip360_room",    4),
    ("data/mip360_stump",   "mip360_stump",   4),
]

CONFIGS = [
    ("base", []),
    ("m3",   ["--dbgs_budget", "300000"]),
]

ITERS = 3000


def run(cmd, log, cwd):
    with open(log, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 60 + "\n$ " + " ".join(cmd) + "\n")
        f.flush()
        p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True,
                             encoding="utf-8", errors="replace")
        for line in p.stdout:
            f.write(line)
            f.flush()
        p.wait()
        return p.returncode


def train(scene_rel, name, res, config_name, extra):
    model = os.path.join(ROOT, "output", f"{name}_{config_name}")
    scene = os.path.join(ROOT, scene_rel)
    cmd = [PY, "train.py", "-s", scene, "-m", model,
           "-r", str(res), "--data_device", "cpu",
           "--iterations", str(ITERS), "--disable_viewer",
           "--test_iterations", "-1"] + extra
    log = os.path.join(ROOT, "output", f"{name}_{config_name}.train.log")
    rc = run(cmd, log, GS)
    return model, rc


def render(model):
    cmd = [PY, "render.py", "-m", model, "--iteration", str(ITERS)]
    log = model + ".render.log"
    rc = run(cmd, log, GS)
    return rc


def count_gaussians(model):
    ply = os.path.join(model, "point_cloud", f"iteration_{ITERS}", "point_cloud.ply")
    if not os.path.exists(ply):
        return -1
    with open(ply, "rb") as f:
        head = f.read(512).decode("latin1")
    for line in head.splitlines():
        if line.startswith("element vertex"):
            return int(line.split()[-1])
    return -1


def compute_psnr(model):
    rd = os.path.join(model, "train", f"ours_{ITERS}", "renders")
    gt = os.path.join(model, "train", f"ours_{ITERS}", "gt")
    if not os.path.isdir(rd):
        return -1.0
    try:
        import numpy as np
        from PIL import Image
    except Exception as e:
        return -2.0
    rs = sorted(glob.glob(os.path.join(rd, "*.png")))
    gs = sorted(glob.glob(os.path.join(gt, "*.png")))
    n = min(len(rs), len(gs))
    if n == 0:
        return -3.0
    tot = 0.0
    for i in range(n):
        a = np.asarray(Image.open(rs[i]).convert("RGB"), dtype=np.float64) / 255.0
        b = np.asarray(Image.open(gs[i]).convert("RGB"), dtype=np.float64) / 255.0
        mse = ((a - b) ** 2).mean()
        tot += 10 * np.log10(1.0 / max(mse, 1e-12)) if mse > 0 else 100.0
    return tot / n


def main():
    results = []
    for scene_rel, name, res in SCENES:
        for config_name, extra in CONFIGS:
            key = f"{name}_{config_name}"
            print(f"===== TRAIN {key} (r={res}) =====", flush=True)
            model, rc = train(scene_rel, name, res, config_name, extra)
            if rc != 0:
                print(f"[FAIL] {key} train rc={rc}", flush=True)
                results.append((key, "train_fail", rc, 0, 0.0))
                continue
            rc2 = render(model)
            if rc2 != 0:
                print(f"[FAIL] {key} render rc={rc2}", flush=True)
                results.append((key, "render_fail", rc2, count_gaussians(model), 0.0))
                continue
            npts = count_gaussians(model)
            ps = compute_psnr(model)
            results.append((key, "ok", 0, npts, ps))
            print(f"[OK] {key}: gaussians={npts} PSNR={ps:.3f} dB", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("MULTISCENE RESULTS (3000 iters)", flush=True)
    print(f"{'config':<22} {'status':<12} {'gaussians':>10} {'PSNR':>8}", flush=True)
    for key, status, rc, npts, ps in results:
        print(f"{key:<22} {status:<12} {npts:>10} {ps:>8.3f}", flush=True)
    print("MULTISCENE_DONE", flush=True)


if __name__ == "__main__":
    main()
