#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""方向1 对偶上升软剪枝 vs 硬预算对照实验（Lagrangian dual ascent ablation）。

对照：--dbgs_budget(硬预算, 0.9 soft margin) vs --dbgs_dual_budget(对偶软剪枝, λ自动调节)。
均 3000 iter、-r 1/4、--data_device cpu。硬预算结果已有(见 多场景实验结果.md)。
"""
import subprocess, os, glob, re

PY = r"D:\DSH_Project\AI_Projects\venv_gs\Scripts\python.exe"
GS = r"D:\DSH_Project\AI_Projects\gaussian-splatting"
ROOT = r"D:\DSH_Project\AI_Projects"
ITERS = 3000
BUDGET = 300000
DUAL_LR = 0.05

# 选 4 个场景（覆盖小/中/大）：与硬预算 300K 的结果对比
SCENES = [
    ("data/smoke_truck",  "dual_truck",    1),
    ("data/db_drjohnson", "dual_drjohnson", 1),
    ("data/tandt_train",  "dual_train",    1),
    ("data/mip360_room",  "dual_room",     4),
]


def run(cmd, log):
    with open(log, "a", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n"); f.flush()
        p = subprocess.Popen(cmd, cwd=GS, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True,
                             encoding="utf-8", errors="replace")
        for line in p.stdout:
            f.write(line); f.flush()
        p.wait()
        return p.returncode


def count_psnr(model):
    ply = os.path.join(model, "point_cloud", f"iteration_{ITERS}", "point_cloud.ply")
    npts = -1
    if os.path.exists(ply):
        with open(ply, "rb") as f:
            head = f.read(512).decode("latin1")
        for line in head.splitlines():
            if line.startswith("element vertex"):
                npts = int(line.split()[-1]); break
    rd = os.path.join(model, "train", f"ours_{ITERS}", "renders")
    gt = os.path.join(model, "train", f"ours_{ITERS}", "gt")
    ps = -1.0
    if os.path.isdir(rd):
        import numpy as np
        from PIL import Image
        rs = sorted(glob.glob(os.path.join(rd, "*.png")))
        gs = sorted(glob.glob(os.path.join(gt, "*.png")))
        n = min(len(rs), len(gs))
        tot = 0.0
        for i in range(n):
            a = np.asarray(Image.open(rs[i]).convert("RGB"), dtype=np.float64) / 255.0
            b = np.asarray(Image.open(gs[i]).convert("RGB"), dtype=np.float64) / 255.0
            mse = ((a - b) ** 2).mean()
            tot += 10 * np.log10(1.0 / max(mse, 1e-12))
        ps = tot / n
    return npts, ps


def extract_lambda_traj(log):
    """从训练日志提取 [DUAL] iter=... lambda=... n_gauss=... 序列。"""
    traj = []
    if os.path.exists(log):
        txt = open(log, encoding="utf-8", errors="replace").read()
        for m in re.finditer(r"\[DUAL\] iter=(\d+) lambda=([\d.]+) n_gauss=(\d+)", txt):
            traj.append((int(m.group(1)), float(m.group(2)), int(m.group(3))))
    return traj


def main():
    print("=== 方向1 对偶上升 vs 硬预算 对照 ===", flush=True)
    for scene_rel, name, res in SCENES:
        model = os.path.join(ROOT, "output", name)
        scene = os.path.join(ROOT, scene_rel)
        cmd = [PY, "train.py", "-s", scene, "-m", model, "-r", str(res),
               "--data_device", "cpu", "--iterations", str(ITERS),
               "--disable_viewer", "--test_iterations", "-1",
               "--dbgs_dual_budget", str(BUDGET), "--dbgs_dual_lr", str(DUAL_LR)]
        log = os.path.join(ROOT, "output", name + ".train.log")
        print(f"TRAIN {name} dual_budget={BUDGET}", flush=True)
        rc = run(cmd, log)
        if rc != 0:
            print(f"  FAIL rc={rc}", flush=True); continue
        run([PY, "render.py", "-m", model, "--iteration", str(ITERS)],
            model + ".render.log")
        npts, ps = count_psnr(model)
        traj = extract_lambda_traj(log)
        lam_end = traj[-1][1] if traj else -1
        lam_start = traj[0][1] if traj else -1
        print(f"  OK npts={npts} PSNR={ps:.3f} lambda: {lam_start:.3f}->{lam_end:.3f} "
              f"(traj {len(traj)} 点)", flush=True)
    print("DUAL_DONE", flush=True)


if __name__ == "__main__":
    main()
