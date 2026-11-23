#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""50% 比例预算实验 + 峰值显存测量。"""
import subprocess, os, glob, re

PY = r"D:\DSH_Project\AI_Projects\venv_gs\Scripts\python.exe"
GS = r"D:\DSH_Project\AI_Projects\gaussian-splatting"
ROOT = r"D:\DSH_Project\AI_Projects"
ITERS = 3000

# (数据相对路径, 输出名, -r, 50%预算)  预算=自然点数×0.5
SCENES = [
    ("data/tandt_train",    "tandt_train",    1, 235000),
    ("data/smoke_truck",    "tandt_truck",    1, 524000),
    ("data/db_drjohnson",   "db_drjohnson",   1, 408000),
    ("data/db_playroom",    "db_playroom",    1, 375000),
    ("data/mip360_bicycle", "mip360_bicycle", 4, 628000),
    ("data/mip360_bonsai",  "mip360_bonsai",  4, 451000),
    ("data/mip360_counter", "mip360_counter", 4, 327000),
    ("data/mip360_garden",  "mip360_garden",  4, 882000),
    ("data/mip360_kitchen", "mip360_kitchen", 4, 456000),
    ("data/mip360_room",    "mip360_room",    4, 331000),
    ("data/mip360_stump",   "mip360_stump",   4, 645000),
]

# 重跑 3 个 baseline 测峰值显存（小/中/大 各一）
BASE_RESAMPLE = [
    ("data/tandt_train",   "tandt_train_vram",   1),
    ("data/mip360_room",   "mip360_room_vram",   4),
    ("data/mip360_garden", "mip360_garden_vram", 4),
]


def run(cmd, log):
    with open(log, "a", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n")
        f.flush()
        p = subprocess.Popen(cmd, cwd=GS, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True,
                             encoding="utf-8", errors="replace")
        for line in p.stdout:
            f.write(line); f.flush()
        p.wait()
        return p.returncode


def train(name, scene_rel, res, extra):
    model = os.path.join(ROOT, "output", name)
    scene = os.path.join(ROOT, scene_rel)
    cmd = [PY, "train.py", "-s", scene, "-m", model, "-r", str(res),
           "--data_device", "cpu", "--iterations", str(ITERS),
           "--disable_viewer", "--test_iterations", "-1"] + extra
    log = os.path.join(ROOT, "output", name + ".train.log")
    rc = run(cmd, log)
    # 提取 VRAM
    vram = ""
    if os.path.exists(log):
        txt = open(log, encoding="utf-8", errors="replace").read()
        m = re.search(r"\[VRAM\] peak_allocated=([\d.]+) GB  peak_reserved=([\d.]+) GB", txt)
        if m:
            vram = f"{m.group(1)}/{m.group(2)}"
    return model, rc, vram


def render(model):
    rc = run([PY, "render.py", "-m", model, "--iteration", str(ITERS)],
             model + ".render.log")
    return rc


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


def main():
    print("=== PHASE 1: 50% 比例预算 (11 场景) ===", flush=True)
    for scene_rel, name, res, budget in SCENES:
        print(f"TRAIN {name} budget={budget}", flush=True)
        model, rc, vram = train(name, scene_rel, res, ["--dbgs_budget", str(budget)])
        if rc != 0:
            print(f"  FAIL rc={rc}", flush=True); continue
        render(model)
        npts, ps = count_psnr(model)
        print(f"  OK npts={npts} PSNR={ps:.3f} VRAM(alloc/resv)={vram}", flush=True)

    print("=== PHASE 2: 3 个 baseline 测峰值显存 ===", flush=True)
    for scene_rel, name, res in BASE_RESAMPLE:
        print(f"TRAIN {name} (baseline)", flush=True)
        model, rc, vram = train(name, scene_rel, res, [])
        if rc != 0:
            print(f"  FAIL rc={rc}", flush=True); continue
        render(model)
        npts, ps = count_psnr(model)
        print(f"  OK npts={npts} PSNR={ps:.3f} VRAM(alloc/resv)={vram}", flush=True)

    print("RATIO_DONE", flush=True)


if __name__ == "__main__":
    main()
