"""多 seed 方差 + dual 多场景实验。truck 4 方法 × 3 seed + dual 4 场景。
用 Python 编排 subprocess，带重试（间歇性 0xC0000409 崩溃）。
"""
import subprocess, os, sys, glob
import numpy as np, cv2

PY = r"D:\DSH_Project\AI_Projects\venv_gs\Scripts\python.exe"
GS = r"D:\DSH_Project\AI_Projects\gaussian-splatting"
SCENE = r"D:\DSH_Project\AI_Projects\data\smoke_truck"
OUT = r"D:\DSH_Project\AI_Projects\output"
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"

def run(args, timeout=3600):
    r = subprocess.run([PY] + args, cwd=GS, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=timeout)
    return r.stdout + r.stderr, r.returncode

def train_with_retry(scene, mdir, extra_flags, res="1", iters=3000, max_tries=3):
    """训练 + 返回是否成功。间歇性原生崩溃重跑。"""
    for attempt in range(1, max_tries + 1):
        cmd = ["train.py", "-s", scene, "-m", mdir, "-r", res, "--data_device", "cpu",
               "--iterations", str(iters), "--disable_viewer", "--test_iterations", "-1"] + extra_flags
        if attempt > 1:
            import shutil
            shutil.rmtree(mdir, ignore_errors=True)
        out, rc = run(cmd)
        if "Training complete" in out:
            return out, True
        print(f"  [retry {attempt}] rc={rc} (log tail: {out.strip()[-300:]})")
        sys.stdout.flush()
    return out, False

def psnr_of(model_dir, iters=3000):
    g = sorted(glob.glob(os.path.join(model_dir, "train", f"ours_{iters}", "gt", "*.png")))
    r = sorted(glob.glob(os.path.join(model_dir, "train", f"ours_{iters}", "renders", "*.png")))
    ps = []
    for a, b in zip(g, r):
        ga = cv2.imread(a).astype(np.float64); ra = cv2.imread(b).astype(np.float64)
        mse = np.mean((ga - ra) ** 2)
        ps.append(10 * np.log10(255.0**2 / mse) if mse > 0 else 99)
    return float(np.mean(ps))

def count_of(model_dir, iters=3000):
    ply = os.path.join(model_dir, "point_cloud", f"iteration_{iters}", "point_cloud.ply")
    with open(ply, "rb") as f:
        for line in f:
            if line.startswith(b"element vertex"):
                return int(line.split()[-1])
    return -1

def ply_exists(model_dir, iters=3000):
    return os.path.exists(os.path.join(model_dir, "point_cloud", f"iteration_{iters}", "point_cloud.ply"))

# ---------- Part 1: 多 seed 方差（truck -r 1, 3000 iter） ----------
methods = {
    "base": [],
    "m3": ["--dbgs_budget", "300000"],
    "m2": ["-d", "depths", "--depth_l1_weight_init", "0", "--depth_l1_weight_final", "0", "--dbgs_depth_edge"],
    "dual": ["--dbgs_dual_budget", "300000", "--dbgs_dual_lr", "0.05"],
}
results = {}
print("=== PART 1: multi-seed variance (truck -r 1) ===")
for seed in [0, 1, 2]:
    for name, flags in methods.items():
        tag = f"seed{seed}_{name}"
        mdir = os.path.join(OUT, tag)
        if ply_exists(mdir):
            print(f"[SKIP] {tag} (ply exists)"); sys.stdout.flush()
        else:
            out, ok = train_with_retry(SCENE, mdir, ["--seed", str(seed)] + flags)
            if not ok:
                print(f"[FAIL] {tag}: train incomplete"); sys.stdout.flush(); continue
        run(["render.py", "-m", mdir, "--iteration", "3000"])
        n = count_of(mdir); p = psnr_of(mdir)
        results[tag] = (n, round(p, 3))
        print(f"[OK] {tag}: {n} pts / {p:.3f} dB"); sys.stdout.flush()

print("\n=== SEED SWEEP SUMMARY ===")
for name in methods:
    keys = [f"seed{s}_{name}" for s in [0,1,2]]
    got = [results[k] for k in keys if k in results]
    if len(got) < 3:
        print(f"{name}: only {len(got)}/3 seeds completed")
        continue
    ns = [g[0] for g in got]; ps = [g[1] for g in got]
    print(f"{name}: points {ns} mean={np.mean(ns):.0f} std={np.std(ns):.0f} | "
          f"PSNR {ps} mean={np.mean(ps):.3f} std={np.std(ps):.3f}")

# ---------- Part 2: dual 多场景（4 场景, 300K budget） ----------
print("\n=== PART 2: dual multi-scene (4 scenes, 300K) ===")
dual_scenes = {
    "truck": (r"D:\DSH_Project\AI_Projects\data\smoke_truck", "1"),
    "train": (r"D:\DSH_Project\AI_Projects\data\tandt_train", "1"),
    "drjohnson": (r"D:\DSH_Project\AI_Projects\data\db_drjohnson", "1"),
    "room": (r"D:\DSH_Project\AI_Projects\data\mip360_room", "4"),
}
dual_results = {}
for sname, (spath, res) in dual_scenes.items():
    mdir = os.path.join(OUT, f"dual_ms_{sname}")
    if ply_exists(mdir):
        print(f"[SKIP] dual_ms_{sname} (ply exists)"); sys.stdout.flush()
        out = ""
    else:
        out, ok = train_with_retry(spath, mdir,
            ["--dbgs_dual_budget", "300000", "--dbgs_dual_lr", "0.05"], res=res)
        if not ok:
            print(f"[FAIL] dual_ms_{sname}"); sys.stdout.flush(); continue
    # 提取 λ 轨迹
    lambdas = [l for l in out.splitlines() if "[DUAL]" in l]
    run(["render.py", "-m", mdir, "--iteration", "3000"])
    n = count_of(mdir); p = psnr_of(mdir)
    dual_results[sname] = (n, round(p, 3))
    print(f"[OK] dual_ms_{sname}: {n} pts / {p:.3f} dB | {lambdas[-3:] if lambdas else 'no DUAL log'}")
    sys.stdout.flush()

print("\n=== DUAL MULTI-SCENE SUMMARY ===")
for sname, (n, p) in dual_results.items():
    print(f"{sname}: {n} pts / {p} dB")
print("VARIANCE_DONE")
