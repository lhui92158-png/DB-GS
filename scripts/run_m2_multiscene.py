import subprocess, os, glob
import numpy as np
from PIL import Image

PY = r"D:\DSH_Project\AI_Projects\venv_gs\Scripts\python.exe"
ROOT = r"D:\DSH_Project\AI_Projects"
GS = os.path.join(ROOT, "gaussian-splatting")

# 3 small scenes needing depth + M2 (truck M2 already done: 408,698 / 25.977)
SCENES = [
    ("data/tandt_train",   "tandt_train",   1),
    ("data/db_drjohnson",  "db_drjohnson",  1),
    ("data/db_playroom",   "db_playroom",   1),
]

def gen_depth(rel):
    s = os.path.join(ROOT, rel)
    print(f"DEPTH {rel}", flush=True)
    r = subprocess.run([PY, "dbgs/generate_depths.py", "-s", s],
                       cwd=ROOT, capture_output=True, text=True, errors="replace")
    print(r.stdout[-500:] if r.stdout else "", flush=True)
    print(r.stderr[-500:] if r.stderr else "", flush=True)
    if r.returncode != 0:
        print(f"DEPTH_FAIL {rel} rc={r.returncode}", flush=True)
        return False
    return True

def train_m2(rel, name, res):
    s = os.path.join(ROOT, rel)
    m = os.path.join(ROOT, "output", f"{name}_m2")
    args = [PY, "train.py", "-s", s, "-m", m, "-r", str(res),
            "--data_device", "cpu", "--iterations", "3000",
            "--disable_viewer", "--test_iterations", "-1",
            "-d", "depths", "--depth_l1_weight_init", "0",
            "--depth_l1_weight_final", "0", "--dbgs_depth_edge"]
    print(f"TRAIN_M2 {name}", flush=True)
    r = subprocess.run(args, cwd=GS, capture_output=True, text=True, errors="replace")
    if r.returncode != 0:
        print(f"FAIL {name} rc={r.returncode}", flush=True)
        print((r.stdout+r.stderr)[-1500:], flush=True)
        return
    subprocess.run([PY, "render.py", "-m", m, "--iteration", "3000"],
                   cwd=GS, capture_output=True)
    ply = os.path.join(m, "point_cloud", "iteration_3000", "point_cloud.ply")
    npts = -1
    with open(ply, "rb") as f:
        for line in f.read(4096).decode("latin1").splitlines():
            if line.startswith("element vertex"):
                npts = int(line.split()[-1])
    rd = os.path.join(m, "train", "ours_3000", "renders")
    gt = os.path.join(m, "train", "ours_3000", "gt")
    rs = sorted(glob.glob(os.path.join(rd, "*.png")))
    gs = sorted(glob.glob(os.path.join(gt, "*.png")))
    n = min(len(rs), len(gs)); tot = 0.0
    for i in range(n):
        a = np.asarray(Image.open(rs[i]).convert("RGB"), dtype=np.float64)/255.0
        b = np.asarray(Image.open(gs[i]).convert("RGB"), dtype=np.float64)/255.0
        tot += 10*np.log10(1.0/max(((a-b)**2).mean(), 1e-12))
    print(f"OK_M2 {name} npts={npts} PSNR={tot/n:.3f}", flush=True)

if __name__ == "__main__":
    for rel, name, res in SCENES:
        if gen_depth(rel):
            train_m2(rel, name, res)
    print("M2_MULTI_DONE", flush=True)
