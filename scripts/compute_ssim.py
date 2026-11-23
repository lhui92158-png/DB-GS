import os, glob
import numpy as np
import cv2

OUT = r"D:\DSH_Project\AI_Projects\output"

MODELS = [
    ("baseline(SfM)", "tandt_truck_base", 1048564, 23.50),
    ("M2 depth-gate", "truck_m2_r1", 820125, 23.47),
    ("M3 budget300K", "truck_m3_grad2", 269992, 21.76),
    ("M2+M3",         "truck_m23_r1", 269997, 21.06),
    ("M1 depth-init", "truck_m1_r1", 831262, 21.48),
    ("M1+M3",         "truck_m13_r1", 269996, 20.97),
    ("dual300K",      "truck_dual_grad", 261810, 21.83),
]

def ssim_gray(a, b):
    a = a.astype(np.float32); b = b.astype(np.float32)
    C1 = (0.01*255.0)**2; C2 = (0.03*255.0)**2
    mu_a = cv2.GaussianBlur(a, (11,11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11,11), 1.5)
    mu_a2 = mu_a*mu_a; mu_b2 = mu_b*mu_b; mu_ab = mu_a*mu_b
    sig_a2 = cv2.GaussianBlur(a*a, (11,11), 1.5) - mu_a2
    sig_b2 = cv2.GaussianBlur(b*b, (11,11), 1.5) - mu_b2
    sig_ab = cv2.GaussianBlur(a*b, (11,11), 1.5) - mu_ab
    m = ((2*mu_ab+C1)*(2*sig_ab+C2)) / ((mu_a2+mu_b2+C1)*(sig_a2+sig_b2+C2))
    return float(m.mean())

def psnr_rgb(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    mse = ((a-b)**2).mean()
    if mse == 0: return 100.0
    return 10*np.log10(255.0**2/mse)

for label, d, npts, psnr_ref in MODELS:
    rdir = os.path.join(OUT, d, "train", "ours_3000", "renders")
    gdir = os.path.join(OUT, d, "train", "ours_3000", "gt")
    if not os.path.isdir(rdir):
        print(f"[SKIP] {label}: {rdir}"); continue
    rfiles = sorted(glob.glob(os.path.join(rdir, "*.png")))
    gfiles = sorted(glob.glob(os.path.join(gdir, "*.png")))
    if len(rfiles) == 0 or len(rfiles) != len(gfiles):
        print(f"[WARN] {label}: r={len(rfiles)} g={len(gfiles)}"); continue
    ps, ss = [], []
    for rf, gf in zip(rfiles, gfiles):
        r = cv2.imread(rf, cv2.IMREAD_COLOR)
        g = cv2.imread(gf, cv2.IMREAD_COLOR)
        if r is None or g is None: continue
        ps.append(psnr_rgb(r, g))
        ss.append(ssim_gray(cv2.cvtColor(r, cv2.COLOR_BGR2GRAY),
                            cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)))
    print(f"{label}: n={len(ps)} PSNR={np.mean(ps):.3f} SSIM={np.mean(ss):.4f}  (ref PSNR={psnr_ref})")

print("SSIM_DONE")
