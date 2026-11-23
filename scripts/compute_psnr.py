"""Compute mean PSNR between rendered images and GT (numpy fallback, no lpips)."""
import os, sys, glob
import numpy as np
import cv2

model_dir = sys.argv[1] if len(sys.argv) > 1 else 'output/smoke_truck'
iteration = sys.argv[2] if len(sys.argv) > 2 else '3000'

renders = sorted(glob.glob(os.path.join(model_dir, 'train', f'ours_{iteration}', 'renders', '*.png')))
gts = sorted(glob.glob(os.path.join(model_dir, 'train', f'ours_{iteration}', 'gt', '*.png')))

if not renders:
    print('NO_RENDERS', flush=True)
    sys.exit(0)

n = min(len(renders), len(gts))
psnrs = []
for i in range(n):
    r = cv2.imread(renders[i]).astype(np.float32) / 255.0
    g = cv2.imread(gts[i]).astype(np.float32) / 255.0
    mse = np.mean((r - g) ** 2)
    if mse < 1e-10:
        psnrs.append(100.0)
    else:
        psnrs.append(10 * np.log10(1.0 / mse))

mean_psnr = float(np.mean(psnrs))
print(f'PSNR mean={mean_psnr:.3f} dB (n={n})', flush=True)

# 点数统计
ply = os.path.join(model_dir, 'point_cloud', f'iteration_{iteration}', 'point_cloud.ply')
if os.path.exists(ply):
    with open(ply, 'rb') as f:
        head = f.read(256).decode('ascii', 'ignore')
    for line in head.split('\n'):
        if line.startswith('element vertex'):
            print(f'GAUSSIANS {line.strip()}', flush=True)
            break
