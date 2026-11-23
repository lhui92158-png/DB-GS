# DB-GS: Depth-gated Budget-constrained 3D Gaussian Splatting

**Training-time memory-constrained 3D Gaussian Splatting** via depth-aware gating and Lagrangian-dual pruning. Trains high-quality 3DGS scenes on **consumer GPUs (4 GB VRAM)** that would otherwise exceed device memory — without the "train a full model first, then compress it" detour of post-hoc lightweighting methods.

> **Paper (Chinese)**: [docs/paper_zh.md](docs/paper_zh.md) — 显存约束三维高斯溅射的深度门控与对偶剪枝

---

## Highlights

| Result | Value |
|---|---|
| Peak-VRAM model | `V ≈ 2.706×10⁻⁶ · N` GB (R²=0.987), 4 GB ⇒ N ≤ 1.49 M Gaussians |
| Largest scene (garden) baseline | 4.77 GB → **2.69 GB** (−44%) under 50% budget |
| BGP budget pruning (truck, −r 1) | **−74.2% points @ −1.74 dB** (fixed 300 K budget) |
| 11-scene 50%-budget | **−51% points @ −1.25 dB** (−0.74 ~ −1.65) |
| DAG depth gating (truck, −r 1) | **−21.8% points, lossless** (−0.03 dB) |
| LDB Lagrangian dual (truck, 300 K) | **261.8 K points / 21.83 dB**, auto-converges without a hard budget |
| 30 K iterations (truck, 300 K budget) | **269.9 K points / 25.18 dB / SSIM 0.968**, 78 min stable |
| Multi-seed variance (3 seeds) | PSNR std ≤ 0.03 dB (quality is seed-robust) |

All numbers are reproduced on a **NVIDIA GTX 1050 Ti (4 GB)** — see [docs/](docs/) for full tables.

---

## Method

DB-GS formulates 3DGS training as a constrained optimization `min L(Θ,G) s.t. M(G) ≤ B` and introduces three plug-and-play modules:

| Module | Name | What it does | Flag |
|---|---|---|---|
| **DAG** | Depth-Aware Gating | Gates densification with a monocular-depth edge map `E = ‖∇D‖`, suppressing redundant Gaussians in flat/textureless regions (geometry, not appearance). | `--dbgs_depth_edge` |
| **BGP** | Budget-Guided Pruning | Enforces a hard Gaussian-count budget by pruning the lowest-gradient-norm Gaussians (`s_i = ‖∇_i‖`), keeping the count within `soft_budget = 0.9 × B`. | `--dbgs_budget` |
| **LDB** | Lagrangian Dual Budget | Solves the constraint via Lagrangian duality and dual ascent `λ ← clip(λ + η(|G|/B − 1), 0, 1)` — converges to the budget *without* a hard cap. | `--dbgs_dual_budget` |

A fourth line of work is reported as a **negative/diagnostic result**: replacing SfM initialization with back-projected monocular depth priors degrades quality (−2.0 dB) because single-view relative depth is too noisy for geometric initialization — depth is best used for *gating* (DAG) or *regularization*, not initialization. See `docs/theory_analysis.md` §6.

---

## Repository structure

```
DB-GS/
├── dbgs/                    # Core library (self-contained)
│   ├── depth_estimator.py   #   Depth-Anything-V2 wrapper + depth-prior generation
│   ├── depth_init.py        #   Depth back-projection → point cloud (Module: init)
│   ├── densify_prune.py     #   DAG edge gating + BGP/LDB pruning scores
│   ├── generate_depth_prior.py / generate_depths.py
│   └── test_dbgs.py         #   Unit tests (back-projection geometry, pruning)
├── patch/                   # Integration patch for the vanilla 3DGS codebase
│   ├── gaussian-splatting.patch          # 5 modified files (train/args/scene/model/utils)
│   └── {diff-gaussian-rasterization,fused-ssim,simple-knn}.patch  # build-flag fixes
├── scripts/                 # Reproduce experiments + figures
│   ├── run_variance.py / run_multiscene.py / run_ratio_budget.py
│   ├── run_dual_ablation.py / run_m2_multiscene.py / run_improvements.py
│   ├── make_figures_final.py           # all 7 paper figures (matplotlib)
│   └── compute_psnr.py / compute_ssim.py / download_dataset.py
├── docs/                    # paper + theory + full result tables
├── figures/                 # 7 publication figures (PNG, 300 dpi)
├── requirements.txt
└── LICENSE
```

---

## Installation

Tested environment: **Windows 10 / Python 3.10 / CUDA 12.1 / PyTorch 2.3.1+cu121 / NVIDIA GTX 1050 Ti (sm_61)**.

### 1. Install the vanilla 3D Gaussian Splatting

```bash
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
cd gaussian-splatting
conda create -n dbgs python=3.10 && conda activate dbgs
pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install numpy plyfile tqdm opencv-python
```

### 2. Build the CUDA submodules

```bash
# NVIDIA Pascal (sm_61) needs these flags against MSVC 14.44 / CUDA 12.1:
export TORCH_CUDA_ARCH_LIST="6.1"
pip install submodules/diff-gaussian-rasterization submodules/simple-knn submodules/fused-ssim --no-build-isolation
```

> **Pascal (sm_61) + CUDA 12.1 + MSVC 14.44 caveat** (documented in `patch/*.patch`): add
> `-allow-unsupported-compiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH` to the
> `nvcc` `extra_compile_args` in each submodule's `setup.py`, and add the include paths for
> `cub`/`thrust` (CUDA 12.1 ships without them). See `patch/` for the exact diffs.

### 3. Apply the DB-GS integration patch

```bash
cd gaussian-splatting
git apply ../DB-GS/patch/gaussian-splatting.patch
```

### 4. Install `dbgs`

```bash
pip install -e ../DB-GS        # or add the DB-GS dir to PYTHONPATH
```

### 5. (Optional) Depth-Anything-V2 for DAG

```bash
git clone https://github.com/DepthAnything/Depth-Anything-V2
# download depth_anything_v2_vits.pth into its checkpoints/ dir
```

---

## Usage

### Train with a hard budget (BGP)

```bash
python train.py -s data/<scene> -m output/<scene> -r 1 --data_device cpu \
    --dbgs_budget 300000 --disable_viewer --test_iterations -1
```

### Train with depth-aware gating (DAG)

```bash
# first generate the monocular depth files (DVA-V2)
python -m dbgs.generate_depths --source data/<scene>

python train.py -s data/<scene> -m output/<scene> -r 1 --data_device cpu \
    -d data/<scene>/depths --depth_l1_weight_init 0 --depth_l1_weight_final 0 \
    --dbgs_depth_edge --disable_viewer --test_iterations -1
```

### Train with Lagrangian dual budget (LDB)

```bash
python train.py -s data/<scene> -m output/<scene> -r 1 --data_device cpu \
    --dbgs_dual_budget 300000 --dbgs_dual_lr 0.05 --disable_viewer --test_iterations -1
```

### Render + metrics

```bash
python render.py -m output/<scene> --iteration 30000
python scripts/compute_psnr.py -m output/<scene>      # PSNR (numpy)
python scripts/compute_ssim.py -m output/<scene>      # PSNR + SSIM (numpy)
```

### Command-line flags

| Flag | Default | Module |
|---|---|---|
| `--dbgs_budget N` | 0 (off) | BGP hard budget (soft cap = 0.9·N) |
| `--dbgs_dual_budget N` | 0 (off) | LDB Lagrangian dual |
| `--dbgs_dual_lr` | 0.05 | LDB dual-ascent step size |
| `--dbgs_depth_edge` | off | DAG depth-aware gating |
| `--dbgs_tau_edge` | 0.05 | DAG edge threshold |
| `--dbgs_tau_grad_floor` | 0.001 | DAG gradient floor |
| `--dbgs_init_ply` | "" | depth-prior initialization (negative result) |
| `--seed N` | 0 | RNG seed (multi-seed variance) |

---

## Reproducing the paper results

```bash
python scripts/run_variance.py       # multi-seed variance (truck, 4 configs × 3 seeds)
python scripts/run_multiscene.py     # baseline vs fixed-300K budget (11 scenes)
python scripts/run_ratio_budget.py   # 50%-budget + peak-VRAM (11 scenes)
python scripts/run_dual_ablation.py  # hard vs dual budget
python scripts/run_m2_multiscene.py  # DAG multi-scene
python scripts/run_improvements.py   # image-edge gate + depth regularization
python scripts/make_figures_final.py # regenerate all 7 figures
```

Datasets follow the standard 3DGS layout (`images/` + `sparse/0/{cameras,images,points3D}.bin`); `scripts/download_dataset.py` fetches a pre-computed COLMAP copy from a HuggingFace mirror.

---

## Citation

If you use DB-GS in your research, please cite:

```bibtex
@misc{dbgs2026,
  title   = {显存约束三维高斯溅射的深度门控与对偶剪枝},
  author  = {DB-GS Contributors},
  year    = {2026},
  note    = {Under submission. See docs/paper_zh.md}
}
```

---

## License

MIT License — see [LICENSE](LICENSE). The integration patch (`patch/`) targets
[graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting),
which is distributed under its own license.
