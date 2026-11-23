# DB-GS — Depth-Guided Budget-Aware Training for Memory-Efficient 3D Gaussian Splatting

DB-GS 在官方 3DGS（`graphdeco-inria/gaussian-splatting`）之上，把「训练时显存」作为一阶目标，
通过**深度先验**在稀疏高斯初始化、稠密化、剪枝三个阶段同时节流，让 4GB 显存也能端到端训练。

> 关键定位：官方 3DGS 已内置深度正则化损失（`train.py` 的 `--depths`）与 SparseAdam（Taming-3dgs），
> 因此 DB-GS 的创新**不是**加一个深度损失，而是把深度先验用于**训练时资源控制**。

## 四个模块

| 模块 | 目标 | 实现文件 | 训练 flag |
|---|---|---|---|
| 1. 深度先验初始化 | 用单目深度反投影点云替代 SfM 稀疏点云，起点更准、收敛更快 | `dbgs/depth_init.py` + `dbgs/generate_depth_prior.py` | `--dbgs_init_ply <ply>` |
| 2. 深度-不确定性感知稠密化 | 只在深度一致/高显著度处增点，抑制平坦区过密化 | `dbgs/densify_prune.py::depth_consistency_gate` | （未线程化，待 GPU 接入）|
| 3. 预算感知剪枝 | 硬性高斯数上限，超预算按重要性（梯度×不透明度/尺度）剪枝 | `dbgs/densify_prune.py::budget_prune_mask` | `--dbgs_budget <N>` |
| 4. 深度一致性损失 | 非创新（官方已内置），保留作正则 | 官方 `train.py` | `--depths` |

## 目录结构

```
dbgs/
  __init__.py               # 导出接口
  depth_estimator.py        # Depth-Anything-V2 封装 + 深度图生成 + 尺度对齐
  depth_init.py             # 深度反投影 -> 点云 (Module 1)
  densify_prune.py          # 深度门控 + 预算剪枝 (Module 2/3)
  generate_depth_prior.py   # Module 1 CLI
  test_dbgs.py              # 单元测试
gaussian-splatting/         # 官方仓库（已做 4 处集成修改，见下）
```

## 集成修改（gaussian-splatting 内）

1. `scene/gaussian_model.py`：`__init__(..., budget=None)` 新增预算参数；`densify_and_prune` 末尾加预算剪枝块。
2. `arguments/__init__.py`：`OptimizationParams.dbgs_budget = 0`（0=禁用）；`ModelParams._dbgs_init_ply = ""`。
3. `train.py`：`GaussianModel(dataset.sh_degree, opt.optimizer_type, budget=...)`。
4. `scene/__init__.py`：加 `import numpy as np`；`dbgs_init_ply` 非空时读 ply 替换 `scene_info.point_cloud`。

## 用法

```bash
# 0) 准备深度先验点云 (Module 1)
python dbgs/generate_depth_prior.py -s <scene> --encoder vits -o <scene>/dbgs_init.ply

# 1) 训练：深度初始化 + 预算剪枝 (Module 1+3)
python train.py -s <scene> -m output/<scene> \
    --dbgs_init_ply <scene>/dbgs_init.ply \
    --dbgs_budget 1000000 \
    --data_device cpu -r 2

# 2) 消融：仅预算剪枝
python train.py -s <scene> -m output/<scene>_budget --dbgs_budget 1000000 --data_device cpu -r 2

# 3) 基线：原版 3DGS
python train.py -s <scene> -m output/<scene>_base --data_device cpu -r 2
```

## 已完成的验证

- `test_dbgs.py` 反投影几何 3 项全对（平面/旋转/平移）。
- 预算剪枝、深度门控逻辑单元测试通过。
- 全部集成文件 `py_compile` 语法通过。

## 待办

- Module 2 深度门控尚未线程化进 `train.py`（需把渲染深度 + 单目深度从 `render_pkg` 传入 `densify_and_prune`）。
- GPU 端到端训练验证（依赖驱动 + CUDA 就绪）。
