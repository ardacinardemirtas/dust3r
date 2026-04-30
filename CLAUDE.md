# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

DUSt3R (Geometric 3D Vision Made Easy) is a neural network for dense unconstrained stereo 3D reconstruction. Given image pairs, it outputs 3D point clouds, depth maps, and confidence estimates directly without requiring camera calibration or known poses.

## Setup

```bash
git clone --recursive https://github.com/naver/dust3r
conda create -n dust3r python=3.11 cmake=3.14.0
conda activate dust3r
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
pip install -r requirements_optional.txt  # for visual localization tasks

# Optional: compile RoPE CUDA kernels for faster inference
cd croco/models/curope/ && python setup.py build_ext --inplace && cd ../../../
```

There is no `setup.py` or `pyproject.toml`; the project is used directly from source with pip-installed dependencies.

## Running

**Interactive demo (Gradio UI):**
```bash
python3 demo.py --model_name DUSt3R_ViTLarge_BaseDecoder_512_dpt
```

**Programmatic inference:**
```python
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

model = AsymmetricCroCo3DStereo.from_pretrained("naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt").to(device)
images = load_images(['img1.jpg', 'img2.jpg'], size=512)
pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
output = inference(pairs, model, device, batch_size=1)
scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizerWithDepthSmoothing)
```

**Visual localization:**
```bash
python3 visloc.py --dataset [aachen|inloc|...] --model_name DUSt3R_ViTLarge_BaseDecoder_512_dpt
```

**Training (distributed):**
```bash
torchrun --nproc_per_node=8 train.py \
    --train_dataset "..." --test_dataset "..." \
    --model "AsymmetricCroCo3DStereo(...)" \
    --train_criterion "ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)" \
    --test_criterion "Regr3D_ScaleShiftInv(L21, gt_scale=True)" \
    --pretrained checkpoints/CroCo_V2_ViTLarge_BaseDecoder.pth \
    --lr 0.0001 --min_lr 1e-06 --warmup_epochs 10 --epochs 100 \
    --batch_size 16 --accum_iter 1 --output_dir checkpoints/dust3r_224
```

Training is done in 3 stages: 224-res with linear head → 512-res with linear head → 512-res with DPT head, each stage loading the previous checkpoint.

There are no automated tests or linting configurations in this repository.

## Architecture

### Core model: `dust3r/model.py`

`AsymmetricCroCo3DStereo` extends `CroCoNet` (from the `croco/` submodule, a CroCo v2 Vision Transformer). The model:
1. Encodes both images through **shared encoder weights** with RoPE positional embeddings
2. Decodes them through **two asymmetric decoders** that cross-attend to each other's features
3. Outputs via a **downstream head** (linear or DPT) two `(pts3d, conf)` pairs, both in view1's coordinate frame

Both outputs are always expressed in **view1's reference frame** — a key design choice that simplifies downstream fusion.

### Global alignment: `dust3r/cloud_opt/`

After pairwise inference, a differentiable optimizer jointly refines camera poses, focal lengths, and per-image depthmaps to be globally consistent across all image pairs. Entry point: `global_aligner()` in `dust3r/cloud_opt/__init__.py`.

- `optimizer.py` — `PointCloudOptimizer`: main optimizer, uses scene graph of pairwise predictions
- `modular_optimizer.py` — `ModularPointCloudOptimizer`: alternative formulation
- `pair_viewer.py` — `PairViewer`: simple two-image visualization without optimization
- `base_opt.py` — shared base class holding optimizable parameters (poses, focals, depthmaps, principal points)
- `init_im_poses.py` — pose initialization from pairwise predictions via MST/PnP

### Scene graphs: `dust3r/image_pairs.py`

`make_pairs()` generates which image pairs to run through the network. Strategies: `complete` (all pairs), `swin` (sliding window), `logwin` (logarithmic window), `oneref` (one reference image vs. all others). `symmetrize=True` adds both (i,j) and (j,i) directions.

### Heads: `dust3r/heads/`

- `linear_head.py` — simple MLP on decoder tokens; faster, used in early training stages
- `dpt_head.py` — Dense Prediction Transformer head; higher quality, used for final models

### Output heads and losses: `dust3r/losses.py`

Key loss classes:
- `Regr3D` — L2 regression on 3D point clouds in a normalized coordinate frame
- `ConfLoss` — wraps a base loss with learned per-point confidence weights
- `Regr3D_ScaleShiftInv` — scale-and-shift invariant version used for evaluation

### Patch embedding: `dust3r/patch_embed.py`

`PatchEmbedDust3R` and `ManyAR_PatchEmbed` support variable aspect ratios and multi-resolution inputs, unlike the standard fixed-size ViT patch embedding.

### CroCo submodule: `croco/`

The `croco/` directory is a git submodule containing the backbone ViT, RoPE positional embeddings, and the `CroCoNet` base class. Do not edit files here; they come from the upstream `naver/croco` repository.

## Key design notes

- **No camera calibration required at inference** — the model predicts absolute 3D coordinates, not disparity.
- **Confidence maps** are a first-class output; use them to filter low-quality predictions.
- **Patch size flexibility**: the `patch_size != 16` compatibility was recently added (commit `3cc8c88`); when modifying patch embedding logic be aware of both `patch_size=16` and `patch_size=14` (ViT-L/14) paths.
- **Datasets** are loaded via a string-based eval interface in `train.py` (`--train_dataset` is `eval()`d). Dataset classes live in `dust3r/datasets/` and inherit from `BaseStereoViewDataset`.
- **License**: CC BY-NC-SA 4.0 — non-commercial use only.
