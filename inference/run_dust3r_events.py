#!/usr/bin/env python3
"""
Zero-shot DUSt3R stereo inference on event-camera sequences (MVSEC / EventScape).

For each consecutive frame pair (i, i+frame_stride) the script runs two DUSt3R
stereo passes using the same pretrained model:
  1. Event pseudo-RGB  — 5-channel voxel converted to R=pos / G=total / B=neg
  2. Aligned RGB frames — ground-truth reference at the same timestep

Output per pair: one PNG panel
  Row 1 (events): event_viz_L | depth_from_events_L | conf_events_L
  Row 2 (RGB):    rgb_L       | depth_from_rgb_L    | conf_rgb_L
"""

import argparse
import os
import sys
import random
import numpy as np
import torch
import PIL.Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm

# ── path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DUST3R_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, DUST3R_ROOT)

from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.inference import inference
from dust3r.image_pairs import make_pairs
import torchvision.transforms as tvf

ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# ── defaults ──────────────────────────────────────────────────────────────────
MVSEC_ROOT = '/cluster/scratch/ademirtas/datasets/MVSEC/train'
EVENTSCAPE_ROOT = '/cluster/project/cvg/students/ademirtas/data/EventScape'
MVSEC_SEQS = [
    'mvsec_outdoor_day1',
    'mvsec_outdoor_night1',
    'mvsec_outdoor_night2',
    'mvsec_outdoor_night3',
]


# ── image preprocessing ───────────────────────────────────────────────────────

def _resize_pil(img, long_edge):
    S = max(img.size)
    interp = PIL.Image.LANCZOS if S > long_edge else PIL.Image.BICUBIC
    new_size = tuple(int(round(x * long_edge / S)) for x in img.size)
    return img.resize(new_size, interp)


def make_dust3r_dict(img_hw3_f32, idx, size=512, patch_size=16):
    """(H,W,3) float32 [0,1] → DUSt3R image dict matching load_images() output."""
    u8 = (img_hw3_f32 * 255).clip(0, 255).astype(np.uint8)
    pil = PIL.Image.fromarray(u8)
    pil = _resize_pil(pil, size)
    W, H = pil.size
    cx, cy = W // 2, H // 2
    halfw = ((2 * cx) // patch_size) * patch_size // 2
    halfh = ((2 * cy) // patch_size) * patch_size // 2
    pil = pil.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))
    return dict(
        img=ImgNorm(pil)[None],                # (1, 3, H', W')
        true_shape=np.int32([pil.size[::-1]]), # [[H', W']]
        idx=idx,
        instance=str(idx),
    )


# ── event utilities ───────────────────────────────────────────────────────────

def voxel_to_pseudo_rgb(vox):
    """(5,H,W) float32 → (H,W,3) float32 [0,1]  (same formula as eventvdpm)."""
    pos = vox.clip(min=0).sum(0)
    neg = (-vox).clip(min=0).sum(0)
    tot = np.abs(vox).sum(0)
    out = np.stack([pos, tot, neg], axis=-1).astype(np.float32)
    for c in range(3):
        mx = out[..., c].max()
        if mx > 0:
            out[..., c] /= mx
    return out


def voxel_to_viz_rgb(vox):
    """(5,H,W) float32 → (H,W,3) float32 [0,1]  white bg, red=positive, blue=negative."""
    pos = vox.clip(min=0).sum(0)
    neg = (-vox).clip(min=0).sum(0)
    mx = max(pos.max(), neg.max(), 1e-6)
    r = (1.0 - neg / mx).clip(0, 1)
    g = (1.0 - (pos + neg) / mx).clip(0, 1)
    b = (1.0 - pos / mx).clip(0, 1)
    return np.stack([r, g, b], axis=-1).astype(np.float32)


# ── data loading ──────────────────────────────────────────────────────────────

def load_mvsec_frame(seq_dir, idx):
    """Returns (vox (5,H,W) float32, rgb (H,W,3) uint8 or None)."""
    vox_path = os.path.join(seq_dir, 'events', 'voxels', f'event_tensor_{idx:010d}.npy')
    rgb_path = os.path.join(seq_dir, 'rgb', 'davis_left_sync', f'frame_{idx:010d}.png')
    vox = np.load(vox_path).astype(np.float32)
    rgb = None
    if os.path.exists(rgb_path):
        rgb = np.array(PIL.Image.open(rgb_path).convert('RGB'))
    return vox, rgb


# ── inference ─────────────────────────────────────────────────────────────────

def run_stereo(dict_a, dict_b, model, device):
    """
    Run DUSt3R on a single image pair.
    Returns depths: list[np(H,W)], confs: list[np(H,W)] indexed by image (0=a, 1=b).
    """
    imgs = [dict_a, dict_b]
    pairs = make_pairs(imgs, scene_graph='complete', prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=1, verbose=False)

    # output['view1']['idx'] is a Python list of ints: [0, 1]
    # output['pred1']['pts3d'] shape: (N_pairs, H, W, 3)
    # output['pred1']['conf']  shape: (N_pairs, H, W)
    depths = [None, None]
    confs = [None, None]
    idx_list = output['view1']['idx']  # [0, 1]
    for k in range(len(pairs)):
        img_idx = idx_list[k]
        depth_k = output['pred1']['pts3d'][k, :, :, 2].detach().cpu().float().numpy()
        conf_k = output['pred1']['conf'][k].detach().cpu().float().numpy()
        depths[img_idx] = depth_k
        confs[img_idx] = conf_k

    return depths, confs


# ── visualization ─────────────────────────────────────────────────────────────

def colorize_depth(depth, cmap='plasma'):
    """(H,W) → (H,W,3) float32 [0,1], percentile-clipped."""
    d = depth.copy().astype(np.float32)
    valid = d > 0
    if valid.any():
        lo, hi = np.percentile(d[valid], [2, 98])
        d = (d - lo) / max(hi - lo, 1e-6)
    d = d.clip(0, 1)
    return cm.get_cmap(cmap)(d)[..., :3].astype(np.float32)


def colorize_conf(conf, cmap='viridis'):
    """(H,W) → (H,W,3) float32 [0,1]."""
    c = conf.copy().astype(np.float32)
    lo, hi = c.min(), c.max()
    c = (c - lo) / max(hi - lo, 1e-6)
    return cm.get_cmap(cmap)(c)[..., :3].astype(np.float32)


def resize_to_height(img_f32, height):
    """(H,W,3) → (height, W', 3), keeping aspect ratio."""
    H, W = img_f32.shape[:2]
    W2 = max(1, int(round(W * height / H)))
    u8 = (img_f32 * 255).clip(0, 255).astype(np.uint8)
    pil = PIL.Image.fromarray(u8).resize((W2, height), PIL.Image.LANCZOS)
    return np.array(pil).astype(np.float32) / 255


def make_row(imgs, height):
    resized = [resize_to_height(im, height) for im in imgs]
    return np.concatenate(resized, axis=1)


def save_panel(rows, path, panel_height):
    """rows: list of lists of (H,W,3) float32 arrays → stacked PNG."""
    rendered = [make_row(r, panel_height) for r in rows]
    # pad each row to max width
    max_w = max(r.shape[1] for r in rendered)
    padded = []
    for r in rendered:
        if r.shape[1] < max_w:
            r = np.pad(r, ((0, 0), (0, max_w - r.shape[1]), (0, 0)))
        padded.append(r)
    panel = np.concatenate(padded, axis=0)
    PIL.Image.fromarray((panel * 255).clip(0, 255).astype(np.uint8)).save(path)


# ── CLI parsing ───────────────────────────────────────────────────────────────

def parse_seq_spec(spec):
    """'name:start:end' → (name, start, end).  Trailing parts optional."""
    parts = spec.split(':')
    name = parts[0]
    start = int(parts[1]) if len(parts) > 1 else None
    end = int(parts[2]) if len(parts) > 2 else None
    return name, start, end


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='Zero-shot DUSt3R stereo on event-camera data (events vs RGB comparison)')
    ap.add_argument('--dataset', default='mvsec', choices=['mvsec'],
                    help='Dataset to use (EventScape support TBD)')
    ap.add_argument('--checkpoint', required=True,
                    help='Path to DUSt3R .pth checkpoint')
    ap.add_argument('--seq', nargs='+', default=None,
                    help='Sequences to run, e.g. "mvsec_outdoor_day1:3037:3060"')
    ap.add_argument('--n_seq', type=int, default=2,
                    help='Number of random sequences if --seq not given')
    ap.add_argument('--frame_stride', type=int, default=2,
                    help='Frame offset between left and right of each stereo pair')
    ap.add_argument('--clip_stride', type=int, default=2,
                    help='Step between consecutive pairs along the sequence')
    ap.add_argument('--panel_height', type=int, default=256,
                    help='Pixel height of each row in the output strip')
    ap.add_argument('--size', type=int, default=512,
                    help='DUSt3R input long-edge size')
    ap.add_argument('--save_dir', required=True)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    print(f'Loading checkpoint: {args.checkpoint}')
    model = AsymmetricCroCo3DStereo.from_pretrained(args.checkpoint).to(device)
    model.eval()
    print('Model loaded.')

    # resolve sequence specs
    if args.seq:
        seq_specs = [parse_seq_spec(s) for s in args.seq]
    else:
        chosen = random.sample(MVSEC_SEQS, min(args.n_seq, len(MVSEC_SEQS)))
        seq_specs = [(s, None, None) for s in chosen]

    total_pairs = 0
    for seq_name, frame_start, frame_end in seq_specs:
        seq_dir = os.path.join(MVSEC_ROOT, seq_name)
        vox_dir = os.path.join(seq_dir, 'events', 'voxels')
        n_vox = len([f for f in os.listdir(vox_dir) if f.endswith('.npy')])

        start = frame_start if frame_start is not None else 0
        # last pair's right frame must be within bounds
        end = (frame_end if frame_end is not None else n_vox - 1) - args.frame_stride
        end = min(end, n_vox - 1 - args.frame_stride)

        seq_save = os.path.join(args.save_dir, seq_name)
        os.makedirs(seq_save, exist_ok=True)
        print(f'\n=== {seq_name}  frames {start}–{end + args.frame_stride}  ({n_vox} total voxels) ===')

        for i in range(start, end + 1, args.clip_stride):
            j = i + args.frame_stride
            print(f'  pair ({i:06d}, {j:06d})', end=' ... ', flush=True)

            vox_i, rgb_i = load_mvsec_frame(seq_dir, i)
            vox_j, rgb_j = load_mvsec_frame(seq_dir, j)

            ev_pseudo_i = voxel_to_pseudo_rgb(vox_i)
            ev_pseudo_j = voxel_to_pseudo_rgb(vox_j)
            ev_viz_i = voxel_to_viz_rgb(vox_i)

            rgb_i_f = rgb_i.astype(np.float32) / 255 if rgb_i is not None else ev_pseudo_i
            rgb_j_f = rgb_j.astype(np.float32) / 255 if rgb_j is not None else ev_pseudo_j

            ev_dict_i = make_dust3r_dict(ev_pseudo_i, 0, size=args.size)
            ev_dict_j = make_dust3r_dict(ev_pseudo_j, 1, size=args.size)
            rgb_dict_i = make_dust3r_dict(rgb_i_f, 0, size=args.size)
            rgb_dict_j = make_dust3r_dict(rgb_j_f, 1, size=args.size)

            with torch.no_grad():
                ev_depths, ev_confs = run_stereo(ev_dict_i, ev_dict_j, model, device)
                rgb_depths, rgb_confs = run_stereo(rgb_dict_i, rgb_dict_j, model, device)

            tag = f'pair_{i:06d}_{j:06d}'
            np.save(os.path.join(seq_save, f'{tag}_ev_depth.npy'),  ev_depths[0])
            np.save(os.path.join(seq_save, f'{tag}_rgb_depth.npy'), rgb_depths[0])
            np.save(os.path.join(seq_save, f'{tag}_ev_conf.npy'),   ev_confs[0])
            np.save(os.path.join(seq_save, f'{tag}_rgb_conf.npy'),  rgb_confs[0])

            ev_depth_col  = colorize_depth(ev_depths[0])
            ev_conf_col   = colorize_conf(ev_confs[0])
            rgb_depth_col = colorize_depth(rgb_depths[0])
            rgb_conf_col  = colorize_conf(rgb_confs[0])

            save_panel(
                rows=[
                    [ev_viz_i,  ev_depth_col,  ev_conf_col],
                    [rgb_i_f,   rgb_depth_col, rgb_conf_col],
                ],
                path=os.path.join(seq_save, f'{tag}.png'),
                panel_height=args.panel_height,
            )

            print('saved', flush=True)
            total_pairs += 1
            torch.cuda.empty_cache()

    print(f'\nFinished. {total_pairs} pairs processed. Results in: {args.save_dir}')


if __name__ == '__main__':
    main()
