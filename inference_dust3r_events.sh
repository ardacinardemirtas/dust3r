#!/bin/bash
#SBATCH --job-name=dust3r-events
#SBATCH --output=/cluster/scratch/ademirtas/code/dust3r/logs/%x-%j.out
#SBATCH --partition=gpuhe.4h
#SBATCH --gpus=nvidia_geforce_rtx_3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --time=1:00:00

# ===========================================================================
# Configuration — edit before submitting
# ===========================================================================

REPO="/cluster/scratch/ademirtas/code/dust3r"
CKPT="$REPO/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
SAVE_DIR="$REPO/output/dust3r_events_$(date +%Y%m%d_%H%M%S)"

# Sequences: "name:start:end" (both inclusive).
# Available: mvsec_outdoor_day1 | mvsec_outdoor_night1 | mvsec_outdoor_night2 | mvsec_outdoor_night3
SEQS=(
    "mvsec_outdoor_day1:3037:3060"
    "mvsec_outdoor_night1:0:40"
)

# Leave SEQS empty and set N_SEQ to pick random sequences instead:
# SEQS=()
N_SEQ=2

# frame_stride: offset between left and right frame of each stereo pair
# clip_stride:  step between consecutive pairs along the sequence
FRAME_STRIDE=2
CLIP_STRIDE=2

PANEL_HEIGHT=256   # pixel height of each row in the output strip

# ===========================================================================

set -euo pipefail
cd "$REPO"
source "$HOME/.bashrc"
conda activate dust3r

mkdir -p logs "$SAVE_DIR"
echo "Output: $SAVE_DIR"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# Download checkpoint if not already present
if [ ! -f "$CKPT" ]; then
    echo "Downloading DUSt3R checkpoint..."
    mkdir -p "$REPO/checkpoints"
    wget -q --show-progress \
        https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \
        -O "$CKPT"
fi

ARGS=(
    --dataset       mvsec
    --checkpoint    "$CKPT"
    --frame_stride  "$FRAME_STRIDE"
    --clip_stride   "$CLIP_STRIDE"
    --panel_height  "$PANEL_HEIGHT"
    --save_dir      "$SAVE_DIR"
    --n_seq         "$N_SEQ"
)

if [ "${#SEQS[@]}" -gt 0 ]; then
    ARGS+=(--seq "${SEQS[@]}")
fi

echo "python inference/run_dust3r_events.py ${ARGS[*]}"
python inference/run_dust3r_events.py "${ARGS[@]}"

echo "Done. Results in: $SAVE_DIR"
