#!/bin/bash
# ============================================================================
# GVHMR Checkpoint Download Script
# All files from HuggingFace — no Google Drive needed
# ============================================================================

set -e
cd "$(dirname "$0")"  # Run from GVHMR root

mkdir -p inputs/checkpoints/{body_models/smpl,body_models/smplx,dpvo,gvhmr,hmr2,vitpose,yolo}

# ────────────────────────────────────────────────────────────────────────────
# 1. GVHMR pipeline checkpoints (from camenduru/GVHMR on HuggingFace)
# ────────────────────────────────────────────────────────────────────────────

echo "=== Downloading GVHMR checkpoint ==="
wget -c -O inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt \
  "https://huggingface.co/camenduru/GVHMR/resolve/main/gvhmr/gvhmr_siga24_release.ckpt"

echo "=== Downloading HMR2 checkpoint ==="
wget -c -O "inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt" \
  "https://huggingface.co/camenduru/GVHMR/resolve/main/hmr2/epoch%3D10-step%3D25000.ckpt"

echo "=== Downloading ViTPose checkpoint ==="
wget -c -O inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth \
  "https://huggingface.co/camenduru/GVHMR/resolve/main/vitpose/vitpose-h-multi-coco.pth"

echo "=== Downloading DPVO checkpoint ==="
wget -c -O inputs/checkpoints/dpvo/dpvo.pth \
  "https://huggingface.co/camenduru/GVHMR/resolve/main/dpvo/dpvo.pth"

echo "=== Downloading YOLOv8x ==="
wget -c -O inputs/checkpoints/yolo/yolov8x.pt \
  "https://huggingface.co/camenduru/GVHMR/resolve/main/yolo/yolov8x.pt"

# ────────────────────────────────────────────────────────────────────────────
# 2. SMPL body models (from camenduru/SMPLer-X on HuggingFace)
#    These bypass the official MPI signup requirement.
# ────────────────────────────────────────────────────────────────────────────

echo "=== Downloading SMPL_NEUTRAL.pkl ==="
wget -c -O inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl \
  "https://huggingface.co/camenduru/SMPLer-X/resolve/main/SMPL_NEUTRAL.pkl"

echo "=== Downloading SMPLX_NEUTRAL.npz ==="
wget -c -O inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz \
  "https://huggingface.co/camenduru/SMPLer-X/resolve/main/SMPLX_NEUTRAL.npz"

# ────────────────────────────────────────────────────────────────────────────
# OPTIONAL: Male/Female variants (only needed if you do gendered inference)
# ────────────────────────────────────────────────────────────────────────────

# Uncomment if needed:

# echo "=== Downloading SMPLX_MALE.npz ==="
# wget -c -O inputs/checkpoints/body_models/smplx/SMPLX_MALE.npz \
#   "https://huggingface.co/camenduru/SMPLer-X/resolve/main/SMPLX_MALE.npz"

# echo "=== Downloading SMPLX_FEMALE.npz ==="
# wget -c -O inputs/checkpoints/body_models/smplx/SMPLX_FEMALE.npz \
#   "https://huggingface.co/camenduru/SMPLer-X/resolve/main/SMPLX_FEMALE.npz"

echo ""
echo "=== All downloads complete! ==="
echo ""
echo "Final structure:"
find inputs/checkpoints -type f | sort