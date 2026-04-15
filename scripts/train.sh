#!/usr/bin/env bash
# Local reference script — not meant to be run on your laptop.
# Use train.slurm on Oscar instead.
set -euo pipefail

CONFIG="${1:-configs/default.yaml}"

torchrun --nproc_per_node=2 train.py \
    --config "$CONFIG" \
    --overrides "training.output_dir=outputs/run_$(date +%Y%m%d_%H%M%S)"
