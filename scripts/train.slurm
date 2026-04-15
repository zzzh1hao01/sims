#!/usr/bin/env bash
#SBATCH --job-name=subpop-qwen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a5000:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# -----------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------
module load anaconda/2023.09 cuda/12.1

# Create once on Oscar:
#   conda env create -f environment.yml
conda activate sims

# Store HuggingFace model cache on scratch so it persists across jobs.
# Adjust the path to your Oscar scratch directory.
export HF_HOME="/oscar/scratch/${USER}/hf_cache"
export WANDB_API_KEY="${WANDB_API_KEY}"   # set in your ~/.bashrc on Oscar

cd "$SLURM_SUBMIT_DIR"

# -----------------------------------------------------------------------
# Launch
# -----------------------------------------------------------------------
torchrun \
    --nproc_per_node=2 \
    --master_port=29500 \
    train.py \
    --config configs/default.yaml \
    --overrides "training.output_dir=outputs/run_${SLURM_JOB_ID}"
