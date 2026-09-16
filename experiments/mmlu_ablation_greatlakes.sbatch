#!/bin/bash
# Orthogonal-residual ablation (test vs. norm-matched null) for the MMLU group at
# all 8 swept layers of Llama-3.2-3B, on UMich Great Lakes.
#
# Identical to the run behind the paper's layer-21 number except that --layers is
# omitted: null_distribution.py then defaults to the standard depth fractions,
# which for a 28-layer model resolve to [1, 3, 6, 8, 11, 14, 18, 21].
#
# Writes to causal_projection/all_layers/ so the existing single-layer artifacts
# in causal_projection/ (gitignored, not reproducible without a rerun) survive.
#
# Partition spgpu = A40 (48 GB), which fits the model and a generous --chunk.
# On the gpu partition (V100, 16 GB) keep CHUNK low. A GPU partition job that does
# not request a GPU never starts, hence the explicit --gres.
#
# Usage, from the repo root on a login node:
#     sbatch --account=<slurm_account> experiments/mmlu_ablation_greatlakes.sbatch

#SBATCH --job-name=mmlu-null-8layer
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48g
#SBATCH --time=08:00:00
#SBATCH --output=results/%x-%j.out
#SBATCH --mail-type=END,FAIL

set -euo pipefail

REPO=${REPO:-$SLURM_SUBMIT_DIR}
SCRATCH=${SCRATCH:-/scratch/${SLURM_JOB_ACCOUNT}_root/${SLURM_JOB_ACCOUNT}/$USER}

# Weights and the cais/mmlu download go to scratch, not the 80 GB home quota.
export HF_HOME="$SCRATCH/hf"
export TOKENIZERS_PARALLELISM=false

module load python3.10-anaconda/2023.03
source "$SCRATCH/tvenv/bin/activate"

cd "$REPO"
mkdir -p results causal_projection/all_layers

nvidia-smi

# --chunk sets the null-draw batch size. It affects memory only, not results:
# activation memory scales with chunk * seq_len, and MMLU has the longest prompts.
# 4 is the known-safe value; raising it shortens wall time until it OOMs.
python experiments/task_vector_injection/null_distribution.py \
    --group mmlu \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --samples 10 \
    --draws 30 \
    --seed 0 \
    --chunk "${CHUNK:-4}" \
    --out-dir "$REPO/causal_projection/all_layers"

python experiments/task_vector_injection/analyze_null_position.py \
    --group mmlu \
    --stem nulldist \
    --dir "$REPO/causal_projection/all_layers"
