#!/bin/bash
#SBATCH --account=dssc
#SBATCH --job-name=experiment
#SBATCH --partition=DGX
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=450GB
#SBATCH --time=01:00:00

PROJECT_ROOT="$HOME/multimodal_finetuned_representations"

# Activate environment
source "$PROJECT_ROOT/.venv/bin/activate"

cd "$PROJECT_ROOT/tests"

# Set PyTorch memory config
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TOKENIZERS_PARALLELISM=false python test_residual_stream_tracer.py
