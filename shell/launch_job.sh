#!/bin/bash
#SBATCH --account=dssc
#SBATCH --job-name=extract
#SBATCH --partition=DGX
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=40GB
#SBATCH --time=00:30:00

# Conda environment name
CONDA_ENV_NAME=emu3

# Activate conda environment
source ~/scratch/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV_NAME"

# SCRIPT="multiple_choice_benchmarks_evaluation.sh"
SCRIPT="analyze_residual_streams.sh"

# SCRIPT="extract_heads_representations.sh"
# SCRIPT="analyze_heads_representations.sh"
# Command to run
bash "$HOME/multimodal_finetuned_representations/shell/$SCRIPT"
