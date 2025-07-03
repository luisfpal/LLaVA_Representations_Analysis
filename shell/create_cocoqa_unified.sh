#!/bin/bash
#SBATCH --account=dssc
#SBATCH --job-name=experiment
#SBATCH --partition=EPYC
#SBATCH --cpus-per-task=20
#SBATCH --mem=60GB
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.out

PROJECT_ROOT="$HOME/multimodal_finetuned_representations"

# Activate environment
source "$PROJECT_ROOT/.venv/bin/activate"

cd "$PROJECT_ROOT/scripts"

python create_cocoqa_unified.py
