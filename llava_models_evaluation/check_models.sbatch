#!/bin/bash
#SBATCH --job-name=check_models
#SBATCH --account=dssc
#SBATCH --partition=DGX
#SBATCH --output=/u/dssc/lpalaciosflores/multimodal_finetuned_representations/llava_models_evaluation/logs/check_models_%j.out
#SBATCH --error=/u/dssc/lpalaciosflores/multimodal_finetuned_representations/llava_models_evaluation/logs/check_models_%j.out
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1

# Change to project root directory
cd /u/dssc/lpalaciosflores/multimodal_finetuned_representations

# Create logs directory if it doesn't exist
mkdir -p llava_models_evaluation/logs

# Activate Python environment
source .venv/bin/activate

# Add project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/u/dssc/lpalaciosflores/multimodal_finetuned_representations/"

# Disable tokenizer parallelism to avoid fork warnings
export TOKENIZERS_PARALLELISM=false

# Run the model checking script (checks all collections)
python llava_models_evaluation/check_models.py

