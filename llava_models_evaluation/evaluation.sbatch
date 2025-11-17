#!/bin/bash
#SBATCH --job-name=evaluation
#SBATCH --account=dssc
#SBATCH --partition=DGX
#SBATCH --output=/u/dssc/lpalaciosflores/multimodal_finetuned_representations/llava_models_evaluation/logs/evaluation_%j.out
#SBATCH --error=/u/dssc/lpalaciosflores/multimodal_finetuned_representations/llava_models_evaluation/logs/evaluation_%j.out
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
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

# Launch the evaluation script with optional collection name argument
# Default is "Llava1.5-7B" if not provided
COLLECTION_NAME=${1:-"Llava1.5-7B-2-12"}
python llava_models_evaluation/evaluation.py --collection_name "$COLLECTION_NAME"
