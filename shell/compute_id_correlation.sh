#!/bin/bash
# This script runs ID correlation computation between multimodal models

# --- Configuration Section ---
DATASET_PATH_OR_NAME="~/scratch/datasets/cocoqa_unified"

# Analysis configuration
DATASET_TYPE="cocoqa_txt,cocoqa_img"

# ID correlation parameters
ID_NN_RANK="100"
ID_CORRELATION_PERMUTATIONS="100"

# Processing parameters
BATCH_SIZE="1"
SEED="42"

# Directory configuration
PROJECT_DIR="$HOME/multimodal_finetuned_representations"
RESULTS_DIR="$PROJECT_DIR/results_id_correlation"
MODEL_CACHE_DIR="$HOME/scratch/huggingface/hub"

# Script configuration
FILENAME="compute_id_correlation.py"
FILEPATH="$PROJECT_DIR/scripts/$FILENAME"

# --- Argument Construction ---
SCRIPT_ARGS=()
SCRIPT_ARGS+=(--results-dir "$RESULTS_DIR")
SCRIPT_ARGS+=(--dataset-path-or-name "$DATASET_PATH_OR_NAME")
SCRIPT_ARGS+=(--model-cache-dir "$MODEL_CACHE_DIR")
SCRIPT_ARGS+=(--batch-size "$BATCH_SIZE")
SCRIPT_ARGS+=(--dataset-type "$DATASET_TYPE")
SCRIPT_ARGS+=(--id-nn-rank "$ID_NN_RANK")
SCRIPT_ARGS+=(--id-correlation-permutations "$ID_CORRELATION_PERMUTATIONS")
SCRIPT_ARGS+=(--seed "$SEED")

# --- Debugger ---
DEBUGGER_ARGS=()
if [ "$1" = "--debug" ]; then
    DEBUGGER_ARGS=(-m debugpy --listen localhost:5678 --wait-for-client)
    echo "Debugger enabled - waiting for client connection on localhost:5678"
fi

# --- Create directories ---
mkdir -p "$RESULTS_DIR"

# Print configuration
echo "====== ID Correlation Analysis Configuration ======"
echo "Dataset: $DATASET_PATH_OR_NAME"
echo "Results directory: $RESULTS_DIR"
echo "Model cache directory: $MODEL_CACHE_DIR"
echo "Dataset type: $DATASET_TYPE"
echo "ID NN rank: $ID_NN_RANK"
echo "ID correlation permutations: $ID_CORRELATION_PERMUTATIONS"
echo "Batch size: $BATCH_SIZE"
echo "Seed: $SEED"
echo "==============================="

# Run analysis
echo -e "\n====== Running ID Correlation Analysis ======"

# Set PyTorch memory config
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run the command
TOKENIZERS_PARALLELISM=false python "${DEBUGGER_ARGS[@]}" "${FILEPATH}" "${SCRIPT_ARGS[@]}"

echo -e "\n====== Analysis Complete ======"
echo "Results saved to: $RESULTS_DIR"
