#!/bin/bash
# This script runs modalities cosine similarity analysis

# --- Configuration Section ---
DATASET_PATH_OR_NAME="~/scratch/datasets/cocoqa_unified"

# Analysis configuration
RESIDUAL_STREAM_TYPES="output_layer,post_mlp"
DATASET_TYPE="coco_captioning,cocoqa_img"

# Processing parameters
BATCH_SIZE="25"
SEED="42"

# Directory configuration
PROJECT_DIR="$HOME/multimodal_finetuned_representations"
RESULTS_DIR="$PROJECT_DIR/results"
MODEL_CACHE_DIR="$HOME/scratch/huggingface/hub"

# Script configuration
FILENAME="modalities_cosine_similarity.py"
FILEPATH="$PROJECT_DIR/scripts/$FILENAME"

# --- Argument Construction ---
SCRIPT_ARGS=()
SCRIPT_ARGS+=(--results-dir "$RESULTS_DIR")
SCRIPT_ARGS+=(--dataset-path-or-name "$DATASET_PATH_OR_NAME")
SCRIPT_ARGS+=(--model-cache-dir "$MODEL_CACHE_DIR")
SCRIPT_ARGS+=(--batch-size "$BATCH_SIZE")
SCRIPT_ARGS+=(--dataset-type "$DATASET_TYPE")
SCRIPT_ARGS+=(--residual-stream-types "$RESIDUAL_STREAM_TYPES")
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
echo "====== Modalities Cosine Similarity Analysis Configuration ======"
echo "Dataset: $DATASET_PATH_OR_NAME"
echo "Results directory: $RESULTS_DIR"
echo "Model cache directory: $MODEL_CACHE_DIR"
echo "Residual stream types: $RESIDUAL_STREAM_TYPES"
echo "Dataset type: $DATASET_TYPE"
echo "Batch size: $BATCH_SIZE"
echo "Seed: $SEED"
echo "==============================="

# Run analysis
echo -e "\n====== Running Modalities Cosine Similarity Analysis ======"

# Set PyTorch memory config
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run the command
TOKENIZERS_PARALLELISM=false python "${DEBUGGER_ARGS[@]}" "${FILEPATH}" "${SCRIPT_ARGS[@]}"

echo -e "\n====== Analysis Complete ======"
echo "Results saved to: $RESULTS_DIR" 