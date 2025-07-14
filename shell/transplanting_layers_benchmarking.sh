#!/bin/bash
# This script runs layer transplantation benchmarking

# --- Configuration Section ---
MULTIMODAL_MODEL_NAME_OR_PATH="llava-hf/llava-1.5-7b-hf"
LANGUAGE_MODEL_NAME_OR_PATH="lmsys/vicuna-7b-v1.5"
PRETRAINED_PROJECTOR_NAME_OR_PATH="liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5"

# Dataset configuration
DATASET_NAME_OR_PATH="~/scratch/datasets/cocoqa_unified"

# Transplantation configuration
TRANSPLANTATION_METHOD="sliding_window"
# TRANSPLANTATION_METHOD="two_parts"

# Benchmarking parameters
BATCH_SIZE="25"
MAX_NEW_TOKENS="10"
SEED="42"

# Directory configuration
PROJECT_DIR="$HOME/multimodal_finetuned_representations"
RESULTS_DIR="$PROJECT_DIR/results"
MODEL_CACHE_DIR="$HOME/scratch/huggingface/hub"

# Script configuration
FILENAME="transplanting_layers_benchmarking.py"
FILEPATH="$PROJECT_DIR/scripts/$FILENAME"

# --- Argument Construction ---
SCRIPT_ARGS=()
SCRIPT_ARGS+=(--multimodal_model_name_or_path "$MULTIMODAL_MODEL_NAME_OR_PATH")
SCRIPT_ARGS+=(--model_cache_dir "$MODEL_CACHE_DIR")
SCRIPT_ARGS+=(--pretrained_projector_name_or_path "$PRETRAINED_PROJECTOR_NAME_OR_PATH")
SCRIPT_ARGS+=(--language_model_name_or_path "$LANGUAGE_MODEL_NAME_OR_PATH")
SCRIPT_ARGS+=(--transplantation_method "$TRANSPLANTATION_METHOD")
SCRIPT_ARGS+=(--batch_size "$BATCH_SIZE")
SCRIPT_ARGS+=(--max_new_tokens "$MAX_NEW_TOKENS")
SCRIPT_ARGS+=(--results_dir "$RESULTS_DIR")
SCRIPT_ARGS+=(--dataset_name_or_path "$DATASET_NAME_OR_PATH")
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
echo "====== Transplanting Layers Benchmarking Configuration ======"
echo "Multimodal model: $MULTIMODAL_MODEL_NAME_OR_PATH"
echo "Language model: $LANGUAGE_MODEL_NAME_OR_PATH"
echo "Pretrained projector: $PRETRAINED_PROJECTOR_NAME_OR_PATH"
echo "Dataset: $DATASET_NAME_OR_PATH"
echo "Transplantation method: $TRANSPLANTATION_METHOD"
echo "Results directory: $RESULTS_DIR"
echo "Model cache directory: $MODEL_CACHE_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Max new tokens: $MAX_NEW_TOKENS"
echo "Seed: $SEED"
echo "==============================="

# Run benchmarking
echo -e "\n====== Running Transplanting Layers Benchmarking ======"

# Set PyTorch memory config
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run the command
TOKENIZERS_PARALLELISM=false python "${DEBUGGER_ARGS[@]}" "${FILEPATH}" "${SCRIPT_ARGS[@]}"

echo -e "\n====== Benchmarking Complete ======"
echo "Results saved to: $RESULTS_DIR"
