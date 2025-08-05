#!/bin/bash
# This script runs layer transplantation benchmarking for captioning tasks

# --- Model Configuration ---
MULTIMODAL_MODEL_NAME_OR_PATH="llava-hf/llava-1.5-7b-hf"
LANGUAGE_MODEL_NAME_OR_PATH="lmsys/vicuna-7b-v1.5"
PRETRAINED_PROJECTOR_NAME_OR_PATH="liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5"

# --- Dataset Configuration ---
DATASET_PATH_OR_NAME="~/scratch/datasets/cocoqa_unified"

# --- Transplantation Configuration ---
TRANSPLANTATION_METHOD="sliding_window"
# TRANSPLANTATION_METHOD="two_parts"
STRIDE="2"
WINDOW_SIZE="2"

# --- Benchmarking Parameters ---
BATCH_SIZE="10"
MAX_NEW_TOKENS="20"
SEED="42"

# --- CLIP Evaluation Parameters ---
CLIP_MODEL_NAME_OR_PATH="openai/clip-vit-large-patch14-336"
CLIP_CACHE_DIR="$HOME/scratch/huggingface/hub"
CLIP_WEIGHT="2.5"
CLIP_BATCH_SIZE="16"

# --- Output Configuration ---
SAVE_BASELINE_OUTPUTS="true"  # Set to "false" to disable saving baseline outputs

# --- Directory Configuration ---
PROJECT_DIR="$HOME/multimodal_finetuned_representations"
RESULTS_DIR="$PROJECT_DIR/results"
MODEL_CACHE_DIR="$HOME/scratch/huggingface/hub"

# --- Script Configuration ---
SCRIPT_FILENAME="transplanting_layers_caption_benchmarking.py"
SCRIPT_FILEPATH="$PROJECT_DIR/scripts/$SCRIPT_FILENAME"

# --- Argument Construction ---
SCRIPT_ARGS=()
SCRIPT_ARGS+=(--multimodal_model_name_or_path "$MULTIMODAL_MODEL_NAME_OR_PATH")
SCRIPT_ARGS+=(--model_cache_dir "$MODEL_CACHE_DIR")
SCRIPT_ARGS+=(--pretrained_projector_name_or_path "$PRETRAINED_PROJECTOR_NAME_OR_PATH")
SCRIPT_ARGS+=(--language_model_name_or_path "$LANGUAGE_MODEL_NAME_OR_PATH")
SCRIPT_ARGS+=(--transplantation_method "$TRANSPLANTATION_METHOD")
SCRIPT_ARGS+=(--stride "$STRIDE")
SCRIPT_ARGS+=(--window_size "$WINDOW_SIZE")
SCRIPT_ARGS+=(--batch_size "$BATCH_SIZE")
SCRIPT_ARGS+=(--max_new_tokens "$MAX_NEW_TOKENS")
SCRIPT_ARGS+=(--results_dir "$RESULTS_DIR")
SCRIPT_ARGS+=(--dataset_path_or_name "$DATASET_PATH_OR_NAME")
SCRIPT_ARGS+=(--seed "$SEED")

# Add CLIP evaluation parameters
SCRIPT_ARGS+=(--clip_model_name_or_path "$CLIP_MODEL_NAME_OR_PATH")
SCRIPT_ARGS+=(--clip_cache_dir "$CLIP_CACHE_DIR")
SCRIPT_ARGS+=(--clip_weight "$CLIP_WEIGHT")
SCRIPT_ARGS+=(--clip_batch_size "$CLIP_BATCH_SIZE")

# Add save_baseline_outputs flag if enabled
if [ "$SAVE_BASELINE_OUTPUTS" = "true" ]; then
    SCRIPT_ARGS+=(--save_baseline_outputs)
fi

# --- Debug Configuration ---
DEBUGGER_ARGS=()
if [ "$1" = "--debug" ]; then
    DEBUGGER_ARGS=(-m debugpy --listen localhost:5678 --wait-for-client)
    echo "Debugger enabled - waiting for client connection on localhost:5678"
fi

# --- Create directories ---
mkdir -p "$RESULTS_DIR"

# --- Display Configuration ---
echo "====== Transplanting Layers Captioning Benchmarking Configuration ======"
echo "Multimodal model: $MULTIMODAL_MODEL_NAME_OR_PATH"
echo "Language model: $LANGUAGE_MODEL_NAME_OR_PATH"
echo "Pretrained projector: $PRETRAINED_PROJECTOR_NAME_OR_PATH"
echo "Dataset: $DATASET_PATH_OR_NAME"
echo "Transplantation method: $TRANSPLANTATION_METHOD"
echo "Stride: $STRIDE"
echo "Window size: $WINDOW_SIZE"
echo "Results directory: $RESULTS_DIR"
echo "Model cache directory: $MODEL_CACHE_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Max new tokens: $MAX_NEW_TOKENS"
echo "Seed: $SEED"
echo "CLIP model: $CLIP_MODEL_NAME_OR_PATH"
echo "CLIP cache dir: $CLIP_CACHE_DIR"
echo "CLIP weight: $CLIP_WEIGHT"
echo "CLIP batch size: $CLIP_BATCH_SIZE"
echo "Save baseline outputs: $SAVE_BASELINE_OUTPUTS"
echo "==============================="

# --- Run Benchmarking ---
echo -e "\n====== Running Transplanting Layers Captioning Benchmarking ======"

# Set PyTorch memory config
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run the command
TOKENIZERS_PARALLELISM=false python "${DEBUGGER_ARGS[@]}" "${SCRIPT_FILEPATH}" "${SCRIPT_ARGS[@]}"

echo -e "\n====== Captioning Benchmarking Complete ======"
echo "Results saved to: $RESULTS_DIR" 