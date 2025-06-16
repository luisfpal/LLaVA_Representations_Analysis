#!/bin/bash
# This script analyzes residual streams and computes neighborhood overlap
PROJECT_DIR="$HOME/multimodal_finetuned_representations"

# LLaVA 1.5 7B
# MMLU
# RESIDUAL_STREAM_ABS_PATH1="/orfeo/cephfs/scratch/dssc/lpalaciosflores/representations/llava-hf_llava-1.5-7b-hf/mmlu/test/layers_representations/layer-all_token-last_texts-qa_chat-format_qinst-type-singular_continue-fm_downsample-2500_seed-42.safetensors"
# ScienceQA Text
RESIDUAL_STREAM_ABS_PATH1="/orfeo/cephfs/scratch/dssc/lpalaciosflores/representations/llava-hf_llava-1.5-7b-hf/ScienceQA/test/layers_representations/layer-all_token-last_texts-qa_chat-format_qinst-type-singular_continue-fm_seed-42.safetensors"
# ScienceQA Image
# RESIDUAL_STREAM_ABS_PATH1="/orfeo/cephfs/scratch/dssc/lpalaciosflores/representations/llava-hf_llava-1.5-7b-hf/ScienceQA/test/layers_representations/layer-all_token-last_images-qa_chat-format_qinst-type-singular_continue-fm_seed-42.safetensors"
# COCOQA Captioning Restval Text
# RESIDUAL_STREAM_ABS_PATH1="/orfeo/cephfs/scratch/dssc/lpalaciosflores/representations/llava-hf_llava-1.5-7b-hf/cocoqa_captioning_restval/layers_representations/layer-all_token-last_texts-qa_chat-format_continue-fm_downsample-2500_seed-42.safetensors"
# COCOQA Captioning Restval Image
# RESIDUAL_STREAM_ABS_PATH1="/orfeo/cephfs/scratch/dssc/lpalaciosflores/representations/llava-hf_llava-1.5-7b-hf/cocoqa_captioning_restval/layers_representations/layer-all_token-last_images-qa_chat-format_continue-fm_downsample-2500_seed-42.safetensors"

# Vicuna 1.5 7B
# MMLU
# RESIDUAL_STREAM_ABS_PATH2="/orfeo/cephfs/scratch/dssc/lpalaciosflores/representations/lmsys_vicuna-7b-v1.5/mmlu/test/layers_representations/layer-all_token-last_texts-qa_chat-format_qinst-type-singular_continue-fm_downsample-2500_seed-42.safetensors"
# ScienceQA Text
RESIDUAL_STREAM_ABS_PATH2="/orfeo/cephfs/scratch/dssc/lpalaciosflores/representations/lmsys_vicuna-7b-v1.5/ScienceQA/test/layers_representations/layer-all_token-last_texts-qa_chat-format_qinst-type-singular_continue-fm_seed-42.safetensors"
# ScienceQA Image
# RESIDUAL_STREAM_ABS_PATH2="/orfeo/cephfs/scratch/dssc/lpalaciosflores/representations/lmsys_vicuna-7b-v1.5/ScienceQA/test/layers_representations/layer-all_token-last_images-qa_chat-format_qinst-type-singular_continue-fm_seed-42.safetensors"
# COCOQA Captioning Restval Text
# RESIDUAL_STREAM_ABS_PATH2="/orfeo/cephfs/scratch/dssc/lpalaciosflores/representations/lmsys_vicuna-7b-v1.5/cocoqa_captioning_restval/layers_representations/layer-all_token-last_texts-qa_chat-format_continue-fm_downsample-2500_seed-42.safetensors"
# COCOQA Captioning Restval Image
# RESIDUAL_STREAM_ABS_PATH2="/orfeo/cephfs/scratch/dssc/lpalaciosflores/representations/lmsys_vicuna-7b-v1.5/cocoqa_captioning_restval/layers_representations/layer-all_token-last_images-qa_chat-format_continue-fm_downsample-2500_seed-42.safetensors"

MAXK="30"
FILENAME="analyze_residual_streams.py"
FILEPATH="$PROJECT_DIR/scripts/$FILENAME"

# DATASET="MMLU Test SubSet"
DATASET="ScienceQA Text Test Set"
# DATASET="ScienceQA IMG Test Set"
# DATASET="COCOQA Text SubSet"
# DATASET="COCOQA Image SubSet"

DOWNSAMPLE_SIZE=""
if [ -n "$DOWNSAMPLE_SIZE" ]; then
    DATASET="$DATASET [$DOWNSAMPLE_SIZE]"
fi
PLOT_TITLE="LLaVA 1.5 7B vs Vicuna 1.5 7B ($DATASET)"

# --- Argument Construction ---
# Initialize an array to hold script arguments
SCRIPT_ARGS=()
SCRIPT_ARGS+=(--residual-stream-path1 "$RESIDUAL_STREAM_ABS_PATH1")
SCRIPT_ARGS+=(--residual-stream-path2 "$RESIDUAL_STREAM_ABS_PATH2")
SCRIPT_ARGS+=(--result-parent-dir "$PROJECT_DIR")
SCRIPT_ARGS+=(--maxk "$MAXK")
SCRIPT_ARGS+=(--plot-title "$PLOT_TITLE")
if [ -n "$DOWNSAMPLE_SIZE" ]; then
    SCRIPT_ARGS+=(--downsample-size "$DOWNSAMPLE_SIZE")
fi

# --- Debugger ---
# Check if debugging is enabled via command-line argument (e.g., ./script.sh --debug)
DEBUGGER_ARGS=()
if [ "$1" = "--debug" ]; then
    DEBUGGER_ARGS=(-m debugpy --listen localhost:5678 --wait-for-client) # Changed to localhost
    echo "Debugger enabled - waiting for client connection on localhost:5678"
fi

# Print analysis configuration
echo "====== Analysis Configuration ======"
echo "Residual stream path 1: $RESIDUAL_STREAM_ABS_PATH1"
echo "Residual stream path 2: $RESIDUAL_STREAM_ABS_PATH2"
echo "Results parent directory: $PROJECT_DIR"
echo "Max k for neighborhood: $MAXK"
echo "Plot Title: $PLOT_TITLE"
echo "Downsample Size: $DOWNSAMPLE_SIZE"
echo "==============================="

# Analyze and plot neighborhood overlaps
echo -e "\n====== Analyzing Residual Streams ======"

# Run the command with the array to properly pass the arguments
python "${DEBUGGER_ARGS[@]}" "${FILEPATH}" "${SCRIPT_ARGS[@]}"

echo -e "\n====== Analysis Complete ======"
