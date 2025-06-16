#!/bin/bash
PROJECT_DIR="$HOME/multimodal_finetuned_representations"

# LLaVA 1.5 7B
# MMLU
# HEADS_RESIDUAL_STREAM_ABS_PATH1="/orfeo/cephfs/scratch/dssc/lpalaciosflores/representations/llava-hf_llava-1.5-7b-hf/mmlu/test/heads_representations/layer-all_token-last_texts-qa_chat-format_qinst-type-singular_guide-text_continue-fm_downsample-2500_seed-42_batch-20.safetensors"
# ScienceQA Text
# HEADS_RESIDUAL_STREAM_ABS_PATH1="/orfeo/cephfs/scratch/dssc/lpalaciosflores/representations/llava-hf_llava-1.5-7b-hf/ScienceQA/test/heads_representations/layer-all_token-last_texts-qa_chat-format_qinst-type-singular_guide-text_continue-fm_seed-42_batch-20.safetensors"
# ScienceQA Image
# HEADS_RESIDUAL_STREAM_ABS_PATH1="/orfeo/cephfs/scratch/dssc/lpalaciosflores/representations/llava-hf_llava-1.5-7b-hf/ScienceQA/test/heads_representations/layer-all_token-last_images-qa_chat-format_qinst-type-singular_guide-text_continue-fm_seed-42_batch-20.safetensors"
# COCOQA Captioning Restval Text
# HEADS_RESIDUAL_STREAM_ABS_PATH1="/orfeo/cephfs/scratch/dssc/lpalaciosflores/representations/llava-hf_llava-1.5-7b-hf/cocoqa_captioning_restval/heads_representations/layer-all_token-last_texts-qa_chat-format_continue-fm_downsample-2500_seed-42_batch-20.safetensors"
# COCOQA Captioning Restval Image
HEADS_RESIDUAL_STREAM_ABS_PATH1="/orfeo/cephfs/scratch/dssc/lpalaciosflores/representations/llava-hf_llava-1.5-7b-hf/cocoqa_captioning_restval/heads_representations/layer-all_token-last_images-qa_chat-format_continue-fm_downsample-2500_seed-42_batch-20.safetensors"

# Vicuna 1.5 7B
# MMLU
# HEADS_RESIDUAL_STREAM_ABS_PATH2="/orfeo/cephfs/scratch/dssc/lpalaciosflores/representations/lmsys_vicuna-7b-v1.5/mmlu/test/heads_representations/layer-all_token-last_texts-qa_chat-format_qinst-type-singular_guide-text_continue-fm_downsample-2500_seed-42_batch-20.safetensors"
# ScienceQA Text
# HEADS_RESIDUAL_STREAM_ABS_PATH2="/orfeo/cephfs/scratch/dssc/lpalaciosflores/representations/lmsys_vicuna-7b-v1.5/ScienceQA/test/heads_representations/layer-all_token-last_texts-qa_chat-format_qinst-type-singular_guide-text_continue-fm_seed-42_batch-20.safetensors"
# ScienceQA Image
# HEADS_RESIDUAL_STREAM_ABS_PATH2="/orfeo/cephfs/scratch/dssc/lpalaciosflores/representations/lmsys_vicuna-7b-v1.5/ScienceQA/test/heads_representations/layer-all_token-last_images-qa_chat-format_qinst-type-singular_guide-text_continue-fm_seed-42_batch-20.safetensors"
# COCOQA Captioning Restval Text
# HEADS_RESIDUAL_STREAM_ABS_PATH2="/orfeo/cephfs/scratch/dssc/lpalaciosflores/representations/lmsys_vicuna-7b-v1.5/cocoqa_captioning_restval/heads_representations/layer-all_token-last_texts-qa_chat-format_continue-fm_downsample-2500_seed-42_batch-20.safetensors"
# COCOQA Captioning Restval Image
HEADS_RESIDUAL_STREAM_ABS_PATH2="/orfeo/cephfs/scratch/dssc/lpalaciosflores/representations/lmsys_vicuna-7b-v1.5/cocoqa_captioning_restval/heads_representations/layer-all_token-last_images-qa_chat-format_continue-fm_downsample-2500_seed-42_batch-20.safetensors"

MAXK="30"
FILENAME="analyze_heads_representations.py"
FILEPATH="$PROJECT_DIR/scripts/$FILENAME"

# DATASET="MMLU Test SubSet"
# DATASET="ScienceQA Text Test Set"
# DATASET="ScienceQA IMG Test Set"
# DATASET="COCOQA Text SubSet"
# DATASET="COCOQA Image SubSet"
DOWNSAMPLE_SIZE="2500"
if [ -n "$DOWNSAMPLE_SIZE" ]; then
    DATASET="$DATASET [$DOWNSAMPLE_SIZE]"
fi
PLOT_TITLE="LLaVA 1.5 7B vs Vicuna 1.5 7B ($DATASET)"

# --- Argument Construction ---
# Initialize an array to hold script arguments
SCRIPT_ARGS=()
SCRIPT_ARGS+=(--heads-residual-stream-path1 "$HEADS_RESIDUAL_STREAM_ABS_PATH1")
SCRIPT_ARGS+=(--heads-residual-stream-path2 "$HEADS_RESIDUAL_STREAM_ABS_PATH2")
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
echo "Heads Residual Stream Path 1: $HEADS_RESIDUAL_STREAM_ABS_PATH1"
echo "Heads Residual Stream Path 2: $HEADS_RESIDUAL_STREAM_ABS_PATH2"
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
