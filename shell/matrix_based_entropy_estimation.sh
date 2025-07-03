#!/bin/bash
# This script computes Rényi entropy of layer representations
PROJECT_DIR="$HOME/multimodal_finetuned_representations"
REPRESENTATIONS_PATH="$HOME/scratch/representations"

# Configuration for which model and dataset to analyze
# MODEL="llava-hf_llava-1.5-7b-hf"
MODEL="lmsys_vicuna-7b-v1.5"
DATASET="cocoqa_unified"
MODALITY="img"

# Construct the filename based on the configuration
if [ "$DATASET" == "ScienceQA" ]; then
    if [ "$MODALITY" == "txt" ]; then
        FILENAME="layer-all_token-last_pproj_txt-qa_chat_qitype-singular_gtxt_cfm_s42.safetensors"
    elif [ "$MODALITY" == "img" ]; then
        FILENAME="layer-all_token-last_pproj_img-qa_chat_qitype-singular_gtxt_cfm_s42.safetensors"
    fi
    SPLIT="test"
elif [ "$DATASET" == "mmlu" ]; then
    FILENAME="layer-all_token-last_pproj_txt-qa_chat_qitype-singular_gtxt_cfm_ds2500_s42.safetensors"
    SPLIT="test"
elif [ "$DATASET" == "cocoqa_unified" ]; then
    if [ "$MODALITY" == "txt" ]; then
        FILENAME="layer-all_token-last_pproj_txt-qa_chat_gtxt_cfm_ds2500_s42.safetensors"
    elif [ "$MODALITY" == "img" ]; then
        FILENAME="layer-all_token-last_pproj_img-qa_chat_gtxt_cfm_ds2500_s42.safetensors"
    fi
    SPLIT=""
fi

# Construct full path
if [ -n "$SPLIT" ]; then
    RESIDUAL_STREAM_ABS_PATH="$REPRESENTATIONS_PATH/$MODEL/$DATASET/$SPLIT/layers_representations/$FILENAME"
else
    RESIDUAL_STREAM_ABS_PATH="$REPRESENTATIONS_PATH/$MODEL/$DATASET/layers_representations/$FILENAME"
fi

# Rényi entropy parameters
ALPHA="1.0"        # Rényi entropy order (α = 1 for Shannon entropy)
DOWNSAMPLE_SIZE="" # Leave empty for no downsampling, or set a number like 2500

# Script configuration
SCRIPT_FILENAME="matrix_based_entropy_estimation.py"
SCRIPT_FILEPATH="$PROJECT_DIR/scripts/$SCRIPT_FILENAME"

# --- Argument Construction ---
SCRIPT_ARGS=()
SCRIPT_ARGS+=(--residual-stream-path "$RESIDUAL_STREAM_ABS_PATH")
SCRIPT_ARGS+=(--result-parent-dir "$PROJECT_DIR")
SCRIPT_ARGS+=(--alpha "$ALPHA")

if [ -n "$DOWNSAMPLE_SIZE" ]; then
    SCRIPT_ARGS+=(--downsample-size "$DOWNSAMPLE_SIZE")
fi

# --- Debugger ---
# Check if debugging is enabled via command-line argument (e.g., ./script.sh --debug)
DEBUGGER_ARGS=()
if [ "$1" = "--debug" ]; then
    DEBUGGER_ARGS=(-m debugpy --listen localhost:5678 --wait-for-client)
    echo "Debugger enabled - waiting for client connection on localhost:5678"
fi

# Print analysis configuration
echo "====== Rényi Entropy Analysis Configuration ======"
echo "Model: $MODEL"
echo "Dataset: $DATASET ($MODALITY modality)"
echo "Residual stream path: $RESIDUAL_STREAM_ABS_PATH"
echo "Results parent directory: $PROJECT_DIR"
echo "Alpha (Rényi order): $ALPHA"
if [ -n "$DOWNSAMPLE_SIZE" ]; then
    echo "Downsample Size: $DOWNSAMPLE_SIZE"
fi
echo "==============================="

# Check if the residual stream file exists
if [ ! -f "$RESIDUAL_STREAM_ABS_PATH" ]; then
    echo "Error: Residual stream file not found: $RESIDUAL_STREAM_ABS_PATH"
    echo "Please check the model, dataset, and filename configuration."
    exit 1
fi

# Compute Rényi entropy
echo -e "\n====== Computing Rényi Entropy ======"

# Run the command with the array to properly pass the arguments
python "${DEBUGGER_ARGS[@]}" "${SCRIPT_FILEPATH}" "${SCRIPT_ARGS[@]}"

echo -e "\n====== Analysis Complete ======"
