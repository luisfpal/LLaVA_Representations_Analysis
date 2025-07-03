#!/bin/bash
# This script computes intrinsic dimension of layer representations
PROJECT_DIR="$HOME/multimodal_finetuned_representations"
REPRESENTATIONS_PATH="$HOME/scratch/representations"

# Configuration for which model and dataset to analyze
# MODEL="llava-hf_llava-1.5-7b-hf"
MODEL="lmsys_vicuna-7b-v1.5"
DATASET="cocoqa_unified"
MODALITY="txt"

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

# Intrinsic dimension parameters
ALGORITHM="scaling_gride" # Options: twoNN, MLE, scaling_gride
K="16"                    # For MLE/scaling_gride algorithm
FRACTION="0.9"            # For twoNN algorithm
RANGE_MAX="100"           # For scaling_gride algorithm
FULL_OUTPUT="false"       # For MLE/scaling_gride algorithm
DOWNSAMPLE_SIZE=""        # Leave empty for no downsampling, or set a number like 2500

# Script configuration
SCRIPT_FILENAME="intrinsic_dimension_estimation.py"
SCRIPT_FILEPATH="$PROJECT_DIR/scripts/$SCRIPT_FILENAME"

# --- Argument Construction ---
SCRIPT_ARGS=()
SCRIPT_ARGS+=(--residual-stream-path "$RESIDUAL_STREAM_ABS_PATH")
SCRIPT_ARGS+=(--result-parent-dir "$PROJECT_DIR")
SCRIPT_ARGS+=(--algorithm "$ALGORITHM")

if [ "$ALGORITHM" == "twoNN" ]; then
    SCRIPT_ARGS+=(--fraction "$FRACTION")
fi

if [ "$ALGORITHM" == "MLE" ]; then
    SCRIPT_ARGS+=(--k "$K")
    if [ "$FULL_OUTPUT" == "true" ]; then
        SCRIPT_ARGS+=(--full-output)
    fi
elif [ "$ALGORITHM" == "scaling_gride" ]; then
    SCRIPT_ARGS+=(--k "$K")
    SCRIPT_ARGS+=(--range-max "$RANGE_MAX")
fi

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
# Validate algorithm choice
if [ "$ALGORITHM" != "twoNN" ] && [ "$ALGORITHM" != "MLE" ] && [ "$ALGORITHM" != "scaling_gride" ]; then
    echo "Error: Invalid algorithm '$ALGORITHM'. Must be 'twoNN', 'MLE', or 'scaling_gride'."
    exit 1
fi

echo "====== Intrinsic Dimension Analysis Configuration ======"
echo "Model: $MODEL"
echo "Dataset: $DATASET ($MODALITY modality)"
echo "Residual stream path: $RESIDUAL_STREAM_ABS_PATH"
echo "Results parent directory: $PROJECT_DIR"
echo "Algorithm: $ALGORITHM"
if [ "$ALGORITHM" == "MLE" ]; then
    echo "K (for MLE): $K"
    echo "Full output: $FULL_OUTPUT"
elif [ "$ALGORITHM" == "scaling_gride" ]; then
    echo "K (for scaling_gride): $K"
    echo "Range max (for scaling_gride): $RANGE_MAX"
elif [ "$ALGORITHM" == "twoNN" ]; then
    echo "Fraction (for twoNN): $FRACTION"
fi
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

# Compute intrinsic dimension
echo -e "\n====== Computing Intrinsic Dimension ======"

# Run the command with the array to properly pass the arguments
python "${DEBUGGER_ARGS[@]}" "${SCRIPT_FILEPATH}" "${SCRIPT_ARGS[@]}"

echo -e "\n====== Analysis Complete ======"
