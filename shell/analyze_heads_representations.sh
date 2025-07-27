#!/bin/bash
PROJECT_DIR="$HOME/multimodal_finetuned_representations"
REPRESENTATIONS_PATH="$HOME/scratch/representations"
# LLaVA 1.5 7B
# MMLU
# FILENAME1="layer-all_token-last_pproj_txt-qa_chat_qitype-singular_gtxt_cfm_ds2500_s42_batch1.safetensors"
# HEADS_RESIDUAL_STREAM_ABS_PATH1="$REPRESENTATIONS_PATH/llava-hf_llava-1.5-7b-hf/mmlu/test/heads_representations/$FILENAME1"
# ScienceQA Text
# FILENAME1="layer-all_token-last_pproj_txt-qa_chat_qitype-singular_gtxt_cfm_s42_batch1.safetensors"
# HEADS_RESIDUAL_STREAM_ABS_PATH1="$REPRESENTATIONS_PATH/llava-hf_llava-1.5-7b-hf/ScienceQA/test/heads_representations/$FILENAME1"
# ScienceQA Image
# FILENAME1="layer-all_token-last_pproj_img-qa_chat_qitype-singular_gtxt_cfm_s42_batch1.safetensors"
# HEADS_RESIDUAL_STREAM_ABS_PATH1="$REPRESENTATIONS_PATH/llava-hf_llava-1.5-7b-hf/ScienceQA/test/heads_representations/$FILENAME1"
# COCOQA Unified Text
# FILENAME1="layer-all_token-last_pproj_txt-qa_chat_gtxt_cfm_ds2500_s42_batch1.safetensors"
# HEADS_RESIDUAL_STREAM_ABS_PATH1="$REPRESENTATIONS_PATH/llava-hf_llava-1.5-7b-hf/cocoqa_unified/heads_representations/$FILENAME1"
# COCOQA Unified Image
FILENAME1="layer-all_token-last_pproj_img-qa_chat_gtxt_cfm_ds2500_s42_batch1.safetensors"
HEADS_RESIDUAL_STREAM_ABS_PATH1="$REPRESENTATIONS_PATH/llava-hf_llava-1.5-7b-hf/cocoqa_unified/heads_representations/$FILENAME1"

# Vicuna 1.5 7B
# MMLU
# FILENAME2="layer-all_token-last_pproj_txt-qa_chat_qitype-singular_gtxt_cfm_ds2500_s42_batch1.safetensors"
# HEADS_RESIDUAL_STREAM_ABS_PATH2="$REPRESENTATIONS_PATH/lmsys_vicuna-7b-v1.5/mmlu/test/heads_representations/$FILENAME2"
# ScienceQA Text
# FILENAME2="layer-all_token-last_pproj_txt-qa_chat_qitype-singular_gtxt_cfm_s42_batch1.safetensors"
# HEADS_RESIDUAL_STREAM_ABS_PATH2="$REPRESENTATIONS_PATH/lmsys_vicuna-7b-v1.5/ScienceQA/test/heads_representations/$FILENAME2"
# ScienceQA Image
# FILENAME2="layer-all_token-last_pproj_img-qa_chat_qitype-singular_gtxt_cfm_s42_batch1.safetensors"
# HEADS_RESIDUAL_STREAM_ABS_PATH2="$REPRESENTATIONS_PATH/lmsys_vicuna-7b-v1.5/ScienceQA/test/heads_representations/$FILENAME2"
# COCOQA Unified Text
# FILENAME2="layer-all_token-last_pproj_txt-qa_chat_gtxt_cfm_ds2500_s42_batch1.safetensors"
# HEADS_RESIDUAL_STREAM_ABS_PATH2="$REPRESENTATIONS_PATH/lmsys_vicuna-7b-v1.5/cocoqa_unified/heads_representations/$FILENAME2"
# COCOQA Unified Image
FILENAME2="layer-all_token-last_pproj_img-qa_chat_gtxt_cfm_ds2500_s42_batch1.safetensors"
HEADS_RESIDUAL_STREAM_ABS_PATH2="$REPRESENTATIONS_PATH/lmsys_vicuna-7b-v1.5/cocoqa_unified/heads_representations/$FILENAME2"

FILENAME="analyze_heads_representations.py"
FILEPATH="$PROJECT_DIR/scripts/$FILENAME"
MAXK="30"
ACCEPT_RATE="0.95"
# MEASURE="neighborhood_overlap"
MEASURE="svcca"
# MEASURE="linear_cka"
# MEASURE="rbf_cka"
# MEASURE="distance_correlation"

# DATASET="MMLU Test SubSet"
# DATASET="ScienceQA TXT Test Set"
# DATASET="ScienceQA IMG Test Set"
# DATASET="COCOQA TXT SubSet"
DATASET="COCOQA IMG SubSet"
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
SCRIPT_ARGS+=(--plot-title "$PLOT_TITLE")
SCRIPT_ARGS+=(--measure "$MEASURE")
if [ "$MEASURE" = "svcca" ]; then
    SCRIPT_ARGS+=(--accept-rate "$ACCEPT_RATE")
elif [ "$MEASURE" = "neighborhood_overlap" ]; then
    SCRIPT_ARGS+=(--maxk "$MAXK")
fi
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
echo "Measure: $MEASURE"
if [ "$MEASURE" = "svcca" ]; then
    echo "Accept Rate: $ACCEPT_RATE"
elif [ "$MEASURE" = "neighborhood_overlap" ]; then
    echo "Max k for neighborhood overlap: $MAXK"
fi
echo "Plot Title: $PLOT_TITLE"
echo "Downsample Size: $DOWNSAMPLE_SIZE"
echo "==============================="

# Analyze and plot neighborhood overlaps
echo -e "\n====== Analyzing Heads Representations ======"

# Run the command with the array to properly pass the arguments
python "${DEBUGGER_ARGS[@]}" "${FILEPATH}" "${SCRIPT_ARGS[@]}"

echo -e "\n====== Analysis Complete ======"
