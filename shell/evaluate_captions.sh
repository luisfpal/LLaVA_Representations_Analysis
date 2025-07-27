#!/bin/bash

# --- Configuration Section ---
CAPTIONS_FILE="~/scratch/captioning_results/llava-hf_llava-1.5-7b-hf/cocoqa_captioning_restval/captioning_results/captions_chat_gtxt_cfm_s42.jsonl"
DATASET_PATH="~/scratch/datasets/cocoqa_captioning_restval"
OUTPUT_FILE="~/scratch/captioning_results/evaluation_results.json"

# Project configuration
PROJECT_DIR="$HOME/multimodal_finetuned_representations"
FILENAME="evaluate_captions.py"
FILEPATH="$PROJECT_DIR/scripts/$FILENAME"

# --- Argument Construction ---
SCRIPT_ARGS=()
SCRIPT_ARGS+=(--captions-file "$CAPTIONS_FILE")
SCRIPT_ARGS+=(--dataset-path "$DATASET_PATH")
SCRIPT_ARGS+=(--output-file "$OUTPUT_FILE")

# --- Debug Configuration ---
DEBUGGER_ARGS=()
if [ "$1" = "--debug" ]; then
    DEBUGGER_ARGS=("python" "-m" "pdb")
fi

# --- Display Configuration ---
echo "==============================="
echo "CAPTION EVALUATION CONFIGURATION"
echo "==============================="
echo "Captions file: $CAPTIONS_FILE"
echo "Dataset path: $DATASET_PATH"
echo "Output file: $OUTPUT_FILE"
if [ "$1" = "--debug" ]; then
    echo "Debugger: ${DEBUGGER_ARGS[*]}"
fi
echo "==============================="

echo -e "\n====== Starting Caption Evaluation ======"

# Run the command with the array to properly pass the arguments
python "${DEBUGGER_ARGS[@]}" "${FILEPATH}" "${SCRIPT_ARGS[@]}"

echo -e "\n====== Caption Evaluation Complete ======"
echo "Results saved to: $OUTPUT_FILE" 