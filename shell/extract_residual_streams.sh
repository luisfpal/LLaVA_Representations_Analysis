#!/bin/bash
# This script extracts residual streams from models
# and saves them to a specified directory.

# --- Configuration Section ---
CHAT_MODE="true"
MM_NAME="llava-hf/llava-1.5-7b-hf"
LM_NAME="lmsys/vicuna-7b-v1.5"
QUESTION_INSTRUCTION_TYPE="singular"
GUIDE_TEXT=$'\nAnswer ONLY with the option\'s letter from the given choices directly.\n'
DATASET_NAME="cais/mmlu"
# DATASET_NAME="derek-thomas/ScienceQA"
DATASET_SPLIT="test"
TEXTS_QA="true"
IMAGES_QA="false"
LAYER_INDEX="all" # Can be a number, a comma-separated list, or "all"
TOKEN_INDEX="-1"
MEAN_OVER_TOKENS="false"
SEED=42

REPRESENTATIONS_DIR="$HOME/scratch/representations"
PROJECT_DIR="$HOME/multimodal_finetuned_representations"
FILENAME="extract_and_save_residual_streams.py"
FILEPATH="$PROJECT_DIR/scripts/$FILENAME"
MODEL_CACHE_DIR="$HOME/scratch/huggingface/hub"
DATASET_CACHE_DIR="$HOME/scratch/huggingface/datasets"

# --- Argument Construction ---
# Initialize an array to hold script arguments
SCRIPT_ARGS=()
SCRIPT_ARGS+=(--representations-dir "$REPRESENTATIONS_DIR")
SCRIPT_ARGS+=(--model-cache-dir "$MODEL_CACHE_DIR")
SCRIPT_ARGS+=(--dataset-name "$DATASET_NAME")
SCRIPT_ARGS+=(--dataset-cache-dir "$DATASET_CACHE_DIR")
SCRIPT_ARGS+=(--split "$DATASET_SPLIT")
SCRIPT_ARGS+=(--layer-index "$LAYER_INDEX")
SCRIPT_ARGS+=(--token-index "$TOKEN_INDEX")
if [ "$MEAN_OVER_TOKENS" = "true" ]; then
    SCRIPT_ARGS+=(--mean-over-tokens)
fi
if [ "$TEXTS_QA" = "true" ]; then
    SCRIPT_ARGS+=(--texts_qa)
fi
if [ "$IMAGES_QA" = "true" ]; then
    SCRIPT_ARGS+=(--images_qa)
fi
if [ "$CHAT_MODE" = "true" ]; then
    SCRIPT_ARGS+=(--chat-mode)
fi
SCRIPT_ARGS+=(--mm-name-or-path "$MM_NAME")
SCRIPT_ARGS+=(--lm-name-or-path "$LM_NAME")
SCRIPT_ARGS+=(--question-instruction-type "$QUESTION_INSTRUCTION_TYPE")
SCRIPT_ARGS+=(--seed "$SEED")

# --- Debugger ---
# Check if debugging is enabled via command-line argument (e.g., ./script.sh --debug)
DEBUGGER_ARGS=()
if [ "$1" = "--debug" ]; then
    DEBUGGER_ARGS=(-m debugpy --listen localhost:5678 --wait-for-client) # Changed to localhost
    echo "Debugger enabled - waiting for client connection on localhost:5678"
fi

# --- Create directories ---
mkdir -p "$REPRESENTATIONS_DIR"

# Print extraction configuration
echo "====== Extraction Configuration ======"
echo "Representations directory: $REPRESENTATIONS_DIR"
echo "MM model: $MM_NAME"
echo "LM model: $LM_NAME"
echo "Dataset: $DATASET_NAME"
echo "Split: $DATASET_SPLIT"
echo "Layer specification: $LAYER_INDEX"
echo "Token index: $TOKEN_INDEX"
echo "Mean over tokens: $MEAN_OVER_TOKENS"
echo "Question instruction type: $QUESTION_INSTRUCTION_TYPE"
echo "Chat mode: $CHAT_MODE"
echo "Guide text: $GUIDE_TEXT"
if [ "$TEXTS_QA" = "true" ]; then
    echo "Texts QA: $TEXTS_QA"
fi
if [ "$IMAGES_QA" = "true" ]; then
    echo "Images QA: $IMAGES_QA"
fi
echo "Seed: $SEED"
if [ "$1" = "--debug" ]; then
    echo "Debugger: ${DEBUGGER_ARGS[*]}"
fi
echo "==============================="

# Extract residual streams
echo -e "\n====== Extracting Residual Streams ======"

# Run the command with the array to properly pass the arguments
TOKENIZERS_PARALLELISM=false python "${DEBUGGER_ARGS[@]}" "${FILEPATH}" "${SCRIPT_ARGS[@]}"

echo -e "\n====== Extraction Complete ======"
echo "Residual streams saved to: $REPRESENTATIONS_DIR"
