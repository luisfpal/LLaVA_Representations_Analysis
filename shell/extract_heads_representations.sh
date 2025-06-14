#!/bin/bash

# --- Configuration Section ---
DATASET_NAME="cais/mmlu"
# DATASET_NAME="derek-thomas/ScienceQA"
DATASET_SPLIT="test"
QUESTION_INSTRUCTION_TYPE="plural"
GUIDE_TEXT=$'\nAnswer ONLY with the option\'s letter from the given choices directly.\n'
CONTINUE_FINAL_MESSAGE="true"
TEXTS_QA="true"
IMAGES_QA="false"
LAYER_INDEX="all"
TOKENS_MODE="last"
BATCH_SIZE=10
MM_NAME="llava-hf/llava-1.5-7b-hf"
LM_NAME="lmsys/vicuna-7b-v1.5"
SEED=42

REPRESENTATIONS_DIR="$HOME/scratch/representations"
PROJECT_DIR="$HOME/multimodal_finetuned_representations"
FILENAME="extract_and_save_heads_representations.py"
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
SCRIPT_ARGS+=(--tokens-mode "$TOKENS_MODE")
if [ "$TEXTS_QA" = "true" ]; then
    SCRIPT_ARGS+=(--texts_qa)
fi
if [ "$IMAGES_QA" = "true" ]; then
    SCRIPT_ARGS+=(--images_qa)
fi
SCRIPT_ARGS+=(--mm-name-or-path "$MM_NAME")
SCRIPT_ARGS+=(--lm-name-or-path "$LM_NAME")
SCRIPT_ARGS+=(--question-instruction-type "$QUESTION_INSTRUCTION_TYPE")
SCRIPT_ARGS+=(--continue-final-message)
SCRIPT_ARGS+=(--guide-text "$GUIDE_TEXT")
if [ "$CONTINUE_FINAL_MESSAGE" = "true" ]; then
    SCRIPT_ARGS+=(--continue-final-message)
fi
SCRIPT_ARGS+=(--batch-size "$BATCH_SIZE")
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
echo "Tokens mode: $TOKENS_MODE"
echo "Question instruction type: $QUESTION_INSTRUCTION_TYPE"
echo "Guide text: $GUIDE_TEXT"
echo "Continue final message: $CONTINUE_FINAL_MESSAGE"
if [ "$TEXTS_QA" = "true" ]; then
    echo "Texts QA: $TEXTS_QA"
fi
if [ "$IMAGES_QA" = "true" ]; then
    echo "Images QA: $IMAGES_QA"
fi
echo "Batch size: $BATCH_SIZE"
echo "Seed: $SEED"
if [ "$1" = "--debug" ]; then
    echo "Debugger: ${DEBUGGER_ARGS[*]}"
fi
echo "==============================="

# Extract residual streams
echo -e "\n====== LLMs Heads Residual Stream Extraction ======"

# Set PyTorch memory config
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run the command with the array to properly pass the arguments
TOKENIZERS_PARALLELISM=false python "${DEBUGGER_ARGS[@]}" "${FILEPATH}" "${SCRIPT_ARGS[@]}"

echo -e "\n====== Extraction Complete ======"
echo "Residual streams saved to: $REPRESENTATIONS_DIR"
