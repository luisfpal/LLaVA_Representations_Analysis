#!/bin/bash

# --- Configuration Section ---
CHAT_MODE="true"
GUIDE_TEXT=$'\nAnswer ONLY with the option\'s letter from the given choices directly.\n'
QUESTION_INSTRUCTION_TYPE="singular"
TEXT_MODEL="false"
REPLACE_LLAVA_LM="false"
MODEL_NAME="llava-hf/llava-1.5-7b-hf"
# MODEL_NAME="lmsys/vicuna-7b-v1.5"
# MODEL_NAME="meta-llama/Llama-2-7b-hf"
# MODEL_NAME="meta-llama/Llama-2-7b-chat-hf"
DATASET_NAME="cais/mmlu"
# DATASET_NAME="derek-thomas/ScienceQA"
DATASET_SPLIT="test"
TEXTS_QA="true"
IMAGES_QA="false"
REPLACEMENT_LM_NAME="lmsys/vicuna-7b-v1.5"
BATCH_SIZE=26

PROJECT_DIR="$HOME/multimodal_finetuned_representations"
FILENAME="multiple_choice_benchmarks_evaluation.py"
FILEPATH="$PROJECT_DIR/scripts/$FILENAME"
BASE_DIR="$PROJECT_DIR/benchmarks_evaluation/"
MODEL_CACHE_DIR="$HOME/scratch/huggingface/hub"
DATASET_CACHE_DIR="$HOME/scratch/huggingface/datasets"
SEED=42

# --- Argument Construction ---
# Initialize an array to hold script arguments
SCRIPT_ARGS=()
SCRIPT_ARGS+=(--model-name-or-path "$MODEL_NAME")
SCRIPT_ARGS+=(--model-cache-dir "$MODEL_CACHE_DIR")
SCRIPT_ARGS+=(--dataset-name "$DATASET_NAME")
SCRIPT_ARGS+=(--dataset-cache-dir "$DATASET_CACHE_DIR")
SCRIPT_ARGS+=(--split "$DATASET_SPLIT")
SCRIPT_ARGS+=(--base-dir "$BASE_DIR")
if [ "$TEXT_MODEL" = "true" ]; then
    SCRIPT_ARGS+=(--text-model)
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
SCRIPT_ARGS+=(--question-instruction-type "$QUESTION_INSTRUCTION_TYPE")
SCRIPT_ARGS+=(--guide-text "$GUIDE_TEXT")
SCRIPT_ARGS+=(--batch-size "$BATCH_SIZE")
SCRIPT_ARGS+=(--seed "$SEED")
if [ "$REPLACE_LLAVA_LM" = "true" ]; then
    SCRIPT_ARGS+=(--replacement-lm-name-or-path "$REPLACEMENT_LM_NAME")
fi

# --- Debugger ---
# Check if debugging is enabled via command-line argument (e.g., ./script.sh --debug)
DEBUGGER_ARGS=()
if [ "$1" = "--debug" ]; then
    DEBUGGER_ARGS=(-m debugpy --listen localhost:5678 --wait-for-client) # Changed to localhost
    echo "Debugger enabled - waiting for client connection on localhost:5678"
fi

# --- Ensure the MODEL_NAME AND REPLACEMENT_LM_NAME are not the same if REPLACE_LLAVA_LM is true ---
if [ "$REPLACE_LLAVA_LM" = "true" ] && [ "$MODEL_NAME" = "$REPLACEMENT_LM_NAME" ]; then
    echo "Error: MODEL_NAME and REPLACEMENT_LM_NAME cannot be the same when REPLACE_LLAVA_LM is true."
    exit 1
fi

# --- Execution ---
# Print evaluation configuration
echo "====== Evaluation Configuration ======"
echo "Model: $MODEL_NAME"
echo "Text model: $TEXT_MODEL"
echo "Dataset: $DATASET_NAME"
echo "Dataset split: $DATASET_SPLIT"
echo "Base dir: $BASE_DIR"
echo "Chat mode: $CHAT_MODE"
echo "Question instruction type: $QUESTION_INSTRUCTION_TYPE"
echo "Guide text: $GUIDE_TEXT"
echo "Batch size: $BATCH_SIZE"
if [ "$TEXTS_QA" = "true" ]; then
    echo "Texts QA: $TEXTS_QA"
fi
if [ "$IMAGES_QA" = "true" ]; then
    echo "Images QA: $IMAGES_QA"
fi
echo "Seed: $SEED"
if [ "$REPLACE_LLAVA_LM" = "true" ]; then
    echo "Replacing Llava LM with: $REPLACEMENT_LM_NAME"
fi
if [ "$1" = "--debug" ]; then
    echo "Debugger: ${DEBUGGER_ARGS[*]}"
fi
echo "==============================="

echo -e "\n====== Evaluating Model ======"

# Set PyTorch memory config
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run the command with the array to properly pass the arguments
TOKENIZERS_PARALLELISM=false python "${DEBUGGER_ARGS[@]}" "${FILEPATH}" "${SCRIPT_ARGS[@]}"

echo -e "\n====== Evaluation Complete ======"
echo "Results saved under: $BASE_DIR"
