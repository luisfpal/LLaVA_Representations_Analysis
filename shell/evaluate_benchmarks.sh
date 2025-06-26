#!/bin/bash

# --- Configuration Section ---
GUIDE_TEXT=$'\nAnswer ONLY with the option\'s letter from the given choices directly.\n'
# GUIDE_TEXT=$'Answer the question using a single word or phrase.\n'
DOWNSAMPLE_SIZE=""
MODEL_NAME="llava-hf/llava-1.5-7b-hf"
# MODEL_NAME="lmsys/vicuna-7b-v1.5"
# MODEL_NAME="meta-llama/Llama-2-7b-hf"
# MODEL_NAME="meta-llama/Llama-2-7b-chat-hf"
TEXT_MODEL="false"
REPLACE_MULTIMODAL_LM="true"
REPLACEMENT_LM_NAME="lmsys/vicuna-7b-v1.5"
REPLACE_MULTIMODAL_PROJECTOR="true"
PRETRAINED_PROJECTOR_NAME_OR_PATH="liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5"
# DATASET_NAME="cais/mmlu"
DATASET_NAME="derek-thomas/ScienceQA"
# DATASET_NAME="~/scratch/datasets/cocoqa_captioning_restval"
DATASET_SPLIT="test"
QUESTION_INSTRUCTION_TYPE="singular"
MAX_NEW_TOKENS=1
TEXTS_QA="false"
IMAGES_QA="true"

BATCH_SIZE=25
CHAT_MODE="true"
CONTINUE_FINAL_MESSAGE="true"
SEED=42
PROJECT_DIR="$HOME/multimodal_finetuned_representations"
FILENAME="evaluate_benchmarks.py"
FILEPATH="$PROJECT_DIR/scripts/$FILENAME"
BASE_DIR="$PROJECT_DIR/benchmarks_evaluation/"
MODEL_CACHE_DIR="$HOME/scratch/huggingface/hub"
DATASET_CACHE_DIR="$HOME/scratch/huggingface/datasets"

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
if [ "$CONTINUE_FINAL_MESSAGE" = "true" ]; then
    SCRIPT_ARGS+=(--continue-final-message)
fi
SCRIPT_ARGS+=(--question-instruction-type "$QUESTION_INSTRUCTION_TYPE")
SCRIPT_ARGS+=(--guide-text "$GUIDE_TEXT")
SCRIPT_ARGS+=(--batch-size "$BATCH_SIZE")
SCRIPT_ARGS+=(--max-new-tokens "$MAX_NEW_TOKENS")
if [ -n "$DOWNSAMPLE_SIZE" ]; then
    SCRIPT_ARGS+=(--downsample-size "$DOWNSAMPLE_SIZE")
fi
SCRIPT_ARGS+=(--seed "$SEED")
if [ "$REPLACE_MULTIMODAL_LM" = "true" ]; then
    SCRIPT_ARGS+=(--replacement-lm-name-or-path "$REPLACEMENT_LM_NAME")
fi
if [ "$REPLACE_MULTIMODAL_PROJECTOR" = "true" ] && [ -n "$PRETRAINED_PROJECTOR_NAME_OR_PATH" ]; then
    SCRIPT_ARGS+=(--pretrained-projector-name-or-path "$PRETRAINED_PROJECTOR_NAME_OR_PATH")
fi

# --- Debugger ---
# Check if debugging is enabled via command-line argument (e.g., ./script.sh --debug)
DEBUGGER_ARGS=()
if [ "$1" = "--debug" ]; then
    DEBUGGER_ARGS=(-m debugpy --listen localhost:5678 --wait-for-client) # Changed to localhost
    echo "Debugger enabled - waiting for client connection on localhost:5678"
fi

# --- Ensure the MODEL_NAME AND REPLACEMENT_LM_NAME are not the same if REPLACE_MULTIMODAL_LM is true ---
if [ "$REPLACE_MULTIMODAL_LM" = "true" ] && [ "$MODEL_NAME" = "$REPLACEMENT_LM_NAME" ]; then
    echo "Error: MODEL_NAME and REPLACEMENT_LM_NAME cannot be the same when REPLACE_MULTIMODAL_LM is true."
    exit 1
fi

# --- Ensure projector replacement is only used with multimodal models ---
if [ "$REPLACE_MULTIMODAL_PROJECTOR" = "true" ] && [ "$TEXT_MODEL" = "true" ]; then
    echo "Error: Cannot replace multimodal projector when TEXT_MODEL is true."
    exit 1
fi

# --- Ensure projector path is provided when replacement is enabled ---
if [ "$REPLACE_MULTIMODAL_PROJECTOR" = "true" ] && [ -z "$PRETRAINED_PROJECTOR_NAME_OR_PATH" ]; then
    echo "Error: PRETRAINED_PROJECTOR_NAME_OR_PATH must be provided when REPLACE_MULTIMODAL_PROJECTOR is true."
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
echo "Question instruction type: $QUESTION_INSTRUCTION_TYPE"
echo "Guide text: $GUIDE_TEXT"
echo "Batch size: $BATCH_SIZE"
echo "Max new tokens: $MAX_NEW_TOKENS"
if [ "$CHAT_MODE" = "true" ]; then
    echo "Chat mode: $CHAT_MODE"
fi
if [ "$CONTINUE_FINAL_MESSAGE" = "true" ]; then
    echo "Continue final message: $CONTINUE_FINAL_MESSAGE"
fi
if [ -n "$DOWNSAMPLE_SIZE" ]; then
    echo "Downsample size: $DOWNSAMPLE_SIZE"
fi
if [ "$TEXTS_QA" = "true" ]; then
    echo "Texts QA: $TEXTS_QA"
fi
if [ "$IMAGES_QA" = "true" ]; then
    echo "Images QA: $IMAGES_QA"
fi
echo "Seed: $SEED"
if [ "$REPLACE_MULTIMODAL_LM" = "true" ]; then
    echo "Replacing Multimodal LM with: $REPLACEMENT_LM_NAME"
fi
if [ "$REPLACE_MULTIMODAL_PROJECTOR" = "true" ]; then
    echo "Replacing Multimodal Projector with: $PRETRAINED_PROJECTOR_NAME_OR_PATH"
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
