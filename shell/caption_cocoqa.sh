#!/bin/bash

# --- Configuration Section ---
MODEL_NAME="llava-hf/llava-1.5-7b-hf"
# MODEL_NAME="llava-hf/llava-1.5-13b-hf"
# MODEL_NAME="llava-hf/llava-v1.6-34b-hf"

# Dataset configuration
DATASET_PATH="~/scratch/datasets/cocoqa_captioning_restval"
BASE_DIR="~/scratch/captioning_results"

# Generation parameters for captioning
MAX_NEW_TOKENS=50
DO_SAMPLE="true"
TEMPERATURE=0.7
TOP_P=0.9
NUM_BEAMS=1

# Model replacement options (optional)
REPLACE_MULTIMODAL_LM="false"
REPLACEMENT_LM_NAME="lmsys/vicuna-7b-v1.5"
REPLACE_MULTIMODAL_PROJECTOR="false"
PRETRAINED_PROJECTOR_NAME_OR_PATH="liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5"

# Processing parameters
BATCH_SIZE=4
SEED=42
PROJECT_DIR="$HOME/multimodal_finetuned_representations"
FILENAME="caption_cocoqa_with_llava.py"
FILEPATH="$PROJECT_DIR/scripts/$FILENAME"
MODEL_CACHE_DIR="$HOME/scratch/huggingface/hub"

# --- Argument Construction ---
SCRIPT_ARGS=()
SCRIPT_ARGS+=(--model-name-or-path "$MODEL_NAME")
SCRIPT_ARGS+=(--model-cache-dir "$MODEL_CACHE_DIR")
SCRIPT_ARGS+=(--dataset-path "$DATASET_PATH")
SCRIPT_ARGS+=(--base-dir "$BASE_DIR")
SCRIPT_ARGS+=(--batch-size "$BATCH_SIZE")
SCRIPT_ARGS+=(--max-new-tokens "$MAX_NEW_TOKENS")
if [ "$DO_SAMPLE" = "true" ]; then
    SCRIPT_ARGS+=(--do-sample)
fi
SCRIPT_ARGS+=(--temperature "$TEMPERATURE")
SCRIPT_ARGS+=(--top-p "$TOP_P")
SCRIPT_ARGS+=(--num-beams "$NUM_BEAMS")
SCRIPT_ARGS+=(--seed "$SEED")
if [ "$REPLACE_MULTIMODAL_LM" = "true" ]; then
    SCRIPT_ARGS+=(--replacement-lm-name-or-path "$REPLACEMENT_LM_NAME")
fi
if [ "$REPLACE_MULTIMODAL_PROJECTOR" = "true" ]; then
    SCRIPT_ARGS+=(--pretrained-projector-name-or-path "$PRETRAINED_PROJECTOR_NAME_OR_PATH")
fi

# --- Debug Configuration ---
DEBUGGER_ARGS=()
if [ "$1" = "--debug" ]; then
    DEBUGGER_ARGS=("python" "-m" "pdb")
fi

# --- Display Configuration ---
echo "==============================="
echo "COCO-QA Captioning Configuration"
echo "==============================="
echo "Model: $MODEL_NAME"
echo "Dataset path: $DATASET_PATH"
echo "Base dir: $BASE_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Max new tokens: $MAX_NEW_TOKENS"
echo "Do sample: $DO_SAMPLE"
echo "Temperature: $TEMPERATURE"
echo "Top-p: $TOP_P"
echo "Num beams: $NUM_BEAMS"
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

echo -e "\n====== Starting Captioning ======"

# Set PyTorch memory config
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run the command with the array to properly pass the arguments
TOKENIZERS_PARALLELISM=false python "${DEBUGGER_ARGS[@]}" "${FILEPATH}" "${SCRIPT_ARGS[@]}"

echo -e "\n====== Captioning Complete ======"
echo "Results saved under: $BASE_DIR" 