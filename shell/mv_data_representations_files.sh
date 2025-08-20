#!/bin/bash

# Script to organize output layer versions in models_data_measures directories
# This script helps manage different versions of output layer files (chat, plain, last)

# Configuration
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="$PROJECT_DIR/results"

# Datasets to process
DATASETS=("coco_captioning" "cocoqa_img")

# Models to process
MODELS=("llava-1.5-7b-hf" "llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5")

# Function to print usage
print_usage() {
    echo "Usage: $0 [COMMAND] [VERSION]"
    echo ""
    echo "Commands:"
    echo "  organize    - Move all *output_layer_last* files into output_layer_versions folders"
    echo "  select      - Move selected version from output_layer_versions back to model folder"
    echo ""
    echo "Versions (for select command):"
    echo "  chat        - Select chat version (*chat.*)"
    echo "  plain       - Select plain version (*plain.*)"
    echo "  last        - Select last version (*last.*)"
    echo ""
    echo "Examples:"
    echo "  $0 organize"
    echo "  $0 select chat"
    echo "  $0 select plain"
    echo "  $0 select last"
}

# Function to create output_layer_versions directory
create_version_dir() {
    local dataset="$1"
    local model="$2"
    local version_dir="$RESULTS_DIR/$dataset/models_data_measures/$model/output_layer_versions"
    
    if [ ! -d "$version_dir" ]; then
        echo "Creating directory: $version_dir"
        mkdir -p "$version_dir"
    fi
}

# Function to organize files into output_layer_versions folders
organize_files() {
    echo "=== Organizing output layer files into version folders ==="
    
    for dataset in "${DATASETS[@]}"; do
        echo "Processing dataset: $dataset"
        
        for model in "${MODELS[@]}"; do
            local model_dir="$RESULTS_DIR/$dataset/models_data_measures/$model"
            
            if [ ! -d "$model_dir" ]; then
                echo "Model directory not found: $model_dir"
                echo "  Skipping $model (directory not found)"
                continue
            fi
            
            echo "  Processing model: $model"
            
            # Create output_layer_versions directory
            create_version_dir "$dataset" "$model"
            local version_dir="$model_dir/output_layer_versions"
            
            # Find and move all *output_layer_last* files
            local files_to_move=($(find "$model_dir" -maxdepth 1 -name "*output_layer_last*" -type f))
            
            if [ ${#files_to_move[@]} -eq 0 ]; then
                echo "    No *output_layer_last* files found"
                continue
            fi
            
            echo "    Moving ${#files_to_move[@]} files to output_layer_versions:"
            for file in "${files_to_move[@]}"; do
                local filename=$(basename "$file")
                echo "      $filename"
                mv "$file" "$version_dir/"
            done
        done
    done
    
    echo "=== Organization complete ==="
}

# Function to select version and move files back to model folder
select_version() {
    local version="$1"
    
    if [ -z "$version" ]; then
        echo "Error: Version not specified"
        print_usage
        exit 1
    fi
    
    case "$version" in
        "chat"|"plain"|"last")
            ;;
        *)
            echo "Error: Invalid version '$version'. Valid versions: chat, plain, last"
            print_usage
            exit 1
            ;;
    esac
    
    echo "=== Selecting $version version ==="
    
    for dataset in "${DATASETS[@]}"; do
        echo "Processing dataset: $dataset"
        
        for model in "${MODELS[@]}"; do
            local model_dir="$RESULTS_DIR/$dataset/models_data_measures/$model"
            local version_dir="$model_dir/output_layer_versions"
            
            if [ ! -d "$version_dir" ]; then
                echo "  Skipping $model (output_layer_versions directory not found)"
                continue
            fi
            
            echo "  Processing model: $model"
            
            # Find files matching the version pattern
            local pattern="*${version}*"
            local files_to_move=($(find "$version_dir" -maxdepth 1 -name "$pattern" -type f))
            
            if [ ${#files_to_move[@]} -eq 0 ]; then
                echo "    No files matching pattern '$pattern' found"
                continue
            fi
            
            echo "    Moving ${#files_to_move[@]} files to model directory:"
            for file in "${files_to_move[@]}"; do
                local filename=$(basename "$file")
                echo "      $filename"
                mv "$file" "$model_dir/$filename"
            done
        done
    done
    
    echo "=== Version selection complete ==="
}

# Function to list available versions
list_versions() {
    echo "=== Available versions in output_layer_versions folders ==="
    
    for dataset in "${DATASETS[@]}"; do
        echo "Dataset: $dataset"
        
        for model in "${MODELS[@]}"; do
            local version_dir="$RESULTS_DIR/$dataset/models_data_measures/$model/output_layer_versions"
            
            if [ ! -d "$version_dir" ]; then
                echo "  $model: No output_layer_versions directory"
                continue
            fi
            
            echo "  $model:"
            local files=($(find "$version_dir" -maxdepth 1 -name "*.safetensors" -type f))
            
            if [ ${#files[@]} -eq 0 ]; then
                echo "    No files found"
            else
                for file in "${files[@]}"; do
                    local filename=$(basename "$file")
                    echo "    $filename"
                done
            fi
        done
    done
}

# Main script logic
case "$1" in
    "organize")
        organize_files
        ;;
    "select")
        select_version "$2"
        ;;
    "list")
        list_versions
        ;;
    *)
        print_usage
        exit 1
        ;;
esac
