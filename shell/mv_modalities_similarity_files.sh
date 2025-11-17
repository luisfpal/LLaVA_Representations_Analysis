#!/bin/bash

# Script to manage modalities_similarity files for different versions
# Focuses on coco_captioning and cocoqa_img datasets

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="$PROJECT_DIR/results"
DATASETS=("coco_captioning" "cocoqa_img")
MODELS=("llava-1.5-7b-hf" "llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5")

print_usage() {
    echo "Usage: $0 <command> [version]"
    echo ""
    echo "Commands:"
    echo "  organize    - Create output_layer_versions directories and move files"
    echo "  select <v>  - Move files with version <v> back to model directory"
    echo "  list        - List available versions for each model"
    echo ""
    echo "Versions:"
    echo "  chat        - Files with _chat suffix"
    echo "  plain       - Files with _plain suffix"
    echo "  none        - Files without suffix (base version)"
    echo ""
    echo "Examples:"
    echo "  $0 organize"
    echo "  $0 select chat"
    echo "  $0 list"
}

create_version_dir() {
    local modalities_dir="$1"
    local version_dir="$modalities_dir/output_layer_versions"
    
    if [[ ! -d "$version_dir" ]]; then
        echo "Creating directory: $version_dir"
        mkdir -p "$version_dir"
    fi
}

organize_files() {
    echo "Organizing modalities_similarity files..."
    echo "========================================"
    
    for dataset in "${DATASETS[@]}"; do
        echo "Processing dataset: $dataset"
        for model in "${MODELS[@]}"; do
            local modalities_dir="$RESULTS_DIR/$dataset/$model/modalities_similarity"
            
            if [[ ! -d "$modalities_dir" ]]; then
                echo "  Skipping $model - modalities_similarity directory not found"
                continue
            fi
            
            echo "  Processing model: $model"
            create_version_dir "$modalities_dir"
            
            # Move files to output_layer_versions directory
            local files_moved=0
            for file in "$modalities_dir"/output_layer_sample_*.safetensors; do
                if [[ -f "$file" ]]; then
                    local filename=$(basename "$file")
                    echo "    Moving: $filename"
                    mv "$file" "$modalities_dir/output_layer_versions/$filename"
                    ((files_moved++))
                fi
            done
            
            if [[ $files_moved -eq 0 ]]; then
                echo "    No output_layer_sample_*.safetensors files found"
            else
                echo "    Moved $files_moved files to output_layer_versions/"
            fi
        done
        echo ""
    done
    
    echo "Organization complete!"
}

select_version() {
    local version="$1"
    
    if [[ -z "$version" ]]; then
        echo "Error: Please specify a version (chat, plain, or none)"
        exit 1
    fi
    
    echo "Selecting version: $version"
    echo "=========================="
    
    for dataset in "${DATASETS[@]}"; do
        echo "Processing dataset: $dataset"
        for model in "${MODELS[@]}"; do
            local modalities_dir="$RESULTS_DIR/$dataset/$model/modalities_similarity"
            local version_dir="$modalities_dir/output_layer_versions"
            
            if [[ ! -d "$version_dir" ]]; then
                echo "  Skipping $model - output_layer_versions directory not found"
                continue
            fi
            
            echo "  Processing model: $model"
            
            # Find files matching the version pattern
            local files_to_move=()
            if [[ "$version" == "none" ]]; then
                # Files without suffix (base version)
                for file in "$version_dir"/output_layer_sample_*.safetensors; do
                    if [[ -f "$file" ]]; then
                        local filename=$(basename "$file")
                        if [[ ! "$filename" =~ _(chat|plain)\.safetensors$ ]]; then
                            files_to_move+=("$file")
                        fi
                    fi
                done
            else
                # Files with specific suffix
                for file in "$version_dir"/output_layer_sample_*_${version}.safetensors; do
                    if [[ -f "$file" ]]; then
                        files_to_move+=("$file")
                    fi
                done
            fi
            
            if [[ ${#files_to_move[@]} -eq 0 ]]; then
                echo "    No files found for version: $version"
                continue
            fi
            
            echo "    Moving ${#files_to_move[@]} files to modalities_similarity directory:"
            for file in "${files_to_move[@]}"; do
                local filename=$(basename "$file")
                echo "      $filename"
                mv "$file" "$modalities_dir/$filename"
            done
        done
        echo ""
    done
    
    echo "Version selection complete!"
}

list_versions() {
    echo "Available versions for each model:"
    echo "=================================="
    
    for dataset in "${DATASETS[@]}"; do
        echo "Dataset: $dataset"
        for model in "${MODELS[@]}"; do
            local modalities_dir="$RESULTS_DIR/$dataset/$model/modalities_similarity"
            local version_dir="$modalities_dir/output_layer_versions"
            
            if [[ ! -d "$version_dir" ]]; then
                echo "  $model: No output_layer_versions directory found"
                continue
            fi
            
            echo "  $model:"
            
            # Count files by version
            local base_count=0
            local chat_count=0
            local plain_count=0
            
            for file in "$version_dir"/output_layer_sample_*.safetensors; do
                if [[ -f "$file" ]]; then
                    local filename=$(basename "$file")
                    if [[ "$filename" =~ _chat\.safetensors$ ]]; then
                        ((chat_count++))
                    elif [[ "$filename" =~ _plain\.safetensors$ ]]; then
                        ((plain_count++))
                    else
                        ((base_count++))
                    fi
                fi
            done
            
            if [[ $base_count -gt 0 ]]; then
                echo "    none (base): $base_count files"
            fi
            if [[ $chat_count -gt 0 ]]; then
                echo "    chat: $chat_count files"
            fi
            if [[ $plain_count -gt 0 ]]; then
                echo "    plain: $plain_count files"
            fi
            
            if [[ $base_count -eq 0 && $chat_count -eq 0 && $plain_count -eq 0 ]]; then
                echo "    No output_layer_sample_*.safetensors files found"
            fi
        done
        echo ""
    done
}

# Main script logic
case "${1:-}" in
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
