import os
import matplotlib.pyplot as plt
import numpy as np
from safetensors import safe_open

def load_safetensors_data(file_path):
    """Load tensor data from safetensors file"""
    with safe_open(file_path, framework="np") as f:
        # Get the tensor data - assuming it's stored under 'layers_id_correlation'
        tensor_data = f.get_tensor('layers_id_correlation')
        return tensor_data

def create_correlation_plot():
    """Create correlation plots for Visual QA and Text QA datasets"""
    
    # File paths
    cocoqa_img_path = "cocoqa_img/similarities/llava-1.5-7b-hf_vs_vicuna-7b-v1.5_id_correlation_output_layer_last.safetensors"
    cocoqa_txt_path = "cocoqa_txt/similarities/llava-1.5-7b-hf_vs_vicuna-7b-v1.5_id_correlation_output_layer_last.safetensors"
    
    # Load data
    print("Loading Visual QA data...")
    visual_qa_data = load_safetensors_data(cocoqa_img_path)
    
    print("Loading Text QA data...")
    text_qa_data = load_safetensors_data(cocoqa_txt_path)
    
    print(f"Visual QA data shape: {visual_qa_data.shape}")
    print(f"Text QA data shape: {text_qa_data.shape}")
    
    # Create x-axis (layer indices)
    x_visual = np.arange(len(visual_qa_data))
    x_text = np.arange(len(text_qa_data))
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot both datasets
    plt.plot(x_visual, visual_qa_data, 'b-o', label='Visual QA', linewidth=2, markersize=6)
    plt.plot(x_text, text_qa_data, 'r-s', label='Text QA', linewidth=2, markersize=6)
    
    # Customize the plot
    plt.xlabel('Layer Index', fontsize=12)
    plt.ylabel('ID Correlation', fontsize=12)
    plt.title('ID Correlation Across Layers: LLaVA-1.5-7B vs Vicuna-7B', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Set axis limits and ticks
    plt.xlim(-0.5, max(len(visual_qa_data), len(text_qa_data)) - 0.5)
    plt.xticks(range(0, max(len(visual_qa_data), len(text_qa_data)), 2))
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('id_correlation_plot.png', dpi=300, bbox_inches='tight')
    plt.savefig('id_correlation_plot.pdf', bbox_inches='tight')
    
    print("Plot saved as 'id_correlation_plot.png' and 'id_correlation_plot.pdf'")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    create_correlation_plot()