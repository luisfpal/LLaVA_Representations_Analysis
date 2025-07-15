"""
Example script demonstrating how to load and access experimental results data.

This script shows various ways to load and work with the results from the
multimodal finetuned representations analysis.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Add the notebooks directory to path so we can import our data loader
sys.path.append(str(Path(__file__).parent))

from data_loader import (
    load_experimental_data,
    get_entropy_comparison,
)


def main():
    """Main example demonstrating data loading and basic analysis."""

    print("🚀 Loading experimental results...")

    # Method 1: Simple loading with convenience function
    loader = load_experimental_data()

    # Method 2: Manual loading with custom path (if needed)
    # loader = ResultsDataLoader(results_dir="../results")
    # loader.load_all_data()

    print("\n" + "=" * 60)
    print("EXAMPLE DATA ACCESS PATTERNS")
    print("=" * 60)

    # Example 1: Access specific entropy data
    print("\n1️⃣ ACCESSING ENTROPY DATA:")

    # Get dataset entropy for llava_base model on cocoqa_img with output_layer and last pooling
    entropy_data = loader.get_model_entropy(
        dataset="cocoqa_img",
        model="llava_base",
        entropy_type="dataset_entropy",
        stream_type="output_layer",
        pooling="last",
    )

    print(f"Dataset entropy shape: {entropy_data['layers_dataset_entropy'].shape}")
    print(f"First 5 layers entropy: {entropy_data['layers_dataset_entropy'][:5]}")

    # Example 2: Compare entropy between models
    print("\n2️⃣ COMPARING MODELS - ENTROPY:")

    entropy_comparison = get_entropy_comparison(
        loader, dataset="cocoqa_img", stream_type="output_layer", pooling="last"
    )

    llava_base_entropy = entropy_comparison["llava_base"]["layers_dataset_entropy"]
    llava_pretrained_entropy = entropy_comparison["llava_pretrained"][
        "layers_dataset_entropy"
    ]

    print(f"LLaVA Base entropy (layers 0-4): {llava_base_entropy[:5]}")
    print(f"LLaVA Pretrained entropy (layers 0-4): {llava_pretrained_entropy[:5]}")

    # Example 3: Access intrinsic dimension data
    print("\n3️⃣ ACCESSING INTRINSIC DIMENSION DATA:")

    id_data = loader.get_model_intrinsic_dimension(
        dataset="cocoqa_img",
        model="llava_base",
        stream_type="output_layer",
        pooling="last",
    )

    print(f"Intrinsic dimension shape: {id_data['layers_intrinsic_dimension'].shape}")
    print(f"First 5 layers ID: {id_data['layers_intrinsic_dimension'][:5]}")

    # Example 4: Access similarity measures
    print("\n4️⃣ ACCESSING SIMILARITY MEASURES:")

    similarity_data = loader.get_similarity(
        dataset="cocoqa_img",
        stream_type="output_layer",
        pooling="last",
        similarity_measure="linear_cka",
    )

    print(f"Linear CKA similarity shape: {similarity_data.shape}")
    print(f"Linear CKA similarity (layers 0-4): {similarity_data[:5]}")

    # Example 5: Access transplantation results
    print("\n5️⃣ ACCESSING TRANSPLANTATION RESULTS:")

    transplant_data = loader.get_transplantation_results(
        dataset="cocoqa_img", experiment_type="sliding_window"
    )

    print(f"Sliding window results shape: {transplant_data.shape}")
    print(f"Columns: {list(transplant_data.columns)}")
    print(f"First few rows:\n{transplant_data.head()}")

    # Example 6: Iterate through all available data
    print("\n6️⃣ ITERATING THROUGH DATA:")

    print("\nAvailable datasets:", list(loader.data["model_measures"].keys()))

    for dataset in ["cocoqa_img", "cocoqa_txt"]:
        print(f"\n{dataset.upper()}:")

        # Show available models
        models = list(loader.data["model_measures"][dataset].keys())
        print(f"  Models: {models}")

        # Show available similarity comparisons
        similarities = list(loader.data["similarities"][dataset].keys())
        print(f"  Similarity comparisons: {len(similarities)}")

        # Show sample entropy configurations for first model
        if models:
            model = models[0]
            entropy_configs = list(
                loader.data["model_measures"][dataset][model]["dataset_entropy"].keys()
            )
            print(f"  Sample entropy configs for {model}: {entropy_configs[:3]}...")


def plot_entropy_comparison():
    """Example plotting function comparing entropy across models."""
    print("\n" + "=" * 60)
    print("EXAMPLE PLOTTING")
    print("=" * 60)

    loader = load_experimental_data()

    # Compare entropy between models for different configurations
    configs = [
        ("output_layer", "last"),
        ("output_layer", "mean"),
        ("post_mlp", "last"),
        ("post_mlp", "mean"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for i, (stream_type, pooling) in enumerate(configs):
        ax = axes[i]

        # Get data for both models
        entropy_comp = get_entropy_comparison(
            loader, "cocoqa_img", stream_type, pooling
        )

        base_entropy = entropy_comp["llava_base"]["layers_dataset_entropy"]
        pretrained_entropy = entropy_comp["llava_pretrained"]["layers_dataset_entropy"]

        # Plot
        layers = range(len(base_entropy))
        ax.plot(layers, base_entropy, label="LLaVA Base", marker="o", markersize=3)
        ax.plot(
            layers,
            pretrained_entropy,
            label="LLaVA Pretrained",
            marker="s",
            markersize=3,
        )

        ax.set_title(f"{stream_type} - {pooling}")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Entropy")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("entropy_comparison.png", dpi=150, bbox_inches="tight")
    print("📊 Entropy comparison plot saved as 'entropy_comparison.png'")


def demonstrate_data_access_patterns():
    """Show different patterns for accessing the data efficiently."""
    print("\n" + "=" * 60)
    print("ADVANCED DATA ACCESS PATTERNS")
    print("=" * 60)

    loader = load_experimental_data()

    # Pattern 1: Batch access for all pooling methods
    print("\n🔍 Pattern 1: Compare all pooling methods for a stream type")

    dataset = "cocoqa_img"
    model = "llava_base"
    stream_type = "output_layer"

    pooling_methods = ["last", "mean"]
    entropy_by_pooling = {}

    for pooling in pooling_methods:
        try:
            entropy_data = loader.get_model_entropy(
                dataset, model, "dataset_entropy", stream_type, pooling
            )
            entropy_by_pooling[pooling] = entropy_data["layers_dataset_entropy"]
            print(f"  {pooling}: {entropy_by_pooling[pooling].shape}")
        except KeyError:
            print(f"  {pooling}: Not available")

    # Pattern 2: Compare across stream types
    print("\n🔍 Pattern 2: Compare across stream types")

    stream_types = ["output_layer", "post_mlp"]
    entropy_by_stream = {}

    for stream_type in stream_types:
        try:
            entropy_data = loader.get_model_entropy(
                dataset, model, "dataset_entropy", stream_type, "last"
            )
            entropy_by_stream[stream_type] = entropy_data["layers_dataset_entropy"]
            print(f"  {stream_type}: {entropy_by_stream[stream_type].shape}")
        except KeyError:
            print(f"  {stream_type}: Not available")

    # Pattern 3: Access all similarity measures for a configuration
    print("\n🔍 Pattern 3: All similarity measures for a configuration")

    stream_type = "output_layer"
    pooling = "last"

    similarity_measures = ["neighborhood_overlap", "linear_cka", "svcca"]
    similarities = {}

    for measure in similarity_measures:
        try:
            sim_data = loader.get_similarity(dataset, stream_type, pooling, measure)
            similarities[measure] = sim_data
            print(f"  {measure}: {similarities[measure].shape}")
        except KeyError:
            print(f"  {measure}: Not available")


if __name__ == "__main__":
    # Run the main examples
    main()

    # Run plotting example
    plot_entropy_comparison()

    # Show advanced patterns
    demonstrate_data_access_patterns()

    print("\n🎉 Examples completed successfully!")
