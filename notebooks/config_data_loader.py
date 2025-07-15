"""
Configuration-based data loader for multimodal finetuned representations analysis.

This version uses YAML configuration files for paths and settings, providing
a more flexible approach to data organization.
"""

import os
import torch
import yaml
from pathlib import Path
from safetensors.torch import load_file
from typing import Dict, Any, Union, Optional
import pandas as pd


class ConfigBasedDataLoader:
    """
    Data loader that uses YAML configuration for paths and settings.

    This provides a more flexible alternative to hardcoded paths and allows
    easy customization of model names, paths, and plotting configurations.
    """

    def __init__(self, config_path: Union[str, Path] = None):
        """
        Initialize the config-based data loader.

        Args:
            config_path: Path to YAML config file. If None, uses config.yaml in same directory.
        """
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        else:
            config_path = Path(config_path)

        # Load configuration
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Set up paths relative to config file directory
        self.base_dir = config_path.parent
        self.results_dir = self.base_dir / self.config["paths"]["results_dir"]

        # Initialize data container
        self.data = {}

    def load_all_data(self) -> Dict[str, Any]:
        """Load all available data using configuration settings."""
        print("🚀 Loading experimental results using configuration...")

        # Load model data measures
        self.data["model_measures"] = self._load_model_measures()

        # Load similarity measures
        self.data["similarities"] = self._load_similarities()

        # Load transplantation benchmarking
        self.data["transplantation"] = self._load_transplantation_results()

        print(
            f"✅ Config-based data loading complete. Available data: {list(self.data.keys())}"
        )
        return self.data

    def _load_model_measures(self) -> Dict[str, Any]:
        """Load model measures using config-defined datasets and models."""
        model_measures = {}

        datasets = self.config["datasets"]["types"]
        model_mapping = self.config["models"]["mapping"]

        for dataset in datasets:
            model_measures[dataset] = {}
            models_dir = self.results_dir / dataset / "models_data_measures"

            if not models_dir.exists():
                continue

            for model_dir in models_dir.iterdir():
                if model_dir.is_dir():
                    model_key = model_mapping.get(model_dir.name, model_dir.name)
                    model_measures[dataset][model_key] = (
                        self._load_single_model_measures(model_dir)
                    )

        return model_measures

    def _load_single_model_measures(self, model_dir: Path) -> Dict[str, Any]:
        """Load measures for a single model using config patterns."""
        measures = {
            "dataset_entropy": {},
            "prompt_entropy": {},
            "intrinsic_dimension": {},
        }

        # Get file patterns from config
        entropy_patterns = self.config["file_patterns"]["entropy"]
        id_pattern = self.config["file_patterns"]["intrinsic_dimension"]

        for file_path in model_dir.glob("*.safetensors"):
            filename = file_path.stem
            data = load_file(str(file_path))

            # Parse based on configured patterns
            if filename.startswith("dataset_entropy_"):
                parts = filename.split("_")
                stream_type = parts[2]
                pooling = parts[3]
                measures["dataset_entropy"][f"{stream_type}_{pooling}"] = data

            elif filename.startswith("prompt_entropy_"):
                parts = filename.split("_")
                stream_type = parts[2]
                pooling = parts[3]
                measures["prompt_entropy"][f"{stream_type}_{pooling}"] = data

            elif filename.startswith("id_"):
                parts = filename.split("_")
                rank, range_max = parts[1], parts[2]
                stream_type = parts[3]
                pooling = parts[4]
                key = f"{stream_type}_{pooling}_k{rank}_r{range_max}"
                measures["intrinsic_dimension"][key] = data

        return measures

    def _load_similarities(self) -> Dict[str, Any]:
        """Load similarity measures using configuration."""
        similarities = {}

        datasets = self.config["datasets"]["types"]
        model_mapping = self.config["models"]["mapping"]

        for dataset in datasets:
            similarities[dataset] = {}
            sim_dir = self.results_dir / dataset / "similarities"

            if not sim_dir.exists():
                continue

            for file_path in sim_dir.glob("*.safetensors"):
                filename = file_path.stem
                data = load_file(str(file_path))

                if "_vs_" in filename:
                    parts = filename.split("_")
                    vs_idx = parts.index("vs")

                    stream_type = parts[-2]
                    pooling = parts[-1]

                    # Use model mapping for consistent naming
                    key = f"llava_base_vs_llava_pretrained_{stream_type}_{pooling}"
                    similarities[dataset][key] = data

        return similarities

    def _load_transplantation_results(self) -> Dict[str, Any]:
        """Load transplantation results using configuration."""
        transplantation = {}

        transplant_dir = self.results_dir / "transplanting_layers_benchmarking"
        if not transplant_dir.exists():
            return transplantation

        datasets = self.config["datasets"]["types"]

        for dataset in datasets:
            dataset_dir = transplant_dir / dataset
            if not dataset_dir.exists():
                continue

            transplantation[dataset] = {}

            for file_path in dataset_dir.glob("*.csv"):
                filename = file_path.stem
                data = pd.read_csv(file_path)

                if "sliding_window" in filename:
                    transplantation[dataset]["sliding_window"] = data
                elif "two_parts" in filename:
                    transplantation[dataset]["two_parts"] = data

        return transplantation

    def get_model_entropy(
        self,
        dataset: str,
        model: str,
        entropy_type: str,
        stream_type: str,
        pooling: str,
    ) -> torch.Tensor:
        """Get entropy data using configuration-defined keys."""
        key = f"{stream_type}_{pooling}"
        return self.data["model_measures"][dataset][model][entropy_type][key]

    def get_model_intrinsic_dimension(
        self,
        dataset: str,
        model: str,
        stream_type: str,
        pooling: str,
        k: Optional[int] = None,
        range_max: Optional[int] = None,
    ) -> torch.Tensor:
        """Get intrinsic dimension using config defaults if not specified."""
        if k is None:
            k = self.config["analysis"]["intrinsic_dimension"]["default_k"]
        if range_max is None:
            range_max = self.config["analysis"]["intrinsic_dimension"][
                "default_range_max"
            ]

        key = f"{stream_type}_{pooling}_k{k}_r{range_max}"
        return self.data["model_measures"][dataset][model]["intrinsic_dimension"][key]

    def get_similarity(
        self, dataset: str, stream_type: str, pooling: str, similarity_measure: str
    ) -> torch.Tensor:
        """Get similarity data using configuration."""
        key = f"llava_base_vs_llava_pretrained_{stream_type}_{pooling}"
        similarity_data = self.data["similarities"][dataset][key]
        return similarity_data[similarity_measure]

    def get_transplantation_results(
        self, dataset: str, experiment_type: str
    ) -> pd.DataFrame:
        """Get transplantation results using configuration."""
        return self.data["transplantation"][dataset][experiment_type]

    def get_display_name(self, key_type: str, key: str) -> str:
        """Get display name from configuration."""
        if key_type == "model":
            return self.config["models"]["display_names"].get(key, key)
        elif key_type == "dataset":
            return self.config["datasets"]["display_names"].get(key, key)
        else:
            return key

    def get_plot_config(self, config_type: str) -> Any:
        """Get plotting configuration."""
        return self.config["plotting"].get(config_type, {})

    def print_config_summary(self):
        """Print configuration summary."""
        print("\n" + "=" * 60)
        print("CONFIGURATION SUMMARY")
        print("=" * 60)

        print(f"\n📁 Paths:")
        print(f"  Results directory: {self.results_dir}")

        print(f"\n🤖 Models:")
        for original, mapped in self.config["models"]["mapping"].items():
            display = self.config["models"]["display_names"][mapped]
            print(f"  {original} → {mapped} ({display})")

        print(f"\n📊 Datasets:")
        for dataset in self.config["datasets"]["types"]:
            display = self.config["datasets"]["display_names"][dataset]
            print(f"  {dataset} ({display})")

        print(f"\n🔧 Analysis:")
        print(f"  Stream types: {self.config['analysis']['stream_types']}")
        print(f"  Pooling methods: {self.config['analysis']['pooling_methods']}")
        print(
            f"  Similarity measures: {self.config['analysis']['similarity_measures']}"
        )

        print("\n" + "=" * 60)


# Convenience function
def load_data_with_config(
    config_path: Union[str, Path] = None,
) -> ConfigBasedDataLoader:
    """
    Convenience function to load data using configuration.

    Args:
        config_path: Path to config file. If None, uses default config.yaml

    Returns:
        Configured data loader with all data loaded
    """
    loader = ConfigBasedDataLoader(config_path)
    loader.print_config_summary()
    loader.load_all_data()
    return loader


# Helper function for plotting with config
def create_comparison_plot_with_config(
    loader: ConfigBasedDataLoader, dataset: str, configs: list, save_path: str = None
):
    """
    Create comparison plots using configuration styling.

    Args:
        loader: Configured data loader
        dataset: Dataset to plot
        configs: List of (stream_type, pooling) tuples
        save_path: Where to save plot (optional)
    """
    import matplotlib.pyplot as plt

    # Get plot configuration
    plot_config = loader.get_plot_config("default_figsize")
    colors = loader.get_plot_config("colors")
    markers = loader.get_plot_config("markers")
    style = loader.get_plot_config("style")

    fig, axes = plt.subplots(2, 2, figsize=plot_config)
    axes = axes.flatten()

    for i, (stream_type, pooling) in enumerate(configs):
        ax = axes[i]

        # Get data for both models
        for model in ["llava_base", "llava_pretrained"]:
            try:
                entropy_data = loader.get_model_entropy(
                    dataset, model, "dataset_entropy", stream_type, pooling
                )
                entropy_values = entropy_data["layers_dataset_entropy"]

                display_name = loader.get_display_name("model", model)
                layers = range(len(entropy_values))

                ax.plot(
                    layers,
                    entropy_values,
                    label=display_name,
                    color=colors.get(model, None),
                    marker=markers.get(model, "o"),
                    markersize=3,
                )
            except KeyError:
                print(f"Data not available for {model}, {stream_type}, {pooling}")

        ax.set_title(f"{stream_type.replace('_', ' ').title()} - {pooling.title()}")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Entropy")
        ax.legend()

        if style.get("grid", False):
            ax.grid(True, alpha=style.get("grid_alpha", 0.3))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=style.get("dpi", 150), bbox_inches="tight")
        print(f"📊 Plot saved to {save_path}")

    return fig
