import os
import torch
from pathlib import Path
from safetensors.torch import load_file
from typing import Dict, Any, Union
import pandas as pd


class ResultsDataLoader:
    """
    Data loader for multimodal finetuned representations analysis results.

    Organizes and loads all results from the analysis pipeline including:
    - Model data measures (entropy, intrinsic dimension)
    - Similarity measures between models
    - Layer transplantation benchmarking results
    """

    def __init__(self, results_dir: Union[str, Path] = None):
        """
        Initialize the data loader.

        Args:
            results_dir: Path to results directory. If None, uses ../results relative to this file.
        """
        if results_dir is None:
            # Get path relative to this notebook file
            self.results_dir = Path(__file__).parent.parent / "results"
        else:
            self.results_dir = Path(results_dir)

        # Define model name mappings for cleaner variable names
        self.model_name_mapping = {
            "llava-1.5-7b-hf": "llava_base",
            "llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5": "llava_pretrained",
        }

        # Define dataset types
        self.datasets = ["cocoqa_img", "cocoqa_txt"]

        # Define analysis types
        self.stream_types = ["output_layer", "post_mlp", "heads_projection"]
        self.pooling_methods = ["last", "mean", "none"]

        # Initialize data containers
        self.data = {}

    def load_all_data(self) -> Dict[str, Any]:
        """
        Load all available data from results directory.

        Returns:
            Dictionary containing all loaded data organized by type
        """
        print("Loading all experimental results...")

        # Load model data measures
        self.data["model_measures"] = self._load_model_measures()

        # Load similarity measures
        self.data["similarities"] = self._load_similarities()

        # Load transplantation benchmarking
        self.data["transplantation"] = self._load_transplantation_results()

        print(
            f"✅ Data loading complete. Available data types: {list(self.data.keys())}"
        )
        return self.data

    def _load_model_measures(self) -> Dict[str, Any]:
        """Load all model-specific measures (entropy, intrinsic dimension)."""
        model_measures = {}

        for dataset in self.datasets:
            model_measures[dataset] = {}
            models_dir = self.results_dir / dataset / "models_data_measures"

            if not models_dir.exists():
                continue

            for model_dir in models_dir.iterdir():
                if model_dir.is_dir():
                    model_key = self.model_name_mapping.get(
                        model_dir.name, model_dir.name
                    )
                    model_measures[dataset][model_key] = (
                        self._load_single_model_measures(model_dir)
                    )

        return model_measures

    def _load_single_model_measures(self, model_dir: Path) -> Dict[str, Any]:
        """Load measures for a single model."""
        measures = {
            "dataset_entropy": {},
            "prompt_entropy": {},
            "intrinsic_dimension": {},
        }

        for file_path in model_dir.glob("*.safetensors"):
            filename = file_path.stem
            data = load_file(str(file_path))

            # Parse filename to extract measure type and parameters
            if filename.startswith("dataset_entropy_"):
                # Format: dataset_entropy_{stream_type}_{pooling}
                parts = filename.split("_")
                stream_type = parts[2]
                pooling = parts[3]
                measures["dataset_entropy"][f"{stream_type}_{pooling}"] = data

            elif filename.startswith("prompt_entropy_"):
                # Format: prompt_entropy_{stream_type}_{pooling}
                parts = filename.split("_")
                stream_type = parts[2]
                pooling = parts[3]
                measures["prompt_entropy"][f"{stream_type}_{pooling}"] = data

            elif filename.startswith("id_"):
                # Format: id_{rank}_{range}_{stream_type}_{pooling}
                parts = filename.split("_")
                rank, range_max = parts[1], parts[2]
                stream_type = parts[3]
                pooling = parts[4]
                key = f"{stream_type}_{pooling}_k{rank}_r{range_max}"
                measures["intrinsic_dimension"][key] = data

        return measures

    def _load_similarities(self) -> Dict[str, Any]:
        """Load similarity measures between models."""
        similarities = {}

        for dataset in self.datasets:
            similarities[dataset] = {}
            sim_dir = self.results_dir / dataset / "similarities"

            if not sim_dir.exists():
                continue

            for file_path in sim_dir.glob("*.safetensors"):
                filename = file_path.stem
                data = load_file(str(file_path))

                # Parse filename: {model1}_vs_{model2}_{stream_type}_{pooling}
                # Example: llava-1.5-7b-hf_vs_vicuna-7b-v1.5_output_layer_last
                if "_vs_" in filename:
                    parts = filename.split("_")
                    # Find the vs index to split model names
                    vs_idx = parts.index("vs")

                    # Extract stream type and pooling (last 2 parts)
                    stream_type = parts[-2]
                    pooling = parts[-1]

                    # Create meaningful key
                    key = f"llava_base_vs_llava_pretrained_{stream_type}_{pooling}"
                    similarities[dataset][key] = data

        return similarities

    def _load_transplantation_results(self) -> Dict[str, Any]:
        """Load layer transplantation benchmarking results."""
        transplantation = {}

        transplant_dir = self.results_dir / "transplanting_layers_benchmarking"
        if not transplant_dir.exists():
            return transplantation

        for dataset in self.datasets:
            dataset_dir = transplant_dir / dataset
            if not dataset_dir.exists():
                continue

            transplantation[dataset] = {}

            for file_path in dataset_dir.glob("*.csv"):
                filename = file_path.stem
                data = pd.read_csv(file_path)

                # Parse filename to extract experiment type
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
        """
        Get entropy data for specific configuration.

        Args:
            dataset: 'cocoqa_img' or 'cocoqa_txt'
            model: 'llava_base' or 'llava_pretrained'
            entropy_type: 'dataset_entropy' or 'prompt_entropy'
            stream_type: 'output_layer', 'post_mlp', 'heads_projection'
            pooling: 'last', 'mean', 'none'
        """
        key = f"{stream_type}_{pooling}"
        return self.data["model_measures"][dataset][model][entropy_type][key]

    def get_model_intrinsic_dimension(
        self,
        dataset: str,
        model: str,
        stream_type: str,
        pooling: str,
        k: int = 16,
        range_max: int = 100,
    ) -> torch.Tensor:
        """Get intrinsic dimension data for specific configuration."""
        key = f"{stream_type}_{pooling}_k{k}_r{range_max}"
        return self.data["model_measures"][dataset][model]["intrinsic_dimension"][key]

    def get_similarity(
        self, dataset: str, stream_type: str, pooling: str, similarity_measure: str
    ) -> torch.Tensor:
        """
        Get similarity data between models.

        Args:
            dataset: 'cocoqa_img' or 'cocoqa_txt'
            stream_type: 'output_layer', 'post_mlp', 'heads_projection'
            pooling: 'last', 'mean'
            similarity_measure: 'neighborhood_overlap', 'linear_cka', 'svcca'
        """
        key = f"llava_base_vs_llava_pretrained_{stream_type}_{pooling}"
        similarity_data = self.data["similarities"][dataset][key]
        return similarity_data[similarity_measure]

    def get_transplantation_results(
        self, dataset: str, experiment_type: str
    ) -> pd.DataFrame:
        """
        Get transplantation benchmarking results.

        Args:
            dataset: 'cocoqa_img' or 'cocoqa_txt'
            experiment_type: 'sliding_window' or 'two_parts'
        """
        return self.data["transplantation"][dataset][experiment_type]

    def print_data_summary(self):
        """Print a summary of all loaded data."""
        print("\n" + "=" * 60)
        print("DATA SUMMARY")
        print("=" * 60)

        # Model measures summary
        if "model_measures" in self.data:
            print("\n📊 MODEL MEASURES:")
            for dataset, models in self.data["model_measures"].items():
                print(f"  {dataset}:")
                for model, measures in models.items():
                    print(f"    {model}:")
                    for measure_type, configs in measures.items():
                        print(f"      {measure_type}: {len(configs)} configurations")

        # Similarities summary
        if "similarities" in self.data:
            print("\n🤝 SIMILARITY MEASURES:")
            for dataset, similarities in self.data["similarities"].items():
                print(f"  {dataset}: {len(similarities)} comparisons")

        # Transplantation summary
        if "transplantation" in self.data:
            print("\n🔄 TRANSPLANTATION RESULTS:")
            for dataset, experiments in self.data["transplantation"].items():
                print(f"  {dataset}: {list(experiments.keys())}")

        print("\n" + "=" * 60)


# Example usage function
def load_experimental_data() -> ResultsDataLoader:
    """
    Convenience function to load all experimental data.

    Returns:
        Configured DataLoader with all data loaded
    """
    loader = ResultsDataLoader()
    loader.load_all_data()
    loader.print_data_summary()
    return loader


# Quick access functions for common data patterns
def get_entropy_comparison(
    loader: ResultsDataLoader, dataset: str, stream_type: str, pooling: str
) -> Dict[str, torch.Tensor]:
    """Get entropy data for both models for easy comparison."""
    return {
        "llava_base": loader.get_model_entropy(
            dataset, "llava_base", "dataset_entropy", stream_type, pooling
        ),
        "llava_pretrained": loader.get_model_entropy(
            dataset, "llava_pretrained", "dataset_entropy", stream_type, pooling
        ),
    }


def get_id_comparison(
    loader: ResultsDataLoader, dataset: str, stream_type: str, pooling: str
) -> Dict[str, torch.Tensor]:
    """Get intrinsic dimension data for both models for easy comparison."""
    return {
        "llava_base": loader.get_model_intrinsic_dimension(
            dataset, "llava_base", stream_type, pooling
        ),
        "llava_pretrained": loader.get_model_intrinsic_dimension(
            dataset, "llava_pretrained", stream_type, pooling
        ),
    }
