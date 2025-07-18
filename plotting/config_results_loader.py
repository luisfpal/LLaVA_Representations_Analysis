"""
Configuration-based results loader for multimodal analysis.
Loads experimental results using YAML configuration for flexible data access.
"""

import torch
import yaml
from pathlib import Path
from safetensors.torch import load_file
from typing import Dict, Any, Union, Optional
import pandas as pd


class ResultsLoader:
    """
    Load and organize experimental results using YAML configuration.

    Provides clean access to model measures, similarities, and transplantation results.

    The results data are organized in a dictionary with the following structure:
    - data["model_data_measures"]
        - dataset
            - model
                - dataset_entropy
                    - residual_stream_type_pooling_args e.g,
                        - output_layer_mean ...
                - prompt_entropy, e.g.,
                    - output_layer_mean ...
                - intrinsic_dimension, e.g.,
                    - output_layer_mean_k<k>_r<r>
    - data["models_similarities"]
        - dataset
            - residual_stream_type_pooling dictionary
                - measure_name, e.g., neighborhood_overlap, linear_cka, svcca
                    - similarity_matrix tensor of shape (num_heads, num_layers) or (num_layers,)
    - data["transplanting_layers_benchmarking"]
        - dataset
            - experiment_name, e.g.,
                - sliding_window_ws<ws>_s<s>
                - two_parts_s<s>
    """

    def __init__(self, config_path: Union[str, Path] = None):
        """Initialize loader with configuration file."""
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        else:
            config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.results_dir = config_path.parent / self.config["paths"]["results"]
        if not self.results_dir.exists():
            raise FileNotFoundError(f"Results directory not found: {self.results_dir}")

        self.data = {}

    def load_all(self) -> Dict[str, Any]:
        """Load all available results."""
        print("\n🚀 Loading results...")

        # Renamed data structure keys for clarity
        self.data["model_data_measures"] = self._load_model_data_measures()
        self.data["models_similarities"] = self._load_models_similarities()
        self.data["transplanting_layers_benchmarking"] = (
            self._load_transplanting_layers_benchmarking()
        )

        print(f"✅ Loaded: {list(self.data.keys())}")
        return self.data

    def _load_model_data_measures(self) -> Dict[str, Any]:
        """Load model measures (entropy, intrinsic dimension)."""
        model_data_measures = {}

        for dataset in self.config["datasets"]:
            model_data_measures[dataset] = {}
            models_dir = self.results_dir / dataset / "models_data_measures"

            if not models_dir.exists():
                print(f"⚠️  No model data measures found for {dataset}")
                continue

            for model_dir in models_dir.iterdir():
                if model_dir.is_dir():
                    model_key = self.config["models"].get(
                        model_dir.name, model_dir.name
                    )
                    model_data_measures[dataset][model_key] = self._load_model_measures(
                        model_dir
                    )

        return model_data_measures

    def _load_model_measures(self, model_dir: Path) -> Dict[str, Any]:
        """Load measures for a single model."""
        model_data = {
            "dataset_entropy": {},
            "prompt_entropy": {},
            "intrinsic_dimension": {},
        }

        for file_path in model_dir.glob("*.safetensors"):
            filename = file_path.stem

            try:
                data = load_file(str(file_path))
            except Exception as e:
                print(f"⚠️  Failed to load {filename}: {e}")
                continue

            if filename.startswith("dataset_entropy_"):
                key = filename.replace("dataset_entropy_", "")
                model_data["dataset_entropy"][key] = data

            elif filename.startswith("prompt_entropy_"):
                key = filename.replace("prompt_entropy_", "")
                model_data["prompt_entropy"][key] = data

            elif filename.startswith("id_"):
                parts = filename.split("_")
                k, r = parts[1], parts[2]
                key = f"{filename.replace(f'id_{k}_{r}_', '')}_k{k}_r{r}"
                model_data["intrinsic_dimension"][key] = data

        return model_data

    def _load_models_similarities(self) -> Dict[str, Any]:
        """Load similarity measures between models."""
        models_similarities = {}

        for dataset in self.config["datasets"]:
            models_similarities[dataset] = {}
            sim_dir = self.results_dir / dataset / "similarities"

            if not sim_dir.exists():
                print(f"⚠️  No model similarities found for {dataset}")
                continue

            for file_path in sim_dir.glob("*.safetensors"):
                filename = file_path.stem

                try:
                    data = load_file(str(file_path))
                    # dictionary with :
                    #  - neighborhood_overlap
                    #  - linear_cka
                    #  - svcca
                except Exception as e:
                    print(f"⚠️  Failed to load {filename}: {e}")
                    continue

                if "_vs_" in filename:
                    # Extract stream_type and pooling from filename
                    parts = filename.split("_vs_")[-1].split("_")
                    stream_type = "_".join(parts[1:-1])
                    pooling = parts[-1]
                    key = f"{stream_type}_{pooling}"

                    # Initialize nested structure if needed
                    if key not in models_similarities[dataset]:
                        models_similarities[dataset][key] = {}

                    # The loaded data is already a dictionary with measures as keys
                    # Merge the data into our structure
                    for measure_name, measure_data in data.items():
                        models_similarities[dataset][key][measure_name] = measure_data

        return models_similarities

    def create_similarities_difference_dataset(
        self,
        source_dataset1: str = "cocoqa_txt",
        source_dataset2: str = "cocoqa_img",
        target_dataset: str = "cocoqa_txt_minus_cocoqa_img",
        verbose: bool = False,
    ) -> bool:
        """
        Create a difference dataset by subtracting similarities.

        Args:
            source_dataset1: First dataset (e.g., "cocoqa_txt")
            source_dataset2: Second dataset to subtract (e.g., "cocoqa_img")
            target_dataset: Name for the new difference dataset
            verbose: Whether to print detailed progress

        Returns:
            bool: True if successful, False otherwise
        """
        print("\n" + "=" * 50)
        print("➕ Creating similarities difference dataset...")
        print("=" * 50)

        if "models_similarities" not in self.data:
            print("⚠️  No similarity data loaded")
            return False

        models_similarities = self.data["models_similarities"]

        # Check if both source datasets exist
        if (
            source_dataset1 not in models_similarities
            or source_dataset2 not in models_similarities
        ):
            print(
                f"⚠️  Cannot create difference dataset: missing {source_dataset1} or {source_dataset2}"
            )
            return False

        models_similarities[target_dataset] = {}

        # Get all common stream_type_pooling keys
        dataset1_stream_pooling_keys = set(models_similarities[source_dataset1].keys())
        dataset2_stream_pooling_keys = set(models_similarities[source_dataset2].keys())
        common_stream_pooling_keys = dataset1_stream_pooling_keys.intersection(
            dataset2_stream_pooling_keys
        )

        if not common_stream_pooling_keys:
            print(
                f"⚠️  No common stream_pooling_keys found between {source_dataset1} and {source_dataset2}"
            )
            return False

        success_count = 0
        for stream_pooling_key in common_stream_pooling_keys:
            models_similarities[target_dataset][stream_pooling_key] = {}

            # Get all common measure keys
            dataset1_measures = set(
                models_similarities[source_dataset1][stream_pooling_key].keys()
            )
            dataset2_measures = set(
                models_similarities[source_dataset2][stream_pooling_key].keys()
            )
            common_measures = dataset1_measures.intersection(dataset2_measures)

            for measure in common_measures:
                data1 = models_similarities[source_dataset1][stream_pooling_key][
                    measure
                ]
                data2 = models_similarities[source_dataset2][stream_pooling_key][
                    measure
                ]

                # Perform subtraction: dataset1 - dataset2
                try:
                    diff_data = data1 - data2
                    models_similarities[target_dataset][stream_pooling_key][measure] = (
                        diff_data
                    )
                    if verbose:
                        print(f"✅ Created difference: {stream_pooling_key}/{measure}")
                    success_count += 1
                except Exception as e:
                    print(
                        f"⚠️  Failed to compute difference for {stream_pooling_key}/{measure}: {e}"
                    )

        if success_count > 0:
            print(
                f"✅ Similarities difference dataset '{target_dataset}' created successfully with {success_count} measures"
            )
            print("=" * 50)
            return True
        else:
            print(
                f"❌ Failed to create similarities difference dataset '{target_dataset}'"
            )
            return False

    def compute_mean_heads_projections_at_layers(self, verbose: bool = False) -> bool:
        """
        Compute mean across heads (columns) for heads_projection similarity matrices.
        Creates new stream_type_pooling fields: mean_heads_projection_mean, mean_heads_projection_last

        For each heads_projection matrix with shape (layers, heads), computes column mean
        resulting in shape (layers,) and stores as new measure.

        Args:
            verbose: Whether to print detailed progress

        Returns:
            bool: True if successful, False otherwise
        """
        print("\n" + "=" * 50)
        print("➕ Computing mean heads projections at layers...")
        print("=" * 50)

        if "models_similarities" not in self.data:
            print("⚠️  No similarity data loaded")
            return False

        models_similarities = self.data["models_similarities"]
        success_count = 0

        # Process all datasets (including difference datasets)
        for dataset_name, dataset_data in models_similarities.items():
            if verbose:
                print(f"\n📊 Processing heads mean for dataset: {dataset_name}")

            # Look for heads_projection stream types
            heads_projection_keys = [
                key
                for key in dataset_data.keys()
                if key.startswith("heads_projection_")
                and key.endswith(("_mean", "_last"))
            ]

            if not heads_projection_keys:
                if verbose:
                    print(f"  ⚠️  No heads_projection data found for {dataset_name}")
                continue

            for heads_key in heads_projection_keys:
                # Extract pooling type (mean or last)
                pooling_type = heads_key.split("_")[-1]  # "mean" or "last"
                new_key = f"mean_heads_projection_{pooling_type}"

                # Initialize new stream_type_pooling if not exists
                if new_key not in dataset_data:
                    dataset_data[new_key] = {}

                measures_data = dataset_data[heads_key]

                for measure_name, measure_matrix in measures_data.items():
                    try:
                        # Compute mean across columns (heads dimension)
                        # (layers, heads) -> (layers,)
                        if hasattr(measure_matrix, "mean"):
                            heads_mean = measure_matrix.mean(dim=1)

                        dataset_data[new_key][measure_name] = heads_mean

                        if verbose:
                            print(
                                f"  ✅ {heads_key}/{measure_name} -> {new_key}/{measure_name}"
                            )
                            print(
                                f"     Shape: {measure_matrix.shape} -> {heads_mean.shape}"
                            )
                        success_count += 1

                    except Exception as e:
                        print(
                            f"  ⚠️  Failed to compute heads mean for {heads_key}/{measure_name}: {e}"
                        )

        if success_count > 0:
            print(
                f"\n✅ Mean heads projections at layers completed successfully: {success_count} measures processed"
            )
            print("=" * 50)
            return True
        else:
            print("\n❌ Failed to compute mean heads projections at layers")
            return False

    def create_dataset_measures_models_difference(
        self,
        dataset_name: str,
        model1: str,
        model2: str,
        target_model: str = None,
        verbose: bool = False,
    ) -> bool:
        """
        Create difference between model measures: model1 - model2 for a specific dataset.

        Computes differences for all matching measure keys across all measure types:
        - dataset_entropy
        - prompt_entropy
        - intrinsic_dimension

        Args:
            dataset_name: Dataset to process (e.g., "cocoqa_img")
            model1: First model (e.g., "base")
            model2: Second model to subtract (e.g., "pretrained")
            target_model: Name for difference model (default: "model1-model2")
            verbose: Whether to print detailed progress

        Returns:
            bool: True if successful, False otherwise

        Example:
            results.create_dataset_measures_models_difference("cocoqa_img", "base", "pretrained")
            # Creates: data["model_data_measures"]["cocoqa_img"]["base-pretrained"]
        """
        if target_model is None:
            target_model = f"{model1}-{model2}"

        print("\n" + "=" * 50)
        print(f"➕ Creating model measures difference: {model1} - {model2}")
        print(f"   Dataset: {dataset_name}")
        print(f"   Target model: {target_model}")
        print("=" * 50)

        if "model_data_measures" not in self.data:
            print("⚠️  No model data measures loaded")
            return False

        model_data_measures = self.data["model_data_measures"]

        # Check if dataset exists
        if dataset_name not in model_data_measures:
            print(f"⚠️  Dataset '{dataset_name}' not found")
            print(f"   Available datasets: {list(model_data_measures.keys())}")
            return False

        dataset_data = model_data_measures[dataset_name]

        # Check if both models exist
        if model1 not in dataset_data or model2 not in dataset_data:
            print(
                f"⚠️  Cannot create difference: missing model '{model1}' or '{model2}'"
            )
            print(f"   Available models: {list(dataset_data.keys())}")
            return False

        # Check if both models have the same measures
        model1_measures = dataset_data[model1].keys()
        model2_measures = dataset_data[model2].keys()
        if model1_measures != model2_measures:
            print(f"⚠️  Models {model1} and {model2} have different measures")
            return False

        # Initialize target model structure
        dataset_data[target_model] = {
            measure_type: {} for measure_type in model1_measures
        }

        success_count = 0

        for measure_type in model1_measures:
            if verbose:
                print(f"\n📊 Processing {measure_type}...")

            model1_stream_pooling_measures = dataset_data[model1].get(measure_type, {})
            model2_stream_pooling_measures = dataset_data[model2].get(measure_type, {})

            # Get common residual_stream_type_pooling keys
            model1_stream_type_pooling_keys = set(model1_stream_pooling_measures.keys())
            model2_stream_type_pooling_keys = set(model2_stream_pooling_measures.keys())
            common_stream_type_pooling_keys = (
                model1_stream_type_pooling_keys.intersection(
                    model2_stream_type_pooling_keys
                )
            )

            if not common_stream_type_pooling_keys:
                if verbose:
                    print(f"  ⚠️  No common {measure_type} measures found")
                continue

            for stream_type_pooling_key in common_stream_type_pooling_keys:
                try:
                    data1 = next(
                        iter(
                            model1_stream_pooling_measures[
                                stream_type_pooling_key
                            ].values()
                        )
                    )
                    data2 = next(
                        iter(
                            model2_stream_pooling_measures[
                                stream_type_pooling_key
                            ].values()
                        )
                    )

                    # Perform subtraction: model1 - model2
                    diff_data = data1 - data2
                    dataset_data[target_model][measure_type][
                        stream_type_pooling_key
                    ] = diff_data

                    if verbose:
                        print(f"  ✅ {measure_type}/{stream_type_pooling_key}")
                        if hasattr(diff_data, "shape"):
                            print(f"     Shape: {diff_data.shape}")

                    success_count += 1

                except Exception as e:
                    print(
                        f"  ⚠️  Failed to compute difference for {measure_type}/{stream_type_pooling_key}: {e}"
                    )

        if success_count > 0:
            print(
                f"\n✅ Model measures difference '{target_model}' created successfully"
            )
            print(
                f"   Processed {success_count} measures across {len(model1_measures)} types"
            )

            if verbose:
                # Print summary of created measures
                print(f"\n📋 Created measures for '{target_model}':")
                for measure_type in dataset_data[target_model].keys():
                    created_measures = list(
                        dataset_data[target_model][measure_type].keys()
                    )
                    if created_measures:
                        print(f"   {measure_type}: {created_measures} measures")

            print("=" * 50)
            return True
        else:
            print(f"❌ Failed to create model measures difference '{target_model}'")
            return False

    def _load_transplanting_layers_benchmarking(self) -> Dict[str, Any]:
        """Load transplantation experiment results."""
        transplanting_layers_benchmarking = {}
        transplant_dir = self.results_dir / "transplanting_layers_benchmarking"

        if not transplant_dir.exists():
            print("⚠️  No transplanting layers benchmarking results found")
            return transplanting_layers_benchmarking

        for dataset in self.config["datasets"]:
            dataset_dir = transplant_dir / dataset
            if not dataset_dir.exists():
                continue

            transplanting_layers_benchmarking[dataset] = {}
            for file_path in dataset_dir.glob("*.csv"):
                filename = file_path.stem

                try:
                    data = pd.read_csv(file_path)
                except Exception as e:
                    print(f"⚠️  Failed to load {filename}: {e}")
                    continue

                if "sliding_window" in filename:
                    key = filename.split("sliding_window_")[-1]
                    transplanting_layers_benchmarking[dataset][
                        f"sliding_window_{key}"
                    ] = data
                elif "two_parts" in filename:
                    key = filename.split("two_parts_")[-1]
                    transplanting_layers_benchmarking[dataset][f"two_parts_{key}"] = (
                        data
                    )

        return transplanting_layers_benchmarking

    # === Data Access Methods ===

    def get_entropy(
        self, dataset: str, model: str, stream_type: str, pooling: str
    ) -> torch.Tensor:
        """Get dataset entropy data."""
        key = f"{stream_type}_{pooling}"
        try:
            return self.data["model_data_measures"][dataset][model]["dataset_entropy"][
                key
            ]
        except KeyError:
            raise KeyError(f"Entropy data not found: {dataset}/{model}/{key}")

    def get_prompt_entropy(
        self, dataset: str, model: str, stream_type: str, pooling: str
    ) -> torch.Tensor:
        """Get prompt entropy data."""
        key = f"{stream_type}_{pooling}"
        try:
            return self.data["model_data_measures"][dataset][model]["prompt_entropy"][
                key
            ]
        except KeyError:
            raise KeyError(f"Prompt entropy data not found: {dataset}/{model}/{key}")

    def get_intrinsic_dimension(
        self,
        dataset: str,
        model: str,
        stream_type: str,
        pooling: str,
        k: Optional[int] = None,
        range_max: Optional[int] = None,
    ) -> torch.Tensor:
        """Get intrinsic dimension data."""
        if k is None:
            k = self.config["analysis"]["intrinsic_dimension_defaults"]["k"]
        if range_max is None:
            range_max = self.config["analysis"]["intrinsic_dimension_defaults"][
                "range_max"
            ]

        key = f"{stream_type}_{pooling}_k{k}_r{range_max}"
        try:
            return self.data["model_data_measures"][dataset][model][
                "intrinsic_dimension"
            ][key]
        except KeyError:
            raise KeyError(
                f"Intrinsic dimension data not found: {dataset}/{model}/{key}"
            )

    def get_similarity(
        self, dataset: str, stream_type: str, pooling: str, measure: str
    ) -> torch.Tensor:
        """Get similarity data between models."""
        key = f"{stream_type}_{pooling}"
        try:
            return self.data["models_similarities"][dataset][key][measure]
        except KeyError:
            raise KeyError(f"Similarity data not found: {dataset}/{key}/{measure}")

    def get_transplant(self, dataset: str, experiment: str) -> pd.DataFrame:
        """Get transplantation experiment results."""
        try:
            return self.data["transplanting_layers_benchmarking"][dataset][experiment]
        except KeyError:
            raise KeyError(f"Transplant data not found: {dataset}/{experiment}")

    # === Summary Methods ===

    def print_config(self):
        """Print configuration summary."""
        print("\n" + "=" * 50)
        print("CONFIGURATION")
        print("=" * 50)
        print(f"📁 Results: {self.results_dir}")
        print(f"🤖 Models: {list(self.config['models'].values())}")
        print(f"📊 Datasets: {self.config['datasets']}")
        print(f"🔧 Stream types: {self.config['analysis']['stream_types']}")
        print(f"🔄 Pooling: {self.config['analysis']['pooling_methods']}")
        print("=" * 50)

    def print_summary(self):
        """Print loaded data summary."""
        print("\n" + "=" * 50)
        print("LOADED DATA SUMMARY")
        print("=" * 50)

        if "model_data_measures" in self.data:
            print("\n📊 MODEL MEASURES:")
            for dataset, models in self.data["model_data_measures"].items():
                print(f"  {dataset}:")
                for model, data in models.items():
                    dataset_entropy_count = len(data.get("dataset_entropy", {}))
                    prompt_entropy_count = len(data.get("prompt_entropy", {}))
                    intrinsic_dimension_count = len(data.get("intrinsic_dimension", {}))
                    print(
                        f"    {model}: {dataset_entropy_count} dataset entropy, "
                        f"{prompt_entropy_count} prompt entropy, "
                        f"{intrinsic_dimension_count} intrinsic dimension"
                    )

        if "models_similarities" in self.data:
            print("\n🤝 SIMILARITIES:")
            for dataset, sims in self.data["models_similarities"].items():
                # Count regular and mean heads projection stream types separately
                regular_streams = [
                    k for k in sims.keys() if not k.startswith("mean_heads_projection")
                ]
                mean_heads_streams = [
                    k for k in sims.keys() if k.startswith("mean_heads_projection")
                ]

                total_configs = len(sims)
                regular_count = len(regular_streams)
                mean_heads_count = len(mean_heads_streams)

                if mean_heads_count > 0:
                    print(
                        f"  {dataset}: {total_configs} configurations ({regular_count} regular + {mean_heads_count} heads mean)"
                    )
                else:
                    print(f"  {dataset}: {total_configs} configurations")

        if "transplanting_layers_benchmarking" in self.data:
            print("\n🔄 TRANSPLANTATION:")
            for dataset, exps in self.data["transplanting_layers_benchmarking"].items():
                print(f"  {dataset}: {list(exps.keys())}")

        print("\n" + "=" * 50)


def main(config_path: Union[str, Path] = None) -> ResultsLoader:
    """
    Load all experimental results using configuration.

    Args:
        config_path: Path to config file (uses default if None)

    Returns:
        ResultsLoader with all data loaded
    """
    loader = ResultsLoader(config_path)
    loader.print_config()
    loader.load_all()
    loader.print_summary()
    return loader


if __name__ == "__main__":
    main()
