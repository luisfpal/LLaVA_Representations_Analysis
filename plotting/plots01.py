import matplotlib.pyplot as plt
from typing import List, Optional
from functools import partial
import torch
from pathlib import Path
from utils import plot_similarity_measure_matrix
from config_results_loader import ResultsLoader


def plot_similarity_measures_matrix(
    results_loader: ResultsLoader,
    dataset: str,
    stream_type: str,
    pooling: str,
    measures: List[str],
    subplot_width: int = 8,
    plot_height: int = 10,
    percentage_threshold: float = 0.1,
    top_p_percent: float = 0.05,
    bottom_p_percent: float = 0.03,
    max_xticks: Optional[int] = None,
    max_yticks: Optional[int] = None,
    annot_font_size: int = 8,
    colorbar_labelsize: int = 12,
    wspace: float = 0.0,
    save_plots: bool = False,
):
    """
    Create side-by-side heatmaps for multiple similarity measures.

    Args:
        results_loader: ResultsLoader instance with loaded data
        dataset: Dataset name (e.g., "cocoqaqa_img")
        stream_type: Stream type (e.g., "heads_projection")
        pooling: Pooling method (e.g., "last")
        measures: List of measure names to plot
        subplot_width: Width of each subplot
        plot_height: Height of the plot
        percentage_threshold: Threshold for percentage of values to be shown
        top_p_percent: Percentage of top values to be shown
        bottom_p_percent: Percentage of bottom values to be shown
        max_xticks: Maximum number of x-axis ticks to show (None = show all)
        max_yticks: Maximum number of y-axis ticks to show (None = show all)
        annot_font_size: Font size for annotations
        colorbar_labelsize: Font size for colorbar tick labels
        wspace: Width of the space between subplots
        save_plots: Whether to save plots instead of showing them
    """
    num_measures = len(measures)

    # Calculate figure size: width scales with number of measures
    fig_width = num_measures * subplot_width

    # Create the main figure with subplots
    fig, axes = plt.subplots(
        nrows=1,
        ncols=num_measures,
        figsize=(fig_width, plot_height),
    )

    # Handle single subplot case (axes is not a list when ncols=1)
    if num_measures == 1:
        axes = [axes]

    # Create heatmap for each measure using the core plotting function
    for idx, measure in enumerate(measures):
        # Get similarity data for this measure
        matrix_similarities = results_loader.get_similarity(
            dataset=dataset,
            stream_type=stream_type,
            pooling=pooling,
            measure=measure,
        )

        # Create heatmap directly on the subplot axis
        plot_similarity_measure_matrix(
            ax=axes[idx],
            matrix_similarities=matrix_similarities,
            xlabel="Heads",
            ylabel="Layers" if idx == 0 else "",  # Only first subplot gets y-label
            measure=measure,
            max_xticks=max_xticks,
            max_yticks=max_yticks,
            percentage_threshold=percentage_threshold,
            top_p_percent=top_p_percent,
            bottom_p_percent=bottom_p_percent,
            annot_font_size=annot_font_size,
            colorbar_labelsize=colorbar_labelsize,
        )

        # Clean up y-axis elements for all columns except the first
        if idx > 0:
            axes[idx].set_ylabel("")
            axes[idx].set_yticklabels([])
            axes[idx].tick_params(left=False)  # Remove y-axis ticks

    # Configure overall figure layout with reduced spacing
    plt.suptitle(f"{dataset} - {stream_type} - {pooling}", fontsize=16, y=0.97)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, wspace=wspace)  # Reduced wspace for tighter layout

    if save_plots:
        plots_dir = Path(__file__).parent.parent / "plots"
        plots_dir.mkdir(exist_ok=True)
        filename = f"similarity_matrix_{dataset}_{stream_type}_{pooling}_{'_'.join(measures)}.png"
        plt.savefig(plots_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_all_similarity_measures_matrices(
    results_loader: ResultsLoader, save_plots: bool = False
):
    template_plot = partial(
        plot_similarity_measures_matrix,
        results_loader=results_loader,
        measures=["neighborhood_overlap", "linear_cka"],
        max_xticks=10,
        max_yticks=10,
        percentage_threshold=0.08,
        top_p_percent=0.03,
        bottom_p_percent=0.03,
        annot_font_size=6,
        colorbar_labelsize=14,
        subplot_width=10,
        plot_height=8,
        wspace=-0.05,
        save_plots=save_plots,
    )
    template_plot(
        dataset="cocoqa_txt",
        stream_type="heads_projection",
        pooling="last",
    )
    template_plot(
        dataset="cocoqa_img",
        stream_type="heads_projection",
        pooling="last",
    )
    template_plot(
        dataset="cocoqa_txt_minus_cocoqa_img",
        stream_type="heads_projection",
        pooling="last",
    )
    template_plot(
        dataset="cocoqa_txt",
        stream_type="heads_projection",
        pooling="mean",
    )
    template_plot(
        dataset="cocoqa_img",
        stream_type="heads_projection",
        pooling="mean",
    )
    template_plot(
        dataset="cocoqa_txt_minus_cocoqa_img",
        stream_type="heads_projection",
        pooling="mean",
    )


def plot_all_mean_heads_projection_similarity_measures(
    results_loader: ResultsLoader,
    save_plots: bool = False,
):
    """
    Plot similarity measures for mean heads projection data.
    Uses the computed mean across heads: (layers,) matrices.
    Creates a single 1x2 plot with both pooling methods differentiated by markers/linestyles.
    """
    print("\n📊 Plotting mean_heads_projection with combined pooling methods...")
    template_plot_similarity_measures(
        results_loader=results_loader,
        stream_type="mean_heads_projection",
        pooling_methods=["last", "mean"],
        measures=["neighborhood_overlap", "linear_cka"],
        save_plots=save_plots,
    )


def template_ax_plot(
    ax: plt.Axes,
    data,
    **kwargs,
):
    """
    Plot data on a single axis.
    """
    # For 1D data (mean heads projection), plot as a line
    if data.ndim == 1:
        ax.plot(range(len(data)), data, **kwargs)
    else:
        print(f"⚠️  Unexpected data shape: {data.shape}")


def template_plot_similarity_measures(
    results_loader: ResultsLoader,
    stream_type: str,
    pooling_methods: List[str],
    measures: List[str],
    save_plots: bool = False,
):
    """
    Template plot function for similarity measures with multiple pooling methods.
    """
    from matplotlib.colors import TABLEAU_COLORS
    from matplotlib.lines import Line2D

    # Set style for better plots
    plt.style.use("default")
    hex_colors = list(TABLEAU_COLORS.values())

    # Define markers and linestyles for different pooling methods
    pooling_styles = {
        "last": {"marker": "o", "linestyle": "-"},
        "mean": {"marker": "s", "linestyle": "--"},
        "none": {
            "marker": "^",
            "linestyle": ":",
        },  # Triangle marker, dotted line for "none"
    }

    height = 8
    width = 6 * len(measures)
    fig, axes = plt.subplots(
        nrows=1, ncols=len(measures), figsize=(width, height), sharey=True
    )

    # Handle single subplot case (axes is not a list when ncols=1)
    if len(measures) == 1:
        axes = [axes]

    datasets = results_loader.config["datasets"].copy()
    datasets.append("cocoqa_txt_minus_cocoqa_img")

    # Collect all data to determine Y limits
    min_all_data_values = []

    # Iterate through each measure (subplot)
    for m_idx, measure in enumerate(measures):
        ax = axes[m_idx]

        # Plot each combination of dataset and pooling method
        for d_idx, dataset in enumerate(datasets):
            for pooling in pooling_methods:
                try:
                    data = results_loader.get_similarity(
                        dataset=dataset,
                        stream_type=stream_type,
                        pooling=pooling,
                        measure=measure,
                    )
                    if isinstance(data, torch.Tensor):
                        data = data.numpy()

                    # Collect data for Y limits calculation
                    min_all_data_values.append(data.min())

                    # Set styling: same color for dataset, different marker/linestyle for pooling
                    color = hex_colors[d_idx % len(hex_colors)]
                    marker = pooling_styles[pooling]["marker"]
                    linestyle = pooling_styles[pooling]["linestyle"]

                    template_ax_plot(
                        ax,
                        data,
                        color=color,
                        marker=marker,
                        markersize=4,
                        linewidth=2,
                        linestyle=linestyle,
                    )

                except Exception as e:
                    print(
                        f"⚠️  Failed to plot {dataset}/{stream_type}_{pooling}/{measure}: {e}"
                    )

        # Configure subplot
        ax.set_title(f"{measure}", fontsize=14)
        ax.grid(True, alpha=0.3)

        # Configure axis labels
        if m_idx == 0:
            ax.set_ylabel("Similarity", fontsize=12)
        else:
            # Remove Y-axis elements for non-first subplots
            ax.tick_params(left=False)
            ax.set_ylabel("")  # Set an empty string as ylabel

        ax.set_xlabel("Layer", fontsize=12)

    # Set Y limits: min = 0 or minimum data value, max = 1
    if min_all_data_values:
        y_min = min(0, min(min_all_data_values))
        y_max = 1.0
        for ax in axes:
            ax.set_ylim(y_min, y_max)

    # === ADD CUSTOM LEGENDS ONLY AT THE END ===
    # Only add legends to the first subplot (0,0)
    first_ax = axes[0]

    # Create custom legend for datasets (colors)
    dataset_legend_elements = []
    for d_idx, dataset in enumerate(datasets):
        color = hex_colors[d_idx % len(hex_colors)]
        dataset_legend_elements.append(
            Line2D([0], [0], color=color, linewidth=3, label=dataset)
        )

    # Create custom legend for pooling methods (markers/linestyles)
    pooling_legend_elements = []
    for pooling in pooling_methods:
        marker = pooling_styles[pooling]["marker"]
        linestyle = pooling_styles[pooling]["linestyle"]
        pooling_legend_elements.append(
            Line2D(
                [0],
                [0],
                color="black",
                marker=marker,
                linestyle=linestyle,
                markersize=8,  # Larger marker for better visibility
                linewidth=2.5,  # Thicker line for better visibility
                markerfacecolor="black",  # Fill marker with black
                markeredgecolor="black",  # Black marker edge
                markeredgewidth=1,  # Visible marker edge
                label=pooling,
            )
        )

    # Add dataset legend first
    dataset_legend = first_ax.legend(
        handles=dataset_legend_elements,
        # title="",
        loc="upper left",
        fontsize=9,
        title_fontsize=10,
        frameon=True,
        fancybox=True,
        shadow=True,
        bbox_to_anchor=(0.0, 1),
        ncol=len(datasets),
    )

    # Add dataset legend as artist to preserve it when adding second legend
    first_ax.add_artist(dataset_legend)

    # Add pooling methods legend second
    first_ax.legend(
        handles=pooling_legend_elements,
        # title="Pooling Methods",
        loc="upper left",
        fontsize=9,
        title_fontsize=10,
        frameon=True,
        fancybox=True,
        shadow=True,
        bbox_to_anchor=(0.0, 0.95),
        ncol=len(pooling_methods),
        handlelength=5.0,  # Increase line length in legend
        handletextpad=1.0,  # Add space between line and text
    )

    # Set overall title
    plt.suptitle(f"{stream_type}", fontsize=16, y=0.98)
    plt.tight_layout()

    if save_plots:
        plots_dir = Path(__file__).parent.parent / "plots"
        plots_dir.mkdir(exist_ok=True)
        filename = f"similarity_measures_{stream_type}_{'_'.join(pooling_methods)}_{'_'.join(measures)}.png"
        plt.savefig(plots_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_all_output_layer_similarity_measures(
    results_loader: ResultsLoader,
    save_plots: bool = False,
):
    """
    Plot similarity measures for output layer data.
    """
    print("\n📊 Plotting output_layer with combined pooling methods...")
    template_plot_similarity_measures(
        results_loader=results_loader,
        stream_type="output_layer",
        pooling_methods=["last", "mean"],
        measures=["neighborhood_overlap", "linear_cka"],
        save_plots=save_plots,
    )


def plot_all_post_mlp_similarity_measures(
    results_loader: ResultsLoader,
    save_plots: bool = False,
):
    """
    Plot similarity measures for post-mlp data.
    """
    print("\n📊 Plotting post_mlp with combined pooling methods...")
    template_plot_similarity_measures(
        results_loader=results_loader,
        stream_type="post_mlp",
        pooling_methods=["last", "mean"],
        measures=["neighborhood_overlap", "linear_cka"],
        save_plots=save_plots,
    )


def template_plot_model_data_measures(
    results_loader: ResultsLoader,
    dataset_measure: str,
    stream_type: str,
    pooling_methods: List[str],
    save_plots: bool = False,
):
    """
    Template plot function for model data measures.
    Creates subplots for each model, showing dataset measures across layers.
    """
    from matplotlib.colors import TABLEAU_COLORS
    from matplotlib.lines import Line2D

    # Set style for better plots
    plt.style.use("default")
    hex_colors = list(TABLEAU_COLORS.values())

    # Define markers and linestyles for different pooling methods
    pooling_styles = {
        "last": {"marker": "o", "linestyle": "-"},
        "mean": {"marker": "s", "linestyle": "--"},
        "none": {
            "marker": "^",
            "linestyle": ":",
        },  # Triangle marker, dotted line for "none"
    }

    if "model_data_measures" not in results_loader.data:
        print("⚠️  No model data measures loaded")
        return

    model_data_measures = results_loader.data["model_data_measures"]

    # Get all available models from all datasets
    all_models = set()
    datasets = []
    for dataset_name, dataset_data in model_data_measures.items():
        datasets.append(dataset_name)
        all_models.update(dataset_data.keys())

    all_models = sorted(list(all_models))

    # Create subplots: 1 row, num_models columns
    height = 8
    width = 6 * len(all_models)
    fig, axes = plt.subplots(
        nrows=1, ncols=len(all_models), figsize=(width, height), sharey=True
    )

    # Handle single subplot case
    if len(all_models) == 1:
        axes = [axes]

    # Collect all data values for Y limits
    all_data_values = []

    # Plot for each model (subplot)
    for m_idx, model in enumerate(all_models):
        ax = axes[m_idx]

        # Plot each combination of dataset and pooling method
        for d_idx, dataset in enumerate(datasets):
            # Check if this model exists in this dataset
            if model not in model_data_measures[dataset]:
                continue

            model_data = model_data_measures[dataset][model]

            # Check if the measure type exists
            if dataset_measure not in model_data:
                continue

            measure_data = model_data[dataset_measure]

            for pooling in pooling_methods:
                # Handle different pooling methods for different measure types
                if dataset_measure == "prompt_entropy":
                    # Prompt entropy uses "none" as pooling method
                    key = f"{stream_type}_none"
                else:
                    # Construct the key: stream_type_pooling
                    key = f"{stream_type}_{pooling}"

                # For intrinsic dimension, we need to handle additional parameters
                if dataset_measure == "intrinsic_dimension":
                    # Look for keys that start with our prefix (e.g., "output_layer_last")
                    matching_keys = [
                        k for k in measure_data.keys() if k.startswith(key)
                    ]
                    if not matching_keys:
                        continue
                    # Use the first matching key (they should all have the same data anyway)
                    actual_key = matching_keys[0]
                else:
                    actual_key = key
                    if actual_key not in measure_data:
                        continue

                try:
                    # Get the data - it's stored as a dict with one item usually
                    data_dict = measure_data[actual_key]
                    if isinstance(data_dict, dict) and len(data_dict) > 0:
                        # Get the first (and usually only) tensor from the dict
                        data = next(iter(data_dict.values()))
                    else:
                        data = data_dict

                    if isinstance(data, torch.Tensor):
                        data = data.numpy()

                    # Collect data for Y limits calculation
                    all_data_values.extend(data.flatten())

                    # Set styling: same color for dataset, different marker/linestyle for pooling
                    color = hex_colors[d_idx % len(hex_colors)]
                    marker = pooling_styles[pooling]["marker"]
                    linestyle = pooling_styles[pooling]["linestyle"]

                    # Plot the data
                    ax.plot(
                        range(len(data)),
                        data,
                        color=color,
                        marker=marker,
                        markersize=4,
                        linewidth=2,
                        linestyle=linestyle,
                    )

                except Exception as e:
                    print(
                        f"⚠️  Failed to plot {dataset}/{model}/{dataset_measure}/{actual_key}: {e}"
                    )

        # Configure subplot
        ax.set_title(f"{model}", fontsize=14)
        ax.grid(True, alpha=0.3)

        # Configure axis labels
        if m_idx == 0:
            ax.set_ylabel(f"{dataset_measure.replace('_', ' ').title()}", fontsize=12)
        else:
            # Remove Y-axis elements for non-first subplots
            ax.tick_params(left=False)
            ax.set_ylabel("")

        ax.set_xlabel("Layer", fontsize=12)

    # Set Y limits based on all data
    if all_data_values:
        y_min = min(all_data_values)
        y_max = max(all_data_values)
        # Add some padding
        y_range = y_max - y_min
        y_min_padded = y_min - 0.05 * y_range
        y_max_padded = y_max + 0.05 * y_range

        for ax in axes:
            ax.set_ylim(y_min_padded, y_max_padded)

    # === ADD CUSTOM LEGENDS ONLY AT THE END ===
    # Only add legends to the first subplot (0,0)
    if len(all_models) > 0:
        first_ax = axes[0]

        # Create custom legend for datasets (colors)
        dataset_legend_elements = []
        for d_idx, dataset in enumerate(datasets):
            color = hex_colors[d_idx % len(hex_colors)]
            dataset_legend_elements.append(
                Line2D([0], [0], color=color, linewidth=3, label=dataset)
            )

        # Create custom legend for pooling methods (markers/linestyles)
        pooling_legend_elements = []
        for pooling in pooling_methods:
            marker = pooling_styles[pooling]["marker"]
            linestyle = pooling_styles[pooling]["linestyle"]
            pooling_legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    color="black",
                    marker=marker,
                    linestyle=linestyle,
                    markersize=8,
                    linewidth=2.5,
                    markerfacecolor="black",
                    markeredgecolor="black",
                    markeredgewidth=1,
                    label=pooling,
                )
            )

        # Add dataset legend first
        dataset_legend = first_ax.legend(
            handles=dataset_legend_elements,
            loc="lower right",
            fontsize=9,
            title_fontsize=10,
            frameon=True,
            fancybox=True,
            shadow=True,
            bbox_to_anchor=(1, 0.0),
            ncol=len(datasets),
        )

        # Add dataset legend as artist to preserve it when adding second legend
        first_ax.add_artist(dataset_legend)

        # Add pooling methods legend second
        first_ax.legend(
            handles=pooling_legend_elements,
            loc="lower right",
            fontsize=9,
            title_fontsize=10,
            frameon=True,
            fancybox=True,
            shadow=True,
            bbox_to_anchor=(1, 0.05),
            ncol=len(pooling_methods),
            handlelength=5.0,
            handletextpad=1.0,
        )

    # Set overall title
    plt.suptitle(
        f"{dataset_measure.replace('_', ' ').title()} - {stream_type}",
        fontsize=16,
        y=0.98,
    )
    plt.tight_layout()
    if save_plots:
        plots_dir = Path(__file__).parent.parent / "plots"
        plots_dir.mkdir(exist_ok=True)
        filename = f"model_data_measures_{dataset_measure}_{stream_type}_{'_'.join(pooling_methods)}.png"
        plt.savefig(plots_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_all_model_data_measures(
    results_loader: ResultsLoader,
    save_plots: bool = False,
):
    template_plot_model_data_measures(
        results_loader=results_loader,
        dataset_measure="dataset_entropy",
        stream_type="output_layer",
        pooling_methods=["last", "mean"],
        save_plots=save_plots,
    )
    template_plot_model_data_measures(
        results_loader=results_loader,
        dataset_measure="prompt_entropy",
        stream_type="output_layer",
        pooling_methods=["none"],
        save_plots=save_plots,
    )
    template_plot_model_data_measures(
        results_loader=results_loader,
        dataset_measure="intrinsic_dimension",
        stream_type="output_layer",
        pooling_methods=["last", "mean"],
        save_plots=save_plots,
    )
    template_plot_model_data_measures(
        results_loader=results_loader,
        dataset_measure="dataset_entropy",
        stream_type="post_mlp",
        pooling_methods=["last", "mean"],
        save_plots=save_plots,
    )
    template_plot_model_data_measures(
        results_loader=results_loader,
        dataset_measure="prompt_entropy",
        stream_type="post_mlp",
        pooling_methods=["none"],
        save_plots=save_plots,
    )
    template_plot_model_data_measures(
        results_loader=results_loader,
        dataset_measure="intrinsic_dimension",
        stream_type="post_mlp",
        pooling_methods=["last", "mean"],
        save_plots=save_plots,
    )


def plot_transplanting_layers_benchmarking(
    results_loader: ResultsLoader,
    transplantation_method: str,
    save_plots: bool = False,
):
    """
    Plot benchmarking and transplanting layers.

    Args:
        results_loader: ResultsLoader instance with loaded data
        transplantation_method: Either 'sliding_window_ws2_s2' or 'two_parts_s2'
    """

    # Load transplanting data for both datasets
    datasets = ["cocoqa_txt", "cocoqa_img"]

    # Validate transplantation method
    if "transplanting_layers_benchmarking" not in results_loader.data:
        print("⚠️  No transplanting layers benchmarking data loaded")
        return

    print(f"\n📊 Plotting transplanting layers benchmarking: {transplantation_method}")

    # Create figure with 1x2 subplots (shared y-axis)
    fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharey=True)

    # Process each dataset
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]

        try:
            # Get transplant data for this dataset and method
            data_df = results_loader.get_transplant(dataset, transplantation_method)

            # Extract layer data (excluding mm_model and mm_pretrained_connector)
            layer_prefix = (
                "start_layer_"
                if "sliding_window" in transplantation_method
                else "split_layer_"
            )

            # Get layer entries - the layer names are in the first column (Unnamed: 0)
            layer_names_column = data_df.columns[0]  # "Unnamed: 0"
            layer_rows_mask = data_df[layer_names_column].str.startswith(layer_prefix)
            layer_data = data_df[layer_rows_mask]

            layer_numbers = []
            layer_accuracies = []

            for _, row in layer_data.iterrows():
                # Extract layer number from string like "start_layer_4" or "split_layer_7"
                layer_name = row[layer_names_column]
                layer_num = int(layer_name.split("_")[-1])
                accuracy = row["accuracy"]

                layer_numbers.append(layer_num)
                layer_accuracies.append(accuracy)

            # Sort by layer number
            sorted_data = sorted(zip(layer_numbers, layer_accuracies))
            layer_numbers, layer_accuracies = zip(*sorted_data)

            # Get full model and pretrained model accuracies
            fm_row = data_df[data_df[layer_names_column] == "mm_model"]
            pm_row = data_df[data_df[layer_names_column] == "mm_pretrained_connector"]

            fm_accuracy = fm_row["accuracy"].iloc[0]
            pm_accuracy = pm_row["accuracy"].iloc[0]

            # Prepare data for plotting
            x_positions = list(range(len(layer_numbers) + 2))  # +2 for FM and PM
            all_accuracies = list(layer_accuracies) + [fm_accuracy, pm_accuracy]
            x_labels = [str(num) for num in layer_numbers] + ["FM", "PM"]

            # Create bar plot
            bars = ax.bar(x_positions, all_accuracies, alpha=0.7, edgecolor="black")

            # Color coding: layers in blue, FM in green, PM in red
            for i, bar in enumerate(bars):
                if i < len(layer_numbers):
                    bar.set_color("skyblue")
                elif x_labels[i] == "FM":
                    bar.set_color("lightgreen")
                else:  # PM
                    bar.set_color("lightcoral")

            # Customize subplot
            ax.set_xlabel("Layer Number / Model Type", fontsize=12)
            if idx == 0:  # Only first subplot gets y-label
                ax.set_ylabel("Accuracy", fontsize=12)

            ax.set_title(f"{dataset}", fontsize=14, fontweight="bold")
            ax.set_xticks(x_positions)
            ax.set_xticklabels(x_labels, rotation=45 if len(x_labels) > 10 else 0)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3, axis="y")

            # Add value labels on top of bars
            for i, (pos, acc) in enumerate(zip(x_positions, all_accuracies)):
                ax.text(
                    pos,
                    acc + 0.01,
                    f"{acc:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        except Exception as e:
            print(f"⚠️  Failed to plot {dataset}/{transplantation_method}: {e}")
            ax.text(
                0.5,
                0.5,
                f"No data for\n{dataset}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=14,
                color="red",
            )
            ax.set_title(f"{dataset}", fontsize=14, fontweight="bold")

    # Add overall title
    method_display = transplantation_method.replace("_", " ").title()
    plt.suptitle(
        f"Transplanting Layers Benchmarking: {method_display}",
        fontsize=16,
        fontweight="bold",
        y=0.95,
    )

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="skyblue", label="Layer Transplantation"),
        Patch(facecolor="lightgreen", label="Full Model (FM)"),
        Patch(facecolor="lightcoral", label="Pretrained Model (PM)"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=3,
        fontsize=11,
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Make room for title and legend
    if save_plots:
        plots_dir = Path(__file__).parent.parent / "plots"
        plots_dir.mkdir(exist_ok=True)
        filename = f"transplanting_layers_benchmarking_{transplantation_method}.png"
        plt.savefig(plots_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_all_transplanting_layers_benchmarking(
    results_loader: ResultsLoader, save_plots: bool = False
):
    """
    Plot all available transplanting layers benchmarking experiments.
    """
    print("\n📊 Plotting all transplanting layers benchmarking experiments...")

    # Available transplantation methods
    methods = ["sliding_window_ws2_s2", "two_parts_s2"]

    for method in methods:
        plot_transplanting_layers_benchmarking(
            results_loader, method, save_plots=save_plots
        )


def main(save_plots: bool = False):
    """Main function to demonstrate similarity measures heatmap."""
    # Initialize results loader and load all data
    results = ResultsLoader()
    results.load_all()
    results.print_summary()

    # Create similarities difference dataset using renamed method
    results.create_similarities_difference_dataset(
        source_dataset1="cocoqa_txt",
        source_dataset2="cocoqa_img",
        target_dataset="cocoqa_txt_minus_cocoqa_img",
        verbose=False,
    )

    # Compute mean heads projections at layers using renamed method
    results.compute_mean_heads_projections_at_layers()

    # Create model measures differences for each dataset
    print("\n🔄 Creating model measures differences...")

    # Create model measures differences using renamed method
    results.create_dataset_measures_models_difference(
        dataset_name="cocoqa_txt",
        model1="pretrained",
        model2="full",
    )
    results.create_dataset_measures_models_difference(
        dataset_name="cocoqa_img",
        model1="pretrained",
        model2="full",
    )

    results.print_summary()

    # ! do not touch this code anymore
    print("🎉 All processing completed!")

    # plot_all_similarity_measures_matrices(results, save_plots=save_plots)
    plot_all_output_layer_similarity_measures(results, save_plots=save_plots)
    plot_all_post_mlp_similarity_measures(results, save_plots=save_plots)
    # plot_all_mean_heads_projection_similarity_measures(results, save_plots=save_plots)
    # plot_all_model_data_measures(results, save_plots=save_plots)
    # plot_all_transplanting_layers_benchmarking(results, save_plots=save_plots)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate multimodal analysis plots")
    parser.add_argument(
        "--save", action="store_true", help="Save plots instead of showing them"
    )
    args = parser.parse_args()

    main(save_plots=args.save)
