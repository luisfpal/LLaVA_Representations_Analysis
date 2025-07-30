from .argparsing_utils import (
    parse_layer_index,
    parse_question_instruction,
    parse_tokens_mode,
)
from .constants import (
    GUIDE_TEXT,
    ANSWER_TEXT,
    SYSTEM_ROLE,
    ASSISTANT_ROLE,
    SQA_ANSWER_CHOICES,
    MMLU_ANSWER_CHOICES,
    COCOQA_VI_DIGITS_MAP,
)
from .dataset import (
    get_dataloader,
    preprocess_batch,
    format_prompts,
)
from .io_utils import (
    save_extracted_residual_stream_data,
    load_layers_residual_stream,
)
from .metrics_utils import (
    compute_neighborhood_overlap,
    compute_similarity,
    compute_intrinsic_dimension,
    compute_matrix_based_renyi_entropy,
    compute_layers_residual_stream_entropy,
    compute_layers_residual_stream_similarities,
    compute_heads_projection_residual_stream_similarities,
    compute_layers_intrinsic_dimension,
)
from .model_utils import (
    load_hf_model_and_processor_or_tokenizer,
    resolve_layer_indices,
    replace_multimodal_lm,
    replace_multimodal_projector,
    ProcessorType,
    ModelType,
    get_hidden_size,
)
from .naming_utils import (
    create_filename,
    generate_filename_suffix,
    setup_directories,
    create_filename_from_paths,
)
from .operations_utils import seed_all, sample_unique_row_indices
from .multimodal_extraction_setup import setup_multimodal_model
from .benchmarking_utils import (
    parse_predicted_answer,
    benchmark_model_vqa_processed_dataloader,
)
from .layers_weights_transplantation import transplant_layers_weights
from .plot_utils import plot_similarity_measure_heatmap, plot_similarity_measure_matrix

__all__ = [
    "parse_layer_index",
    "parse_question_instruction",
    "parse_tokens_mode",
    "GUIDE_TEXT",
    "ANSWER_TEXT",
    "SYSTEM_ROLE",
    "ASSISTANT_ROLE",
    "SQA_ANSWER_CHOICES",
    "MMLU_ANSWER_CHOICES",
    "COCOQA_VI_DIGITS_MAP",
    "get_dataloader",
    "preprocess_batch",
    "format_prompts",
    "save_extracted_residual_stream_data",
    "load_layers_residual_stream",
    "compute_neighborhood_overlap",
    "compute_similarity",
    "compute_intrinsic_dimension",
    "compute_matrix_based_renyi_entropy",
    "compute_layers_residual_stream_entropy",
    "compute_layers_residual_stream_similarities",
    "compute_heads_projection_residual_stream_similarities",
    "compute_layers_intrinsic_dimension",
    "load_hf_model_and_processor_or_tokenizer",
    "resolve_layer_indices",
    "replace_multimodal_lm",
    "replace_multimodal_projector",
    "ProcessorType",
    "ModelType",
    "get_hidden_size",
    "create_filename",
    "generate_filename_suffix",
    "setup_directories",
    "create_filename_from_paths",
    "seed_all",
    "sample_unique_row_indices",
    "setup_multimodal_model",
    "parse_predicted_answer",
    "benchmark_model_vqa_processed_dataloader",
    "transplant_layers_weights",
    "plot_similarity_measure_heatmap",
    "plot_similarity_measure_matrix",
]
