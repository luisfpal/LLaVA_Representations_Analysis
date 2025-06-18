import torch
import os
from typing import Tuple, Type, Union, List, Optional
import argparse
from transformers import (
    LlavaProcessor,
    LlavaForConditionalGeneration,
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
    AutoModelForCausalLM,
    AutoTokenizer,
    # LlamaForCausalLM,  # Consider removing if AutoModelForCausalLM is sufficient
    # LlamaTokenizerFast, # Consider removing if AutoTokenizer is sufficient
)
from PIL import Image
from .constants import (
    ANSWER_TEXT,
    SYSTEM_ROLE,
    ASSISTANT_ROLE,
)

# Define supported vision-language model types and their corresponding classes
SUPPORTED_VL_MODELS = {
    "llava-1.5": {
        "processor": LlavaProcessor,
        "model": LlavaForConditionalGeneration,
    },
    "llava-v1.6": {
        "processor": LlavaNextProcessor,
        "model": LlavaNextForConditionalGeneration,
    },
    "paligemma2": {
        "processor": PaliGemmaProcessor,
        "model": PaliGemmaForConditionalGeneration,
    },
}

# Define types for clarity
ProcessorType = Union[LlavaProcessor, PaliGemmaProcessor]
ModelType = Union[
    LlavaForConditionalGeneration,
    PaliGemmaForConditionalGeneration,
    AutoModelForCausalLM,
]


def _get_vl_model_classes(
    model_name_or_path: str,
) -> Tuple[Type[ProcessorType], Type[ModelType]]:
    """
    Helper function to determine the processor and model classes based on the model name.

    Args:
        model_name_or_path (str): The name or path of the model.

    Returns:
        Tuple[Type[ProcessorType], Type[ModelType]]: The processor and model classes.

    Raises:
        ValueError: If the model type is not supported.
    """
    for key, classes in SUPPORTED_VL_MODELS.items():
        if key in model_name_or_path.lower():
            return classes["processor"], classes["model"]
    raise ValueError(
        f"Model type for '{model_name_or_path}' is not supported. "
        f"Supported types: {', '.join(SUPPORTED_VL_MODELS.keys())}"
    )


def load_hf_model_and_processor_or_tokenizer(
    model_name_or_path: str,
    cache_dir: str,
    text_model: bool = False,
    device_map: str = "auto",
    dtype: torch.dtype = torch.float16,
    attn_implementation: str = "flash_attention_2",
    low_cpu_mem_usage: bool = True,
    use_fast: bool = True,
) -> Tuple[ModelType, Union[ProcessorType, AutoTokenizer]]:
    """
    Load a Hugging Face model and its processor or tokenizer.

    Args:
        model_name_or_path (str): Model identifier or local path.
        cache_dir (str): Directory for caching the model and processor.
        text_model (bool): Whether to load a text-only model.
        device_map (str): Device map strategy for model loading.
        dtype (torch.dtype): Data type for model parameters.
        attn_implementation (str): Attention implementation type.
        low_cpu_mem_usage (bool): Optimize for low CPU memory usage.
        use_fast (bool): Use the fast version of the processor if available.

    Returns:
        Tuple[ModelType, Union[ProcessorType, AutoTokenizer]]: The loaded model and processor/tokenizer.

    Raises:
        ValueError: If the model type is unsupported.
        RuntimeError: If loading the model or processor fails.
    """
    cache_dir = os.path.expanduser(cache_dir)

    try:
        if not text_model:
            ProcessorClass, ModelClass = _get_vl_model_classes(model_name_or_path)
        else:
            ProcessorClass = AutoTokenizer
            ModelClass = AutoModelForCausalLM
    except ValueError as e:
        raise e  # Re-raise the specific ValueError

    processor_kwargs = {
        "pretrained_model_name_or_path": model_name_or_path,
        "cache_dir": cache_dir,
    }

    model_kwargs = {
        **processor_kwargs,
        "device_map": device_map,
        "low_cpu_mem_usage": low_cpu_mem_usage,
        "attn_implementation": attn_implementation,
        "torch_dtype": dtype,
    }

    try:
        processor = ProcessorClass.from_pretrained(
            **processor_kwargs, use_fast=use_fast
        )

        if "vicuna" in model_name_or_path.lower():
            chat_template = """
            {% for message in messages %}{% if message['role'] != 'system' %}{{ message['role'].upper() + ': '}}{% endif %}{# Render all images first #}{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}{{ '<image>\n' }}{% endfor %}{# Render all text next #}{% if message['role'] != 'assistant' %}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{{ content['text'] + ' '}}{% endfor %}{% else %}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{% generation %}{{ content['text'] + ' '}}{% endgeneration %}{% endfor %}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}
            """.strip()
            processor.chat_template = chat_template
    except Exception as e:
        raise RuntimeError(
            "Failed to load "
            f"{('processor' if not text_model else 'tokenizer')} "
            f"for {model_name_or_path} from cache {cache_dir}. Error: {e}"
        )
    try:
        model = ModelClass.from_pretrained(**model_kwargs)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load model for '{model_name_or_path}' from cache '{cache_dir}'. Error: {e}"
        )

    return model, processor


def replace_multimodal_lm(
    multimodal_model: ModelType,
    replacement_lm_name_or_path: str,
    cache_dir: str,
) -> ModelType:
    """
    Replace the Multimodal language model with the Vicuna model.

    Args:
        model (ModelType): The original Multimodal model.
        replacement_lm_name_or_path (str): The name or path of the language model to replace with (e.g., Vicuna).
        cache_dir (str): Directory for caching the model.

    Returns:
        ModelType: The updated Multimodal model.
    """
    # Original Multimodal model device
    multimodal_model_device = next(multimodal_model.parameters()).device

    print(
        f"\nLoading {replacement_lm_name_or_path} model for language model replacement..."
    )
    # Load the replacement language model (e.g., Vicuna)
    replacement_lm, _ = load_hf_model_and_processor_or_tokenizer(
        model_name_or_path=replacement_lm_name_or_path,
        cache_dir=cache_dir,
        text_model=True,
        device_map="cpu",
        dtype=torch.float16,
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
    )

    # Check if the type of the replacement model is compatible
    if type(replacement_lm.model) is not type(multimodal_model.language_model.model):
        raise TypeError(
            f"\nReplacement model type {type(replacement_lm.model)} is not compatible "
            f"with Multimodal model type {type(multimodal_model.language_model.model)}."
        )
    print(f"\nSuccessfully loaded {replacement_lm_name_or_path} mode")

    # Move Multimodal model to CPU
    multimodal_model.to("cpu")

    print("Replacing Multimodal language model...")
    min_vocab_size = min(
        multimodal_model.language_model.model.embed_tokens.weight.size(0),
        replacement_lm.model.embed_tokens.weight.size(0),
    )

    with torch.no_grad():
        multimodal_model.language_model.model.embed_tokens.weight[:min_vocab_size] = (
            replacement_lm.model.embed_tokens.weight[:min_vocab_size].clone()
        )
        multimodal_model.language_model.model.layers.load_state_dict(
            replacement_lm.model.layers.state_dict()
        )
        multimodal_model.language_model.model.norm.load_state_dict(
            replacement_lm.model.norm.state_dict()
        )
        multimodal_model.language_model.lm_head.weight[:min_vocab_size] = (
            replacement_lm.lm_head.weight[:min_vocab_size].clone()
        )

    # Explanation for LLaVA:
    # - `multimodal_model.language_model.model` is the LLaVA language model.
    # - `replacement_lm` contains a `model` attribute that is the language model
    # - The LLaVA language model includes additional parameters, that act as placeholders
    #  for image embeddings, in the embed_tokens and lm_head layers.
    # !These additional parameters do not influence the language model's behavior but
    # !are necessary for the model to function correctly.
    # !These parameters are changed for image embeddings when there is an image as context.
    # !For more details check the modelling_llava.py of the transformers library.
    # !Specifically, lines 413-430 under the class `LlavaForConditionalGeneration`
    # !in transformers/models/llava/modeling_llava.py, version 4.50.1.

    # Remove the vicuna model and its tokenizer from memory
    del replacement_lm, _
    torch.cuda.empty_cache()

    # Move the updated llava model back to its original device
    multimodal_model.to(multimodal_model_device)

    print("Successfully replaced multimodal language model.")
    return multimodal_model


def format_prompts(
    questions_and_options: List[str],
    images: Optional[Image.Image],
    args: argparse.Namespace,
    processor: ProcessorType,
    chat_template_exists: bool = False,
) -> List:
    """
    Format input questions and images into structured prompts for a model.

    Parameters:
        questions_and_options: List of question and options in string format.
        images: List of images associated with each question.
        args: Arguments namespace with a `chat_mode` boolean attribute.

    Returns:
        list: Formatted prompts either as conversations (dict format) or plain strings.
    """

    if args.chat_mode and chat_template_exists:
        conversations_list = _format_as_conversations(
            questions_and_options, images, args
        )
        kwargs_chat_template = {
            "conversation": conversations_list,
            "tokenize": False,
        }
        if args.continue_final_message:
            kwargs_chat_template["continue_final_message"] = True
            kwargs_chat_template["add_generation_prompt"] = False
        else:
            kwargs_chat_template["add_generation_prompt"] = True
        # Apply the chat template to format the conversations
        return processor.apply_chat_template(**kwargs_chat_template)

    else:
        return _format_as_plain_prompts(questions_and_options, images)


def _format_as_conversations(questions_and_options, images, args):
    formatted_conversations = []
    continue_final_message = (
        hasattr(args, "continue_final_message") and args.continue_final_message
    )
    for question_text, image in zip(questions_and_options, images):
        content = [{"type": "text", "text": f"{question_text}{args.guide_text}"}]

        if image is not None:
            content.append({"type": "image"})

        conversation = [{"role": "user", "content": content}]
        conversation.insert(0, SYSTEM_ROLE)
        if continue_final_message:
            conversation.append(ASSISTANT_ROLE)
        formatted_conversations.append(conversation)
    return formatted_conversations


def _format_as_plain_prompts(questions_and_options, images):
    formatted_prompts = []
    for image, question_text in zip(images, questions_and_options):
        prompt_prefix = "<image>\n" if image is not None else ""
        prompt = f"{prompt_prefix}{question_text}{ANSWER_TEXT}"
        formatted_prompts.append(prompt)
    return formatted_prompts


def resolve_layer_indices(
    model,
    layer_index: Union[int, List[int], str],
) -> List[int]:
    """
    Resolves the layer indices to extract based on the input specification.
    Handles "all", a list of indices, or a single index.
    Indices are expected to be 1-based for transformer layers (excluding embeddings).

    Args:
        model: The model to extract from.
        layer_index: Specification of which layers to extract from.

    Returns:
        List[int]: A list of layer indices to extract from, in the range 1 to num_layers.
        For further applications, one might want to subtract 1 to get the indices
        in the range 0 to num_layers - 1.

    Raises:
        ValueError: If the layers cannot be determined.
    """
    num_layers = None
    # Get the total number of layers in the model
    if hasattr(model, "config") and hasattr(model.config, "num_hidden_layers"):
        num_layers = model.config.num_hidden_layers
    elif hasattr(model, "language_model") and hasattr(
        model.language_model.config, "num_hidden_layers"
    ):
        num_layers = model.language_model.config.num_hidden_layers
    else:
        raise ValueError(
            "Could not determine the number of layers: No 'num_hidden_layers' attribute found "
            "in model.config or model.language_model.config."
        )
    if isinstance(layer_index, str) and layer_index.lower() == "all":
        return list(range(1, num_layers + 1))
    elif isinstance(layer_index, list):
        if not all(1 <= idx <= num_layers for idx in layer_index):
            raise ValueError(
                f"Invalid layer indices: {layer_index}. "
                f"Expected integers in the range 1 to {num_layers}."
            )
        return layer_index
    elif isinstance(layer_index, int):
        if not 1 <= layer_index <= num_layers:
            raise ValueError(
                f"Invalid layer index: {layer_index}. "
                f"Expected an integer in the range 1 to {num_layers}."
            )
        return [layer_index]
    else:
        raise ValueError(
            f"Invalid layer_index type: {type(layer_index)}. Expected int, List[int], or 'all'."
            f" Received: {layer_index}"
        )
