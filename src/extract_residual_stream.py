from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, Union, List
import torch
from tqdm import tqdm
from PIL import Image
from utils import format_prompts, resolve_layer_indices


def extract_residual_stream(
    model,
    processor,
    dataloader: DataLoader,
    args: Optional[Dict[str, Any]] = None,
    text_model: Optional[bool] = False,
) -> Dict[str, Any]:
    """
    Extracts the residual stream from a model for a given dataset.

    Args:
        model: The model from which to extract the residual stream.
        processor: The processor or tokenizer to use for the model.
        dataloader: The dataloader containing the dataset.

    Returns:
        A dictionary containing:
        - 'residual_stream_data': Either a tensor (for single layer) or
                                  a dictionary mapping layer indices to tensors (for multiple layers)
        - 'prompts': List of input prompts that were processed
    """
    # Input validation
    if args.token_index is not None and args.mean_over_tokens:
        raise ValueError(
            "Cannot specify both token_index and mean_over_tokens. Choose one."
        )

    # Get model dimensions
    hidden_size = _get_hidden_size(model)
    num_samples = len(dataloader.dataset)

    print(f"Extracting residual streams for {num_samples} samples...")

    # Resolve which layers to extract from
    layer_indices = resolve_layer_indices(model, args.layer_index)

    # Initialize storage for results
    multi_layer_data = {
        f"layer_{idx}": torch.zeros(num_samples, hidden_size, dtype=torch.float16)
        for idx in layer_indices
    }

    input_counter = 0
    prompts = []

    # Create a single progress bar for all samples
    progress_bar = tqdm(total=num_samples, desc="Processing samples", unit="sample")

    # Check if not empty chat_template exists
    chat_template_exists = False
    if text_model and hasattr(processor, "chat_template"):
        chat_template_exists = bool(processor.chat_template)
    elif not text_model:
        chat_template_exists = True

    # Process each batch
    for batch in dataloader:
        questions_and_options = batch["questions"]
        images = batch["images"]

        # Process each question in the batch
        full_prompts = format_prompts(
            questions_and_options=questions_and_options,
            images=images,
            args=args,
            processor=processor,
            chat_template_exists=chat_template_exists,
        )
        for prompt, image in zip(full_prompts, images):
            prompts.append(prompt)

            # Extract hidden states for the sample
            hidden_states_dict = _process_sample_multi_layer(
                model=model,
                text_model=text_model,
                processor=processor,
                prompt=prompt,
                image=image,
                layer_indices=layer_indices,
                token_index=args.token_index,
                mean_over_tokens=args.mean_over_tokens,
            )  # Dict of len(layer_indices) with tensors of shape (hidden_size,)

            # Store results for each layer
            for layer_idx, state_tensor in hidden_states_dict.items():
                if f"layer_{layer_idx}" in multi_layer_data:
                    multi_layer_data[f"layer_{layer_idx}"][input_counter] = state_tensor
                else:
                    # This case should not happen if multi_layer_data is initialized correctly
                    print(
                        f"Warning: Layer {layer_idx} not found in multi_layer_data keys."
                    )

            input_counter += 1
            progress_bar.update(1)

        del (
            questions_and_options,
            images,
            full_prompts,
        )
        torch.cuda.empty_cache()

        if input_counter >= num_samples:  # Check after processing a batch
            break

    if input_counter != num_samples:
        # This warning is more accurate now
        print(
            f"Warning: Processed {input_counter} samples, but expected {num_samples} samples. "
            "This might be due to batching or dataset issues."
        )

    # Close the progress bar
    progress_bar.close()

    # Format return value for backward compatibility
    return {"residual_stream_data": multi_layer_data, "prompts": prompts}


def _get_hidden_size(model) -> int:
    """
    Retrieves the hidden dimension size from the model configuration.
    Tries to access `config.hidden_size` and then `language_model.config.hidden_size`.

    Args:
        model: The model to inspect.

    Returns:
        int: The hidden size dimension.

    Raises:
        ValueError: If the hidden size cannot be determined.
    """
    hidden_size = getattr(getattr(model, "config", None), "hidden_size", None)
    if hidden_size is not None:
        return hidden_size

    hidden_size = getattr(
        getattr(getattr(model, "language_model", None), "config", None),
        "hidden_size",
        None,
    )
    if hidden_size is not None:
        return hidden_size

    raise ValueError(
        "Could not determine model hidden size: No valid config.hidden_size found"
    )


def _process_sample_multi_layer(
    model,
    text_model,
    processor,
    prompt: str,
    image: Union[Image.Image, None],
    layer_indices: List[int],
    token_index: Optional[int],
    mean_over_tokens: bool,
) -> Dict[int, torch.Tensor]:
    """
    Processes a single text sample to extract hidden states from specified layers.

    Args:
        model: The model to extract from.
        processor: The text processor or tokenizer.
        questions: The input text or question.
        images: The input image or images (if applicable).
        layer_indices: List of layer indices to extract from.
        token_index: Index of the token to extract (None or -1 for the last token).
        mean_over_tokens: Whether to average across all tokens.

    Returns:
        Dict[int, torch.Tensor]: Dictionary mapping layer indices to their hidden states.

    Raises:
        ValueError: If the token index is out of range.
    """

    # Tokenize and prepare input
    processor_kwargs = {
        "text": prompt,
        "return_tensors": "pt",
    }

    if text_model:
        inputs = processor(**processor_kwargs).to(model.device)
    else:
        # For image models, we need to pass the image as well
        inputs = processor(**processor_kwargs, images=image).to(model.device)

    # Run forward pass with gradient tracking disabled
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        # List of hidden states for each layer: num_hidden_layers + 1
        # Each layer's hidden state is a tensor of shape (batch_size, sequence_length, hidden_size)

    # Extract hidden states for each requested layer
    hidden_states_dict = {}
    for layer_idx in layer_indices:
        # hidden_states is a tuple where hidden_states[0] are embeddings,
        # hidden_states[1] is output of layer 1, ..., hidden_states[num_layers] is output of last layer.
        # So, if layer_idx is 1-based (as per resolve_layer_indices for "all"), it directly maps.
        if layer_idx < 0 or layer_idx >= len(hidden_states):
            raise ValueError(
                f"Layer index {layer_idx} is out of range for available hidden_states (len: {len(hidden_states)}). Ensure layer indices are valid."
            )
        layer_hidden_state = hidden_states[layer_idx].squeeze(0)
        # shape: (sequence_length, hidden_size)

        if mean_over_tokens:
            processed_hidden_state = (
                layer_hidden_state.mean(dim=0).detach().cpu().to(torch.float16)
            )
        else:
            seq_len = layer_hidden_state.shape[0]
            actual_token_index = None
            if token_index < 0:  # Convention for last token
                actual_token_index = seq_len + token_index
                if actual_token_index < 0:
                    raise ValueError(
                        f"Token index {token_index} (derived as {actual_token_index}) is out of range for sequence length {seq_len} in layer {layer_idx}."
                    )

            # Validate token index
            if not (0 <= actual_token_index < seq_len):
                # If original token_index was used for error, it might be confusing if it was -1
                raise ValueError(
                    f"Token index {actual_token_index} (derived from input {token_index}) is out of range "
                    f"for sequence length {seq_len} in layer {layer_idx}."
                )
            processed_hidden_state = (
                layer_hidden_state[actual_token_index].detach().cpu().to(torch.float16)
            )

        hidden_states_dict[layer_idx] = processed_hidden_state

    return hidden_states_dict  # Dict of len(layer_indices) with tensors of shape (hidden_size,)
