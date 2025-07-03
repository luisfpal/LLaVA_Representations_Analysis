import torch
from typing import Optional, Tuple
from .dataset import get_dataloader
from .model_utils import (
    replace_multimodal_projector,
    replace_multimodal_lm,
    load_hf_model_and_processor_or_tokenizer,
)
from .model_utils import ModelType, ProcessorType
from torch.utils.data import DataLoader


def setup_multimodal_extraction(
    multimodal_model_name_or_path: str,
    dataset_path_or_name: str,
    model_cache_dir: str,
    dataset_cache_dir: Optional[str] = None,
    batch_size: int = 1,
    seed: int = 42,
    language_model_name_or_path: Optional[str] = None,
    pretrained_projector_name_or_path: Optional[str] = None,
    model_dtype: torch.dtype = torch.float16,
    attn_implementation: str = "flash_attention_2",
    question_instruction_type: Optional[str] = None,
    texts_qa: bool = False,
    images_qa: bool = False,
    split: Optional[str] = None,
    downsample_size: Optional[int] = None,
    replace_language_model: bool = False,
    replace_pretrained_projector: bool = False,
    guide_text: str = "",
) -> Tuple[ModelType, ProcessorType, DataLoader]:
    model, processor = load_hf_model_and_processor_or_tokenizer(
        model_name_or_path=multimodal_model_name_or_path,
        cache_dir=model_cache_dir,
        device_map="cuda:0",
        dtype=model_dtype,
        attn_implementation=attn_implementation,
    )

    if replace_pretrained_projector and pretrained_projector_name_or_path is not None:
        model = replace_multimodal_projector(
            multimodal_model=model,
            pretrained_projector_model_name_or_path=pretrained_projector_name_or_path,
            cache_dir=model_cache_dir,
        )

    # Replace multimodal language model
    if replace_language_model and language_model_name_or_path is not None:
        model = replace_multimodal_lm(
            multimodal_model=model,
            replacement_lm_name_or_path=language_model_name_or_path,
            cache_dir=model_cache_dir,
        )

    if not texts_qa and not images_qa:
        raise ValueError("Either texts_qa or images_qa must be True")

    dataloader = get_dataloader(
        dataset_path_or_name=dataset_path_or_name,
        cache_dir=dataset_cache_dir,
        split=split,
        texts_qa=texts_qa,
        images_qa=images_qa,
        question_instruction_type=question_instruction_type,
        downsample_size=downsample_size,
        seed=seed,
        batch_size=batch_size,
        processor=processor,
        guide_text=guide_text,
    )

    return model, processor, dataloader
