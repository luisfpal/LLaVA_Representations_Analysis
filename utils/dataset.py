import datasets
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, load_from_disk
import os
import random
from typing import Dict, Any, Optional
from .constants import (
    SQA_ANSWER_CHOICES,
    MMLU_ANSWER_CHOICES,
)

MULTIPLE_CHOICE_BENCHMARKS = ["mmlu", "scienceqa"]


def get_hf_dataset_split(
    dataset_path_or_name: str,
    cache_dir: str,
    split: str = "test",
) -> datasets.Dataset:
    """
    Loads a specific split from a dataset, potentially from a local path or Hugging Face Hub.

    Args:
        dataset_path: Path to the dataset directory or Hugging Face dataset identifier.
        split: The name of the dataset split to load (e.g., 'train', 'validation', 'test').

    Returns:
        The loaded dataset split.

    Raises:
        Exception: If the dataset fails to load (e.g., path not found, invalid identifier).
    """
    try:
        dataset_kwargs = {
            "path": dataset_path_or_name,
            "split": split,
            "cache_dir": cache_dir,
        }
        if "mmlu" in dataset_path_or_name.lower():
            # MMLU dataset requires a specific name argument
            dataset_kwargs["name"] = "all"

        dataset = load_dataset(**dataset_kwargs)
        return dataset
    except Exception as e:
        # Catching a broad exception, but could be more specific depending on expected errors
        # from load_dataset (e.g., FileNotFoundError, datasets.exceptions.DatasetNotFoundError)
        raise RuntimeError(
            f"Failed to load dataset split '{split}' from '{dataset_path_or_name}'. Error: {e}"
        )


class MultipleChoiceDatasetBenchmark(Dataset):
    def __init__(
        self,
        dataset_path_or_name: str,
        cache_dir: str,
        split: str = "test",
        texts_qa: bool = False,
        images_qa: bool = False,
        question_instruction_type: Optional[str] = None,
        downsample_size: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        if (
            "scienceqa" not in dataset_path_or_name.lower()
            and "mmlu" not in dataset_path_or_name.lower()
        ):
            raise ValueError(
                f"Dataset {dataset_path_or_name} is not supported. "
                "Only 'scienceqa' and 'mmlu' datasets are supported."
            )

        if texts_qa and images_qa:
            raise ValueError("texts_qa and images_qa cannot be both True.")

        self.dataset_path_or_name = dataset_path_or_name
        self.question_instruction_type = question_instruction_type

        random.seed(seed)  # Set random seed for reproducibility

        # Load the initial dataset
        dataset = get_hf_dataset_split(
            dataset_path_or_name=dataset_path_or_name,
            cache_dir=cache_dir,
            split=split,
        )

        # Apply downsampling if requested and if dataset is larger than requested size
        if downsample_size and downsample_size < len(dataset):
            print(
                f"Downsampling dataset from {len(dataset)} to {downsample_size} samples"
            )
            dataset = dataset.shuffle(seed=seed).select(range(downsample_size))

        mm_dataset = "image" in dataset.features

        # Apply filters if not mm_dataset mode
        if mm_dataset:
            if images_qa:
                dataset = dataset.filter(lambda example: example["image"] is not None)
            if (
                texts_qa
            ):  # This implies multimodal dataset but we only want text samples from it
                dataset = dataset.filter(lambda example: example["image"] is None)

        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        question_data = self.dataset[idx]
        question_text = question_data["question"]
        choices_list = question_data["choices"]
        context_text = question_data.get("hint", None)
        answer_letter = None
        answer_idx = question_data["answer"]
        subject = question_data.get("subject", None)

        def format_subject(subject):
            return subject.replace("_", " ")

        if "scienceqa" in self.dataset_path_or_name.lower():
            answer_letter = SQA_ANSWER_CHOICES[answer_idx]
        elif "mmlu" in self.dataset_path_or_name.lower():
            answer_letter = MMLU_ANSWER_CHOICES[answer_idx]

        full_question_prompt = ""
        if self.question_instruction_type == "singular":
            full_question_prompt = f"The following is a multiple choice question (with answers) about {format_subject(subject)}.\n\n"
        elif self.question_instruction_type == "plural":
            full_question_prompt = f"The following are multiple choice questions (with answers) about {format_subject(subject)}.\n\n"

        if context_text is not None and context_text != "":
            full_question_prompt += f"Context: {context_text}\n"
        full_question_prompt += f"{question_text}\n"

        if choices_list is not None:
            last_choice = choices_list[-1]
            for letter_ord, choice_text in zip(
                range(ord("A"), ord("Z") + 1), choices_list
            ):
                full_question_prompt += f"{chr(letter_ord)}: {choice_text}"
                full_question_prompt += "\n" if choice_text != last_choice else ""

        sample = {
            "question": full_question_prompt,
            "answer_letter": answer_letter,
        }
        # Retrieve the image. It might be None if the sample doesn't have one.
        sample["image"] = question_data.get("image", None)

        return sample


class OpenVQADataset(Dataset):
    """
    Dataset for Open-ended VQA datasets.

    Specifically, it is used for a version of the COCOQA dataset that has been modified to
    remove the images or the captions of the original dataset.

    Args:
        dataset_path_or_name: Path to the dataset directory or Hugging Face dataset identifier.
        downsample_size: Maximum number of samples to use
        seed: Random seed for reproducibility
        remove_images: Whether to remove images from the dataset
    """

    def __init__(
        self,
        dataset_path_or_name: str,
        downsample_size: Optional[int] = None,
        seed: int = 42,
        remove_images: bool = False,
    ):
        self.remove_images = remove_images
        random.seed(seed)  # Set random seed for reproducible question selection

        dataset_dir = os.path.expanduser(dataset_path_or_name)

        if not os.path.exists(dataset_dir):
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

        try:
            # Load the initial dataset
            dataset = load_from_disk(dataset_dir)
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset from {dataset_dir}: {e}")

        # Filter images or captions based on keep_images flag
        if remove_images:
            dataset = dataset.remove_columns("image")
        else:
            dataset = dataset.remove_columns("captions")

        # Apply downsampling if requested and if dataset is larger than requested size
        if downsample_size and downsample_size < len(dataset):
            print(
                f"Downsampling dataset from {len(dataset)} to {downsample_size} samples"
            )
            dataset = dataset.shuffle(seed=seed).select(range(downsample_size))

        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        full_question_prompt = ""

        question_data = self.dataset[idx]
        questions_and_answers = question_data["questions_answers"]

        # Add context if not using images
        if self.remove_images:
            captions = question_data.get("captions")
            if captions:
                # Choose the longest caption as context
                context = max(captions, key=len)
                full_question_prompt += f"{context}\n"

        full_question_prompt += "Answer the question using a single word or phrase.\n"

        # Choose one question-answer pair at random
        question_answer = random.choice(questions_and_answers)
        full_question_prompt += f"{question_answer['question']}\n"

        sample = {
            "question": full_question_prompt,
            "answer_letter": question_answer["answer"],
        }

        if not self.remove_images:
            sample["image"] = question_data.get("image", None)

        return sample


def get_dataloader(
    dataset_path_or_name: str,
    cache_dir: Optional[str] = None,
    split: Optional[str] = None,
    texts_qa: Optional[bool] = None,
    images_qa: Optional[bool] = None,
    question_instruction_type: Optional[str] = None,
    batch_size: int = 1,
    num_workers: int = 4,
    shuffle: bool = False,
    remove_images: Optional[bool] = None,
    downsample_size: Optional[int] = None,
    seed: Optional[int] = None,
):
    """
    Get a DataLoader for either multiple choice benchmarks or open-ended VQA datasets.

    Args:
        dataset_path_or_name: Path to dataset or HuggingFace dataset name
        cache_dir: Cache directory for HuggingFace datasets (required for benchmark datasets)
        split: Dataset split to use
        texts_qa: Whether to use text-only samples (for multimodal datasets)
        images_qa: Whether to use image samples (for multimodal datasets)
        question_instruction_type: Instruction type for multiple choice questions
        batch_size: Batch size for DataLoader
        num_workers: Number of workers for DataLoader
        shuffle: Whether to shuffle the data
        remove_images: Whether to remove images in samples (for OpenVQA datasets)
        downsample_size: Maximum number of samples to use
        seed: Random seed for reproducibility
    """

    # Determine dataset type based on name/path
    is_multiple_choice_benchmark = any(
        benchmark in dataset_path_or_name.lower()
        for benchmark in MULTIPLE_CHOICE_BENCHMARKS
    )

    if is_multiple_choice_benchmark:
        if cache_dir is None:
            raise ValueError("cache_dir is required for benchmark datasets")
        dataset = MultipleChoiceDatasetBenchmark(
            dataset_path_or_name=dataset_path_or_name,
            cache_dir=cache_dir,
            split=split,
            texts_qa=texts_qa,
            images_qa=images_qa,
            question_instruction_type=question_instruction_type,
            downsample_size=downsample_size,
            seed=seed,
        )
    else:
        # Use OpenVQADataset for local datasets
        dataset = OpenVQADataset(
            dataset_path_or_name=dataset_path_or_name,
            downsample_size=downsample_size,
            seed=seed,
            remove_images=remove_images,
        )

    def collate_fn(batch):
        # Custom collate function to handle images and text
        questions = [item["question"] for item in batch]
        answer_letters = [item["answer_letter"] for item in batch]
        images = [item.get("image", None) for item in batch]

        return {
            "questions": questions,
            "answer_letters": answer_letters,
            "images": images,
        }

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=collate_fn,
        pin_memory=True,
    )
