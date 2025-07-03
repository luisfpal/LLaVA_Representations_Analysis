import hashlib
import datasets
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, load_from_disk
import os
import numpy as np
from typing import Dict, Any, Optional, List, Union
from .model_utils import ProcessorType
from .constants import (
    SQA_ANSWER_CHOICES,
    MMLU_ANSWER_CHOICES,
    SYSTEM_ROLE,
    ASSISTANT_ROLE,
    ANSWER_TEXT,
)
from PIL import Image
import argparse


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
        images_qa: Whether to remove images from the dataset
        texts_qa: Whether to remove captions from the dataset
    """

    def __init__(
        self,
        dataset_path_or_name: str,
        downsample_size: Optional[int] = None,
        seed: int = 42,
        images_qa: bool = False,
        texts_qa: bool = False,
    ):
        if images_qa and texts_qa:
            raise ValueError("images_qa and texts_qa cannot be both True.")

        self.images_qa = images_qa
        self.texts_qa = texts_qa
        # Random number generator for reproducibility and stable one-to-one
        # question-answer pair selection between images_qa and texts_qa datasets
        self.rng = np.random.RandomState(seed)

        dataset_dir = os.path.expanduser(dataset_path_or_name)

        if not os.path.exists(dataset_dir):
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

        try:
            # Load the initial dataset
            dataset = load_from_disk(dataset_dir)
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset from {dataset_dir}: {e}")

        # Add an idx column to the dataset
        dataset = dataset.add_column("idx", list(range(len(dataset))))

        # Apply downsampling if requested and if dataset is larger than requested size
        if downsample_size and downsample_size < len(dataset):
            print(
                f"Downsampling dataset from {len(dataset)} to {downsample_size} samples"
            )
            dataset = dataset.shuffle(seed=seed).select(range(downsample_size))

        # Filter images or captions based on keep_images flag
        if images_qa:
            dataset = dataset.remove_columns("captions")
        elif texts_qa:
            dataset = dataset.remove_columns("image")

        # Sort the dataset by idx to guarantee images_qa and texts_qa are in the same order
        dataset = dataset.sort("idx")

        self.dataset = dataset
        self._seen_hashes = set()

    def _hash_qa(self, qa_dict):
        """
        Create a SHA-256 hash for a question-answer dict.
        Normalizes by stripping whitespace and converting to lowercase.
        Returns the hex digest string.
        """
        question = qa_dict["question"].strip().lower()
        answer = qa_dict["answer"].strip().lower()
        combined = f"{question}|||{answer}"
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        full_question_prompt = ""
        question_data = self.dataset[idx]
        questions_and_answers = question_data["questions_and_answers"]

        # Find a unique question-answer pair using shuffled indices
        unique_qa_dict = None
        shuffled_indices = list(range(len(questions_and_answers)))
        self.rng.shuffle(shuffled_indices)
        for idx in shuffled_indices:
            qa_dict = questions_and_answers[idx]
            hash_qa = self._hash_qa(qa_dict)
            if hash_qa not in self._seen_hashes:
                self._seen_hashes.add(hash_qa)
                unique_qa_dict = qa_dict
                break
        # If no unique question-answer pair found, use the first one from shuffled list
        if unique_qa_dict is None:
            unique_qa_dict = questions_and_answers[shuffled_indices[0]]
            # Despite the effort to make sure that the same (question, answer) pair is not repeated
            # !Note: multiple images can have the shared (question, answer) pairs
            # e.g., (what is the color of the car?, red) can be found in different images

        if self.texts_qa:
            captions = question_data.get("captions")
            if captions:
                # Concatenate all the captions
                full_question_prompt += f"Below are {len(captions)} descriptions of an image. Use them to answer the question.\n\n"
                for idx, caption in zip(range(len(captions)), captions):
                    full_question_prompt += f"{idx + 1}: {caption}\n"
            else:
                raise ValueError("No captions found in the dataset")

        full_question_prompt += f"\n{unique_qa_dict['question']}?\n"

        sample = {
            "idx": question_data.get("idx", None),
            "question": full_question_prompt,
            "answer_letter": unique_qa_dict["answer"],
            "image_id": question_data.get("image_id", None),
            # `answer_letter` is actually a word but it's called letter for convenience
            # and alignment with the multiple choice benchmarks
        }

        if self.images_qa:
            sample["image"] = question_data.get("image", None)

        return sample


def format_prompts(
    questions: List[str],
    images: Optional[List[Image.Image]],
    args: argparse.Namespace,
    processor: ProcessorType,
    chat_template_exists: bool = False,
) -> List[str]:
    """
    Format input questions and images into structured prompts for a model.

    Parameters:
        questions: List of question and options in string format.
        images: List of images associated with each question.
        args: Arguments namespace with a `chat_mode` boolean attribute.
        processor: A processor used to tokenize and encode text+image inputs.
        chat_template_exists: Whether the chat template exists.

    Returns:
        list: Formatted prompts either as conversations (dict format) or plain strings.
    """

    if args.chat_mode and chat_template_exists:
        conversations_list = _format_as_conversations(questions, images, args)
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
        return _format_as_plain_prompts(questions, images)


def _format_as_conversations(questions, images, args):
    formatted_conversations = []
    continue_final_message = (
        hasattr(args, "continue_final_message") and args.continue_final_message
    )
    for question_text, image in zip(questions, images):
        content = [{"type": "text", "text": f"{question_text}{args.guide_text}"}]

        if image is not None:
            content.append({"type": "image"})

        conversation = [{"role": "user", "content": content}]
        conversation.insert(0, SYSTEM_ROLE)
        if continue_final_message:
            conversation.append(ASSISTANT_ROLE)
        formatted_conversations.append(conversation)
    return formatted_conversations


def _format_as_plain_prompts(questions, images):
    formatted_prompts = []
    for image, question_text in zip(images, questions):
        prompt_prefix = "<image>\n" if image is not None else ""
        prompt = f"{prompt_prefix}{question_text}{ANSWER_TEXT}"
        formatted_prompts.append(prompt)
    return formatted_prompts


# todo: check if this should be removed
# its functionality was absorbed in the get_dataloader function
def preprocess_batch(
    batch: Union[
        List[str],
        Dict[str, Union[List[str], List[Optional[Image.Image]]]],
    ],
    processor: ProcessorType,
    guide_text: str = "",
):
    try:
        if isinstance(batch, list):
            return batch, len(batch)
        elif isinstance(batch, dict):
            questions = batch["questions"]
            images = batch.get("images", None)

            # !For simplicity, define args for backward compatibility with previous code
            # !todo: redesign the interface if used in the future
            args = argparse.Namespace(
                chat_mode=True,
                continue_final_message=True,
                guide_text=guide_text,
            )  # !"enforced" but consistent for these experiments

            full_prompts = format_prompts(
                questions=questions,
                images=images,
                args=args,
                processor=processor,
                chat_template_exists=True,
                # !"enforced" but consistent for these experiments
            )
            return full_prompts, len(questions)
    except Exception as e:
        raise ValueError(f"Invalid batch format: {e}")


def get_dataloader(
    dataset_path_or_name: str,
    cache_dir: Optional[str] = None,
    split: Optional[str] = None,
    texts_qa: Optional[bool] = None,
    images_qa: Optional[bool] = None,
    question_instruction_type: Optional[str] = None,
    batch_size: int = 1,
    num_workers: int = 4,
    downsample_size: Optional[int] = None,
    seed: Optional[int] = None,
    processor: Optional[ProcessorType] = None,
    guide_text: str = "Answer the question using a single word or phrase.\n",
    answer_letters_with_processed_batch: bool = False,
):
    """
    Get a DataLoader for either multiple choice benchmarks or open-ended VQA datasets.
    For convenience, if a processor is provided, the dataset will be preprocessed and returned as
    model-ready tensors for residual stream tracing, remaining to move to the model device.
    If answer_letters_with_processed_batch is True, in addition to the processed batch, the raw answer letters
    will be returned as a list of strings in a tuple with the processed batch.
    Otherwise, the dataset will be returned as a dictionary with "questions", "answer_letters",
    and "images" keys.

    Args:
        dataset_path_or_name: Path to dataset or HuggingFace dataset name.
        cache_dir: Cache directory for HuggingFace datasets (required for benchmark datasets).
        split: Dataset split to use.
        texts_qa: Whether to use text-only samples (for multimodal datasets).
        images_qa: Whether to use image samples (for multimodal datasets).
        question_instruction_type: Instruction type for multiple choice questions.
        batch_size: Batch size for DataLoader.
        num_workers: Number of workers for DataLoader.
        downsample_size: Maximum number of samples to use.
        seed: Random seed for reproducibility.
        processor: A processor used to tokenize and encode text+image inputs.
        guide_text: Optional text to append to questions during prompt formatting.
        answer_letters_with_processed_batch: Whether to include answer letters in the processed batch,
            it just applies when a processor is provided.
    """

    print("\nLoading dataloader...")

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
        dataset = OpenVQADataset(
            dataset_path_or_name=dataset_path_or_name,
            downsample_size=downsample_size,
            seed=seed,
            images_qa=images_qa,
            texts_qa=texts_qa,
        )

    def collate_fn(batch):
        # Custom collate function to handle images and text
        questions = [item["question"] for item in batch]
        answer_letters = [item["answer_letter"] for item in batch]
        images = [item.get("image", None) for item in batch]
        indices = [item.get("indices", None) for item in batch]
        image_ids = [item.get("image_id", None) for item in batch]

        if processor is not None:
            # For simplicity, define args for backward code compatibility
            # todo: redesign the interface if used in the future
            args = argparse.Namespace(
                chat_mode=True,
                continue_final_message=True,
                guide_text=guide_text,
                # !"enforced" but consistent for these experiments
            )

            prompts = format_prompts(
                questions=questions,
                images=images,
                args=args,
                processor=processor,
                chat_template_exists=True,
                # !"enforced" but consistent for these experiments
            )

            processor_kwargs = {
                "text": prompts,
                "return_tensors": "pt",
                "padding": (True if len(prompts) > 1 else False),
            }

            if any(image is not None for image in images):
                processor_kwargs["images"] = images

            tokenized = processor(**processor_kwargs)
            if answer_letters_with_processed_batch:
                # !not so neat but it's a backward compatibility feature
                return tokenized, answer_letters
            return tokenized

        # Raw mode (default)
        return {
            "questions": questions,
            "answer_letters": answer_letters,
            "images": images,
            "indices": indices,
            "image_ids": image_ids,
        }

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
    )
