import torch
import gc
from typing import Dict
from torch.utils.data import DataLoader
from tqdm import tqdm
from .model_utils import ModelType, ProcessorType
from .constants import COCOQA_VI_DIGITS_MAP


def process_digits(text: str) -> str:
    """
    Replace digit strings in the text with their word equivalents.
    Handles ambiguity by processing longer digits first.
    """
    # Sort keys in descending order of length to prevent substring replacement issues
    for key in sorted(COCOQA_VI_DIGITS_MAP.keys(), key=len, reverse=True):
        text = text.replace(key, COCOQA_VI_DIGITS_MAP[key])
    return text


def parse_predicted_answer(predicted_text: str, answer: str) -> str:
    """
    Parses the model's predicted answer and checks for a match against the ground truth.

    This function handles two types of evaluation:
    - Multiple-choice (single character answers, e.g., "A")
    - Open-ended (free-form text answers, e.g., "apple")

    It performs case-insensitive and trimmed matching, and handles partial matches for open-ended answers.

    Args:
        predicted_text (str): The raw output from the model.
        answer (str): The correct answer to compare against.

    Returns:
        str: The parsed answer if it matches expectations, otherwise 'FAILED'.
    """
    predicted_text = predicted_text.strip()
    answer = answer.strip()

    # Multiple-choice: expect exact match or contained match (e.g., "Answer: A")
    if len(answer) == 1:
        if predicted_text == answer:
            return answer
        if answer in predicted_text:
            return answer
        return "FAILED"

    # Open-ended: allow substring match (e.g., answer="apple", predicted="a green apple")
    answer = answer.lower()
    predicted_text = process_digits(predicted_text.lower())
    if answer in predicted_text or predicted_text in answer:
        return answer
    return "FAILED"


def benchmark_model_vqa_processed_dataloader(
    model: ModelType,
    dataloader: DataLoader,
    processor: ProcessorType,
    max_new_tokens: int = 1,
) -> Dict[str, float]:
    correct_answers = 0
    incorrect_answers = 0

    dataset_size = len(dataloader)
    progress_bar = tqdm(total=dataset_size, desc="Processing samples", unit="sample")
    update_count = 0
    update_every = max(1, int(0.1 * dataset_size))
    samples_processed = 0

    for batch, answer_letters in dataloader:
        current_batch_size, input_ids_length = batch.input_ids.shape[:2]
        with torch.no_grad():
            kwargs_for_generate = {
                **batch,
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
            }

            output_ids_tensor = model.generate(
                **kwargs_for_generate,
            )
        # Slice to get only the generated tokens (excluding the prompt)
        generated_ids_only_tensor = output_ids_tensor[:, input_ids_length:]
        decoded_outputs_list = processor.batch_decode(
            generated_ids_only_tensor,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        # Decode the outputs and remove leading/trailing whitespace
        predicted_answers = [
            decoded_output.strip() for decoded_output in decoded_outputs_list
        ]

        for predicted_text, answer in zip(predicted_answers, answer_letters):
            parsed_answer = parse_predicted_answer(predicted_text, answer)
            if parsed_answer == answer:
                correct_answers += 1
            else:
                incorrect_answers += 1

        samples_processed += current_batch_size
        update_count += current_batch_size

        if update_count % update_every == 0 or samples_processed == dataset_size:
            progress_bar.update(update_count)
            update_count = 0

    progress_bar.close()
    torch.cuda.empty_cache()
    gc.collect()

    return {
        "num_correct_answers": correct_answers,
        "num_incorrect_answers": incorrect_answers,
        "accuracy": correct_answers / (correct_answers + incorrect_answers),
    }
