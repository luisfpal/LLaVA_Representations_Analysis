import os
import json
import torch
import argparse
from tqdm import tqdm
from rich import print
from utils import (
    seed_all,
    load_hf_model_and_processor_or_tokenizer,
    generate_filename_suffix,
    replace_multimodal_lm,
    replace_multimodal_projector,
    parse_question_instruction,
    format_prompts,
    get_dataloader,
    COCOQA_VI_DIGITS_MAP,
)


def prepare_results_directory(args: argparse.Namespace) -> str:
    """
    Prepare the directory for saving results and return the answers file path.

    Args:
        args (argparse.Namespace): The parsed command-line arguments.
        suffix (str): The filename suffix.

    Returns:
        str: Path to the answers file.
    """

    dataset_name = None
    if "datasets" in args.dataset_name:
        dataset_name = args.dataset_name.split("/")[-1]
    else:
        dataset_name = args.dataset_name.replace("/", "_")

    args.base_dir = os.path.expanduser(args.base_dir)
    results_dir = os.path.join(
        args.base_dir,
        args.model_name_or_path.replace("/", "_"),
        dataset_name,
        args.split,
    )

    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def process_predictions(predictions_file: str) -> dict:
    """
    Process predictions to calculate accuracy and categorize results.

    Args:
        predictions_file (str): Path to the file containing predictions.

    Returns:
        dict: Evaluation results including accuracy and categorized results.
    """
    with open(predictions_file, "r") as pred_file:
        predictions_list = [json.loads(line) for line in pred_file]

    evaluation_results = {
        "accuracy": None,
        "correct": None,
        "count": None,
        "correct_results": [],
        "incorrect_results": [],
    }

    for problem_data in tqdm(predictions_list, desc="Scoring predictions"):
        predicted_text = problem_data["output"]
        answer = problem_data["answer_letter"]

        # Parse the predicted answer
        parsed_answer_char = parse_predicted_answer(predicted_text, answer)

        analysis_entry = {
            "prompt": problem_data["prompt"],
            "raw_prediction": predicted_text,
            "parsed_answer": parsed_answer_char,
            "ground_truth": answer,
        }

        if parsed_answer_char == answer:
            evaluation_results["correct_results"].append(analysis_entry)
        else:
            evaluation_results["incorrect_results"].append(analysis_entry)

    # Calculate overall accuracy
    num_correct = len(evaluation_results["correct_results"])
    num_incorrect = len(evaluation_results["incorrect_results"])
    total_questions = num_correct + num_incorrect

    accuracy = num_correct / total_questions if total_questions > 0 else 0
    evaluation_results["accuracy"] = accuracy
    evaluation_results["correct"] = num_correct
    evaluation_results["count"] = total_questions

    print("Evaluation results:")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Correct: {num_correct}")
    print(f"Count: {total_questions}")

    return evaluation_results


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


def save_evaluation_results(results: dict, results_file: str):
    """
    Save evaluation results to a JSON file.

    Args:
        results (dict): The evaluation results.
        results_file (str): Path to the results file.
    """
    with open(results_file, "w") as f_results:
        json.dump(results, f_results, indent=4)
        f_results.write("\n")  # Add a newline for POSIX compatibility


def eval_model(args: argparse.Namespace):
    """
    Evaluate the model on the dataset and save results.

    Args:
        args (argparse.Namespace): The parsed command-line arguments.
    """
    # Generate filename suffix and prepare results directory
    filenames_suffix = generate_filename_suffix(args)
    results_dir = prepare_results_directory(args)
    answers_file = os.path.join(
        results_dir,
        f"answers{filenames_suffix}.jsonl",
    )

    try:
        # Load the model and processor
        print("Loading model and processor...")
        model, processor = load_hf_model_and_processor_or_tokenizer(
            model_name_or_path=args.model_name_or_path,
            cache_dir=args.model_cache_dir,
            device_map="cuda:0",
            dtype=torch.float16,
            text_model=args.text_model,
        )

        # Set the model to evaluation mode
        model.eval()

        # Replace multimodal projector (if specified and not text model)
        if (
            hasattr(args, "pretrained_projector_name_or_path")
            and args.pretrained_projector_name_or_path is not None
            and not args.text_model
        ):
            model = replace_multimodal_projector(
                multimodal_model=model,
                pretrained_projector_model_name_or_path=args.pretrained_projector_name_or_path,
                cache_dir=args.model_cache_dir,
            )

        # Replace multimodal language model
        if args.replacement_lm_name_or_path is not None:
            model = replace_multimodal_lm(
                multimodal_model=model,
                replacement_lm_name_or_path=args.replacement_lm_name_or_path,
                cache_dir=args.model_cache_dir,
            )

        # Prepare the dataloader
        print("Preparing dataloader...")
        dataloader = get_dataloader(
            dataset_path_or_name=args.dataset_name,
            cache_dir=args.dataset_cache_dir,
            split=args.split,
            texts_qa=args.texts_qa,
            images_qa=args.images_qa,
            batch_size=args.batch_size,
            question_instruction_type=args.question_instruction_type,
            downsample_size=args.downsample_size,
            seed=args.seed,
        )

        # Add padding token if processing a batch
        if args.batch_size > 1:
            if args.text_model:
                processor.padding_side = "left"
                if processor.pad_token is None:
                    processor.add_special_tokens({"pad_token": "[PAD]"})
                    model.resize_token_embeddings(len(processor))
            else:
                processor.tokenizer.padding_side = "left"
                if processor.tokenizer.pad_token is None:
                    processor.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                    model.language_model.resize_token_embeddings(
                        len(processor.tokenizer)
                    )

        # Process each batch and save predictions
        process_batches(dataloader, model, processor, answers_file, args)

    except Exception as e:
        print(f"Error during {args.model_name_or_path} evaluation: {str(e)}")
        raise

    # Score the predictions and save results
    evaluation_results = process_predictions(answers_file)
    results_file = os.path.join(results_dir, f"results{filenames_suffix}.json")
    save_evaluation_results(evaluation_results, results_file)


def process_batches(dataloader, model, processor, answers_file, args):
    """
    Process each batch in the dataloader and save predictions to a file.

    Args:
        dataloader: The dataloader for the dataset.
        model: The loaded model.
        processor: The loaded processor.
        answers_file (str): Path to the answers file.
        args: Parsed command-line arguments.
    """
    num_samples = len(dataloader.dataset)
    progress_bar = tqdm(total=num_samples, desc="Processing samples", unit="sample")
    counter = 0
    update_every = max(1, int(0.05 * num_samples))

    # Check if not empty chat_template exists
    chat_template_exists = False
    if args.text_model and hasattr(processor, "chat_template"):
        chat_template_exists = bool(processor.chat_template)
    elif not args.text_model:
        chat_template_exists = True

    print(f"\nChat template exists: {chat_template_exists}")

    with open(answers_file, "w") as ans_file_handle:
        # Process each batch
        for batch in dataloader:
            questions_and_options = batch["questions"]
            images = batch.get("images", None)
            answer_letters = batch["answer_letters"]

            # Process each question in the batch
            full_prompts = format_prompts(
                questions_and_options=questions_and_options,
                images=images,
                args=args,
                processor=processor,
                chat_template_exists=chat_template_exists,
            )

            processor_kwargs = {
                "text": full_prompts,
                "padding": (True if args.batch_size > 1 else False),
                "return_tensors": "pt",
            }

            if args.text_model or all(image is None for image in images):
                model_inputs = processor(
                    **processor_kwargs,
                ).to(model.device)
            else:
                model_inputs = processor(
                    **processor_kwargs,
                    images=images,
                ).to(model.device)

            # Get the length of the input_ids to separate prompt from generation
            input_ids_length = model_inputs.input_ids.shape[1]

            with torch.no_grad():
                kwargs_for_generate = {
                    **model_inputs,
                    "max_new_tokens": args.max_new_tokens,
                    "do_sample": False,
                    # do_sample doesn't influence much for multiple choice benchmarks
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
            final_outputs = [
                decoded_output.strip() for decoded_output in decoded_outputs_list
            ]

            # Write the prediction to the answers file
            for prompt, output, answer_letter in zip(
                full_prompts,
                final_outputs,
                answer_letters,
            ):
                ans_file_handle.write(
                    json.dumps(
                        {
                            "prompt": prompt,
                            "output": output,
                            "answer_letter": answer_letter,
                        }
                    )
                    + "\n"
                )
                ans_file_handle.flush()  # Ensure data is written to disk periodically

            counter += len(answer_letters)
            if counter % update_every == 0:
                progress_bar.update(update_every)

            del (
                questions_and_options,
                images,
                answer_letters,
                full_prompts,
                processor_kwargs,
                kwargs_for_generate,
            )
            del model_inputs, output_ids_tensor, generated_ids_only_tensor
            del decoded_outputs_list, final_outputs
            torch.cuda.empty_cache()

    # Close the progress bar
    progress_bar.close()


def main():
    """
    Main function to parse arguments and initiate a LLM or LMM model evaluation on a benchmark.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a LLM or LMM model on a multiple choice benchmark."
    )
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--model-cache-dir", type=str, default=None)
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--dataset-cache-dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--base-dir", type=str, required=True)
    parser.add_argument("--text-model", action="store_true", default=False)
    parser.add_argument("--texts_qa", action="store_true", default=False)
    parser.add_argument("--images_qa", action="store_true", default=False)
    parser.add_argument("--chat-mode", action="store_true", default=False)
    parser.add_argument(
        "--question-instruction-type", type=parse_question_instruction, default=None
    )
    parser.add_argument("--continue-final-message", action="store_true", default=False)
    parser.add_argument("--guide-text", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--downsample-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replacement-lm-name-or-path", type=str, default=None)
    parser.add_argument("--pretrained-projector-name-or-path", type=str, default=None)
    args = parser.parse_args()

    # Set the random seed for reproducibility
    seed_all(args.seed)

    # Run the evaluation
    eval_model(args)


if __name__ == "__main__":
    main()
