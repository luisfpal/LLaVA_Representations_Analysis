import os
import json
import torch
import argparse
from tqdm import tqdm
from rich import print
from typing import List, Dict, Any
from datasets import load_from_disk
from utils import (
    seed_all,
    load_hf_model_and_processor_or_tokenizer,
    generate_filename_suffix,
    replace_multimodal_lm,
    replace_multimodal_projector,
)


def prepare_results_directory(args: argparse.Namespace) -> str:
    """
    Prepare the directory for saving captioning results.
    
    Args:
        args (argparse.Namespace): The parsed command-line arguments.
        
    Returns:
        str: Path to the results directory.
    """
    dataset_name = "cocoqa_captioning_restval"
    
    args.base_dir = os.path.expanduser(args.base_dir)
    results_dir = os.path.join(
        args.base_dir,
        args.model_name_or_path.replace("/", "_"),
        dataset_name,
        "captioning_results",
    )
    
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def create_captioning_prompt(image_description: str = "") -> str:
    """
    Create an optimized prompt for image captioning.
    
    The prompt is designed to:
    1. Encourage the model to look carefully at the image
    2. Generate short, descriptive captions similar to COCO style
    3. Focus on key visual elements
    
    Args:
        image_description (str): Optional additional context about the image.
        
    Returns:
        str: The captioning prompt.
    """
    base_prompt = (
        "Please look carefully at this image and provide a short, descriptive caption. "
        "Focus on the main objects, actions, and scene elements. "
        "Keep the caption concise but informative, similar to how you would describe "
        "the image to someone who cannot see it. "
        "Be specific about what you see without being overly detailed."
    )
    
    if image_description:
        base_prompt += f"\n\nAdditional context: {image_description}"
    
    return base_prompt


def process_captioning_batches(dataloader, model, processor, captions_file, args):
    """
    Process each batch in the dataloader and generate captions.
    
    Args:
        dataloader: The dataloader for the dataset.
        model: The loaded LLaVA model.
        processor: The loaded processor.
        captions_file (str): Path to the captions output file.
        args: Parsed command-line arguments.
    """
    num_samples = len(dataloader.dataset)
    progress_bar = tqdm(total=num_samples, desc="Generating captions", unit="sample")
    counter = 0
    update_every = max(1, int(0.05 * num_samples))
    
    # Check if chat template exists
    chat_template_exists = hasattr(processor, "chat_template") and processor.chat_template is not None
    
    print(f"\nChat template exists: {chat_template_exists}")
    
    with open(captions_file, "w") as captions_file_handle:
        # Process each batch
        for batch in dataloader:
            images = batch["images"]
            image_ids = batch["image_ids"]
            
            # Create prompts for captioning
            prompts = []
            for i in range(len(images)):
                prompt = create_captioning_prompt()
                prompts.append(prompt)
            
            # Format prompts for the model
            if chat_template_exists:
                # Use chat template for structured conversation
                formatted_prompts = []
                for prompt in prompts:
                    messages = [
                        {"role": "user", "content": [{"type": "text", "text": prompt}]}
                    ]
                    formatted_prompt = processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    formatted_prompts.append(formatted_prompt)
            else:
                # Simple text prompts
                formatted_prompts = prompts
            
            # Process images and text
            processor_kwargs = {
                "text": formatted_prompts,
                "padding": True if args.batch_size > 1 else False,
                "return_tensors": "pt",
                "images": images,
            }
            
            model_inputs = processor(**processor_kwargs).to(model.device)
            
            # Get the length of the input_ids to separate prompt from generation
            input_ids_length = model_inputs.input_ids.shape[1]
            
            with torch.no_grad():
                kwargs_for_generate = {
                    **model_inputs,
                    "max_new_tokens": args.max_new_tokens,
                    "do_sample": args.do_sample,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "num_beams": args.num_beams,
                }
                
                output_ids_tensor = model.generate(**kwargs_for_generate)
            
            # Slice to get only the generated tokens (excluding the prompt)
            generated_ids_only_tensor = output_ids_tensor[:, input_ids_length:]
            decoded_outputs_list = processor.batch_decode(
                generated_ids_only_tensor,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            
            # Clean up the generated captions
            final_captions = [
                caption.strip() for caption in decoded_outputs_list
            ]
            
            # Write the captions to the output file
            for image_id, caption, original_prompt in zip(
                image_ids, final_captions, prompts
            ):
                captions_file_handle.write(
                    json.dumps(
                        {
                            "image_id": image_id,
                            "generated_caption": caption,
                            "prompt": original_prompt,
                        }
                    )
                    + "\n"
                )
                captions_file_handle.flush()
            
            counter += len(image_ids)
            if counter % update_every == 0:
                progress_bar.update(update_every)
            
            # Clean up memory
            del (
                images,
                image_ids,
                prompts,
                formatted_prompts,
                processor_kwargs,
                kwargs_for_generate,
            )
            del model_inputs, output_ids_tensor, generated_ids_only_tensor
            del decoded_outputs_list, final_captions
            torch.cuda.empty_cache()
    
    progress_bar.close()


def create_captioning_dataloader(dataset_path: str, batch_size: int = 1):
    """
    Create a dataloader for captioning from the COCO-QA dataset.
    
    Args:
        dataset_path (str): Path to the dataset.
        batch_size (int): Batch size for processing.
        
    Returns:
        DataLoader: The dataloader for captioning.
    """
    # Load the dataset
    dataset = load_from_disk(dataset_path)
    
    # Create a simple dataset class for captioning
    class CaptioningDataset:
        def __init__(self, dataset):
            self.dataset = dataset
        
        def __len__(self):
            return len(self.dataset)
        
        def __getitem__(self, idx):
            item = self.dataset[idx]
            return {
                "image": item["image"],
                "image_id": item["image_id"],
            }
    
    captioning_dataset = CaptioningDataset(dataset)
    
    # Create dataloader
    from torch.utils.data import DataLoader
    
    def collate_fn(batch):
        images = [item["image"] for item in batch]
        image_ids = [item["image_id"] for item in batch]
        return {
            "images": images,
            "image_ids": image_ids,
        }
    
    dataloader = DataLoader(
        captioning_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    
    return dataloader


def caption_model(args: argparse.Namespace):
    """
    Generate captions for COCO-QA images using LLaVA.
    
    Args:
        args (argparse.Namespace): The parsed command-line arguments.
    """
    # Generate filename suffix and prepare results directory
    filenames_suffix = generate_filename_suffix(args)
    results_dir = prepare_results_directory(args)
    captions_file = os.path.join(
        results_dir,
        f"captions{filenames_suffix}.jsonl",
    )
    
    try:
        # Load the model and processor
        print("Loading LLaVA model and processor...")
        model, processor = load_hf_model_and_processor_or_tokenizer(
            model_name_or_path=args.model_name_or_path,
            cache_dir=args.model_cache_dir,
            device_map="cuda:0",
            dtype=torch.float16,
            text_model=False,  # Always False for captioning
        )
        
        # Set the model to evaluation mode
        model.eval()
        
        # Replace multimodal projector (if specified)
        if args.pretrained_projector_name_or_path is not None:
            model = replace_multimodal_projector(
                multimodal_model=model,
                pretrained_projector_model_name_or_path=args.pretrained_projector_name_or_path,
                cache_dir=args.model_cache_dir,
            )
        
        # Replace multimodal language model (if specified)
        if args.replacement_lm_name_or_path is not None:
            model = replace_multimodal_lm(
                multimodal_model=model,
                replacement_lm_name_or_path=args.replacement_lm_name_or_path,
                cache_dir=args.model_cache_dir,
            )
        
        # Prepare the dataloader
        print("Preparing captioning dataloader...")
        dataloader = create_captioning_dataloader(
            dataset_path=args.dataset_path,
            batch_size=args.batch_size,
        )
        
        # Add padding token if processing a batch
        if args.batch_size > 1:
            processor.tokenizer.padding_side = "left"
            if processor.tokenizer.pad_token is None:
                processor.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                model.language_model.resize_token_embeddings(len(processor.tokenizer))
        
        # Process each batch and generate captions
        process_captioning_batches(dataloader, model, processor, captions_file, args)
        
    except Exception as e:
        print(f"Error during captioning with {args.model_name_or_path}: {str(e)}")
        raise
    
    print(f"\nCaptioning complete! Results saved to: {captions_file}")


def main():
    """
    Main function to parse arguments and initiate captioning.
    """
    parser = argparse.ArgumentParser(
        description="Generate captions for COCO-QA images using LLaVA."
    )
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--model-cache-dir", type=str, default=None)
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--base-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--do-sample", action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replacement-lm-name-or-path", type=str, default=None)
    parser.add_argument("--pretrained-projector-name-or-path", type=str, default=None)
    args = parser.parse_args()
    
    # Set the random seed for reproducibility
    seed_all(args.seed)
    
    # Run the captioning
    caption_model(args)


if __name__ == "__main__":
    main() 