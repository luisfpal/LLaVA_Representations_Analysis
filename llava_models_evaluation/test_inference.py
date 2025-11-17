import os
import torch
from pathlib import Path
from utils import load_hf_model_and_processor_or_tokenizer
from utils.constants import SYSTEM_ROLE, ASSISTANT_ROLE
from PIL import Image

# Configuration
MODEL_NAME = "lbasile/llava-1000"
MODEL_CACHE_DIR = os.path.expanduser("~/scratch/huggingface/hub")
IMAGE_PATH = Path(__file__).parent / "000000039769.jpg"

# Load model and processor
model, processor = load_hf_model_and_processor_or_tokenizer(
    MODEL_NAME,
    MODEL_CACHE_DIR,
)

# Fix missing patch_size in processor
if hasattr(processor, 'image_processor') and hasattr(processor.image_processor, 'patch_size'):
    processor.patch_size = processor.image_processor.patch_size
elif hasattr(processor, 'image_processor') and hasattr(processor.image_processor, 'size'):
    processor.patch_size = processor.image_processor.size.get('patch_size', 14)

# Prepare conversation
user_role = {
    "role": "user",
    "content": [
        {"type": "text", "text": "What are these?"},
        {"type": "image"},
    ],
}
conversation = [SYSTEM_ROLE, user_role, ASSISTANT_ROLE]
prompt = processor.apply_chat_template(conversation, continue_final_message=True, add_generation_prompt=False)

# Load image and run inference
raw_image = Image.open(IMAGE_PATH)
inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(model.device, torch.float16)
output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
print(processor.decode(output[0], skip_special_tokens=True))

