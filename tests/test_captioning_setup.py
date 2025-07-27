#!/usr/bin/env python3
"""
Test script to verify the captioning and evaluation setup.
"""

import os
import json
import tempfile
from datasets import Dataset
from PIL import Image
import numpy as np

def create_test_dataset():
    """Create a small test dataset for captioning."""
    # Create a simple test image (random noise)
    test_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    
    # Create test data
    test_data = [
        {
            "image_id": "test_001",
            "image": test_image,
            "captions": [
                "A colorful abstract image with random patterns",
                "An image containing various colored pixels arranged randomly",
                "A digital artwork with vibrant colors and geometric shapes"
            ],
            "questions_answers": [
                {"question": "What colors are in this image?", "answer": "various"}
            ]
        },
        {
            "image_id": "test_002", 
            "image": test_image,
            "captions": [
                "A bright and colorful digital image",
                "An abstract composition with many colors",
                "A vibrant picture with random color distribution"
            ],
            "questions_answers": [
                {"question": "Is this image colorful?", "answer": "yes"}
            ]
        }
    ]
    
    return Dataset.from_list(test_data)

def test_captioning_prompt():
    """Test the captioning prompt function."""
    from scripts.caption_cocoqa_with_llava import create_captioning_prompt
    
    # Test basic prompt
    prompt = create_captioning_prompt()
    print("✓ Basic captioning prompt created")
    print(f"Prompt length: {len(prompt)} characters")
    
    # Test prompt with additional context
    prompt_with_context = create_captioning_prompt("This is a test image")
    print("✓ Captioning prompt with context created")
    
    return True

def test_evaluation_metrics():
    """Test the evaluation metrics functions."""
    from scripts.evaluate_captions import (
        preprocess_text, 
        compute_bleu_score, 
        compute_rouge_score,
        compute_cider_score
    )
    
    # Test text preprocessing
    test_text = "A colorful image with many bright colors!"
    tokens = preprocess_text(test_text)
    print(f"✓ Text preprocessing: '{test_text}' -> {tokens}")
    
    # Test BLEU score
    references = [["a", "colorful", "image"], ["bright", "colors", "present"]]
    candidate = ["a", "colorful", "image", "with", "colors"]
    bleu_score = compute_bleu_score(references, candidate)
    print(f"✓ BLEU score computed: {bleu_score:.4f}")
    
    # Test ROUGE score
    rouge_score = compute_rouge_score(references, candidate, n=1)
    print(f"✓ ROUGE-1 score computed: {rouge_score:.4f}")
    
    # Test CIDEr score
    cider_score = compute_cider_score(references, candidate)
    print(f"✓ CIDEr score computed: {cider_score:.4f}")
    
    return True

def test_dataset_loading():
    """Test dataset loading functionality."""
    # Create temporary test dataset
    test_dataset = create_test_dataset()
    
    # Save to temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dataset.save_to_disk(temp_dir)
        print(f"✓ Test dataset saved to: {temp_dir}")
        
        # Test loading
        loaded_dataset = Dataset.load_from_disk(temp_dir)
        print(f"✓ Test dataset loaded: {len(loaded_dataset)} samples")
        
        # Test data structure
        sample = loaded_dataset[0]
        required_keys = ["image_id", "image", "captions", "questions_answers"]
        for key in required_keys:
            assert key in sample, f"Missing key: {key}"
        print("✓ Dataset structure verified")
    
    return True

def test_script_imports():
    """Test that all required scripts can be imported."""
    try:
        from scripts.caption_cocoqa_with_llava import (
            create_captioning_prompt,
            prepare_results_directory
        )
        print("✓ Captioning script imports successful")
    except ImportError as e:
        print(f"✗ Captioning script import failed: {e}")
        return False
    
    try:
        from scripts.evaluate_captions import (
            preprocess_text,
            compute_bleu_score,
            evaluate_captions
        )
        print("✓ Evaluation script imports successful")
    except ImportError as e:
        print(f"✗ Evaluation script import failed: {e}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("=" * 50)
    print("CAPTIONING AND EVALUATION SETUP TEST")
    print("=" * 50)
    
    tests = [
        ("Script Imports", test_script_imports),
        ("Captioning Prompt", test_captioning_prompt),
        ("Evaluation Metrics", test_evaluation_metrics),
        ("Dataset Loading", test_dataset_loading),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- Testing {test_name} ---")
        try:
            if test_func():
                print(f"✓ {test_name} PASSED")
                passed += 1
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} FAILED with error: {e}")
    
    print("\n" + "=" * 50)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("=" * 50)
    
    if passed == total:
        print("🎉 All tests passed! Setup is ready for captioning and evaluation.")
    else:
        print("⚠️  Some tests failed. Please check the setup.")
    
    return passed == total

if __name__ == "__main__":
    main() 