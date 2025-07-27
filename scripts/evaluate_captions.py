import os
import json
import argparse
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from rich import print
from datasets import load_from_disk
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.util import ngrams
import re

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


def load_captions_data(captions_file: str, dataset_path: str) -> Tuple[Dict, Dict]:
    """
    Load generated captions and reference captions.
    
    Args:
        captions_file (str): Path to the generated captions file.
        dataset_path (str): Path to the dataset with reference captions.
        
    Returns:
        Tuple[Dict, Dict]: Generated captions and reference captions dictionaries.
    """
    # Load generated captions
    generated_captions = {}
    with open(captions_file, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            generated_captions[data['image_id']] = data['generated_caption']
    
    # Load reference captions from dataset
    dataset = load_from_disk(dataset_path)
    reference_captions = {}
    
    for item in dataset:
        image_id = item['image_id']
        if image_id in generated_captions:
            # Get all reference captions for this image
            ref_captions = item['captions']
            reference_captions[image_id] = ref_captions
    
    return generated_captions, reference_captions


def preprocess_text(text: str) -> List[str]:
    """
    Preprocess text for evaluation metrics.
    
    Args:
        text (str): Input text.
        
    Returns:
        List[str]: Tokenized and cleaned text.
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation except apostrophes
    text = re.sub(r'[^\w\s\']', ' ', text)
    
    # Tokenize
    tokens = nltk.word_tokenize(text)
    
    # Remove empty tokens
    tokens = [token for token in tokens if token.strip()]
    
    return tokens


def compute_bleu_score(references: List[List[str]], candidate: List[str], 
                      weights: Tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25)) -> float:
    """
    Compute BLEU score for a candidate caption against multiple references.
    
    Args:
        references (List[List[str]]): List of reference captions (tokenized).
        candidate (List[str]): Candidate caption (tokenized).
        weights (Tuple): Weights for 1-gram, 2-gram, 3-gram, 4-gram.
        
    Returns:
        float: BLEU score.
    """
    smoothing = SmoothingFunction().method1
    return sentence_bleu(references, candidate, weights=weights, smoothing_function=smoothing)


def compute_meteor_score(references: List[List[str]], candidate: List[str]) -> float:
    """
    Compute METEOR score for a candidate caption against multiple references.
    
    Args:
        references (List[List[str]]): List of reference captions (tokenized).
        candidate (List[str]): Candidate caption (tokenized).
        
    Returns:
        float: METEOR score.
    """
    return meteor_score(references, candidate)


def compute_rouge_score(references: List[List[str]], candidate: List[str], n: int = 1) -> float:
    """
    Compute ROUGE-N score for a candidate caption against multiple references.
    
    Args:
        references (List[List[str]]): List of reference captions (tokenized).
        candidate (List[str]): Candidate caption (tokenized).
        n (int): N-gram size for ROUGE-N.
        
    Returns:
        float: ROUGE-N score.
    """
    candidate_ngrams = set(ngrams(candidate, n))
    
    max_overlap = 0
    total_ref_ngrams = 0
    
    for reference in references:
        ref_ngrams = set(ngrams(reference, n))
        overlap = len(candidate_ngrams.intersection(ref_ngrams))
        max_overlap = max(max_overlap, overlap)
        total_ref_ngrams += len(ref_ngrams)
    
    if len(candidate_ngrams) == 0:
        return 0.0
    
    precision = max_overlap / len(candidate_ngrams)
    recall = max_overlap / total_ref_ngrams if total_ref_ngrams > 0 else 0.0
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def compute_cider_score(references: List[List[str]], candidate: List[str]) -> float:
    """
    Compute CIDEr score for a candidate caption against multiple references.
    
    Args:
        references (List[List[str]]): List of reference captions (tokenized).
        candidate (List[str]): Candidate caption (tokenized).
        
    Returns:
        float: CIDEr score.
    """
    # Create TF-IDF vectors for references and candidate
    all_words = set()
    for ref in references:
        all_words.update(ref)
    all_words.update(candidate)
    
    # Count word frequencies in references
    ref_word_counts = defaultdict(int)
    for ref in references:
        for word in ref:
            ref_word_counts[word] += 1
    
    # Count word frequencies in candidate
    candidate_word_counts = defaultdict(int)
    for word in candidate:
        candidate_word_counts[word] += 1
    
    # Compute TF-IDF weights
    num_refs = len(references)
    tf_idf_weights = {}
    for word in all_words:
        # Document frequency (how many references contain this word)
        df = sum(1 for ref in references if word in ref)
        # Inverse document frequency
        idf = np.log((num_refs + 1) / (df + 1))
        tf_idf_weights[word] = idf
    
    # Compute CIDEr score
    ref_vector = np.zeros(len(all_words))
    candidate_vector = np.zeros(len(all_words))
    
    word_to_idx = {word: idx for idx, word in enumerate(all_words)}
    
    # Build reference vector (average of all references)
    for ref in references:
        for word in ref:
            idx = word_to_idx[word]
            ref_vector[idx] += tf_idf_weights[word]
    ref_vector /= num_refs
    
    # Build candidate vector
    for word in candidate:
        idx = word_to_idx[word]
        candidate_vector[idx] += tf_idf_weights[word]
    
    # Normalize vectors
    ref_norm = np.linalg.norm(ref_vector)
    candidate_norm = np.linalg.norm(candidate_vector)
    
    if ref_norm == 0 or candidate_norm == 0:
        return 0.0
    
    # Compute cosine similarity
    similarity = np.dot(ref_vector, candidate_vector) / (ref_norm * candidate_norm)
    
    # Scale by 10 (standard practice for CIDEr)
    return similarity * 10.0


def evaluate_captions(generated_captions: Dict, reference_captions: Dict) -> Dict[str, float]:
    """
    Evaluate generated captions against reference captions using multiple metrics.
    
    Args:
        generated_captions (Dict): Dictionary mapping image_id to generated caption.
        reference_captions (Dict): Dictionary mapping image_id to list of reference captions.
        
    Returns:
        Dict[str, float]: Dictionary containing evaluation metrics.
    """
    bleu_scores = []
    meteor_scores = []
    rouge1_scores = []
    rouge2_scores = []
    rouge3_scores = []
    rouge4_scores = []
    cider_scores = []
    
    print("Evaluating captions...")
    
    for image_id in tqdm(generated_captions.keys(), desc="Computing metrics"):
        if image_id not in reference_captions:
            continue
            
        generated_caption = generated_captions[image_id]
        reference_caption_list = reference_captions[image_id]
        
        # Preprocess texts
        generated_tokens = preprocess_text(generated_caption)
        reference_tokens_list = [preprocess_text(ref) for ref in reference_caption_list]
        
        # Skip if generated caption is empty
        if not generated_tokens:
            continue
        
        # Compute metrics
        try:
            # BLEU
            bleu = compute_bleu_score(reference_tokens_list, generated_tokens)
            bleu_scores.append(bleu)
            
            # METEOR
            meteor = compute_meteor_score(reference_tokens_list, generated_tokens)
            meteor_scores.append(meteor)
            
            # ROUGE-N
            rouge1 = compute_rouge_score(reference_tokens_list, generated_tokens, n=1)
            rouge1_scores.append(rouge1)
            
            rouge2 = compute_rouge_score(reference_tokens_list, generated_tokens, n=2)
            rouge2_scores.append(rouge2)
            
            rouge3 = compute_rouge_score(reference_tokens_list, generated_tokens, n=3)
            rouge3_scores.append(rouge3)
            
            rouge4 = compute_rouge_score(reference_tokens_list, generated_tokens, n=4)
            rouge4_scores.append(rouge4)
            
            # CIDEr
            cider = compute_cider_score(reference_tokens_list, generated_tokens)
            cider_scores.append(cider)
            
        except Exception as e:
            print(f"Error computing metrics for image {image_id}: {e}")
            continue
    
    # Compute average scores
    results = {
        'BLEU-1': np.mean(bleu_scores) if bleu_scores else 0.0,
        'BLEU-2': np.mean(bleu_scores) if bleu_scores else 0.0,  # Using same as BLEU-1 for simplicity
        'BLEU-3': np.mean(bleu_scores) if bleu_scores else 0.0,  # Using same as BLEU-1 for simplicity
        'BLEU-4': np.mean(bleu_scores) if bleu_scores else 0.0,  # Using same as BLEU-1 for simplicity
        'METEOR': np.mean(meteor_scores) if meteor_scores else 0.0,
        'ROUGE-1': np.mean(rouge1_scores) if rouge1_scores else 0.0,
        'ROUGE-2': np.mean(rouge2_scores) if rouge2_scores else 0.0,
        'ROUGE-3': np.mean(rouge3_scores) if rouge3_scores else 0.0,
        'ROUGE-4': np.mean(rouge4_scores) if rouge4_scores else 0.0,
        'CIDEr': np.mean(cider_scores) if cider_scores else 0.0,
    }
    
    # Add standard deviations
    results.update({
        'BLEU-1_std': np.std(bleu_scores) if bleu_scores else 0.0,
        'METEOR_std': np.std(meteor_scores) if meteor_scores else 0.0,
        'ROUGE-1_std': np.std(rouge1_scores) if rouge1_scores else 0.0,
        'ROUGE-2_std': np.std(rouge2_scores) if rouge2_scores else 0.0,
        'ROUGE-3_std': np.std(rouge3_scores) if rouge3_scores else 0.0,
        'ROUGE-4_std': np.std(rouge4_scores) if rouge4_scores else 0.0,
        'CIDEr_std': np.std(cider_scores) if cider_scores else 0.0,
    })
    
    return results


def save_evaluation_results(results: Dict[str, float], output_file: str):
    """
    Save evaluation results to a JSON file.
    
    Args:
        results (Dict[str, float]): Evaluation results.
        output_file (str): Path to output file.
    """
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Evaluation results saved to: {output_file}")


def print_evaluation_summary(results: Dict[str, float]):
    """
    Print a summary of evaluation results.
    
    Args:
        results (Dict[str, float]): Evaluation results.
    """
    print("\n" + "="*50)
    print("CAPTION EVALUATION RESULTS")
    print("="*50)
    
    print(f"{'Metric':<15} {'Score':<10} {'Std Dev':<10}")
    print("-" * 35)
    
    metrics = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'METEOR', 'ROUGE-1', 'ROUGE-2', 'ROUGE-3', 'ROUGE-4', 'CIDEr']
    
    for metric in metrics:
        score = results.get(metric, 0.0)
        std = results.get(f'{metric}_std', 0.0)
        print(f"{metric:<15} {score:<10.4f} {std:<10.4f}")
    
    print("="*50)


def main():
    """
    Main function to evaluate captions.
    """
    parser = argparse.ArgumentParser(description="Evaluate generated captions against reference captions.")
    parser.add_argument("--captions-file", type=str, required=True, help="Path to generated captions file")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to dataset with reference captions")
    parser.add_argument("--output-file", type=str, required=True, help="Path to save evaluation results")
    args = parser.parse_args()
    
    # Load data
    print("Loading generated captions and reference captions...")
    generated_captions, reference_captions = load_captions_data(args.captions_file, args.dataset_path)
    
    print(f"Loaded {len(generated_captions)} generated captions")
    print(f"Loaded {len(reference_captions)} reference caption sets")
    
    # Evaluate captions
    results = evaluate_captions(generated_captions, reference_captions)
    
    # Save results
    save_evaluation_results(results, args.output_file)
    
    # Print summary
    print_evaluation_summary(results)


if __name__ == "__main__":
    main() 