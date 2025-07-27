# COCO-QA Captioning and Evaluation Guide

This guide explains how to caption COCO-QA images using LLaVA and evaluate the generated captions against reference captions.

## Overview

The process involves two main steps:
1. **Captioning**: Generate captions for COCO-QA images using LLaVA
2. **Evaluation**: Compare generated captions with reference captions using standard metrics

## 1. Captioning with LLaVA

### Prompt Design

The captioning prompt is carefully designed to:
- **Encourage careful observation**: "Please look carefully at this image..."
- **Generate short, descriptive captions**: Similar to COCO style captions
- **Focus on key elements**: Main objects, actions, and scene elements
- **Be concise but informative**: Balance between detail and brevity

**Example Prompt:**
```
Please look carefully at this image and provide a short, descriptive caption. 
Focus on the main objects, actions, and scene elements. 
Keep the caption concise but informative, similar to how you would describe 
the image to someone who cannot see it. 
Be specific about what you see without being overly detailed.
```

### Key Features of the Captioning Script

1. **Optimized Prompting**: The prompt encourages the model to:
   - Look carefully at the image before generating
   - Generate short, descriptive captions
   - Focus on main visual elements
   - Be specific without being overly detailed

2. **Chat Template Support**: Handles both chat-based and direct prompting
3. **Batch Processing**: Efficient processing of multiple images
4. **Memory Management**: Proper GPU memory cleanup
5. **Flexible Generation Parameters**: Configurable temperature, top-p, beam search

### Running the Captioning

```bash
# Run captioning
./shell/caption_cocoqa.sh

# Or with debug mode
./shell/caption_cocoqa.sh --debug
```

**Configuration Options:**
- `MODEL_NAME`: LLaVA model to use (default: `llava-hf/llava-1.5-7b-hf`)
- `BATCH_SIZE`: Number of images to process simultaneously (default: 4)
- `MAX_NEW_TOKENS`: Maximum caption length (default: 50)
- `TEMPERATURE`: Generation randomness (default: 0.7)
- `TOP_P`: Nucleus sampling parameter (default: 0.9)

## 2. Caption Evaluation

### Evaluation Metrics

The evaluation script implements standard caption evaluation metrics:

#### BLEU (Bilingual Evaluation Understudy)
- **Formula**: Measures n-gram overlap between generated and reference captions
- **Range**: 0-1 (higher is better)
- **Interpretation**: Measures precision of word choice and phrase structure

#### METEOR (Metric for Evaluation of Translation with Explicit ORdering)
- **Formula**: Harmonic mean of precision and recall with synonym matching
- **Range**: 0-1 (higher is better)
- **Interpretation**: More robust than BLEU, handles synonyms and paraphrasing

#### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
- **Formula**: Measures n-gram overlap with focus on recall
- **Range**: 0-1 (higher is better)
- **Types**: ROUGE-1, ROUGE-2, ROUGE-3, ROUGE-4 (different n-gram sizes)
- **Interpretation**: Measures how much of the reference content is captured

#### CIDEr (Consensus-based Image Description Evaluation)
- **Formula**: TF-IDF weighted cosine similarity between caption vectors
- **Range**: 0-10 (higher is better)
- **Interpretation**: Measures semantic similarity using TF-IDF weighting

### Evaluation Process

1. **Data Loading**: Load generated captions and reference captions
2. **Text Preprocessing**: Tokenize and clean text for evaluation
3. **Metric Computation**: Calculate each metric for each image
4. **Aggregation**: Compute mean and standard deviation across all images
5. **Results Output**: Save detailed results and print summary

### Running the Evaluation

```bash
# Run evaluation
./shell/evaluate_captions.sh

# Or with debug mode
./shell/evaluate_captions.sh --debug
```

## 3. Understanding the Metrics

### BLEU Score
- **BLEU-1**: Single word overlap
- **BLEU-2**: Two-word phrase overlap
- **BLEU-3**: Three-word phrase overlap
- **BLEU-4**: Four-word phrase overlap

**Formula**: 
```
BLEU = exp(∑(w_n * log(p_n)))
```
where `w_n` are weights and `p_n` are n-gram precisions.

### METEOR Score
**Formula**:
```
METEOR = F_mean * (1 - Penalty)
F_mean = 10 * P * R / (9P + R)
```

### ROUGE Score
**Formula**:
```
ROUGE-N = F1 = 2 * Precision * Recall / (Precision + Recall)
Precision = |Candidate_Ngrams ∩ Reference_Ngrams| / |Candidate_Ngrams|
Recall = |Candidate_Ngrams ∩ Reference_Ngrams| / |Reference_Ngrams|
```

### CIDEr Score
**Formula**:
```
CIDEr = 10 * cosine_similarity(TF-IDF(candidate), TF-IDF(references))
```

## 4. Expected Results

### Typical Score Ranges for Good Captions:
- **BLEU-4**: 0.25-0.35
- **METEOR**: 0.25-0.35
- **ROUGE-L**: 0.45-0.55
- **CIDEr**: 0.8-1.2

### Factors Affecting Scores:
1. **Model Quality**: Better models produce higher scores
2. **Prompt Design**: Well-designed prompts improve caption quality
3. **Dataset Characteristics**: Different datasets have different score ranges
4. **Evaluation Setup**: Tokenization and preprocessing affect scores

## 5. Best Practices

### For Captioning:
1. **Use appropriate temperature**: 0.7-0.8 for balanced creativity/accuracy
2. **Set reasonable max tokens**: 30-50 for short captions
3. **Enable sampling**: Better than greedy decoding for captioning
4. **Use beam search**: For more consistent results (num_beams > 1)

### For Evaluation:
1. **Multiple references**: COCO typically has 5 reference captions per image
2. **Proper preprocessing**: Consistent tokenization is crucial
3. **Statistical significance**: Report standard deviations
4. **Human evaluation**: Complement automatic metrics with human judgments

## 6. Troubleshooting

### Common Issues:
1. **Low BLEU scores**: Check tokenization and preprocessing
2. **Memory errors**: Reduce batch size
3. **Empty captions**: Check generation parameters
4. **Evaluation errors**: Ensure NLTK data is downloaded

### Debugging:
```bash
# Run with debug mode
./shell/caption_cocoqa.sh --debug
./shell/evaluate_captions.sh --debug
```

## 7. Advanced Usage

### Custom Prompts:
Modify the `create_captioning_prompt()` function in `scripts/caption_cocoqa_with_llava.py` to experiment with different prompts.

### Additional Metrics:
Add new metrics to `scripts/evaluate_captions.py` by implementing new functions and adding them to the evaluation loop.

### Model Comparison:
Run captioning with different models and compare evaluation results to find the best performing model for your use case.

## 8. File Structure

```
scripts/
├── caption_cocoqa_with_llava.py    # Main captioning script
└── evaluate_captions.py            # Main evaluation script

shell/
├── caption_cocoqa.sh              # Captioning shell script
└── evaluate_captions.sh           # Evaluation shell script

docs/
└── captioning_and_evaluation_guide.md  # This guide
```

## 9. References

- [BLEU Paper](https://aclanthology.org/P02-1040/)
- [METEOR Paper](https://aclanthology.org/W05-0909/)
- [ROUGE Paper](https://aclanthology.org/W04-1013/)
- [CIDEr Paper](https://openaccess.thecvf.com/content_cvpr_2015/papers/Vedantam_CIDEr_Consensus-Based_Image_2015_CVPR_paper.pdf)
- [COCO Captioning Challenge](https://cocodataset.org/#captions-2015) 