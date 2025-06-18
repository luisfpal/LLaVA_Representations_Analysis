# VQA Evaluation Guide

## Overview

This guide explains the implementation of Visual Question Answering (VQA) evaluation for both **Multiple Choice** and **Open-ended** questions, based on the VQA evaluation protocol.

## Key Differences Between Multiple Choice and Open VQA

### 1. **Multiple Choice VQA**
- **Generates exactly 1 token** (A, B, C, D)
- **Simple exact matching** for evaluation
- **Binary accuracy**: either correct (1.0) or incorrect (0.0)
- **Used for**: ScienceQA, MMLU, AI2D, etc.

### 2. **Open VQA** 
- **Generates multiple tokens** (free text answers)
- **Complex text normalization** required before matching
- **VQA accuracy protocol**: `min(1, matching_answers / 3)`
- **Used for**: VQAv2, COCO-QA, GQA, etc.

## Essential Components from the Reference Code

### **VQA Evaluation Protocol** (from `open_vqa_eval.py`)

The core of open VQA evaluation involves sophisticated text processing:

```python
# 1. Text normalization
def normalize_answer(answer):
    answer = answer.replace("\n", " ").replace("\t", " ").strip()
    answer = process_punctuation(answer, punct, commaStrip, periodStrip)
    answer = process_digit_article(answer, manualMap, articles, contractions)
    return answer

# 2. VQA scoring: min(1, number_of_matching_answers / 3)
matching_count = normalized_gt_answers.count(normalized_pred)
vqa_accuracy = min(1.0, float(matching_count) / 3.0)
```

#### **Text Processing Steps:**
1. **Remove newlines/tabs**
2. **Process punctuation** (remove or normalize)
3. **Convert word numbers to digits** ("one" → "1")
4. **Remove articles** ("a", "an", "the")
5. **Expand contractions** ("don't" → "do not")

#### **VQA Accuracy Formula:**
- If 3+ humans gave the same answer as the model: **accuracy = 1.0**
- If 2 humans gave the same answer: **accuracy = 0.67**
- If 1 human gave the same answer: **accuracy = 0.33**
- If 0 humans gave the same answer: **accuracy = 0.0**

## Script Usage

### **Unified Script** (`scripts/unified_vqa_evaluation.py`)

```bash
# Multiple Choice VQA
python scripts/unified_vqa_evaluation.py \
    --model-name-or-path "llava-hf/llava-1.5-7b-hf" \
    --dataset-name "ScienceQA" \
    --dataset-cache-dir "/path/to/cache" \
    --base-dir "/path/to/results" \
    --evaluation-type "multiple_choice" \
    --images_qa \
    --chat-mode \
    --batch-size 1

# Open VQA
python scripts/unified_vqa_evaluation.py \
    --model-name-or-path "llava-hf/llava-1.5-7b-hf" \
    --dataset-name "HuggingFaceM4/VQAv2" \
    --dataset-cache-dir "/path/to/cache" \
    --base-dir "/path/to/results" \
    --evaluation-type "open_vqa" \
    --images_qa \
    --chat-mode \
    --max-new-tokens 50 \
    --split "validation"
```

### **Separate Scripts**

```bash
# Multiple Choice (original)
python scripts/multiple_choice_benchmarks_evaluation.py \
    --model-name-or-path "llava-hf/llava-1.5-7b-hf" \
    --dataset-name "ScienceQA" \
    --dataset-cache-dir "/path/to/cache" \
    --base-dir "/path/to/results" \
    --images_qa \
    --chat-mode

# Open VQA (new)
python scripts/open_vqa_evaluation.py \
    --model-name-or-path "llava-hf/llava-1.5-7b-hf" \
    --dataset-name "HuggingFaceM4/VQAv2" \
    --dataset-cache-dir "/path/to/cache" \
    --base-dir "/path/to/results" \
    --images_qa \
    --chat-mode \
    --max-new-tokens 50
```

## Key Parameter Differences

| Parameter | Multiple Choice | Open VQA |
|-----------|----------------|----------|
| `max_new_tokens` | 1 (fixed) | 50+ (configurable) |
| `do_sample` | Usually False | Can be True/False |
| `temperature` | N/A | Configurable (1.0 default) |
| `evaluation_type` | "multiple_choice" | "open_vqa" |

## Dataset Format Requirements

### **Multiple Choice Format**
```json
{
    "prompt": "Question with options A, B, C, D",
    "output": "A",
    "answer_letter": "A"
}
```

### **Open VQA Format**  
```json
{
    "question_id": "12345",
    "question": "What color is the car?",
    "prompt": "What color is the car?", 
    "output": "red",
    "ground_truth_answers": [
        {"answer": "red"},
        {"answer": "crimson"}, 
        {"answer": "red color"}
    ]
}
```

## Output Results Structure

Both evaluation types produce similar result structures:

```json
{
    "accuracy": 0.78,
    "correct": 156,
    "count": 200,
    "correct_results": [...],
    "incorrect_results": [...]
}
```

**For Open VQA**, each result also includes:
- `normalized_predicted`: normalized prediction
- `normalized_gt`: normalized ground truth answers  
- `vqa_accuracy`: per-question VQA accuracy score

## When to Use Which Evaluation

- **Multiple Choice**: When you have fixed answer options (A/B/C/D)
- **Open VQA**: When answers are free-form text that need sophisticated matching

The open VQA evaluation is more complex but provides more nuanced accuracy scores that better reflect human agreement patterns. 