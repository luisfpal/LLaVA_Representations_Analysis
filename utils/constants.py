GUIDE_TEXT = "\nAnswer ONLY with the option's letter from the given choices directly.\n"
ANSWER_TEXT = "\nAnswer:"
SYSTEM_ROLE = {
    "role": "system",
    "content": [
        {
            "type": "text",
            "text": "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.",
        },
    ],
}
ASSISTANT_ROLE = {
    "role": "assistant",
    "content": [
        {"type": "text", "text": "Answer:"},
    ],
}
SQA_ANSWER_CHOICES = ["A", "B", "C", "D", "E"]
MMLU_ANSWER_CHOICES = ["A", "B", "C", "D"]
COCOQA_VI_DIGITS_MAP = {
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
    "10": "ten",
}
