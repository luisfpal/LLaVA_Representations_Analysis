"""
Test to verify OpenVQADataset deterministic behavior after the fix.
"""

from utils.dataset import OpenVQADataset


def test_deterministic_behavior():
    """Test that the same idx returns the same data across multiple calls."""
    print("Testing OpenVQADataset deterministic behavior...")

    # Create dataset instance
    dataset = OpenVQADataset(
        dataset_path_or_name="~/scratch/datasets/cocoqa_unified",
        downsample_size=10,
        seed=42,
        images_qa=True,
    )

    print(f"Dataset size: {len(dataset)}")

    # Test multiple indices
    for test_idx in [0, 3, 7]:
        print(f"\n--- Testing idx={test_idx} ---")

        # Get the same index multiple times
        samples = []
        for call_num in range(5):
            sample = dataset[test_idx]
            samples.append(sample)
            print(
                f"Call {call_num + 1}: Q='{sample['question'][:50]}...' A='{sample['answer_letter']}'"
            )

        # Verify all calls return identical data
        first_sample = samples[0]
        all_identical = all(
            sample["question"] == first_sample["question"]
            and sample["answer_letter"] == first_sample["answer_letter"]
            and sample["image_id"] == first_sample["image_id"]
            for sample in samples[1:]
        )

        if all_identical:
            print(
                f"✅ idx={test_idx} is DETERMINISTIC - all calls returned identical data"
            )
        else:
            print(
                f"❌ idx={test_idx} is NOT DETERMINISTIC - calls returned different data"
            )
            return False

    print("\n🎉 SUCCESS: Dataset is now fully deterministic!")
    return True


def test_uniqueness_across_dataset():
    """Test that different indices return different question-answer pairs when possible."""
    print("\n" + "=" * 60)
    print("Testing uniqueness across different dataset indices...")

    dataset = OpenVQADataset(
        dataset_path_or_name="~/scratch/datasets/cocoqa_unified",
        downsample_size=20,
        seed=42,
        images_qa=True,
    )

    qa_pairs = set()
    duplicate_count = 0

    for idx in range(len(dataset)):
        sample = dataset[idx]
        qa_pair = (sample["question"], sample["answer_letter"])

        if qa_pair in qa_pairs:
            duplicate_count += 1
            print(f"Duplicate Q&A found at idx={idx}: '{sample['question'][:50]}...'")
        else:
            qa_pairs.add(qa_pair)

    unique_count = len(qa_pairs)
    total_samples = len(dataset)

    print(f"\nUniqueness Results:")
    print(f"Total samples: {total_samples}")
    print(f"Unique Q&A pairs: {unique_count}")
    print(f"Duplicate Q&A pairs: {duplicate_count}")
    print(f"Uniqueness rate: {unique_count / total_samples * 100:.1f}%")

    return unique_count, duplicate_count


if __name__ == "__main__":
    # Test deterministic behavior
    success = test_deterministic_behavior()

    if success:
        # Test uniqueness
        test_uniqueness_across_dataset()
    else:
        print("❌ Deterministic test failed!")
