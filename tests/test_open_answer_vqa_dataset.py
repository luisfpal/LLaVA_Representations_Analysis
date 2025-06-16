from rich import print
from utils.dataset import OpenVQADataset


def test_open_answer_vqa_dataset():
    """Test function for the OpenVQADataset - both standalone and integrated usage"""
    try:
        print("=" * 80)
        print("TESTING STANDALONE OpenVQADataset CLASS")
        print("=" * 80)

        # Test with images
        print("\n1. Testing with images...")
        dataset_images = OpenVQADataset(
            dataset_path_or_name="~/scratch/datasets/cocoqa_captioning_restval",
            images_qa=True,
            downsample_size=3,
            seed=42,
        )

        print(f"Dataset size: {len(dataset_images)}")
        for i in range(min(2, len(dataset_images))):
            sample = dataset_images[i]
            print(f"\nSample {i + 1}:")
            print(f"Question: {sample['question']}")
            print(f"Answer: {sample['answer_letter']}")
            print(f"Has image: {sample.get('image') is not None}")
            print("-" * 50)

        # Test without images (text-only)
        print("\n2. Testing without images (text-only)...")
        dataset_text = OpenVQADataset(
            dataset_path_or_name="~/scratch/datasets/cocoqa_captioning_restval",
            texts_qa=True,
            downsample_size=3,
            seed=42,
        )

        print(f"Dataset size: {len(dataset_text)}")
        for i in range(min(2, len(dataset_text))):
            sample = dataset_text[i]
            print(f"\nSample {i + 1}:")
            print(f"Question: {sample['question']}")
            print(f"Answer: {sample['answer_letter']}")
            print(f"Has image: {sample.get('image') is not None}")
            print("-" * 50)

        print("\n" + "=" * 80)
        print("TESTING INTEGRATED USAGE WITH dataset.py")
        print("=" * 80)

        # Test integrated usage
        try:
            from utils.dataset import get_dataloader

            print("\n3. Testing integrated dataloader with images...")
            dataloader_images = get_dataloader(
                dataset_path_or_name="~/scratch/datasets/cocoqa_captioning_restval",
                images_qa=True,
                downsample_size=3,
                batch_size=2,
                seed=42,
            )

            print("Created dataloader successfully")
            for batch_idx, batch in enumerate(dataloader_images):
                print(f"\nBatch {batch_idx + 1}:")
                print(f"Questions: {len(batch['questions'])}")
                print(f"Answers: {len(batch['answer_letters'])}")
                print(f"Images: {len(batch['images'])}")

                for i, (question, answer, image) in enumerate(
                    zip(batch["questions"], batch["answer_letters"], batch["images"])
                ):
                    print(f"  Sample {i + 1}: Question: {question}")
                    print(f"  Sample {i + 1}: Answer: {answer}")
                    print(f"  Sample {i + 1}: Has image: {image is not None}")
                if batch_idx >= 1:  # Only show first 2 batches
                    break

            print("\n4. Testing integrated dataloader without images...")
            dataloader_text = get_dataloader(
                dataset_path_or_name="~/scratch/datasets/cocoqa_captioning_restval",
                texts_qa=True,
                downsample_size=3,
                batch_size=2,
                seed=42,
            )

            for batch_idx, batch in enumerate(dataloader_text):
                print(f"\nText-only Batch {batch_idx + 1}:")
                print(f"Questions: {len(batch['questions'])}")
                print(f"Answers: {len(batch['answer_letters'])}")

                for i, (question, answer, image) in enumerate(
                    zip(batch["questions"], batch["answer_letters"], batch["images"])
                ):
                    print(f"  Sample {i + 1}: Question: {question}")
                    print(f"  Sample {i + 1}: Answer: {answer}")
                    print(f"  Sample {i + 1}: Has image: {image is not None}")
                break  # Only show first batch

        except ImportError as e:
            print(f"Could not test integrated usage: {e}")

    except Exception as e:
        print(f"Error during testing: {e}")


if __name__ == "__main__":
    test_open_answer_vqa_dataset()
