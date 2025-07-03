from rich import print
from utils.dataset import OpenVQADataset
from utils import get_dataloader, seed_all


def test_open_answer_vqa_dataset():
    """Test function for the OpenVQADataset - both standalone and integrated usage"""
    print("=" * 80)
    print("TESTING STANDALONE OpenVQADataset CLASS")
    print("=" * 80)

    # Test with images
    print("\n" + "-" * 50)
    print("1. Testing with images...")
    print("-" * 50)
    dataset_images = OpenVQADataset(
        dataset_path_or_name="~/scratch/datasets/cocoqa_unified",
        images_qa=True,
        downsample_size=10,
        seed=42,
    )

    print(f"Dataset size: {len(dataset_images)}")
    for i in range(len(dataset_images)):
        sample = dataset_images[i]
        print(f"\nSample {i + 1}:")
        print(f"Question: {sample['question']}")
        print(f"Answer: {sample['answer_letter']}")
        print(f"Image ID: {sample['image_id']}")
        print(f"Has image: {sample.get('image') is not None}")
        print("-" * 50)

    # Test without images (text-only)
    print("\n" + "-" * 50)
    print("2. Testing without images (text-only)...")
    print("-" * 50)
    dataset_text = OpenVQADataset(
        dataset_path_or_name="~/scratch/datasets/cocoqa_unified",
        texts_qa=True,
        downsample_size=10,
        seed=42,
    )

    print(f"Dataset size: {len(dataset_text)}")
    for i in range(len(dataset_text)):
        sample = dataset_text[i]
        print(f"\nSample {i + 1}:")
        print(f"Question: {sample['question']}")
        print(f"Answer: {sample['answer_letter']}")
        print(f"Image ID: {sample['image_id']}")
        print(f"Has image: {sample.get('image') is not None}")
        print("-" * 50)

    print("\n" + "=" * 80)
    print("TESTING INTEGRATED OpenVQADataset Class WITH get_dataloader")
    print("=" * 80)

    # Test integrated usage
    print("\n" + "-" * 50)
    print("3. Testing integrated dataloader with images...")
    print("-" * 50)
    dataloader_images = get_dataloader(
        dataset_path_or_name="~/scratch/datasets/cocoqa_unified",
        images_qa=True,
        downsample_size=10,
        batch_size=10,
        seed=42,
    )

    print("Created dataloader successfully")
    for batch_idx, batch in enumerate(dataloader_images):
        if batch_idx > 0:
            break
        print("*" * 50)
        print(f"\nBatch {batch_idx}:")
        print(f"Questions: {len(batch['questions'])}")
        print(f"Answers: {len(batch['answer_letters'])}")
        print(f"Image IDs: {len(batch['image_ids'])}")
        print(f"Images: {len(batch['images'])}")
        print("*" * 50 + "\n")

        for i, (question, answer, image_id, image) in enumerate(
            zip(
                batch["questions"],
                batch["answer_letters"],
                batch["image_ids"],
                batch["images"],
            )
        ):
            print(f"  Sample {i + 1}: Question: {question}")
            print(f"  Sample {i + 1}: Answer: {answer}")
            print(f"  Sample {i + 1}: Image ID: {image_id}")
            print(f"  Sample {i + 1}: Image: {image is not None}")
            print("-" * 50)

    print("\n" + "-" * 50)
    print("4. Testing integrated dataloader without images...")
    print("-" * 50)
    dataloader_text = get_dataloader(
        dataset_path_or_name="~/scratch/datasets/cocoqa_unified",
        texts_qa=True,
        downsample_size=10,
        batch_size=10,
        seed=42,
    )

    for batch_idx, batch in enumerate(dataloader_text):
        if batch_idx > 0:
            break
        print("*" * 50)
        print(f"\nBatch {batch_idx}:")
        print(f"Questions: {len(batch['questions'])}")
        print(f"Answers: {len(batch['answer_letters'])}")
        print(f"Image IDs: {len(batch['image_ids'])}")
        print("*" * 50 + "\n")
        for i, (question, answer, image_id, image) in enumerate(
            zip(
                batch["questions"],
                batch["answer_letters"],
                batch["image_ids"],
                batch["images"],
            )
        ):
            print(f"  Sample {i + 1}: Question: {question}")
            print(f"  Sample {i + 1}: Answer: {answer}")
            print(f"  Sample {i + 1}: Image ID: {image_id}")
            print(f"  Sample {i + 1}: Image: {image is not None}")
            print("-" * 50)


if __name__ == "__main__":
    seed_all(42)
    test_open_answer_vqa_dataset()
