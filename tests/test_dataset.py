import unittest
from rich import print
from datasets import load_dataset
from utils import get_dataloader


class TestDatasetLoading(unittest.TestCase):
    def setUp(self):
        """Set up test environment variables"""
        self.cache_dir = "~/scratch/huggingface/datasets"
        self.split = "test"

    def test_scienceqa_with_images(self):
        """Test loading ScienceQA dataset with images"""
        dataset_path_or_name = "derek-thomas/ScienceQA"
        batch_size = 4

        print(
            "[bold blue]Testing ScienceQA (expect images and potentially Nones):[/bold blue]"
        )
        dataloader = get_dataloader(
            dataset_path_or_name=dataset_path_or_name,
            cache_dir=self.cache_dir,
            split=self.split,
            batch_size=batch_size,
            num_workers=4,
            shuffle=False,
            images_qa=True,
        )

        # Test that we can iterate and get expected data structure
        batch = next(iter(dataloader))

        # Basic assertion checks
        self.assertIsNotNone(batch)
        self.assertIn("questions", batch)
        self.assertIn("answer_letters", batch)
        self.assertIn("images", batch)

        # Print out details for visual inspection
        questions = batch["questions"]
        images = batch["images"]
        answer_letters = batch["answer_letters"]

        for idx, question in enumerate(questions):
            print(f"Question {idx}: {question}")
            print(f"Answer letter: {answer_letters[idx]}")
            if images[idx] is not None:
                print(f"Image: {images[idx]}")
                print(f"Image type: {type(images[idx])}")
            else:
                print("No image available.")
            print()  # Add a newline for better readability

    def test_mmlu_loading(self):
        """Test direct loading of MMLU dataset"""
        dataset_path_or_name = "cais/mmlu"

        print("[bold blue]Testing MMLU loading directly:[/bold blue]")

        # Load dataset
        mmlu_dataset = load_dataset(
            path=dataset_path_or_name,
            name="all",
            split=self.split,
            cache_dir=self.cache_dir,
        )

        # Check features and output info
        print(f"Features: {mmlu_dataset.features}")
        print(f"Number of samples: {len(mmlu_dataset)}")

        # Basic assertions
        self.assertIsNotNone(mmlu_dataset)
        self.assertGreater(len(mmlu_dataset), 0)

    def test_mmlu(self):
        """Test loading MMLU through the get_dataloader interface"""
        dataset_path_or_name = "cais/mmlu"
        batch_size = 4

        print("[bold blue]Testing MMLU (expect only text samples):[/bold blue]")
        dataloader = get_dataloader(
            dataset_path_or_name=dataset_path_or_name,
            cache_dir=self.cache_dir,
            split=self.split,
            batch_size=batch_size,
            num_workers=4,
            shuffle=False,
            texts_qa=True,  # Fixed from mm_dataset=True which doesn't exist
        )

        # Test that we can iterate and get expected data structure
        batch = next(iter(dataloader))

        # Basic assertion checks
        self.assertIsNotNone(batch)
        self.assertIn("questions", batch)
        self.assertIn("answer_letters", batch)
        self.assertIn("images", batch)

        # Print out details for visual inspection
        questions = batch["questions"]
        images = batch["images"]
        answer_letters = batch["answer_letters"]

        for idx, question in enumerate(questions):
            print(f"Question {idx}: {question}")
            print(f"Answer letter: {answer_letters[idx]}")
            if images[idx] is not None:
                print(f"Image: {images[idx]}")
                print(f"Image type: {type(images[idx])}")
            else:
                print("No image available.")
            print()  # Add a newline for better readability


if __name__ == "__main__":
    unittest.main()
