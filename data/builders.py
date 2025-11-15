"""Dataset builder wrappers for Tinker."""

from tinker_cookbook.supervised.data import (
    FromConversationFileBuilder,
    SupervisedDatasetFromHFDataset,
)


def create_dataset_builder(config):
    """
    Create appropriate dataset builder based on config.

    Educational note:
    - FromConversationFileBuilder: For local JSONL files
    - SupervisedDatasetFromHFDataset: For HuggingFace datasets
    - StreamingSupervisedDatasetFromHFDataset: For large HF datasets (not shown)

    Args:
        config: SFTConfig object with data_path and dataset_type

    Returns:
        Dataset builder instance

    Raises:
        ValueError: If dataset_type is unknown
    """
    if config.dataset_type == "jsonl":
        return FromConversationFileBuilder(
            file_path=config.data_path,
            # This builder automatically handles the "messages" field format
        )

    elif config.dataset_type == "huggingface":
        return SupervisedDatasetFromHFDataset(
            dataset_name=config.data_path,  # e.g., "HuggingFaceH4/no_robots"
            split="train",
            # Optionally specify message_field if not "messages"
        )

    else:
        raise ValueError(f"Unknown dataset_type: {config.dataset_type}")
