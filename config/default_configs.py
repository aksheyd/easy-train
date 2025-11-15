"""Default configuration presets for different use cases."""

from .base_config import SFTConfig, RLConfig, InferenceConfig


def get_tiny_test_config():
    """
    Quick test configuration (for debugging).

    Use this to quickly verify the pipeline works end-to-end
    with minimal compute time.
    """
    return {
        "sft": SFTConfig(
            batch_size=8,
            num_epochs=1,
            save_every=10,
            eval_every=5,
            log_path="logs/tiny_test/sft",
        ),
        "rl": RLConfig(
            groups_per_batch=4,
            num_steps=10,
            save_every=5,
            log_path="logs/tiny_test/rl",
        ),
        "inference": InferenceConfig(
            output_dir="downloaded_weights/tiny_test",
        ),
    }


def get_quick_experiment_config():
    """
    Quick experiment configuration (small dataset).

    Good for experimentation and learning. Trains on a subset
    of data with reasonable hyperparameters.
    """
    return {
        "sft": SFTConfig(
            batch_size=32,
            num_epochs=1,
            save_every=25,
            eval_every=10,
            log_path="logs/quick/sft",
        ),
        "rl": RLConfig(
            groups_per_batch=16,
            num_steps=100,
            save_every=25,
            log_path="logs/quick/rl",
        ),
        "inference": InferenceConfig(
            output_dir="downloaded_weights/quick",
        ),
    }


def get_full_training_config():
    """
    Full training configuration.

    Production-quality training with full dataset and
    recommended hyperparameters.
    """
    return {
        "sft": SFTConfig(
            batch_size=64,
            num_epochs=3,
            save_every=50,
            eval_every=25,
            log_path="logs/full/sft",
        ),
        "rl": RLConfig(
            groups_per_batch=32,
            num_steps=500,
            save_every=50,
            log_path="logs/full/rl",
        ),
        "inference": InferenceConfig(
            output_dir="downloaded_weights/full",
        ),
    }
