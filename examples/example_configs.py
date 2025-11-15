"""
Example configuration presets for different use cases.

These are additional examples beyond the default configs.
"""

from config.base_config import SFTConfig, RLConfig, InferenceConfig


def get_debug_config():
    """
    Minimal configuration for quick debugging.

    Use this when you just want to verify the pipeline works
    without waiting for training.
    """
    return {
        "sft": SFTConfig(
            batch_size=4,
            num_epochs=1,
            save_every=5,
            eval_every=2,
            log_path="logs/debug/sft",
        ),
        "rl": RLConfig(
            groups_per_batch=2,
            num_steps=5,
            save_every=2,
            log_path="logs/debug/rl",
        ),
    }


def get_large_model_config():
    """
    Configuration for larger models (e.g., Llama 3.2 8B).

    Adjusts batch sizes and LoRA rank for larger models.
    """
    return {
        "sft": SFTConfig(
            model_name="meta-llama/Llama-3.2-8B",
            batch_size=16,  # Smaller batch size for larger model
            lora_rank=64,  # Higher rank for larger model
            num_epochs=2,
            log_path="logs/large/sft",
        ),
        "rl": RLConfig(
            base_model_name="meta-llama/Llama-3.2-8B",
            lora_rank=64,
            groups_per_batch=16,  # Smaller batch size
            num_steps=300,
            log_path="logs/large/rl",
        ),
        "inference": InferenceConfig(
            base_model_name="meta-llama/Llama-3.2-8B",
            gpu_type="A10G",  # Need more powerful GPU
        ),
    }


def get_huggingface_dataset_config():
    """
    Configuration using a HuggingFace dataset.

    Example: Using the "HuggingFaceH4/no_robots" dataset.
    """
    return {
        "sft": SFTConfig(
            dataset_type="huggingface",
            data_path="HuggingFaceH4/no_robots",
            batch_size=64,
            num_epochs=1,
            log_path="logs/hf_dataset/sft",
        ),
        "rl": RLConfig(
            groups_per_batch=32,
            num_steps=200,
            log_path="logs/hf_dataset/rl",
        ),
    }


def get_ppo_config():
    """
    Configuration using PPO instead of importance sampling.

    PPO is more complex but can be more stable for some tasks.
    """
    return {
        "sft": SFTConfig(
            batch_size=64,
            num_epochs=3,
            log_path="logs/ppo/sft",
        ),
        "rl": RLConfig(
            loss_fn="ppo",  # Use PPO instead of importance sampling
            learning_rate=3e-5,  # Slightly lower LR for PPO
            groups_per_batch=32,
            num_steps=500,
            log_path="logs/ppo/rl",
        ),
    }


def get_long_context_config():
    """
    Configuration for long-context training.

    Adjusts max_length for longer conversations.
    """
    return {
        "sft": SFTConfig(
            max_length=4096,  # Longer context window
            batch_size=32,  # Smaller batch size due to longer sequences
            num_epochs=2,
            log_path="logs/long_context/sft",
        ),
        "rl": RLConfig(
            max_tokens=1024,  # Generate longer responses
            groups_per_batch=16,
            num_steps=300,
            log_path="logs/long_context/rl",
        ),
    }
