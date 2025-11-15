"""Base configuration dataclasses for training and inference."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SFTConfig:
    """Configuration for supervised fine-tuning."""

    # Model
    model_name: str = "meta-llama/Llama-3.2-1B"

    # Data
    data_path: str = "examples/example_data.jsonl"  # Local JSONL or HF dataset
    dataset_type: str = "jsonl"  # "jsonl" or "huggingface"

    # Training
    learning_rate: float = 5e-4  # Higher for LoRA (10x full fine-tune)
    batch_size: int = 64
    num_epochs: int = 3
    max_length: int = 2048  # Max sequence length

    # LoRA
    lora_rank: int = 32  # Rank for LoRA adapters
    # Note: lora_alpha and lora_dropout are not currently supported by Tinker API
    # The API only accepts the rank parameter

    # Training objective
    train_on_what: str = "LAST_ASSISTANT_MESSAGE"  # Most efficient for chat

    # Checkpointing
    save_every: int = 50  # Save checkpoint every N steps
    eval_every: int = 25  # Evaluate every N steps
    log_path: str = "logs/sft"

    # Logging
    wandb_project: Optional[str] = "tinker-posttraining-sft"
    wandb_entity: Optional[str] = None


@dataclass
class RLConfig:
    """Configuration for RL training with importance sampling."""

    # Model - load from SFT checkpoint
    load_checkpoint_path: str = "tinker://{uuid}/sampler_weights/final"
    base_model_name: str = "meta-llama/Llama-3.2-1B"

    # LoRA (should match SFT)
    lora_rank: int = 32
    # Note: lora_alpha and lora_dropout are not currently supported by Tinker API
    # The API only accepts the rank parameter

    # RL Algorithm
    loss_fn: str = "importance_sampling"  # or "ppo"
    learning_rate: float = 4e-5  # Lower than SFT (~10x reduction)

    # Environment
    env_type: str = "preference"  # Type of RL environment
    group_size: int = 2  # Generate 2 responses per prompt for comparison
    groups_per_batch: int = 32  # Number of prompt groups per batch

    # Reward
    reward_type: str = "length_preference"  # "length_preference" or "reward_model"
    target_length: int = 150  # Target response length (for length-based)
    reward_model_path: Optional[str] = None  # Path to reward model if using

    # Generation
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9

    # Training
    num_steps: int = 500
    num_substeps: int = 1  # Gradient accumulation
    kl_penalty_coef: float = 0.01  # KL divergence penalty

    # Checkpointing
    save_every: int = 50
    log_path: str = "logs/rl"

    # Logging
    wandb_project: Optional[str] = "tinker-posttraining-rl"
    wandb_entity: Optional[str] = None


@dataclass
class InferenceConfig:
    """Configuration for inference and deployment."""

    # Model
    base_model_name: str = "meta-llama/Llama-3.2-1B"
    checkpoint_path: str = "tinker://{uuid}/sampler_weights/final"

    # Download
    output_dir: str = "downloaded_weights"
    merge_adapters: bool = True  # Merge LoRA for faster inference

    # Modal deployment
    modal_app_name: str = "tinker-chat-model"
    gpu_type: str = "T4"  # "T4", "A10G", "A100"

    # HuggingFace upload (optional)
    upload_to_hf: bool = False
    hf_repo_name: Optional[str] = None

    # Generation defaults
    default_max_tokens: int = 512
    default_temperature: float = 0.7
