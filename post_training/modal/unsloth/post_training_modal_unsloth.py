import pathlib
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import modal

app = modal.App("example-unsloth-finetune")

train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "accelerate==1.9.0",
        "datasets==3.6.0",
        "hf-transfer==0.1.9",
        "huggingface_hub==0.34.2",
        "peft==0.16.0",
        "transformers==4.54.0",
        "trl==0.19.1",
        "unsloth[cu128-torch270]==2025.7.8",
        "unsloth_zoo==2025.7.10",
        "wandb==0.21.0",
    )
    .env({"HF_HOME": "/model_cache"})
)

with train_image.imports():
    # unsloth must be first!
    import unsloth  # noqa: F401,I001
    import datasets
    import torch
    import wandb  # type: ignore
    from transformers import TrainingArguments
    from trl import SFTTrainer  # type: ignore
    from unsloth import FastLanguageModel  # type: ignore
    from unsloth.chat_templates import standardize_sharegpt  # type: ignore

model_cache_volume = modal.Volume.from_name(
    "unsloth-model-cache", create_if_missing=True
)
dataset_cache_volume = modal.Volume.from_name(
    "unsloth-dataset-cache", create_if_missing=True
)
checkpoint_volume = modal.Volume.from_name(
    "unsloth-checkpoints", create_if_missing=True
)

GPU_TYPE = "L40S"
TIMEOUT_HOURS = 6
MAX_RETRIES = 3

CONVERSATION_COLUMN = "conversations"  # ShareGPT format column name
TEXT_COLUMN = "text"  # Output column for formatted text
TRAIN_SPLIT_RATIO = 0.9  # 90% train, 10% eval split
PREPROCESSING_WORKERS = 2  # Number of workers for dataset processing


def format_chat_template(examples, tokenizer):
    texts = []
    for conversation in examples[CONVERSATION_COLUMN]:
        formatted_text = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        )
        texts.append(formatted_text)
    return {TEXT_COLUMN: texts}


def load_or_cache_dataset(config: "TrainingConfig", paths: dict, tokenizer):
    dataset_cache_path = paths["dataset_cache"]

    if dataset_cache_path.exists():
        print(f"Loading cached dataset from {dataset_cache_path}")
        train_dataset = datasets.load_from_disk(dataset_cache_path / "train")
        eval_dataset = datasets.load_from_disk(dataset_cache_path / "eval")
    else:
        print(f"Downloading and processing dataset: {config.dataset_name}")

        # Load and standardize the dataset format
        dataset = datasets.load_dataset(config.dataset_name, split="train")
        dataset = standardize_sharegpt(dataset)

        # Split into training and evaluation sets with fixed seed for reproducibility
        dataset = dataset.train_test_split(
            test_size=1.0 - TRAIN_SPLIT_RATIO, seed=config.seed
        )
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]

        # Apply chat template formatting to convert conversations to text
        print("Formatting datasets with chat template...")
        train_dataset = train_dataset.map(
            lambda examples: format_chat_template(examples, tokenizer),
            batched=True,
            num_proc=PREPROCESSING_WORKERS,
            remove_columns=train_dataset.column_names,
        )

        eval_dataset = eval_dataset.map(
            lambda examples: format_chat_template(examples, tokenizer),
            batched=True,
            num_proc=PREPROCESSING_WORKERS,
            remove_columns=eval_dataset.column_names,
        )

        # Cache the processed datasets for future runs
        print(f"Caching processed datasets to {dataset_cache_path}")
        dataset_cache_path.mkdir(parents=True, exist_ok=True)
        train_dataset.save_to_disk(dataset_cache_path / "train")
        eval_dataset.save_to_disk(dataset_cache_path / "eval")

        # Commit the dataset cache to the volume
        dataset_cache_volume.commit()

    return train_dataset, eval_dataset


def load_or_cache_model(config: "TrainingConfig", paths: dict):
    print(f"Downloading and caching model: {config.model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        dtype=None,
        load_in_4bit=config.load_in_4bit,
        load_in_8bit=config.load_in_8bit,
    )

    return model, tokenizer


LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


@dataclass
class TrainingConfig:
    # Model and dataset selection
    model_name: str
    dataset_name: str
    max_seq_length: int
    load_in_4bit: bool
    load_in_8bit: bool

    # LoRA configuration for efficient finetuning
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_bias: str
    use_rslora: bool

    # Training hyperparameters
    optim: str
    batch_size: int
    gradient_accumulation_steps: int
    packing: bool
    use_gradient_checkpointing: str
    learning_rate: float
    lr_scheduler_type: str
    warmup_ratio: float
    weight_decay: float
    max_steps: int
    save_steps: int
    eval_steps: int
    logging_steps: int

    # Experiment management
    seed: int
    experiment_name: Optional[str] = None
    enable_wandb: bool = True

    # For testing purposes
    skip_eval: bool = False

    def __post_init__(self):
        # Generate a unique experiment name if not provided
        if self.experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            model_short = self.model_name.split("/")[-1]
            self.experiment_name = f"{model_short}-r{self.lora_r}-{timestamp}"


@app.function(
    image=train_image,
    gpu=GPU_TYPE,
    volumes={
        "/model_cache": model_cache_volume,
        "/dataset_cache": dataset_cache_volume,
        "/checkpoints": checkpoint_volume,
    },
    secrets=[modal.Secret.from_name("wandb-secret")],
    timeout=TIMEOUT_HOURS * 60 * 60,
    retries=modal.Retries(initial_delay=0.0, max_retries=MAX_RETRIES),
    max_inputs=1,  # Ensure we get a fresh container on retry
)
def finetune(config: TrainingConfig):
    # Get structured paths for organized file storage
    paths = get_structured_paths(config)

    # Initialize Weights & Biases for experiment tracking if enabled
    if config.enable_wandb:
        wandb.init(
            project="unsloth-finetune",
            name=config.experiment_name,
            config=config.__dict__,
        )

    # Load or cache model and datasets with progress indicators
    print("Setting up model and data...")
    model, tokenizer = load_or_cache_model(config, paths)
    train_dataset, eval_dataset = load_or_cache_dataset(config, paths, tokenizer)

    # Configure the model for LoRA training
    model = setup_model_for_training(model, config)

    # Prepare checkpoint directory and check for existing checkpoints
    checkpoint_path = paths["checkpoints"]
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    resume_from_checkpoint = check_for_existing_checkpoint(paths)

    # Create training configuration
    training_args = create_training_arguments(config, str(checkpoint_path))

    # Initialize the supervised finetuning trainer
    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field=TEXT_COLUMN,
        max_seq_length=config.max_seq_length,
        dataset_num_proc=PREPROCESSING_WORKERS,
        packing=config.packing,  # Sequence packing for efficiency
        args=training_args,
    )

    # Display training information for transparency
    print(f"Training dataset size: {len(train_dataset):,}")
    print(f"Evaluation dataset size: {len(eval_dataset):,}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(
        f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )
    print(f"Experiment: {config.experiment_name}")

    # Start training or resume from checkpoint
    if resume_from_checkpoint:
        print(f"Resuming training from {resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        print("Starting training from scratch...")
        trainer.train()

    # Save the final trained model and tokenizer
    print("Saving final model...")
    final_model_path = checkpoint_path / "final_model"
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)

    # Clean up experiment tracking
    if config.enable_wandb:
        wandb.finish()

    print(f"Training completed! Model saved to: {final_model_path}")
    return config.experiment_name


@app.function(
    image=train_image,
    gpu=GPU_TYPE,
    volumes={
        "/model_cache": model_cache_volume,
        "/checkpoints": checkpoint_volume,
    },
    secrets=[modal.Secret.from_name("huggingface-secret-write")],
    timeout=2 * 60 * 60,  # 2 hours for merging and uploading
)
def merge_and_upload(
    experiment_name: str,
    hf_repo_name: str,
    max_seq_length: int = 32768,
):
    """Merge LoRA adapter weights with base model and upload to Hugging Face Hub."""
    checkpoint_path = pathlib.Path("/checkpoints") / "experiments" / experiment_name
    final_model_path = checkpoint_path / "final_model"

    if not final_model_path.exists():
        raise ValueError(
            f"Model not found at {final_model_path}. "
            f"Make sure training completed successfully."
        )

    print(f"Loading LoRA adapters from: {final_model_path}")

    # Load base model with LoRA adapters
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(final_model_path),  # Load base model
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )

    # Save merged model locally first
    merged_path = checkpoint_path / "merged_model"
    merged_path.mkdir(parents=True, exist_ok=True)

    print(f"Saving merged model to {merged_path}...")
    model.save_pretrained_merged(
        str(merged_path),
        tokenizer,
        save_method="merged_16bit",
    )

    # Upload to Hugging Face Hub
    print(f"Uploading merged model to Hugging Face: {hf_repo_name}")
    model.push_to_hub_merged(
        hf_repo_name,
        tokenizer,
        token=True,
    )

    print(f"✓ Successfully uploaded to https://huggingface.co/{hf_repo_name}")
    return hf_repo_name


@app.local_entrypoint()
def main(
    # Model and dataset configuration
    model_name: str = "unsloth/Qwen3-32B",
    dataset_name: str = "mlabonne/FineTome-100k",
    max_seq_length: int = 32768,
    load_in_4bit: bool = True,  # unsloth: use 4bit quant for frozen model weights
    load_in_8bit: bool = False,  # unsloth: use 8bit quant for frozen model weights
    # LoRA hyperparameters for finetuning efficiency
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    lora_bias: str = "none",  # unsloth: optimized lora kernel
    use_rslora: bool = False,
    # Training hyperparameters for optimization
    optim: str = "adamw_8bit",  # unsloth: 8bit optimizer
    batch_size: int = 16,
    gradient_accumulation_steps: int = 1,
    packing: bool = False,
    use_gradient_checkpointing: str = "unsloth",  # unsloth: optimized gradient offloading
    learning_rate: float = 2e-4,
    lr_scheduler_type: str = "cosine",
    warmup_ratio: float = 0.06,
    weight_decay: float = 0.01,
    max_steps: int = 5,  # increase!
    save_steps: int = 2,  # increase!
    eval_steps: int = 2,  # increase!
    logging_steps: int = 1,  # increase!
    # Optional experiment configuration
    seed: int = 105,
    experiment_name: Optional[str] = None,
    disable_wandb: bool = False,
    skip_eval: bool = False,
):
    # Create configuration object from command line arguments
    config = TrainingConfig(
        model_name=model_name,
        dataset_name=dataset_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_bias=lora_bias,
        lora_dropout=lora_dropout,
        use_rslora=use_rslora,
        optim=optim,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        packing=packing,
        use_gradient_checkpointing=use_gradient_checkpointing,
        learning_rate=learning_rate,
        max_steps=max_steps,
        lr_scheduler_type=lr_scheduler_type,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        save_steps=save_steps,
        eval_steps=eval_steps,
        logging_steps=logging_steps,
        seed=seed,
        experiment_name=experiment_name,
        enable_wandb=not disable_wandb,
        skip_eval=skip_eval,
    )

    # Display experiment configuration for user verification
    print(f"Starting finetuning experiment: {config.experiment_name}")
    print(f"Model: {config.model_name}")
    print(f"Dataset: {config.dataset_name}")
    print(f"LoRA configuration: rank={config.lora_r}, alpha={config.lora_alpha}")
    print(
        f"Effective batch size: {config.batch_size * config.gradient_accumulation_steps}"
    )
    print(f"Training steps: {config.max_steps}")

    # Launch the training job on Modal infrastructure
    experiment_name = finetune.remote(config)
    print(f"Training completed successfully: {experiment_name}")


def get_structured_paths(config: TrainingConfig):
    """
    Create structured paths within the mounted volumes for organized storage.

    This function maps the configuration to specific directory paths that allow
    multiple models, datasets, and experiments to coexist without conflicts.
    """
    # Replace forward slashes in names to create valid directory names
    dataset_cache_path = (
        pathlib.Path("/dataset_cache")
        / "datasets"
        / config.dataset_name.replace("/", "--")
    )
    checkpoint_path = (
        pathlib.Path("/checkpoints") / "experiments" / config.experiment_name  # type: ignore
    )

    return {
        "dataset_cache": dataset_cache_path,
        "checkpoints": checkpoint_path,
    }


def setup_model_for_training(model, config: TrainingConfig):
    """
    Configure the model with LoRA adapters for efficient finetuning.

    LoRA (Low-Rank Adaptation) allows us to finetune large models efficiently
    by only training a small number of additional parameters. This significantly
    reduces memory usage and training time.
    """
    print("Configuring LoRA for training...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,  # LoRA rank - higher values = more parameters
        target_modules=LORA_TARGET_MODULES,  # Which layers to apply LoRA to
        lora_alpha=config.lora_alpha,  # LoRA scaling parameter
        lora_dropout=config.lora_dropout,  # Dropout for LoRA layers
        bias=config.lora_bias,  # Bias configuration
        use_gradient_checkpointing=config.use_gradient_checkpointing,  # Memory optimization
        random_state=config.seed,  # Fixed seed for reproducibility
        use_rslora=config.use_rslora,  # Rank-stabilized LoRA
        loftq_config=None,  # LoFTQ quantization config
    )
    return model


def create_training_arguments(config: TrainingConfig, output_dir: str):
    """
    Create training arguments for the SFTTrainer.

    These arguments control the training process, including optimization settings,
    evaluation frequency, and checkpointing behavior.
    """
    print("SKIP_EVAL", config.skip_eval)
    return TrainingArguments(
        # Core training configuration
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        max_steps=config.max_steps,
        warmup_ratio=config.warmup_ratio,
        # Evaluation and checkpointing
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        eval_strategy="no" if config.skip_eval else "steps",
        save_strategy="steps",
        do_eval=not config.skip_eval,
        # Optimization settings based on hardware capabilities
        fp16=not torch.cuda.is_bf16_supported(),  # Use fp16 if bf16 not available
        bf16=torch.cuda.is_bf16_supported(),  # Prefer bf16 when available
        optim=config.optim,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        # Logging and output configuration
        logging_steps=config.logging_steps,
        output_dir=output_dir,
        report_to="wandb" if config.enable_wandb else None,
        seed=config.seed,
    )


def check_for_existing_checkpoint(paths: dict):
    """
    Check if there's an existing checkpoint to resume training from.

    This enables resumable training, which is crucial for long-running experiments
    that might be interrupted by infrastructure issues or resource limits.
    """
    checkpoint_dir = paths["checkpoints"]
    if not checkpoint_dir.exists():
        return None

    # Look for the most recent checkpoint directory
    checkpoints = list(checkpoint_dir.glob("checkpoint-*"))
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda p: int(p.name.split("-")[1]))
        print(f"Found existing checkpoint: {latest_checkpoint}")
        return str(latest_checkpoint)

    return None


@app.local_entrypoint()
def upload(
    experiment_name: str,
    base_model_name: str,
    hf_repo_name: str,
    max_seq_length: int = 32768,
):
    """
    Upload a trained model to Hugging Face Hub.

    Usage:
        modal run post_training_modal_unsloth.py::upload \
            --experiment-name "Qwen3-32B-r16-20240123-120000" \
            --base-model-name "unsloth/Qwen3-32B" \
            --hf-repo-name "your-username/your-model-name" \
            --max-seq-length 32768 \
            --quantization-method "q4_k_m"  # Optional

    Args:
        experiment_name: The experiment name from training
        base_model_name: Base model used for training (e.g., "unsloth/Qwen3-32B")
        hf_repo_name: Target Hugging Face repo (e.g., "username/model-name")
        max_seq_length: Maximum sequence length used during training (default: 32768)
    """
    print(f"Starting merge and upload for experiment: {experiment_name}")
    print(f"Base model: {base_model_name}")
    print(f"Target repository: {hf_repo_name}")

    result = merge_and_upload.remote(
        experiment_name=experiment_name,
        hf_repo_name=hf_repo_name,
        max_seq_length=max_seq_length,
    )

    print(f"✓ Model successfully uploaded to: https://huggingface.co/{result}")
