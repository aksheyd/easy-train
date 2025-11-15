"""Supervised fine-tuning training loop."""

import tinker
from tinker_cookbook import renderers
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.supervised.data import TrainOnWhat
from tinker_cookbook.tokenizer_utils import get_tokenizer

from data.builders import create_dataset_builder
from utils.logging_utils import setup_wandb, log_metrics


def train_sft(config):
    """
    Supervised fine-tuning with LoRA.

    Educational notes:
    ----------------
    1. TrainOnWhat.LAST_ASSISTANT_MESSAGE:
       - Only applies loss to the final assistant response
       - More efficient: doesn't waste compute on context
       - Prevents overfitting to user/system messages
       - Standard practice for chat models

    2. LoRA Learning Rate:
       - Typically 10x higher than full fine-tuning
       - For 1B model: ~5e-4 (vs 5e-5 for full fine-tune)
       - LoRA has fewer parameters, needs stronger signal

    3. Cross-entropy loss:
       - Standard language modeling objective
       - Maximizes likelihood of correct next token
       - NLL (negative log likelihood) metric tracks this

    Args:
        config: SFTConfig object with training parameters

    Returns:
        Final checkpoint path (tinker://{uuid}/sampler_weights/final)
    """
    print("=" * 50)
    print("SUPERVISED FINE-TUNING")
    print("=" * 50)

    # Setup logging
    if config.wandb_project:
        setup_wandb(config.wandb_project, config, stage="sft")

    # Create Tinker service client
    service_client = tinker.ServiceClient()

    # Create training client with LoRA configuration
    print(f"\n📦 Loading model: {config.model_name}")
    training_client = service_client.create_training_client(
        base_model=config.model_name,
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
    )

    # Get tokenizer and renderer
    tokenizer = get_tokenizer(config.model_name)
    renderer_name = get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    print(f"Using renderer: {renderer_name}")

    # Create dataset
    print(f"\n📊 Loading data: {config.data_path}")
    dataset_builder = create_dataset_builder(config)
    dataset = dataset_builder.build(
        renderer=renderer,
        train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,  # Educational: Only train on last response
    )

    print(f"Dataset size: {len(dataset)} examples")

    # Training loop
    print(f"\n🚀 Starting training...")
    print(f"  - Learning rate: {config.learning_rate}")
    print(f"  - Batch size: {config.batch_size}")
    print(f"  - Epochs: {config.num_epochs}")
    print(f"  - LoRA rank: {config.lora_rank}")

    step = 0
    total_steps = (len(dataset) // config.batch_size) * config.num_epochs

    for epoch in range(config.num_epochs):
        print(f"\n📖 Epoch {epoch + 1}/{config.num_epochs}")

        for batch_idx, batch_start in enumerate(range(0, len(dataset), config.batch_size)):
            batch_end = min(batch_start + config.batch_size, len(dataset))
            batch = dataset[batch_start:batch_end]

            # Forward-backward pass
            # Educational: Cross-entropy loss measures how well model predicts next token
            metrics = training_client.forward_backward(
                batch=batch,
                loss_fn="cross_entropy",
            )

            # Optimizer step
            # Educational: Adam optimizer with standard betas (0.9, 0.999)
            training_client.optim_step(
                learning_rate=config.learning_rate,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-8,
            )

            step += 1

            # Log metrics
            if step % 10 == 0:
                nll = metrics.get("nll", 0.0)
                print(f"  Step {step}/{total_steps} - NLL: {nll:.4f}")

                if config.wandb_project:
                    log_metrics({"train/nll": nll, "step": step})

            # Evaluate
            if step % config.eval_every == 0:
                # Educational: Evaluation on held-out data
                # In production, you'd have a separate eval dataset
                print(f"  💡 Evaluation at step {step} (skipped for brevity)")

            # Save checkpoint
            if step % config.save_every == 0:
                checkpoint_name = f"step_{step}"
                checkpoint_path = training_client.save_checkpoint(
                    name=checkpoint_name,
                    log_path=config.log_path,
                )
                print(f"  💾 Saved checkpoint: {checkpoint_path}")

    # Save final checkpoint
    print(f"\n✅ Training complete!")
    final_checkpoint = training_client.save_checkpoint(
        name="final",
        log_path=config.log_path,
    )

    print(f"\n📦 Final checkpoint: {final_checkpoint}")
    print(f"   This checkpoint will be used for RL training")

    return final_checkpoint
