"""
Easy Train - Simple fine-tuning with Tinker primitives only

Just put your JSONL file in the data/ folder and run:
    python train.py

This will:
1. Do supervised fine-tuning (SFT) on your data
2. Do reinforcement learning (RL) to improve the model

JSONL format:
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
"""

import json
import os
from pathlib import Path
from typing import List, Dict
import numpy as np
import tinker
from tinker import types


# ============================================================================
# CONFIGURATION - Edit these to customize your training
# ============================================================================

CONFIG = {
    # Model settings
    "model_name": "meta-llama/Llama-3.2-1B",  # Base model to fine-tune
    "lora_rank": 32,  # LoRA rank (higher = more parameters)
    # Data settings
    "data_folder": "data",  # Folder containing JSONL files
    "max_examples": None,  # None = use all data, or set a number to limit
    # SFT settings
    "sft_learning_rate": 5e-4,  # Learning rate for supervised fine-tuning
    "sft_steps": 100,  # Number of SFT training steps
    "sft_batch_size": 4,  # Batch size for SFT
    # RL settings
    "rl_learning_rate": 1e-5,  # Learning rate for RL (lower than SFT)
    "rl_steps": 50,  # Number of RL training steps
    "rl_num_samples": 4,  # Number of responses to generate per prompt
    "rl_max_tokens": 200,  # Max tokens to generate
    "rl_temperature": 0.8,  # Sampling temperature
}


# ============================================================================
# DATA LOADING
# ============================================================================


def load_data(data_folder: str, max_examples: int = None) -> List[Dict]:
    """Load all JSONL files from data folder."""
    data = []
    data_path = Path(data_folder)

    if not data_path.exists():
        raise FileNotFoundError(f"Data folder not found: {data_folder}")

    jsonl_files = list(data_path.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No .jsonl files found in {data_folder}")

    print(f"📂 Loading data from {len(jsonl_files)} file(s)...")

    for jsonl_file in jsonl_files:
        with open(jsonl_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
                    if max_examples and len(data) >= max_examples:
                        return data

    print(f"✓ Loaded {len(data)} examples")
    return data


def render_conversation(messages: List[Dict], tokenizer) -> List[int]:
    """
    Convert messages to tokens using a simple chat template.
    This is a minimal implementation - Tinker Cookbook has more sophisticated renderers.
    """
    text = ""

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "system":
            text += f"System: {content}\n\n"
        elif role == "user":
            text += f"User: {content}\n\n"
        elif role == "assistant":
            text += f"Assistant: {content}\n\n"

    return tokenizer.encode(text, add_special_tokens=True)


def prepare_sft_example(messages: List[Dict], tokenizer) -> types.Datum:
    """
    Prepare a single example for supervised fine-tuning.
    We only train on the last assistant message.
    """
    # Split messages into context (everything except last assistant) and target (last assistant)
    context_messages = []
    target_content = None

    for i, msg in enumerate(messages):
        if msg["role"] == "assistant" and i == len(messages) - 1:
            # This is the last message and it's from assistant - this is our target
            target_content = msg["content"]
        else:
            context_messages.append(msg)

    if target_content is None:
        # If no assistant message at end, just use last message as target
        target_content = messages[-1]["content"]
        context_messages = messages[:-1]

    # Encode context
    context_text = ""
    for msg in context_messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            context_text += f"System: {content}\n\n"
        elif role == "user":
            context_text += f"User: {content}\n\n"
        elif role == "assistant":
            context_text += f"Assistant: {content}\n\n"

    context_text += "Assistant: "  # Add prompt for assistant response
    context_tokens = tokenizer.encode(context_text, add_special_tokens=True)

    # Encode target (what we want to train on)
    target_tokens = tokenizer.encode(target_content + "\n\n", add_special_tokens=False)

    # Combine into full sequence
    all_tokens = context_tokens + target_tokens

    # Create weights: 0 for context, 1 for target
    weights = [0.0] * len(context_tokens) + [1.0] * len(target_tokens)

    # Shift for next-token prediction
    input_tokens = all_tokens[:-1]
    target_tokens_shifted = all_tokens[1:]
    weights_shifted = weights[1:]

    return types.Datum(
        model_input=types.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "target_tokens": target_tokens_shifted,
            "weights": weights_shifted,
        },
    )


# ============================================================================
# SUPERVISED FINE-TUNING (SFT)
# ============================================================================


def train_sft(
    training_client: tinker.TrainingClient, data: List[Dict], config: Dict
) -> str:
    """
    Supervised fine-tuning: Train the model to predict assistant responses.
    Returns: checkpoint path
    """
    print("\n" + "=" * 60)
    print("SUPERVISED FINE-TUNING")
    print("=" * 60)

    tokenizer = training_client.get_tokenizer()

    # Prepare all examples
    print("📊 Preparing training data...")
    examples = [prepare_sft_example(item["messages"], tokenizer) for item in data]
    print(f"✓ Prepared {len(examples)} examples")

    # Training loop
    print(f"\n🚀 Training for {config['sft_steps']} steps...")
    print(f"  Learning rate: {config['sft_learning_rate']}")
    print(f"  Batch size: {config['sft_batch_size']}")

    batch_size = config["sft_batch_size"]

    for step in range(config["sft_steps"]):
        # Sample a batch
        batch_indices = np.random.choice(len(examples), size=batch_size, replace=False)
        batch = [examples[i] for i in batch_indices]

        # Forward-backward pass
        fwd_result = training_client.forward_backward(batch, loss_fn="cross_entropy")

        # Optimizer step
        training_client.optim_step(
            types.AdamParams(
                learning_rate=config["sft_learning_rate"],
                beta1=0.9,
                beta2=0.95,
            )
        )

        # Get result and log
        result = fwd_result.result()
        loss = result.metrics.get("loss:sum", 0.0)

        if step % 10 == 0:
            print(f"  Step {step}/{config['sft_steps']} - Loss: {loss:.4f}")

    # Save checkpoint
    print("\n💾 Saving SFT checkpoint...")
    checkpoint = training_client.save_weights_for_sampler(name="sft_final")
    checkpoint_path = checkpoint.result().path
    print(f"✓ Checkpoint saved: {checkpoint_path}")

    return checkpoint_path


# ============================================================================
# REINFORCEMENT LEARNING (RL)
# ============================================================================


def extract_prompts(data: List[Dict]) -> List[str]:
    """Extract user prompts from conversation data for RL."""
    prompts = []
    for item in data:
        for msg in item["messages"]:
            if msg["role"] == "user":
                prompts.append(msg["content"])
                break  # Just take first user message
    return prompts


def compute_simple_reward(response_text: str) -> float:
    """
    Simple reward function - you can customize this!

    This example rewards:
    - Longer responses (more informative)
    - Responses with good structure (punctuation)
    - Responses that avoid very short replies
    """
    # Length reward (normalize to 0-1 range)
    length_reward = min(len(response_text) / 200.0, 1.0)

    # Structure reward (has punctuation)
    has_period = 1.0 if "." in response_text else 0.0

    # Penalize very short responses
    too_short_penalty = -1.0 if len(response_text) < 20 else 0.0

    total_reward = length_reward + has_period + too_short_penalty

    return total_reward


def train_rl(
    service_client: tinker.ServiceClient,
    sft_checkpoint: str,
    data: List[Dict],
    config: Dict,
) -> str:
    """
    Reinforcement learning: Improve the model using rewards.
    Returns: final checkpoint path
    """
    print("\n" + "=" * 60)
    print("REINFORCEMENT LEARNING")
    print("=" * 60)

    # Load the SFT checkpoint as starting point
    print(f"📦 Loading SFT checkpoint...")
    training_client = service_client.create_lora_training_client(
        base_model=config["model_name"],
        rank=config["lora_rank"],
    )
    training_client.load_state(sft_checkpoint)

    tokenizer = training_client.get_tokenizer()

    # Extract prompts for RL
    prompts = extract_prompts(data)
    print(f"📊 Using {len(prompts)} prompts for RL")

    # RL training loop
    print(f"\n🚀 Starting RL training for {config['rl_steps']} steps...")
    print(f"  Learning rate: {config['rl_learning_rate']}")
    print(f"  Samples per prompt: {config['rl_num_samples']}")

    for step in range(config["rl_steps"]):
        # 1. Create sampling client from current policy
        sampling_checkpoint = training_client.save_weights_for_sampler(
            name=f"rl_step_{step}"
        )
        sampling_path = sampling_checkpoint.result().path
        sampling_client = service_client.create_sampling_client(
            model_path=sampling_path
        )

        # 2. Sample a prompt
        prompt_text = prompts[step % len(prompts)]
        prompt_formatted = f"User: {prompt_text}\n\nAssistant: "
        prompt_tokens = tokenizer.encode(prompt_formatted, add_special_tokens=True)
        prompt_input = types.ModelInput.from_ints(prompt_tokens)

        # 3. Generate multiple responses
        sampling_params = types.SamplingParams(
            max_tokens=config["rl_max_tokens"],
            temperature=config["rl_temperature"],
            top_p=0.9,
        )

        sample_result = sampling_client.sample(
            prompt=prompt_input,
            sampling_params=sampling_params,
            num_samples=config["rl_num_samples"],
        )
        result = sample_result.result()

        # 4. Compute rewards and advantages
        rewards = []
        response_texts = []

        for seq in result.sequences:
            response_text = tokenizer.decode(seq.tokens)
            response_texts.append(response_text)
            reward = compute_simple_reward(response_text)
            rewards.append(reward)

        # Compute advantages (center rewards)
        mean_reward = np.mean(rewards)
        advantages = [r - mean_reward for r in rewards]

        # 5. Prepare training data
        training_data = []
        for seq, advantage in zip(result.sequences, advantages):
            # Compute logprobs for this sequence
            full_tokens = prompt_tokens + seq.tokens

            # For RL, we need logprobs of the actions taken
            # We'll use the sequence logprobs if available, otherwise approximate
            if seq.logprobs is not None:
                logprobs = seq.logprobs
            else:
                # Fallback: use uniform logprobs (not ideal but simple)
                logprobs = [-1.0] * len(seq.tokens)

            # Create datum for importance sampling loss
            datum = types.Datum(
                model_input=types.ModelInput.from_ints(prompt_tokens + seq.tokens[:-1]),
                loss_fn_inputs={
                    "target_tokens": seq.tokens,
                    "logprobs": logprobs,
                    "advantages": [advantage] * len(seq.tokens),
                },
            )
            training_data.append(datum)

        # 6. Update policy
        fwd_result = training_client.forward_backward(
            training_data, loss_fn="importance_sampling"
        )
        training_client.optim_step(
            types.AdamParams(
                learning_rate=config["rl_learning_rate"],
                beta1=0.9,
                beta2=0.95,
            )
        )

        # Log progress
        if step % 10 == 0:
            avg_reward = np.mean(rewards)
            print(f"  Step {step}/{config['rl_steps']} - Avg Reward: {avg_reward:.3f}")
            if step % 20 == 0 and response_texts:
                print(f"    Sample response: {response_texts[0][:100]}...")

    # Save final checkpoint
    print("\n💾 Saving final RL checkpoint...")
    final_checkpoint = training_client.save_weights_for_sampler(name="rl_final")
    checkpoint_path = final_checkpoint.result().path
    print(f"✓ Final checkpoint: {checkpoint_path}")

    return checkpoint_path


# ============================================================================
# MAIN
# ============================================================================


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("EASY TRAIN - Simple Fine-Tuning with Tinker")
    print("=" * 60)

    # Load data
    data = load_data(CONFIG["data_folder"], CONFIG["max_examples"])

    # Initialize Tinker
    print("\n🔧 Initializing Tinker client...")
    service_client = tinker.ServiceClient()

    # Create training client
    print(f"📦 Loading model: {CONFIG['model_name']}")
    training_client = service_client.create_lora_training_client(
        base_model=CONFIG["model_name"],
        rank=CONFIG["lora_rank"],
    )
    print(f"✓ Model loaded with LoRA rank {CONFIG['lora_rank']}")

    # Run SFT
    sft_checkpoint = train_sft(training_client, data, CONFIG)

    # Run RL
    final_checkpoint = train_rl(service_client, sft_checkpoint, data, CONFIG)

    # Done!
    print("\n" + "=" * 60)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nFinal model checkpoint: {final_checkpoint}")
    print("\nYou can now use this checkpoint for inference!")


if __name__ == "__main__":
    main()
