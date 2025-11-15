"""RL training loop with importance sampling."""

import tinker
from tinker_cookbook import renderers
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.rl.rollouts import rollout_trajectories
from tinker_cookbook.rl.data_processing import assemble_training_data

from rl.environments import PreferenceEnvGroupBuilder
from utils.logging_utils import setup_wandb, log_metrics


def train_rl(config, sft_checkpoint_path: str):
    """
    RL training with importance sampling.

    Educational notes:
    ----------------
    1. Importance Sampling:
       - Simpler than PPO, good for beginners
       - Computes loss as: -advantage * log_prob
       - Advantages tell us if action was better/worse than expected
       - High advantage = increase probability of that action

    2. GRPO (Group Relative Policy Optimization):
       - Centers rewards within each group of responses
       - Removes absolute reward scale (only relative matters)
       - More stable training

    3. KL Penalty:
       - Prevents policy from changing too fast
       - Keeps it close to initial policy (SFT model)
       - Controlled by kl_penalty_coef

    4. Why RL after SFT:
       - SFT teaches language and basic capabilities
       - RL fine-tunes for specific behaviors (preferences, accuracy, etc.)
       - Together they produce high-quality aligned models

    Args:
        config: RLConfig object with training parameters
        sft_checkpoint_path: Path to SFT checkpoint to load from

    Returns:
        Final checkpoint path
    """
    print("=" * 50)
    print("REINFORCEMENT LEARNING")
    print("=" * 50)

    # Setup logging
    if config.wandb_project:
        setup_wandb(config.wandb_project, config, stage="rl")

    # Create Tinker service client
    service_client = tinker.ServiceClient()

    # Load SFT checkpoint as initial policy
    print(f"\n📦 Loading SFT checkpoint: {sft_checkpoint_path}")
    training_client = service_client.create_lora_training_client(
        base_model=config.base_model_name,
        load_checkpoint_path=sft_checkpoint_path,  # Start from SFT weights
        rank=config.lora_rank,
    )

    # Get tokenizer and renderer
    tokenizer = get_tokenizer(config.base_model_name)
    renderer_name = get_recommended_renderer_name(config.base_model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    # Load prompts for RL (in practice, load from file)
    # Educational: These are the prompts the model will practice on
    prompts = [
        "Explain what photosynthesis is.",
        "What are the benefits of exercise?",
        "How does the internet work?",
        "Describe the water cycle.",
        "What is artificial intelligence?",
        "How do computers store information?",
        "Explain the concept of gravity.",
        "What causes seasons on Earth?",
    ]

    print(f"\n📊 RL Dataset: {len(prompts)} prompts")
    print(f"  - Group size: {config.group_size}")
    print(f"  - Total rollouts per batch: {len(prompts) * config.group_size}")

    # Training loop
    print(f"\n🚀 Starting RL training...")
    print(f"  - Learning rate: {config.learning_rate}")
    print(f"  - Loss function: {config.loss_fn}")
    print(f"  - Reward type: {config.reward_type}")

    for step in range(config.num_steps):
        print(f"\n📈 RL Step {step + 1}/{config.num_steps}")

        # 1. Create sampling client from current weights
        # Educational: We sample from the latest policy
        sampling_client = service_client.create_sampling_client_from_training_client(
            training_client
        )

        # 2. Create environment group
        # Use a subset of prompts for this batch
        batch_prompts = prompts[: config.groups_per_batch]
        env_group_builder = PreferenceEnvGroupBuilder(
            prompts=batch_prompts,
            group_size=config.group_size,
        )

        # 3. Rollout trajectories
        # Educational: Generate responses and collect (state, action, reward) data
        print(f"  🎲 Rolling out trajectories...")
        trajectories = rollout_trajectories(
            env_group_builder=env_group_builder,
            sampling_client=sampling_client,
            renderer=renderer,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
        )

        # 4. Compute rewards
        responses = [traj.final_response for traj in trajectories]
        rewards = env_group_builder.compute_group_rewards(
            responses,
            reward_fn=None,  # Handled inside compute_group_rewards
            reward_type=config.reward_type,
            target_length=getattr(config, "target_length", 150),
        )

        avg_reward = sum(rewards) / len(rewards)
        print(f"  💰 Average reward: {avg_reward:.4f}")

        # 5. Assemble training data
        # Educational: Convert trajectories to (tokens, logprobs, advantages, mask)
        training_data = assemble_training_data(
            trajectories=trajectories,
            rewards=rewards,
            renderer=renderer,
        )

        # 6. Train on this batch
        # Educational: Update policy to increase probability of high-reward actions
        metrics = training_client.forward_backward(
            batch=training_data,
            loss_fn=config.loss_fn,  # "importance_sampling" or "ppo"
            kl_penalty_coef=config.kl_penalty_coef,
        )

        training_client.optim_step(
            learning_rate=config.learning_rate,
            beta1=0.9,
            beta2=0.999,
        )

        # Log metrics
        loss = metrics.get("loss", 0.0)
        kl_div = metrics.get("kl_divergence", 0.0)
        print(f"  📊 Loss: {loss:.4f}, KL: {kl_div:.4f}")

        if config.wandb_project:
            log_metrics(
                {
                    "rl/loss": loss,
                    "rl/kl_divergence": kl_div,
                    "rl/avg_reward": avg_reward,
                    "step": step,
                }
            )

        # Save checkpoint
        if (step + 1) % config.save_every == 0:
            checkpoint_path = training_client.save_checkpoint(
                name=f"step_{step + 1}",
                log_path=config.log_path,
            )
            print(f"  💾 Saved checkpoint: {checkpoint_path}")

    # Save final checkpoint
    print(f"\n✅ RL training complete!")
    final_checkpoint = training_client.save_checkpoint(
        name="final",
        log_path=config.log_path,
    )

    print(f"\n📦 Final RL checkpoint: {final_checkpoint}")
    return final_checkpoint
