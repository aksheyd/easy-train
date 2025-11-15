"""Reward functions for RL training."""

from typing import List


def length_preference_reward(response: str, target_length: int = 150) -> float:
    """
    Simple length-based reward function.

    Educational note:
    ----------------
    This is a simple reward that encourages responses near target_length.
    In practice, you'd use more sophisticated rewards like:
    - Human preference feedback (RLHF)
    - Reward model trained on preference data
    - Task-specific metrics (accuracy, helpfulness, etc.)

    The negative absolute difference penalizes both too-short and too-long responses.

    Args:
        response: The generated response text
        target_length: Target length in words

    Returns:
        Reward value (higher is better)
    """
    response_length = len(response.split())
    return -abs(response_length - target_length) / 100.0  # Normalize to reasonable scale


def compute_group_rewards(responses: List[str], reward_type: str, **kwargs) -> List[float]:
    """
    Compute rewards for a group of responses.

    Educational note:
    ----------------
    GRPO (Group Relative Policy Optimization) centers rewards within each group.
    This makes training more stable by removing absolute scale of rewards.

    For preference learning with group_size=2:
    - Generate 2 responses per prompt
    - Assign relative rewards (e.g., +0.5 to better, -0.5 to worse)
    - This teaches the model which response is preferred

    Args:
        responses: List of generated responses
        reward_type: Type of reward function to use
        **kwargs: Additional arguments for reward function

    Returns:
        List of centered rewards

    Raises:
        ValueError: If reward_type is unknown
        NotImplementedError: If reward model is not implemented
    """
    if reward_type == "length_preference":
        target_length = kwargs.get("target_length", 150)
        raw_rewards = [length_preference_reward(r, target_length) for r in responses]

    elif reward_type == "reward_model":
        # Educational: In production RLHF, you'd use a trained reward model here
        # reward_model = load_reward_model(kwargs["reward_model_path"])
        # raw_rewards = [reward_model.score(r) for r in responses]
        raise NotImplementedError("Reward model not implemented yet")

    else:
        raise ValueError(f"Unknown reward_type: {reward_type}")

    # Center rewards within group (GRPO)
    # Educational: This removes the absolute scale and focuses on relative preference
    mean_reward = sum(raw_rewards) / len(raw_rewards)
    centered_rewards = [r - mean_reward for r in raw_rewards]

    return centered_rewards
