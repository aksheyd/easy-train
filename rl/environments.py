"""RL environments for preference-based learning."""

from tinker_cookbook.rl.types import Env, EnvGroupBuilder
from typing import List, Dict, Any


class PreferenceEnv(Env):
    """
    Simple preference-based environment.

    Educational note:
    ----------------
    In RL, an Environment represents the task the agent (model) is trying to solve.

    For conversational preference learning:
    - State (observation): A user prompt
    - Action: The model's generated response
    - Reward: How good the response is (from reward function)

    The agent learns to take actions (generate responses) that maximize reward.
    """

    def __init__(self, prompt: str):
        """
        Initialize environment with a prompt.

        Args:
            prompt: The user prompt for this environment
        """
        self.prompt = prompt
        self.done = False

    def initial_observation(self) -> Dict[str, Any]:
        """
        Return the initial prompt as observation.

        Returns:
            Dictionary with messages list containing the user prompt
        """
        return {"messages": [{"role": "user", "content": self.prompt}]}

    def step(self, action: str) -> tuple:
        """
        Take an action (generate response) and return reward.

        For single-turn preference, we're done after one response.

        Args:
            action: The generated response

        Returns:
            Tuple of (reward, info_dict)
        """
        # Reward will be computed later by the group
        # For now, just mark as done
        self.done = True
        return 0.0, {"done": True}


class PreferenceEnvGroupBuilder(EnvGroupBuilder):
    """
    Creates groups of preference environments.

    Educational note:
    ----------------
    GRPO generates multiple responses (group_size) for each prompt.
    These are then ranked/scored, and rewards are centered within the group.

    For group_size=2 (most common for preferences):
    - Generate 2 responses for the same prompt
    - Compare them and assign relative rewards
    - Train policy to prefer better response
    """

    def __init__(self, prompts: List[str], group_size: int = 2):
        """
        Initialize environment group builder.

        Args:
            prompts: List of user prompts
            group_size: Number of responses to generate per prompt
        """
        self.prompts = prompts
        self.group_size = group_size

    def make_envs(self) -> List[PreferenceEnv]:
        """
        Create group_size copies of each prompt's environment.

        Returns:
            List of PreferenceEnv instances
        """
        envs = []
        for prompt in self.prompts:
            for _ in range(self.group_size):
                envs.append(PreferenceEnv(prompt))
        return envs

    def compute_group_rewards(
        self, responses: List[str], reward_fn, **kwargs
    ) -> List[float]:
        """
        Compute rewards for all responses in the group.

        Responses are grouped: [prompt1_resp1, prompt1_resp2, prompt2_resp1, prompt2_resp2, ...]

        Args:
            responses: List of all generated responses
            reward_fn: Reward function (not used, we import from rewards.py)
            **kwargs: Additional arguments for reward computation

        Returns:
            List of rewards for each response
        """
        from rl.rewards import compute_group_rewards

        all_rewards = []

        # Process each prompt's group
        for i in range(0, len(responses), self.group_size):
            group_responses = responses[i : i + self.group_size]
            group_rewards = compute_group_rewards(
                group_responses,
                reward_type=kwargs.get("reward_type", "length_preference"),
                **kwargs,
            )
            all_rewards.extend(group_rewards)

        return all_rewards
