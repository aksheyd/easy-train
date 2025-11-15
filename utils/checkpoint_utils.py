"""Checkpoint path utilities."""

from typing import Dict


def parse_checkpoint_path(checkpoint_path: str) -> Dict[str, str]:
    """
    Parse Tinker checkpoint path.

    Format: tinker://{uuid}/sampler_weights/{name}
    or: tinker://{uuid}/weights/{name}

    Args:
        checkpoint_path: Tinker checkpoint path string

    Returns:
        Dictionary with parsed components (uuid, type, name)
    """
    parts = checkpoint_path.replace("tinker://", "").split("/")
    return {
        "uuid": parts[0],
        "type": parts[1],  # "weights" or "sampler_weights"
        "name": parts[2] if len(parts) > 2 else "final",
    }


def get_checkpoint_path(
    log_path: str, name: str, checkpoint_type: str = "sampler"
) -> str:
    """
    Construct checkpoint path.

    Educational note:
    ----------------
    - sampler_weights: For inference only
    - weights: Full training state (for resuming)

    Args:
        log_path: Log path for the training run
        name: Checkpoint name
        checkpoint_type: Type of checkpoint ("sampler" or "full")

    Returns:
        Tinker checkpoint path string
    """
    # This would be returned by Tinker's save_checkpoint()
    # Placeholder for documentation purposes
    weight_type = "sampler_weights" if checkpoint_type == "sampler" else "weights"
    return f"tinker://{{uuid}}/{weight_type}/{name}"
