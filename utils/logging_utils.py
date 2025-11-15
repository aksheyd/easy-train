"""Logging utilities for Weights & Biases integration."""

from typing import Dict, Any, Optional


def setup_wandb(project: str, config: Any, stage: str):
    """
    Initialize Weights & Biases logging.

    Args:
        project: W&B project name
        config: Configuration object to log
        stage: Stage name (e.g., "sft", "rl")
    """
    try:
        import wandb

        wandb.init(
            project=project,
            config=config.__dict__,
            name=f"{stage}_{config.log_path.replace('/', '_')}",
            tags=[stage],
        )
        print(f"📊 W&B initialized: {project}")
    except ImportError:
        print("⚠️  wandb not installed. Skipping W&B logging.")
        print("   Install with: pip install wandb")


def log_metrics(metrics: Dict[str, Any]):
    """
    Log metrics to W&B.

    Args:
        metrics: Dictionary of metric names and values
    """
    try:
        import wandb

        wandb.log(metrics)
    except ImportError:
        # Silently skip if wandb is not installed
        pass
