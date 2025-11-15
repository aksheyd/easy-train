"""
Main orchestrator for the post-training pipeline.

Educational note:
----------------
This script ties everything together, running:
1. SFT: Supervised fine-tuning on conversational data
2. RL: Reinforcement learning with importance sampling
3. Deployment: Download, merge, and deploy to Modal

You can run individual stages or the full pipeline.
"""

import argparse
from pathlib import Path
from dotenv import load_dotenv

from config.base_config import SFTConfig, RLConfig, InferenceConfig
from config.default_configs import (
    get_tiny_test_config,
    get_quick_experiment_config,
    get_full_training_config,
)
from data.prepare import validate_jsonl, count_examples
from sft.train import train_sft
from rl.train import train_rl
from inference.download import download_checkpoint
from inference.merge_adapter import merge_lora_adapter
from inference.modal_deploy import create_modal_app, create_client_example


def main():
    """Main entry point for the post-training pipeline."""
    parser = argparse.ArgumentParser(
        description="Tinker Post-Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline with default config
  python main.py --mode all --config quick

  # Run only SFT
  python main.py --mode sft --data my_conversations.jsonl

  # Run RL after SFT
  python main.py --mode rl --sft-checkpoint tinker://...

  # Deploy existing checkpoint
  python main.py --mode deploy --checkpoint tinker://...

  # Validate data only
  python main.py --mode validate --data my_conversations.jsonl
        """,
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        choices=["all", "sft", "rl", "deploy", "validate"],
        default="all",
        help="Pipeline stage to run",
    )

    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        choices=["tiny", "quick", "full", "custom"],
        default="quick",
        help="Configuration preset to use",
    )

    # Data
    parser.add_argument("--data", type=str, help="Path to JSONL data file (overrides config)")

    # Checkpoints
    parser.add_argument(
        "--sft-checkpoint", type=str, help="SFT checkpoint path (for RL mode)"
    )

    parser.add_argument("--checkpoint", type=str, help="Checkpoint path (for deploy mode)")

    args = parser.parse_args()

    # Load environment variables (TINKER_API_KEY, HF_TOKEN, etc.)
    load_dotenv()

    print("=" * 60)
    print("TINKER POST-TRAINING PIPELINE")
    print("=" * 60)

    # Load configuration
    if args.config == "tiny":
        configs = get_tiny_test_config()
    elif args.config == "quick":
        configs = get_quick_experiment_config()
    elif args.config == "full":
        configs = get_full_training_config()
    else:
        # Custom config - use defaults
        configs = {
            "sft": SFTConfig(),
            "rl": RLConfig(),
            "inference": InferenceConfig(),
        }

    # Override data path if provided
    if args.data:
        configs["sft"].data_path = args.data

    # Validate data (optional but recommended)
    if args.mode in ["all", "sft", "validate"]:
        print(f"\n🔍 Validating data: {configs['sft'].data_path}")
        if validate_jsonl(configs["sft"].data_path):
            stats = count_examples(configs["sft"].data_path)
            print(f"  Examples: {stats['num_examples']}")
            print(f"  Avg turns: {stats['avg_turns']:.1f}")
            print(f"  Turn range: {stats['min_turns']}-{stats['max_turns']}")
        else:
            print("❌ Validation failed! Please fix data format.")
            return

        if args.mode == "validate":
            print("\n✅ Validation complete!")
            return

    # Run pipeline stages
    sft_checkpoint = args.sft_checkpoint
    rl_checkpoint = args.checkpoint

    if args.mode in ["all", "sft"]:
        print(f"\n{'=' * 60}")
        print("STAGE 1: SUPERVISED FINE-TUNING")
        print(f"{'=' * 60}")
        sft_checkpoint = train_sft(configs["sft"])
        print(f"\n✅ SFT complete! Checkpoint: {sft_checkpoint}")

        if args.mode == "sft":
            print("\n🎉 Pipeline complete!")
            return

    if args.mode in ["all", "rl"]:
        if not sft_checkpoint:
            print("❌ Error: Need SFT checkpoint for RL. Provide --sft-checkpoint")
            return

        print(f"\n{'=' * 60}")
        print("STAGE 2: REINFORCEMENT LEARNING")
        print(f"{'=' * 60}")
        configs["rl"].load_checkpoint_path = sft_checkpoint
        rl_checkpoint = train_rl(configs["rl"], sft_checkpoint)
        print(f"\n✅ RL complete! Checkpoint: {rl_checkpoint}")

        if args.mode == "rl":
            print("\n🎉 Pipeline complete!")
            return

    if args.mode in ["all", "deploy"]:
        if not rl_checkpoint:
            print("❌ Error: Need checkpoint for deployment. Provide --checkpoint")
            return

        print(f"\n{'=' * 60}")
        print("STAGE 3: MODEL DEPLOYMENT")
        print(f"{'=' * 60}")

        # Download checkpoint
        adapter_path = download_checkpoint(rl_checkpoint, configs["inference"].output_dir)

        # Merge adapter
        if configs["inference"].merge_adapters:
            merged_path = merge_lora_adapter(
                base_model_name=configs["inference"].base_model_name,
                adapter_path=adapter_path,
                output_path="merged_model",
                upload_to_hf=configs["inference"].upload_to_hf,
                hf_repo_name=configs["inference"].hf_repo_name,
            )

        # Create Modal deployment
        create_modal_app(configs["inference"])
        create_client_example()

        print(f"\n✅ Deployment setup complete!")
        print(f"\nNext steps:")
        print(f"  1. Deploy to Modal: modal deploy modal_app.py")
        print(f"  2. Test with client: python modal_client.py")

    print(f"\n{'=' * 60}")
    print("🎉 PIPELINE COMPLETE!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
