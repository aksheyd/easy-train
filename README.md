# Tinker Post-Training Pipeline

A complete, educational pipeline for post-training language models using [Tinker](https://www.tinker.so/) by Thinking Machines Lab. This implementation demonstrates the fundamentals of modern post-training techniques including Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) with preference optimization.

## Features

- **Supervised Fine-Tuning (SFT)**: Fine-tune pretrained models on conversational data with LoRA adapters
- **Reinforcement Learning (RL)**: Improve models using importance sampling with preference-based rewards (GRPO)
- **Modal Deployment**: Deploy trained models for serverless inference with automatic scaling
- **Educational Focus**: Extensive comments and documentation explaining post-training fundamentals
- **Flexible Configuration**: Three preset configurations (tiny, quick, full) plus customizable options

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repo-url>
cd easy-train

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env
# Edit .env and add your TINKER_API_KEY
```

### 2. Get Your Tinker Token

1. Sign up at [https://www.tinker.so/](https://www.tinker.so/)
2. Get your API key from the dashboard
3. Add it to your `.env` file:
   ```
   TINKER_API_KEY=your_api_key_here
   ```

### 3. Prepare Data

The pipeline includes example conversational data at `examples/example_data.jsonl`. You can use this to test the pipeline, or create your own JSONL file with this format:

```jsonl
{"messages": [{"role": "user", "content": "Hello!"}, {"role": "assistant", "content": "Hi! How can I help?"}]}
{"messages": [{"role": "user", "content": "Explain AI."}, {"role": "assistant", "content": "AI is..."}]}
```

Validate your data:

```bash
python main.py --mode validate --data examples/example_data.jsonl
```

### 4. Run Training

```bash
# Quick experiment (recommended for learning)
python main.py --mode all --config quick

# Full training
python main.py --mode all --config full

# Tiny test (for debugging)
python main.py --mode all --config tiny
```

### 5. Deploy to Modal (Optional)

After training completes, you can deploy your model:

```bash
# Install Modal
pip install modal

# Setup Modal token
modal token new

# Deploy the generated app
modal deploy modal_app.py

# Test with the generated client
python modal_client.py
```

## Architecture

### Project Structure

```
tinker-posttraining/
├── main.py                     # Main orchestrator script
├── requirements.txt            # Python dependencies
├── .env.example               # Environment variable template
├── README.md                  # This file
│
├── config/                    # Configuration system
│   ├── base_config.py        # Configuration dataclasses
│   └── default_configs.py    # Preset configurations
│
├── data/                      # Data preparation
│   ├── prepare.py            # JSONL validation & utilities
│   └── builders.py           # Dataset builder wrappers
│
├── sft/                       # Supervised fine-tuning
│   └── train.py              # SFT training loop
│
├── rl/                        # Reinforcement learning
│   ├── train.py              # RL training loop
│   ├── environments.py       # Preference-based environments
│   └── rewards.py            # Reward functions
│
├── inference/                 # Inference and deployment
│   ├── download.py           # Download weights from Tinker
│   ├── merge_adapter.py      # Merge LoRA with base model
│   └── modal_deploy.py       # Modal deployment script
│
├── utils/                     # Utilities
│   ├── checkpoint_utils.py   # Checkpoint management
│   └── logging_utils.py      # Logging and metrics
│
└── examples/                  # Examples and samples
    ├── example_data.jsonl    # Sample conversational data
    └── example_configs.py    # Example configuration presets
```

### Training Pipeline

#### SFT Phase
1. Loads pretrained model (e.g., Llama 3.2 1B)
2. Applies LoRA for parameter-efficient fine-tuning
3. Trains on last assistant message only (most efficient)
4. Saves checkpoints to Tinker

#### RL Phase
1. Loads SFT checkpoint as initial policy
2. Generates multiple responses per prompt (GRPO)
3. Computes preference-based rewards
4. Updates policy using importance sampling
5. Saves final checkpoint

#### Deployment
1. Downloads trained LoRA adapter
2. Merges with base model for faster inference
3. Deploys to Modal for serverless scaling

## Usage

### Running Individual Stages

```bash
# Run only SFT
python main.py --mode sft --config quick

# Run only RL (requires SFT checkpoint)
python main.py --mode rl --sft-checkpoint tinker://...

# Deploy existing checkpoint
python main.py --mode deploy --checkpoint tinker://...
```

### Custom Data

```bash
# Use your own JSONL file
python main.py --mode all --config quick --data my_conversations.jsonl
```

### Configuration Presets

1. **Tiny** (`--config tiny`): Minimal training for quick testing
   - Batch size: 8
   - Epochs: 1
   - RL steps: 10

2. **Quick** (`--config quick`): Good for experimentation
   - Batch size: 32
   - Epochs: 1
   - RL steps: 100

3. **Full** (`--config full`): Production-quality training
   - Batch size: 64
   - Epochs: 3
   - RL steps: 500

4. **Custom**: Edit `config/base_config.py` for full control

## Educational Notes

This repository is designed for learning post-training fundamentals:

### Why LoRA?
- Parameter-efficient fine-tuning uses only ~1% of model parameters
- Enables training large models on consumer hardware
- Can be merged with base model for deployment

### Why train on last assistant message only?
- Prevents overfitting to context (user/system messages)
- Focuses compute on what matters: output quality
- Standard practice for chat model fine-tuning

### Why importance sampling?
- Simpler than PPO, excellent for beginners
- Direct policy gradient with advantage weighting
- Good baseline for RL from human feedback (RLHF)

### Why GRPO (Group Relative Policy Optimization)?
- Centering rewards within groups stabilizes training
- Removes absolute reward scale (only relative matters)
- More robust to reward function design

### Why RL after SFT?
- SFT teaches language and basic capabilities
- RL aligns behavior with preferences
- Together they produce high-quality aligned models

## Advanced Topics

### Using HuggingFace Datasets

```python
# In config/base_config.py or custom config
SFTConfig(
    dataset_type="huggingface",
    data_path="HuggingFaceH4/no_robots",
    # ... other params
)
```

### Training Larger Models

```python
# For Llama 3.2 8B
SFTConfig(
    model_name="meta-llama/Llama-3.2-8B",
    batch_size=16,  # Reduce for larger model
    lora_rank=64,   # Increase for larger model
)
```

### Using PPO Instead of Importance Sampling

```python
# In config
RLConfig(
    loss_fn="ppo",
    learning_rate=3e-5,  # Slightly lower for PPO
)
```

### Uploading to HuggingFace

```python
InferenceConfig(
    upload_to_hf=True,
    hf_repo_name="your-username/your-model-name",
)
```

## Troubleshooting

### Out of Memory
- Reduce `batch_size` in config
- Reduce `max_length` (sequence length)
- Use smaller model (e.g., Llama-3.2-1B instead of 8B)

### Low Rewards in RL
- Check reward function (are rewards meaningful?)
- Increase `learning_rate` slightly
- Generate more rollouts per batch
- Try different reward types

### Modal Deployment Fails
- Ensure `modal token new` is set up
- Check GPU type is available in your region
- Verify HuggingFace token for private models

### JSONL Validation Errors
- Ensure each line is valid JSON
- Check for required fields: "messages"
- Verify role is one of: "system", "user", "assistant"
- Use `--mode validate` to check your data

## Resources

- [Tinker Documentation](https://docs.tinker.so/)
- [Tinker Cookbook](https://github.com/thinking-machines/tinker-cookbook)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [GRPO Algorithm](https://arxiv.org/abs/2402.03300)
- [Modal Documentation](https://modal.com/docs)
- [Llama 3.2 Models](https://huggingface.co/meta-llama)

## Contributing

Contributions are welcome! This project is designed to be educational, so:
- Add more educational comments
- Improve documentation
- Add more example configurations
- Share your training results

## License

MIT

## Acknowledgments

Built with [Tinker](https://www.tinker.so/) by Thinking Machines Lab - making post-training accessible to everyone.
