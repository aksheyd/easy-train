# Easy Train - Simple Fine-Tuning with Tinker

The simplest way to fine-tune an LLM. Just put your JSONL file in the `data/` folder and run!

## Quick Start

### 1. Install Dependencies

```bash
pip install tinker-sdk
```

### 2. Set Your API Key

```bash
export TINKER_API_KEY=your_key_here
```

Or create a `.env` file:
```
TINKER_API_KEY=your_key_here
```

### 3. Prepare Your Data

Put a JSONL file in the `data/` folder with this format:

```jsonl
{"messages": [{"role": "user", "content": "Hello!"}, {"role": "assistant", "content": "Hi there!"}]}
{"messages": [{"role": "user", "content": "Explain AI."}, {"role": "assistant", "content": "AI is..."}]}
```

We've included an example file at `data/example.jsonl` to get you started.

### 4. Run Training

```bash
python train.py
```

That's it! The script will:
1. **SFT (Supervised Fine-Tuning)**: Train the model to replicate your assistant responses
2. **RL (Reinforcement Learning)**: Improve the model using a simple reward function

## What This Does

### Supervised Fine-Tuning (SFT)
- Trains the model to predict assistant responses
- Only trains on the last assistant message (more efficient)
- Uses cross-entropy loss (standard language modeling)

### Reinforcement Learning (RL)
- Generates multiple responses per prompt
- Rewards better responses (longer, well-structured)
- Uses importance sampling to update the policy

## Customization

Edit the `CONFIG` dictionary in `train.py`:

```python
CONFIG = {
    # Model settings
    "model_name": "meta-llama/Llama-3.2-1B",  # Change model here
    "lora_rank": 32,  # Higher = more parameters

    # SFT settings
    "sft_learning_rate": 5e-4,
    "sft_steps": 100,
    "sft_batch_size": 4,

    # RL settings
    "rl_learning_rate": 1e-5,
    "rl_steps": 50,
    "rl_num_samples": 4,
    "rl_max_tokens": 200,
}
```

### Custom Reward Function

Edit the `compute_simple_reward()` function in `train.py` to customize how RL evaluates responses:

```python
def compute_simple_reward(response_text: str) -> float:
    # Your custom logic here!
    # Higher reward = better response
    return some_score
```

## Available Models

See the [Tinker docs](https://docs.tinker.so/) for available models. Some popular options:

- `meta-llama/Llama-3.2-1B` (fastest, smallest)
- `meta-llama/Llama-3.2-3B`
- `meta-llama/Llama-3.1-8B`
- `Qwen/Qwen3-8B`
- `Qwen/Qwen3-30B-A3B` (MoE, efficient)

## Project Structure

```
easy-train/
├── train.py           # Main training script (only file you need!)
├── data/              # Put your JSONL files here
│   └── example.jsonl  # Example data
└── README.md          # This file
```

## How It Works

This project uses **only Tinker primitives** - no complicated libraries or abstractions:

- `tinker.ServiceClient()` - Connect to Tinker
- `create_lora_training_client()` - Create a trainable model with LoRA
- `forward_backward()` - Compute gradients
- `optim_step()` - Update weights
- `save_weights_for_sampler()` - Save checkpoint
- `create_sampling_client()` - Generate text

Everything is in one readable file (`train.py`) so you can understand and modify it easily.

## Tips

### Out of Memory?
- Use a smaller model (e.g., `Llama-3.2-1B`)
- Reduce `sft_batch_size`
- Reduce `rl_num_samples`

### Training Too Slow?
- Reduce `sft_steps` and `rl_steps`
- Use fewer examples (set `max_examples` in CONFIG)

### Want Better RL?
- Customize the reward function in `compute_simple_reward()`
- Increase `rl_num_samples` (more diversity)
- Increase `rl_steps` (more training)

## Learn More

- [Tinker Documentation](https://docs.tinker.so/)
- [Tinker Cookbook](https://github.com/thinking-machines/tinker-cookbook) (for advanced features)

## License

MIT
