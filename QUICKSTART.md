# Quick Start Guide

Get your model trained in 3 steps!

## Step 1: Install

```bash
pip install tinker-sdk python-dotenv numpy
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

## Step 2: Set Your API Key

```bash
export TINKER_API_KEY=your_key_here
```

Get your key from [tinker.so](https://www.tinker.so/)

## Step 3: Run Training

```bash
python train.py
```

That's it! The example data is already in the `data/` folder.

## Using Your Own Data

1. Create a JSONL file with your conversations:

```jsonl
{"messages": [{"role": "user", "content": "Hello!"}, {"role": "assistant", "content": "Hi!"}]}
{"messages": [{"role": "user", "content": "How are you?"}, {"role": "assistant", "content": "I'm good!"}]}
```

2. Put it in the `data/` folder

3. Run `python train.py`

## What Happens During Training

### Phase 1: Supervised Fine-Tuning (SFT)
- Trains for 100 steps (configurable)
- Learns to replicate your assistant's responses
- Saves checkpoint: `tinker://xxx/sft_final`

### Phase 2: Reinforcement Learning (RL)
- Trains for 50 steps (configurable)
- Improves responses based on rewards
- Saves checkpoint: `tinker://xxx/rl_final`

## Customizing Training

Edit the `CONFIG` dictionary at the top of `train.py`:

```python
CONFIG = {
    "model_name": "meta-llama/Llama-3.2-1B",  # Try different models!
    "sft_steps": 100,                         # More steps = more training
    "rl_steps": 50,                           # More steps = better RL
    "sft_learning_rate": 5e-4,                # Higher = faster but less stable
    "rl_learning_rate": 1e-5,                 # Keep lower than SFT
}
```

## Customizing Rewards (RL)

Edit the `compute_simple_reward()` function in `train.py`:

```python
def compute_simple_reward(response_text: str) -> float:
    # Example: Reward responses with specific keywords
    reward = 0.0

    if "helpful" in response_text.lower():
        reward += 1.0

    if len(response_text) > 50:
        reward += 0.5

    return reward
```

## Troubleshooting

### "No .jsonl files found in data"
- Make sure you have a `.jsonl` file in the `data/` folder
- Check the file extension is exactly `.jsonl`

### "TINKER_API_KEY not set"
- Run `export TINKER_API_KEY=your_key`
- Or create a `.env` file with `TINKER_API_KEY=your_key`

### Training is slow
- Reduce `sft_steps` and `rl_steps` in the CONFIG
- Use fewer examples (set `max_examples` in CONFIG)
- Try a smaller model like `Llama-3.2-1B`

### Out of memory
- Use a smaller model
- Reduce `sft_batch_size` in CONFIG
- Reduce `rl_num_samples` in CONFIG

## Next Steps

After training completes, you'll get a checkpoint path like:
```
Final model checkpoint: tinker://abc123/rl_final
```

You can use this checkpoint to:
- Generate text with the Tinker API
- Download the model weights
- Deploy for inference

See the [Tinker docs](https://docs.tinker.so/) for more details on using your trained model!
