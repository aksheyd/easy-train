# Easy Train

I was using Tinker to run basic SFT via on datasets like no robots and tulu3 on LLaMa-3.1-8B. After training (LoRA rank 32, batch size 128, learning rate 1e-4), I merge the adapter weights with the base model and uploading the merged model to Hugging Face. I'm running inference via 4/8-bit quantized MLX locally and vLLM on Modal on an L4 GPU.

But not anymore: I ran out of my $150 free credits so now, I'm trying to train on Modal using Unsloth QLora with WanDB for visibility, then will upload weights to Hugging Face.

First SFT on No Robots dataset:

[llama-3.1-8b-instruct-no-robots](https://huggingface.co/aksheyd/llama-3.1-8b-instruct-no-robots)

[llama-3.1-8b-instruct-no-robots-mlx](https://huggingface.co/aksheyd/llama-3.1-8b-instruct-no-robots-mlx)

First SFT on Tulu3 dataset:

[llama-3.1-8b-tulu3-sft](https://huggingface.co/aksheyd/llama-3.1-8b-tulu3-sft)

[llama-3.1-8b-tulu3-sft-mlx](https://huggingface.co/aksheyd/llama-3.1-8b-tulu3-sft-mlx)

## Findings

On extremely short prompts, my fine-tunes produce gibberish responses. I'm currently trying to figure out if this is due to the training hyperparameters, or the model itself. A simple Hello turns into 1000s of words of the model hinting keyowrds to itself then generating text that exactly matches the dataset.

However, on longer prompts, the model seems to be able to generate more coherent responses.

## Future Work

I want to explore RL (maybe the SFT checkpoint is supposed to be poor alone?).

I want to try PPO + RLHF and GRPO + RLVR to see if I can get a better instruct model and coding model potentially. Obviously, this requires a ton of compute and resources. I am interested in scaling RL and tackle problems like in this [paper](https://arxiv.org/pdf/2511.14617) from Moonshot.

## Folers

```
local -> local vLLM setup to test
modal -> Modal vLLM setup to test
train -> Tinker training setup
tinker_cookbook -> Tinker cookbook git submodule
client.py -> Simple client to test the vLLM server
```
