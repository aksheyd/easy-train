# Easy Train

How I'm learning post-training.

Currently, I'm using Tinker to run basic SFT via the [no robots dataset](https://huggingface.co/datasets/HuggingFaceH4/no_robots) on [LLaMa 3.1 8B Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct).

Then, I download the sampler weights from Tinker, downloaded the full model, merged the adapter weights, and uploaded the merged model to Hugging Face.

[Merged Model](https://huggingface.co/aksheyd/llama-3.1-8b-instruct-no-robots)

[MLX Model](https://huggingface.co/aksheyd/llama-3.1-8b-instruct-no-robots-mlx)
Then, I'm running inference via 4-bit quantized MLX locally and vLLM on Modal on an L4 GPU.

## Folers

local -> local vLLM setup to test
modal -> Modal vLLM setup to test
train -> Tinker training setup
tinker_cookbook -> Tinker cookbook git submodule
client.py -> Simple client to test the vLLM server
