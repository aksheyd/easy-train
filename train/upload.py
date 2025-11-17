# Merge LoRA adapter and upload to HuggingFace
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print("Loading base model...")
# Load the base model (NOT quantized for merging)
# Remove device_map to avoid PEFT conflicts
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

print("Loading LoRA adapter...")
# Load and merge your adapter
model = PeftModel.from_pretrained(model, "sampler_weights")

print("Merging adapter with base model...")
model = model.merge_and_unload()  # type: ignore

print("Saving merged model...")
# Save locally first
output_dir = "merged_model"
model.save_pretrained(output_dir, safe_serialization=True)
tokenizer.save_pretrained(output_dir)

print("Uploading to HuggingFace: aksheyd/llama-3.1-8b-instruct-no-robots")
# Upload to HuggingFace
model.push_to_hub("aksheyd/llama-3.1-8b-instruct-no-robots")
tokenizer.push_to_hub("aksheyd/llama-3.1-8b-instruct-no-robots")

print(
    "Done! Model uploaded to: https://huggingface.co/aksheyd/llama-3.1-8b-instruct-no-robots"
)
