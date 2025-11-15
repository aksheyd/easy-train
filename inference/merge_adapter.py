"""Merge LoRA adapter with base model."""

import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel


def merge_lora_adapter(
    base_model_name: str,
    adapter_path: str,
    output_path: str = "merged_model",
    upload_to_hf: bool = False,
    hf_repo_name: str = None,
):
    """
    Merge LoRA adapter with base model for faster inference.

    Educational note:
    ----------------
    LoRA adapters are separate from the base model.
    For inference, you can either:
    1. Load base + adapter separately (uses more memory, slightly slower)
    2. Merge them into one model (this function - faster, more memory efficient)

    Merging is recommended for deployment.

    Args:
        base_model_name: Name of the base model (e.g., "meta-llama/Llama-3.2-1B")
        adapter_path: Path to LoRA adapter weights
        output_path: Where to save merged model
        upload_to_hf: Whether to upload to HuggingFace
        hf_repo_name: HuggingFace repo name (required if upload_to_hf=True)

    Returns:
        Path to merged model
    """
    print(f"\n🔗 Merging LoRA adapter with base model")
    print(f"  Base model: {base_model_name}")
    print(f"  Adapter: {adapter_path}")

    # Load base model
    print(f"  Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load LoRA adapter
    print(f"  Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    # Merge and unload
    # Educational: This combines LoRA weights into base model weights
    print(f"  Merging adapter weights...")
    merged_model = model.merge_and_unload()

    # Save merged model
    print(f"  Saving merged model to {output_path}...")
    merged_model.save_pretrained(output_path)

    # Also save tokenizer
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    tokenizer = get_tokenizer(base_model_name)
    tokenizer.save_pretrained(output_path)

    print(f"✅ Merged model saved to {output_path}")

    # Optional: Upload to HuggingFace
    if upload_to_hf and hf_repo_name:
        print(f"\n📤 Uploading to HuggingFace: {hf_repo_name}")
        merged_model.push_to_hub(hf_repo_name)
        tokenizer.push_to_hub(hf_repo_name)
        print(f"✅ Uploaded to https://huggingface.co/{hf_repo_name}")

    return output_path
