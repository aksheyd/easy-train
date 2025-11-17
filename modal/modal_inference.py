import modal

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.11.0",
        "huggingface-hub==0.36.0",
        "flashinfer-python==0.3.1",
        "transformers>=4.45.0",
        "peft>=0.13.0",
        "torch>=2.5.0",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

MODEL_NAME = "aksheyd/llama-3.1-8b-instruct-no-robots"

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)
lora_cache_vol = modal.Volume.from_name("lora-cache", create_if_missing=True)

FAST_BOOT = True

app = modal.App("example-vllm-inference")


MINUTES = 60
VLLM_PORT = 8000


@app.function(
    image=vllm_image,
    gpu="L4",
    scaledown_window=15 * MINUTES,
    timeout=10 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
        "/root/.cache/lora": lora_cache_vol,
    },
    secrets=[modal.Secret.from_name("tinker-secret")],
)
@modal.concurrent(max_inputs=4)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
def serve():
    import subprocess

    model_to_serve = MODEL_NAME
    model_name = MODEL_NAME.split("/")[-1]

    # Full precision inference on GPU - no quantization
    cmd = [
        "vllm",
        "serve",
        "--uvicorn-log-level=info",
        model_to_serve,
        "--served-model-name",
        model_name,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--dtype=auto",  # Auto-detect best dtype (bfloat16/float16)
        "--max-model-len=8192",  # Increase context length
        "--gpu-memory-utilization=0.95",  # Use 95% of GPU memory
    ]

    # enforce-eager disables both Torch compilation and CUDA graph capture
    # For production, use --no-enforce-eager for better performance
    cmd += ["--enforce-eager" if FAST_BOOT else "--no-enforce-eager"]

    print("Starting vLLM with FULL PRECISION on GPU...")
    print(f"Command: {' '.join(cmd)}")

    subprocess.Popen(" ".join(cmd), shell=True)
