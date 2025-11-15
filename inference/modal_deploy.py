"""Modal deployment script for the trained model."""

# Modal app template that will be written to a file
MODAL_APP_CODE = '''"""
Modal deployment for Tinker-trained chat model.

Educational note:
----------------
Modal is a serverless compute platform perfect for ML inference:
- Automatically scales with demand (0 to many instances)
- Pay only for actual compute time
- Easy GPU access (T4, A10G, A100)
- No infrastructure management

This script creates a Modal app that:
1. Loads the merged model in a container
2. Exposes a FastAPI endpoint for inference
3. Handles concurrent requests efficiently
"""

import modal
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Create Modal app
app = modal.App("tinker-chat-model")

# Define compute resources
# Educational: T4 is cost-effective, A10G is faster, A100 is for large models
gpu_config = modal.gpu.T4()  # Will be replaced with actual GPU type

# Create container image with dependencies
image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "sentencepiece",  # For some tokenizers
    )
)


# Define the model class
@app.cls(
    image=image,
    gpu=gpu_config,
    secrets=[modal.Secret.from_name("huggingface-secret")],  # If model is private
)
class ChatModel:
    """
    Modal class for chat model inference.

    Educational note:
    ----------------
    - @modal.enter: Runs once when container starts (load model here)
    - @modal.method: Creates an endpoint that can be called
    - Modal handles scaling, GPU allocation, and cleanup automatically
    """

    @modal.enter()
    def load_model(self):
        """Load model once when container starts."""
        print("Loading model...")

        # Load from HuggingFace or local path
        model_path = "your-username/your-model-name"  # Will be replaced

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

        print("Model loaded successfully!")

    @modal.method()
    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        Generate a response given conversation history.

        Args:
            messages: List of {"role": "...", "content": "..."} dicts
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = greedy)
            top_p: Nucleus sampling threshold

        Returns:
            Generated assistant response
        """
        # Format prompt using chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode (skip input prompt)
        input_length = inputs["input_ids"].shape[1]
        response = self.tokenizer.decode(
            outputs[0][input_length:],
            skip_special_tokens=True,
        )

        return response


# Create FastAPI app for REST endpoint
web_app = FastAPI()


class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9


class ChatResponse(BaseModel):
    response: str


@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    """
    Expose FastAPI app via Modal.

    Educational note:
    ----------------
    This creates a public HTTPS endpoint that you can call from anywhere.
    Modal handles SSL, load balancing, and auto-scaling.
    """
    model = ChatModel()

    @web_app.post("/chat", response_model=ChatResponse)
    async def chat_endpoint(request: ChatRequest):
        """Chat endpoint for inference."""
        response = model.generate.remote(
            messages=request.messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )
        return ChatResponse(response=response)

    @web_app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy"}

    return web_app
'''

# Client example code
CLIENT_CODE = '''"""
Example client for calling the Modal endpoint.
"""

import requests


def chat_with_model(messages, url="https://your-app.modal.run/chat"):
    """
    Call the Modal endpoint to chat with your model.

    Args:
        messages: List of {"role": "...", "content": "..."} dicts
        url: Modal endpoint URL

    Returns:
        Generated response
    """
    response = requests.post(
        url,
        json={
            "messages": messages,
            "max_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
        },
    )
    response.raise_for_status()
    return response.json()["response"]


# Example usage
if __name__ == "__main__":
    messages = [{"role": "user", "content": "Explain machine learning in simple terms."}]

    response = chat_with_model(messages)
    print(f"Assistant: {response}")

    # Multi-turn conversation
    messages.append({"role": "assistant", "content": response})
    messages.append({"role": "user", "content": "Can you give an example?"})

    response = chat_with_model(messages)
    print(f"Assistant: {response}")
'''


def create_modal_app(config):
    """
    Create Modal deployment script.

    Educational note:
    ----------------
    This writes the Modal app code to a file that you can deploy with:
        modal deploy modal_app.py

    Then access via:
        https://your-app.modal.run/chat

    Args:
        config: InferenceConfig object

    Returns:
        Path to created modal app file
    """
    output_file = "modal_app.py"

    # Customize the template with config
    modal_code = MODAL_APP_CODE.replace(
        "gpu_config = modal.gpu.T4()", f"gpu_config = modal.gpu.{config.gpu_type}()"
    )

    # Write to file
    with open(output_file, "w") as f:
        f.write(modal_code)

    print(f"\n✅ Created Modal app: {output_file}")
    print(f"\nTo deploy:")
    print(f"  1. Install Modal: pip install modal")
    print(f"  2. Setup token: modal token new")
    print(f"  3. Deploy: modal deploy {output_file}")
    print(f"\nThen you can call your endpoint at:")
    print(f"  https://{config.modal_app_name}.modal.run/chat")

    return output_file


def create_client_example():
    """
    Create example client code for calling the Modal endpoint.

    Returns:
        Path to created client file
    """
    client_file = "modal_client.py"

    with open(client_file, "w") as f:
        f.write(CLIENT_CODE)

    print(f"✅ Created example client: {client_file}")

    return client_file
