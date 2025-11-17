import argparse
from mlx_lm.utils import load
from mlx_lm.generate import generate
from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="Run local inference with MLX (Apple Silicon optimized)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="aksheyd/llama-3.1-8b-instruct-no-robots-mlx",
        help="HuggingFace model ID (MLX format)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Prompt to generate from",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate",
    )

    args = parser.parse_args()

    print(f"Loading model: {args.model}")

    # Load model with 4-bit quantization
    model_and_tokenizer_tuple = load(
        args.model,
    )
    if len(model_and_tokenizer_tuple) == 2:
        model, tokenizer = model_and_tokenizer_tuple
    else:
        model, tokenizer, _ = model_and_tokenizer_tuple

    print("Model loaded successfully!")

    if args.interactive:
        print("\nInteractive mode. Type 'quit' to exit.\n")
        while True:
            try:
                prompt = input("You: ")
                if prompt.lower() in ["quit", "exit", "q"]:
                    break

                response = generate(
                    model,
                    tokenizer,
                    prompt=prompt,
                    max_tokens=args.max_tokens,
                    verbose=True,
                )
                print(f"\nAssistant: {response}\n")

            except KeyboardInterrupt:
                print("\nExiting...")
                break

    elif args.prompt:
        response = generate(
            model,
            tokenizer,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            verbose=True,
        )
        print(f"\nResponse: {response}")

    else:
        print("Error: Either --interactive or --prompt must be specified")
        parser.print_help()


if __name__ == "__main__":
    main()
