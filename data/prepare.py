"""Data validation and preparation utilities for JSONL datasets."""

import json
from pathlib import Path
from typing import List, Dict, Any


def validate_jsonl(file_path: str) -> bool:
    """
    Validate JSONL format for Tinker.

    Expected format:
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

    Args:
        file_path: Path to JSONL file

    Returns:
        True if validation passes, False otherwise
    """
    try:
        with open(file_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())

                    # Check for 'messages' field
                    if "messages" not in data:
                        print(f"Line {line_num}: Missing 'messages' field")
                        return False

                    # Validate messages structure
                    messages = data["messages"]
                    if not isinstance(messages, list):
                        print(f"Line {line_num}: 'messages' must be a list")
                        return False

                    # Check each message has role and content
                    for msg_idx, msg in enumerate(messages):
                        if "role" not in msg or "content" not in msg:
                            print(
                                f"Line {line_num}, message {msg_idx}: Missing 'role' or 'content'"
                            )
                            return False

                        if msg["role"] not in ["system", "user", "assistant"]:
                            print(
                                f"Line {line_num}, message {msg_idx}: Invalid role '{msg['role']}'"
                            )
                            return False

                except json.JSONDecodeError as e:
                    print(f"Line {line_num}: Invalid JSON - {e}")
                    return False

        print(f"✓ Validation passed: {file_path}")
        return True

    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        return False


def count_examples(file_path: str) -> Dict[str, Any]:
    """
    Count examples and compute statistics.

    Args:
        file_path: Path to JSONL file

    Returns:
        Dictionary with statistics (num_examples, avg_turns, min_turns, max_turns)
    """
    num_examples = 0
    num_turns = []

    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line.strip())
            num_examples += 1
            num_turns.append(len(data["messages"]))

    return {
        "num_examples": num_examples,
        "avg_turns": sum(num_turns) / len(num_turns) if num_turns else 0,
        "min_turns": min(num_turns) if num_turns else 0,
        "max_turns": max(num_turns) if num_turns else 0,
    }


def create_example_data(output_path: str = "examples/example_data.jsonl"):
    """
    Create example conversational data for testing.

    Args:
        output_path: Where to save the example data
    """
    examples = [
        {
            "messages": [
                {"role": "user", "content": "What is machine learning?"},
                {
                    "role": "assistant",
                    "content": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing algorithms that can access data and use it to learn for themselves.",
                },
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Explain supervised learning."},
                {
                    "role": "assistant",
                    "content": "Supervised learning is a type of machine learning where the algorithm learns from labeled training data. The model is trained on input-output pairs, learning to map inputs to correct outputs. Common examples include classification and regression tasks.",
                },
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "What's the difference between AI and ML?"},
                {
                    "role": "assistant",
                    "content": "AI (Artificial Intelligence) is the broader concept of machines being able to carry out tasks in a smart way. ML (Machine Learning) is a subset of AI that focuses on the idea that systems can learn from data, identify patterns, and make decisions with minimal human intervention.",
                },
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What's reinforcement learning?"},
                {
                    "role": "assistant",
                    "content": "Reinforcement learning is a type of machine learning where an agent learns to make decisions by interacting with an environment. The agent receives rewards or penalties based on its actions and learns to maximize cumulative reward over time. It's used in robotics, game playing, and autonomous systems.",
                },
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "How does neural network training work?"},
                {
                    "role": "assistant",
                    "content": "Neural network training involves adjusting the network's weights to minimize a loss function. The process uses backpropagation to compute gradients and an optimizer (like Adam or SGD) to update weights. Through many iterations on training data, the network learns to make better predictions.",
                },
            ]
        },
    ]

    # Ensure directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")

    print(f"✓ Created example data: {output_path}")
