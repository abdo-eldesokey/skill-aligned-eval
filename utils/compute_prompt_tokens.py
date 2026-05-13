import tiktoken
import argparse
from pathlib import Path
from tiktoken.model import MODEL_TO_ENCODING


def compute_tokens(file_path: str, encoding_name: str = "cl100k_base") -> int:
    """
    Compute the number of tokens in a system prompt file.

    Args:
        file_path: Path to the system prompt file
        encoding_name: Tiktoken encoding to use (default: cl100k_base for GPT-4/GPT-3.5-turbo)

    Returns:
        Number of tokens in the file
    """
    # Read the file content
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Get the encoding
    encoding = tiktoken.get_encoding(encoding_name)

    # Encode and count tokens
    tokens = encoding.encode(content)
    return len(tokens)


def main():
    parser = argparse.ArgumentParser(description="Compute the number of tokens in a system prompt file using tiktoken")
    parser.add_argument("prompt_version", type=str, help="Prompt version (e.g., v1.0)")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5",
        choices=["gpt-5", "gpt-4o", "gpt-4", "gpt-3.5-turbo", "gpt-2"],
        help="Tiktoken encoding to use (default: cl100k_base for GPT-4/GPT-3.5-turbo)",
    )

    args = parser.parse_args()

    # Construct the full path
    prompt_dir = Path(__file__).parent / "assets" / "system_prompts"
    file_path = prompt_dir / f"prompt_v{args.prompt_version}.txt"

    # Check if file exists
    if not file_path.exists():
        print(f"Error: File 'prompt_v{args.prompt_version}.txt' not found in {prompt_dir}")
        print(f"\nAvailable files:")
        for f in sorted(prompt_dir.glob("*.txt")):
            print(f"  - {f.name}")
        return

    # Compute tokens
    token_count = compute_tokens(file_path, MODEL_TO_ENCODING[args.model])

    # Display results
    print(f"File: prompt_v{args.prompt_version}.txt")
    print(f"Model: {args.model}")
    print(f"Token count: {token_count:,}")


if __name__ == "__main__":
    main()
