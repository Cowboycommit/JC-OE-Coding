#!/usr/bin/env python3
"""
Setup script for downloading local quantized Mistral models.

This script downloads GGUF-format quantized Mistral models from HuggingFace
for use as a fallback when the Mistral API is unavailable.

Usage:
    python scripts/setup_local_models.py [--model MODEL_SIZE] [--output-dir DIR]

Options:
    --model       Model size to download: '7b' or '8x7b' (default: 7b)
    --output-dir  Directory to save models (default: models/)
    --quantization  Quantization level: 'Q4_K_M' or 'Q5_K_M' (default: Q4_K_M)

Examples:
    # Download Mistral 7B (smallest, ~4GB)
    python scripts/setup_local_models.py --model 7b

    # Download Mixtral 8x7B (best quality, ~26GB)
    python scripts/setup_local_models.py --model 8x7b

    # Download with higher quantization quality
    python scripts/setup_local_models.py --model 7b --quantization Q5_K_M
"""

import argparse
import os
import sys
from pathlib import Path


# Model configurations
MODELS = {
    "7b": {
        "repo_id": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        "filename_template": "mistral-7b-instruct-v0.2.{quant}.gguf",
        "description": "Mistral 7B Instruct v0.2",
        "size_q4": "~4.1 GB",
        "size_q5": "~4.8 GB",
    },
    "8x7b": {
        "repo_id": "TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF",
        "filename_template": "mixtral-8x7b-instruct-v0.1.{quant}.gguf",
        "description": "Mixtral 8x7B Instruct v0.1 (MoE)",
        "size_q4": "~26.4 GB",
        "size_q5": "~32.2 GB",
    },
}

QUANTIZATIONS = ["Q4_K_M", "Q5_K_M"]


def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        from huggingface_hub import hf_hub_download
        return True
    except ImportError:
        return False


def install_dependencies():
    """Install required dependencies."""
    print("Installing required dependencies...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface-hub>=0.20.0"])
    print("Dependencies installed successfully.")


def download_model(
    model_size: str,
    quantization: str,
    output_dir: str,
    force: bool = False
) -> Path:
    """
    Download a quantized model from HuggingFace.

    Args:
        model_size: Model size ('7b' or '8x7b')
        quantization: Quantization level ('Q4_K_M' or 'Q5_K_M')
        output_dir: Directory to save the model
        force: Force re-download even if file exists

    Returns:
        Path to the downloaded model file
    """
    from huggingface_hub import hf_hub_download

    if model_size not in MODELS:
        raise ValueError(f"Unknown model size: {model_size}. Choose from: {list(MODELS.keys())}")

    if quantization not in QUANTIZATIONS:
        raise ValueError(f"Unknown quantization: {quantization}. Choose from: {QUANTIZATIONS}")

    model_config = MODELS[model_size]
    filename = model_config["filename_template"].format(quant=quantization)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    local_path = output_path / filename

    if local_path.exists() and not force:
        print(f"Model already exists at {local_path}")
        print("Use --force to re-download")
        return local_path

    print(f"Downloading {model_config['description']}...")
    print(f"  Repository: {model_config['repo_id']}")
    print(f"  File: {filename}")
    print(f"  Quantization: {quantization}")

    size_key = f"size_{quantization.lower().split('_')[0]}"
    print(f"  Estimated size: {model_config.get(size_key, 'Unknown')}")
    print()

    # Download the model
    downloaded_path = hf_hub_download(
        repo_id=model_config["repo_id"],
        filename=filename,
        local_dir=str(output_path),
        local_dir_use_symlinks=False
    )

    print(f"\nModel downloaded successfully to: {downloaded_path}")
    return Path(downloaded_path)


def verify_model(model_path: Path) -> bool:
    """Verify that the downloaded model can be loaded."""
    try:
        from llama_cpp import Llama
        print(f"\nVerifying model: {model_path.name}...")

        # Just check if it loads, don't generate anything
        model = Llama(
            model_path=str(model_path),
            n_ctx=512,  # Small context for quick test
            verbose=False
        )
        del model

        print("Model verified successfully!")
        return True

    except ImportError:
        print("\nNote: llama-cpp-python not installed. Skipping verification.")
        print("Install with: pip install llama-cpp-python")
        return True  # Return True since download succeeded

    except Exception as e:
        print(f"\nWarning: Model verification failed: {e}")
        print("The model file may be corrupted. Try re-downloading with --force")
        return False


def create_gitignore(output_dir: str):
    """Create .gitignore in models directory to prevent accidental commits."""
    gitignore_path = Path(output_dir) / ".gitignore"
    if not gitignore_path.exists():
        gitignore_path.write_text("# Ignore downloaded model files\n*.gguf\n*.bin\n")
        print(f"Created {gitignore_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Download local quantized Mistral models for LLM interpretation fallback",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --model 7b                    # Download Mistral 7B (~4GB)
  %(prog)s --model 8x7b                  # Download Mixtral 8x7B (~26GB)
  %(prog)s --model 7b --quantization Q5_K_M  # Higher quality quantization

Available models:
  7b    - Mistral 7B Instruct v0.2 (smaller, faster)
  8x7b  - Mixtral 8x7B Instruct v0.1 (larger, better quality)

Quantization options:
  Q4_K_M - 4-bit quantization (smaller, default)
  Q5_K_M - 5-bit quantization (larger, higher quality)
        """
    )

    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        default="7b",
        help="Model size to download (default: 7b)"
    )

    parser.add_argument(
        "--quantization",
        choices=QUANTIZATIONS,
        default="Q4_K_M",
        help="Quantization level (default: Q4_K_M)"
    )

    parser.add_argument(
        "--output-dir",
        default="models/",
        help="Directory to save models (default: models/)"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if model exists"
    )

    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip model verification after download"
    )

    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit"
    )

    args = parser.parse_args()

    if args.list_models:
        print("Available models:\n")
        for size, config in MODELS.items():
            print(f"  {size}:")
            print(f"    Description: {config['description']}")
            print(f"    Repository:  {config['repo_id']}")
            print(f"    Size (Q4):   {config['size_q4']}")
            print(f"    Size (Q5):   {config['size_q5']}")
            print()
        return

    # Check/install dependencies
    if not check_dependencies():
        print("Required dependencies not found.")
        response = input("Install huggingface-hub? [Y/n] ").strip().lower()
        if response in ("", "y", "yes"):
            install_dependencies()
        else:
            print("Cannot proceed without huggingface-hub. Exiting.")
            sys.exit(1)

    # Create models directory and .gitignore
    create_gitignore(args.output_dir)

    # Download the model
    try:
        model_path = download_model(
            model_size=args.model,
            quantization=args.quantization,
            output_dir=args.output_dir,
            force=args.force
        )

        # Verify the model
        if not args.skip_verify:
            verify_model(model_path)

        print("\n" + "=" * 60)
        print("Setup complete!")
        print("=" * 60)
        print(f"\nModel location: {model_path}")
        print(f"\nTo use this model, ensure your .env file contains:")
        print(f"  LLM_LOCAL_MODEL_PATH={args.output_dir}")
        print("\nThe system will automatically use this model as a fallback")
        print("when the Mistral API is unavailable.")

    except KeyboardInterrupt:
        print("\n\nDownload cancelled.")
        sys.exit(1)

    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
