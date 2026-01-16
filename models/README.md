# Local LLM Models

This directory stores local quantized Mistral models for LLM-assisted cluster interpretation.

## Setup

Run the setup script to download a local model:

```bash
# Download Mistral 7B (recommended for most systems, ~4GB)
python scripts/setup_local_models.py --model 7b

# Download Mixtral 8x7B (best quality, requires ~26GB disk space)
python scripts/setup_local_models.py --model 8x7b
```

## Model Options

| Model | Size | Quality | RAM Needed |
|-------|------|---------|------------|
| mistral-7b-instruct-v0.2.Q4_K_M.gguf | ~4 GB | Good | ~6 GB |
| mistral-7b-instruct-v0.2.Q5_K_M.gguf | ~5 GB | Better | ~7 GB |
| mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf | ~26 GB | Best | ~32 GB |
| mixtral-8x7b-instruct-v0.1.Q5_K_M.gguf | ~32 GB | Best+ | ~40 GB |

## Dependencies

To use local models, install llama-cpp-python:

```bash
pip install llama-cpp-python
```

For GPU acceleration (optional):

```bash
# CUDA (NVIDIA)
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

# Metal (Apple Silicon)
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

## Notes

- Model files (`.gguf`) are automatically excluded from git
- The system will automatically detect and use any `.gguf` file in this directory
- If Mistral API is configured, it will be used first; local models are the fallback
