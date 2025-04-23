# Consyn AI Language Model Family

Consyn AI ("Conscious Synthesis") is a suite of transformer-based language models designed for versatility, efficiency, and advanced reasoning capabilities. The suite includes three model sizes to meet different deployment requirements:

## Model Variants

- **Consyn Verse** - Small (~125M–350M parameters)
- **Consyn Stanza** - Medium (~1B–7B parameters)
- **Consyn Epic** - Large (~13B–65B+ parameters)

## Key Features

### Core Architecture

- Transformer-based architecture with configurable parameters
- Multiple attention mechanisms (standard, rotary, sparse)
- Advanced embedding systems with robust positional encoding
- Optimized feed-forward networks with various activation functions
- Efficient normalization layers (LayerNorm, RMSNorm)

### Unique Differentiators

1. **Context-Aware Memory Module**
   - Maintains persistent memory of past contexts beyond the attention window
   - Enables handling of much longer contexts than standard models
   - Memory compression and routing for efficient information retrieval

2. **Intent Parsing System**
   - Internal intent detection and classification
   - Query decomposition for handling complex instructions
   - State tracking for improved conversation coherence

3. **Lightweight Retrieval-Augmented Generation (RAG-Lite)**
   - Internal knowledge indexing without external databases
   - Self-retrieval during generation for improved factuality
   - Adaptive weighting of retrieved information

### Training & Inference

- Full training pipeline with mixed precision and gradient accumulation
- Efficient inference with key-value caching and optimized attention
- Support for quantization (INT8, INT4) and model compression
- ONNX export for deployment across different frameworks

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/consynai/consyn.git
cd consyn

# Install the package
pip install -e .
```

### Training a Model

```bash
# Train from scratch
consyn-train --model verse --data /path/to/data --output ./models/consyn_verse

# Fine-tune an existing model
consyn-train --model stanza --data /path/to/data --checkpoint ./models/consyn_stanza --output ./models/consyn_stanza_ft
```

### Text Generation

```bash
# Generate text using the Verse model
consyn-generate --model verse --prompt "In the beginning" --max_tokens 128

# Run interactive generation with the Epic model
consyn-generate --model epic --interactive
```

### Serving Models

```bash
# Start the API server
consyn-serve --models verse,stanza,epic --port 8000
```

## API Usage

Once the server is running, you can use the API to generate text:

```python
import requests

response = requests.post(
    "http://localhost:8000/generate",
    json={
        "prompt": "Once upon a time",
        "max_new_tokens": 128,
        "temperature": 0.7,
        "model_name": "verse"
    }
)

print(response.json()["generated_text"])
```

## Model Export

You can export models to various formats:

```bash
# Export to ONNX
consyn-convert --model verse --format onnx --output ./exported/verse

# Export for TensorRT
consyn-convert --model epic --format tensorrt --output ./exported/epic
```

## System Requirements

### Minimum Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.7+ (for GPU acceleration)
- 8GB RAM for Verse models
- 16GB RAM for Stanza models
- 32GB+ RAM for Epic models

### Recommended

- 16GB+ RAM
- NVIDIA GPU with 8GB+ VRAM
- CUDA 12.0+
- PyTorch 2.1+ with CUDA support

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Built with ❤️

## Website

Coming soon: [https://consyn.ai](https://consyn.ai)
Want early access or partnership? Email: `hello@consyn.ai`
