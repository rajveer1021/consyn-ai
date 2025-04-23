# Base image with Python and PyTorch
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install basic dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Optional: Install optimizations
RUN pip install --no-cache-dir \
    ninja \
    flash-attn \
    bitsandbytes \
    optimum \
    onnx \
    onnxruntime \
    onnxruntime-gpu

# Copy the source code
COPY . /app/

# Install the package
RUN pip install -e .

# Create model and data directories
RUN mkdir -p /app/models /app/data

# Set environment variables
ENV PYTHONPATH=/app
ENV MODEL_DIR=/app/models
ENV DATA_DIR=/app/data

# Default port for the API
EXPOSE 8000

# Entry point for the API
CMD ["uvicorn", "consyn.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
