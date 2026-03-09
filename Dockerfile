# Z-Image Gradio Application with CUDA Support
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_HOME=/app/models/torch \
    HF_HOME=/app/models/huggingface \
    ZIMAGE_ATTENTION="_native_flash"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    pip \
    git \
    curl \
    wget \
    libssl-dev \
    libffi-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Create app directory
WORKDIR /app

# Copy project files
COPY . .

# Create directories for model weights
RUN mkdir -p /app/models/torch /app/models/huggingface /app/ckpts

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install torch>=2.5.0 transformers>=4.51.0 safetensors loguru pillow accelerate huggingface_hub>=0.25.0 gradio>=4.0.0 && \
    pip install -e . --no-build-isolation

# Expose Gradio port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:7860/info || exit 1

# Run Gradio app
CMD ["python", "app_gradio.py"]
