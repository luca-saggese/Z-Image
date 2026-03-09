# Z-Image Gradio Application with CUDA Support
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_HOME=/huggingface/torch \
    HF_HOME=/huggingface \
    ZIMAGE_ATTENTION="_native_flash" \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3.11-distutils \
    git \
    curl \
    wget \
    libssl-dev \
    libffi-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default
RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3


ENV PATH=/opt/conda/bin:${PATH}

# Install pip for Python 3.11
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.11 get-pip.py && \
    rm get-pip.py

# Create app directory
WORKDIR /app

# Copy project files
COPY . .

# Create directories for model weights
RUN mkdir -p /huggingface /app/ckpts

# Install Python dependencies using Python 3.11
RUN python3.11 -m pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
RUN python3.11 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install remaining dependencies
RUN python3.11 -m pip install transformers>=4.51.0 safetensors loguru pillow accelerate huggingface_hub>=0.25.0 gradio>=4.0.0

# Try to install Flash Attention 3 from prebuilt wheel (aarch64), fallback to flash-attn v2
RUN if [ "$(uname -m)" = "aarch64" ]; then \
            python3.11 -m pip install "https://huggingface.co/datasets/malaysia-ai/Flash-Attention3-wheel/resolve/main/flash_attn_3-3.0.0b1-cp39-abi3-linux_aarch64-2.7.1-12.8.whl" \
            || (echo "Flash Attention 3 wheel install failed, trying flash-attn==2.8.3" && python3.11 -m pip install flash-attn==2.8.3) \
            || echo "flash-attn installation failed, will use native backends"; \
        else \
            echo "Skipping Flash Attention 3 wheel (requires aarch64); trying flash-attn==2.8.3" && \
            python3.11 -m pip install flash-attn==2.8.3 \
            || echo "flash-attn installation failed, will use native backends"; \
        fi

# Install the project in development mode
RUN python3.11 -m pip install -e . --no-build-isolation

# Expose Gradio port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:7860/info || exit 1

# Run Gradio app
CMD ["python", "app_gradio.py"]
