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

# Install Miniconda for fast flash-attn installation
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

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

# Install flash-attn via conda (much faster than compiling)
RUN conda install -y -c conda-forge flash-attn || echo "flash-attn installation failed, will use native backends"

# Install the project in development mode
RUN python3.11 -m pip install -e . --no-build-isolation

# Expose Gradio port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:7860/info || exit 1

# Run Gradio app
CMD ["python", "app_gradio.py"]
