# Multi-stage Dockerfile for locknessapi optimized for NVIDIA RTX 4090

FROM nvidia/cuda:12.6.2-devel-ubuntu22.04 AS base
# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    build-essential \
    cmake \
    ninja-build \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libegl1 \
    libegl1-mesa \
    libgl1-mesa-dev \
    libgles2-mesa-dev \
    libosmesa6-dev \
    xvfb \
    libcudnn9-cuda-12 \
    libnvinfer8 \
    libnvinfer-plugin8 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py310_24.7.1-0-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh
ENV PATH="/opt/conda/bin:${PATH}"

# Initialize conda
RUN conda init bash

# Create conda environment with Python 3.10
RUN conda create -n locknessapi python=3.10 -y

# Make conda environment activation persistent
SHELL ["conda", "run", "-n", "locknessapi", "/bin/bash", "-c"]

# Install PyTorch with CUDA 12.6 support for RTX 4090
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Set working directory
WORKDIR /app

# Copy the entire project
COPY . .
# Install Hunyuan3D 2.1 requirements
WORKDIR /app/thirdparty/Hunyuan3D-2.1
ENV TORCH_CUDA_ARCH_LIST="8.9"

# Ensure nvidia-smi is in PATH for Conda environment
ENV PATH="/usr/bin:${PATH}"
RUN pip install huggingface_hub
RUN pip install -r requirements-inference.txt --verbose
# Install Hunyuan3D 2.1 dependencies
WORKDIR /app/thirdparty/Hunyuan3D-2.1/hy3dpaint/custom_rasterizer
ENV TORCH_CUDA_ARCH_LIST="8.9"

# Ensure nvidia-smi is in PATH for Conda environment
ENV PATH="/usr/bin:${PATH}"

# Verify CUDA environment before building

RUN pip install -e .

# Build differentiable renderer for Hunyuan3D 2.1
WORKDIR /app/thirdparty/Hunyuan3D-2.1/hy3dpaint/DifferentiableRenderer
RUN bash compile_mesh_painter.sh




# Install HoloPart dependencies
# WORKDIR /app/thirdparty/HoloPart
# RUN pip install -r requirements.txt

# Install UniRig dependencies
# WORKDIR /app/thirdparty/UniRig
# RUN pip install spconv-cu126 pyrender fast-simplification python-box timm

# Install PartPacker dependencies
# WORKDIR /app/thirdparty/PartPacker
# RUN pip install meshiki fpsample kiui pymcubes einops

# Install PartCrafter dependencies (if requirements exist)
# WORKDIR /app/thirdparty/PartCrafter
# RUN if [ -f "requirements.txt" ]; then pip install -r requirements.txt; fi

# Install main project dependencies
WORKDIR /app
RUN pip install -r requirements.txt
RUN pip install -r requirements-test.txt
RUN pip install huggingface_hub

# Create necessary directories
RUN mkdir -p /app/uploads /app/data

# Set environment variables for runtime
ENV PYTHONPATH="/app:${PYTHONPATH}"
ENV CONDA_DEFAULT_ENV=locknessapi
# Ada Lovelace architecture for RTX 4090
ENV TORCH_CUDA_ARCH_LIST="8.9"  
# Optimize for cuDNN 9
ENV CUDNN_V9_PATH=/usr/lib/x86_64-linux-gnu  

# Expose port for FastAPI
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command to run the FastAPI server
CMD ["conda", "run", "-n", "locknessapi", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
