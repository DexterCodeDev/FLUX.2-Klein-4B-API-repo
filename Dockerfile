# Optimized for CUDA 12.1 Execution on NVIDIA L4
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

WORKDIR /app

# Install base Python dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip first to ensure modern wheel support
RUN pip3 install --no-cache-dir --upgrade pip

COPY requirements.txt .

# Use PyTorch's native index url to pull the correct Linux CUDA binary
RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install --no-cache-dir -r requirements.txt

# --- CI PIPELINE BUILD CACHING ---
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# Flattened python string execution to guarantee no indentation errors
RUN python3 -c "import os; from diffusers import DiffusionPipeline; import torch; DiffusionPipeline.from_pretrained('black-forest-labs/FLUX.2-klein-4B', torch_dtype=torch.bfloat16, token=os.getenv('HF_TOKEN'))"
# ----------------------------------

COPY app.py .

EXPOSE 8080

CMD uvicorn app:app --host 0.0.0.0 --port ${PORT}
