# Use PyTorch 2.4 to satisfy the transformers requirement
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

# Enable Xet high-performance transfers to beat the Cloud Run startup timeout
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    HF_XET_HIGH_PERFORMANCE=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["python", "app.py"]
