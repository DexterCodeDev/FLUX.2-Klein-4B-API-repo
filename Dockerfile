FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip3 install --upgrade pip

RUN pip3 install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu124 \
    torch torchvision

RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD uvicorn app:app \
    --host 0.0.0.0 \
    --port ${PORT}
