# Base image with CUDA support
FROM nvidia/cuda:12.1.105-cudnn8-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3-pip git curl && \
    rm -rf /var/lib/apt/lists/*

# Set Python 3.11
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Set workdir
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY app.py .

# Expose port
EXPOSE 8080

# Start command
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
