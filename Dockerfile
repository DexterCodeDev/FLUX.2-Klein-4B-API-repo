# Use an official PyTorch runtime with CUDA 12.1 support out-of-the-box
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    HF_HUB_ENABLE_HF_TRANSFER=0

# Set working directory
WORKDIR /app

# Install system dependencies required for image processing and git
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application source code
COPY . .

# Expose the port Cloud Run expects (defaults to 8080)
EXPOSE 8080

# Run the web application
CMD ["python", "app.py"]
