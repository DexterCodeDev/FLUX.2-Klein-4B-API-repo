# Use a lightweight Python base image
FROM python:3.10-slim

# Install system dependencies required for image processing libraries
# Updated to use 'libgl1' instead of the deprecated 'libgl1-mesa-glx'
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY app.py .

# Expose the default Cloud Run port
ENV PORT=8080
EXPOSE 8080

# Start the application
CMD ["python", "app.py"]
