FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- NEW: Download weights into the Docker image ---
# You will need to pass your HF token as a build argument
ARG HF_TOKEN
ENV HF_TOKEN=$HF_TOKEN

RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='black-forest-labs/FLUX.2-klein-4B', token='$HF_TOKEN')"
# ---------------------------------------------------

COPY app.py .

ENV PORT=8080
EXPOSE 8080

CMD ["python", "app.py"]
