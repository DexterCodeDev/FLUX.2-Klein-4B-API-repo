#!/bin/bash

echo "Starting FLUX API..."

# Show GPU info (helpful for Cloud Run logs)
python3 -c "
import torch
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
"

exec uvicorn app:app \
    --host 0.0.0.0 \
    --port ${PORT:-8080} \
    --workers 1
