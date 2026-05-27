# FLUX.2 Klein 4B Cloud Run API

Production-ready API deployment for:

black-forest-labs/FLUX.2-klein-4B

Built for:

- Google Cloud Run
- NVIDIA L4 GPU
- FastAPI
- Diffusers
- GitHub auto deploy

---

# Features

- Warm model loading
- Full parameter control
- Seed support
- Health endpoint
- Binary image responses
- Cloud Run optimized
- GPU inference
- Ready for Cloudflare middleware later

---

# API Endpoints

## Health Check

GET /health

Response:

```json
{
  "status": "healthy",
  "service": "flux2-klein-4b"
}
```

---

## Generate Image

POST /generate

### Request JSON

```json
{
  "prompt": "luxury black evening dress",
  "negative_prompt": "",
  "width": 1024,
  "height": 1024,
  "steps": 4,
  "guidance_scale": 3.5,
  "seed": 12345,
  "output_format": "webp"
}
```

### Parameters

| Field | Type | Default |
|-------|------|----------|
| prompt | string | required |
| negative_prompt | string | "" |
| width | int | 1024 |
| height | int | 1024 |
| steps | int | 4 |
| guidance_scale | float | 3.5 |
| seed | int/null | null |
| output_format | string | webp |

Supported formats:

- webp
- png
- jpeg
- jpg

Returns:

Binary image response.

Content type example:

```http
image/webp
```

---

# Google Cloud Run Settings

Recommended:

## Resources

CPU:
```txt
8
```

RAM:
```txt
32Gi
```

GPU:
```txt
NVIDIA L4
```

---

## Scaling

Concurrency:
```txt
1
```

Minimum instances:
```txt
1
```

Maximum instances:
```txt
2
```

CPU allocation:
```txt
CPU always allocated
```

Request timeout:
```txt
3600
```

---

# Environment Variables

Set inside Cloud Run:

```env
HF_TOKEN=your_huggingface_token
MODEL_ID=black-forest-labs/FLUX.2-klein-4B
HF_HOME=/tmp/huggingface
MAX_IMAGE_SIZE=2048
```

---

# Deploy

1. Push repo to GitHub

2. Open Google Cloud Run

3. Click:

Create Service → Continuously deploy from repository

4. Select your GitHub repo

5. Configure:

Build Type:
```txt
Dockerfile
```

Port:
```txt
8080
```

6. Enable GPU:

```txt
NVIDIA L4
```

7. Set environment variables

8. Deploy

---

# Test Health Endpoint

Example:

```bash
curl https://YOUR_URL/health
```

Expected:

```json
{
  "status": "healthy",
  "service": "flux2-klein-4b"
}
```

---

# Generate Example

WEBP output:

```bash
curl -X POST "https://YOUR_URL/generate" \
-H "Content-Type: application/json" \
-o result.webp \
-d '{
  "prompt":"luxury black evening dress ecommerce photography",
  "steps":4,
  "guidance_scale":3.5,
  "width":1024,
  "height":1024,
  "seed":12345,
  "output_format":"webp"
}'
```

PNG output:

```bash
curl -X POST "https://YOUR_URL/generate" \
-H "Content-Type: application/json" \
-o result.png \
-d '{
  "prompt":"fashion photoshoot luxury red gown",
  "steps":6,
  "guidance_scale":4,
  "width":1024,
  "height":1024,
  "output_format":"png"
}'
```

---

# Recommended Defaults

Fast + good quality:

```json
{
  "steps": 4,
  "guidance_scale": 3.5,
  "width": 1024,
  "height": 1024
}
```

Higher quality:

```json
{
  "steps": 8,
  "guidance_scale": 4,
  "width": 1024,
  "height": 1024
}
```

---

# Future Improvements

After successful deployment:

- Cloudflare Worker prompt rewriting
- Authentication
- API keys
- Rate limiting
- Async job queue
- Cloudflare R2 storage
- LoRA support
- Image editing endpoint
- Multi-reference generation
