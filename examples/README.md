# Examples

This directory contains example code and configurations for using Aspida.

## Contents

### Clients

- `openai_client.py` — Python client using OpenAI SDK
- `openai_client.js` — Node.js client using OpenAI SDK
- `curl_examples.sh` — cURL commands for testing

### Configurations

- `docker-compose.yml` — Docker Compose setup
- `nginx.conf` — Nginx reverse proxy with TLS
- `systemd/` — Systemd service files

## Quick Start

### 1. Start the Server

```bash
# Set model path
export QWEN_MODEL_PATH=/path/to/model.gguf

# Start server
./obj/secure_server 8080
```

### 2. Test with cURL

```bash
# List models
curl http://localhost:8080/v1/models

# Chat completion
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen", "messages": [{"role": "user", "content": "Hello!"}]}'
```

### 3. Use with OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="your-token"  # ASPIDA_CLIENT_TOKEN
)

response = client.chat.completions.create(
    model="qwen",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content, end="")
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ASPIDA_GPU` | Enable GPU offload | (disabled) |
| `ASPIDA_MODELS_DIR` | Model search paths | `.` |
| `ASPIDA_CLIENT_TOKEN` | Auth token | (none) |
| `ASPIDA_BIND` | Bind address | `0.0.0.0` |
| `QWEN_MODEL_PATH` | Default model | (required) |

## Docker

```bash
# Build image
docker build -t aspida .

# Run container
docker run -p 8080:8080 \
  -v /path/to/models:/models \
  -e QWEN_MODEL_PATH=/models/qwen.gguf \
  aspida
```

## Production Setup

See `nginx.conf` for TLS termination and `systemd/` for service management.
