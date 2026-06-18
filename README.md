# Aspida

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ada/SPARK](https://img.shields.io/badge/Language-Ada%2FSPARK-blue.svg)](https://ada-lang.io)
[![GitHub](https://img.shields.io/github/stars/chabanov/aspida?style=social)](https://github.com/chabanov/aspida)

**End-to-end encrypted LLM inference engine, built from scratch in Ada/SPARK.**

Aspida is a high-performance LLM inference engine with its own GGUF parser, quantized weight kernels, and cryptographic stack — no third-party crypto or ML libraries. Prompts are sealed in the client and stay sealed across the network; only the inference boundary ever opens them.

## Features

- 🚀 **Native Performance** — Written in Ada/SPARK, compiled to native code
- 🔐 **End-to-End Encryption** — X25519 key exchange + ChaCha20-Poly1305 AEAD
- 🎯 **OpenAI-Compatible API** — Drop-in replacement for OpenAI SDKs
- ⚡ **GPU Offload** — CUDA kernels for Q4_K/Q6_K quantization
- 🧠 **Multiple Backends** — Qwen3.5-MoE+SSM, Llama 3.x, Gemma 3n
- 📦 **GGUF Support** — Direct loading from quantized GGUF files
- 🔒 **SPARK Proven** — Cryptographic core verified with SPARK contracts

## Supported Models

| Architecture | Models | Quantization | Status |
|--------------|--------|--------------|--------|
| **Qwen3.5-MoE+SSM** | Qwen3.5-35B-A3B | Q5_K, Q8_K | ✅ Production |
| **Llama 3.x** | Llama 3.1/3.2, Mistral | Q4_K, Q6_K | ✅ Production |
| **Gemma 3n** | Gemma 3n E4B | Q4_K, Q5_K | ⚠️ Validation |

## Quick Start

### Prerequisites

- GNAT Community 2021+ or GNAT Pro
- CUDA Toolkit 12+ (optional, for GPU offload)
- Make, Git

### Build

```bash
# Clone the repository
git clone https://github.com/chabanov/aspida.git
cd aspida

# Build the server
make server

# (Optional) Build CUDA kernels
make gpu
```

### Run

```bash
# Start the server with a model
QWEN_MODEL_PATH=/path/to/model.gguf ./obj/secure_server 8080

# Or use the convenience target
make serve
```

### Connect

```bash
# Interactive client
make chat

# Or point any OpenAI SDK at http://localhost:8080/v1
```

## Architecture

```
src/
├── llm/           — Inference engine (GGUF, backends, tokenizer, GPU)
├── crypto/        — Cryptographic primitives (ChaCha20, Poly1305, X25519)
├── secure/        — Secure channel (handshake, AEAD records)
├── server/        — HTTP/WebSocket servers (OpenAI API, WebSocket bridge)
├── session/       — Session management, at-rest encryption
└── train/         — Training/distillation infrastructure
```

## API

### OpenAI-Compatible Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Streaming chat completions (SSE) |
| `/v1/models` | GET | List available models |

### Example Request

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ASPIDA_CLIENT_TOKEN" \
  -d '{
    "model": "qwen3.5-35b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

## Configuration

### Environment Variables

| Variable | Description |
|----------|-------------|
| `ASPIDA_GPU` | Enable GPU offload (any non-empty value) |
| `ASPIDA_MODELS_DIR` | Colon-separated paths to scan for GGUF models |
| `ASPIDA_CLIENT_TOKEN` | Authentication token for API access |
| `ASPIDA_BIND` | Bind address (default: 0.0.0.0) |
| `QWEN_MODEL_PATH` | Path to default model GGUF |

## Testing

```bash
# Run all tests
make test

# Crypto-specific tests (RFC test vectors)
make test-crypto

# LLM unit tests
make test-llm

# SPARK verification
make prove
```

## Security

- **No third-party crypto** — All cryptographic primitives implemented from scratch
- **SPARK contracts** — Cryptographic core verified for absence of runtime errors
- **Constant-time operations** — Timing-safe comparisons, secure memory wiping
- **Forward secrecy** — Ephemeral key exchange per session

## Performance

- **Streaming matvec** — No full FP32 materialization, quantized weights stay quantized
- **Thread pool** — Persistent workers avoid spawn/join overhead
- **GPU kernels** — Block-level parallelism for Q4_K/Q6_K
- **Stack allocation** — Hot-path decode uses stack-allocated blocks

## Roadmap

- [ ] Gemma 3n validation
- [ ] SSM selective scan (Mamba)
- [ ] mRoPE positional encoding
- [ ] Multi-GPU support
- [ ] Quantization-aware training

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting PRs.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [Ada/SPARK](https://ada-lang.io)
- Inspired by [llama.cpp](https://github.com/ggerganov/llama.cpp)
- Cryptographic primitives follow [RFC 8439](https://tools.ietf.org/html/rfc8439) (ChaCha20-Poly1305)

---

**Note:** This project is under active development. APIs may change between versions.
