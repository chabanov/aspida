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

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full request lifecycle, backends,
context management, security model, and configuration.

## API

### OpenAI-Compatible Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completions (non-stream + SSE streaming) |
| `/v1/models` | GET | List available models |

Responses follow the OpenAI schema: real `usage` (`prompt_tokens` /
`completion_tokens` / `total_tokens`) and a correct `finish_reason` —
`"stop"` for a natural end-of-turn, `"length"` when the token cap was hit (so a
truncated reply is never reported as complete). Streaming emits a final chunk
carrying `finish_reason` + `usage`. Client `max_tokens` is honored.

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

> **Why `http://` and not `https://`?** The OpenAI-compatible endpoint is a
> **local proxy** that binds `127.0.0.1` only. This hop is plaintext **on
> loopback (the same machine)** — it never touches the network. The proxy then
> seals every request into the AEAD-encrypted, pinned-key channel for the
> **network** hop to the (possibly remote) server. So encryption is real and
> end-to-end *on the wire*; the only plaintext is the same-machine loopback step
> that exists for OpenAI-SDK compatibility. For full end-to-end with **no
> plaintext anywhere** (the client performs the handshake itself), use the
> native CLI (`secure_client`) or the browser client.

## Configuration

### Environment Variables

**Model & serving**

| Variable | Description |
|----------|-------------|
| `QWEN_MODEL_PATH` | Active model GGUF (any supported architecture) |
| `ASPIDA_MODELS_DIR` | Colon-separated paths to scan for GGUF models |
| `ASPIDA_GPU` | Enable GPU offload (any non-empty value) |
| `ASPIDA_MAX_TOKENS` | Per-turn generation cap (default 2048) |
| `ASPIDA_AUTORELOAD` | Allow runtime model switching (supervisor restart) |

**Context window** (Llama backend)

| Variable | Description |
|----------|-------------|
| `ASPIDA_CTX` | Served window (default 4096, clamped to the model's trained context) |
| `ASPIDA_NO_CTX_SHIFT` | Disable context-shift ("infinite generation"); default on |
| `ASPIDA_STRICT_CTX` | Return `context_length_exceeded` instead of trimming an over-long prompt |
| `ASPIDA_ROPE_NTK` | NTK-aware context extension factor for an unscaled model |

**Security & network**

| Variable | Description |
|----------|-------------|
| `ASPIDA_CLIENT_TOKEN` | Require this shared-secret client auth token |
| `ASPIDA_BIND` | Listener address (default `0.0.0.0`; e.g. `127.0.0.1` behind a proxy) |
| `ASPIDA_RATE_MAX` / `ASPIDA_RATE_WINDOW` | New-connection rate limit (default off) |
| `ASPIDA_IDLE_TIMEOUT` | Seconds before a silent connection is reclaimed (default 600) |

**Sampling** (default greedy)

| Variable | Description |
|----------|-------------|
| `ASPIDA_TEMP` / `ASPIDA_TOP_P` / `ASPIDA_TOP_K` | Temperature / nucleus / top-k |
| `ASPIDA_REPEAT_PENALTY` / `ASPIDA_REPEAT_LAST_N` | Repetition penalty + window |
| `ASPIDA_SEED` | RNG seed |

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
- **Trust boundary** — Network traffic is AEAD-sealed end to end; the only
  plaintext is the same-machine loopback hop on the OpenAI proxy (binds
  `127.0.0.1` only). Native/browser clients have no plaintext hop at all.

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

## Further Reading

- [What the Model Sees: The First Level Solved](https://the-platform.example/blog/what-the-model-sees-the-first-level-solved) — Deep dive into tokenization, context windows, and how LLMs process text

---

**Note:** This project is under active development. APIs may change between versions.
