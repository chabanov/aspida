# Aspida

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ada/SPARK](https://img.shields.io/badge/Language-Ada%2FSPARK-blue.svg)](https://ada-lang.io)
[![GitHub](https://img.shields.io/github/stars/chabanov/aspida?style=social)](https://github.com/chabanov/aspida)

**End-to-end encrypted LLM inference and training engine, built from scratch in Ada/SPARK.**

Aspida is a from-scratch LLM stack: its own GGUF parser, dequantization kernels, tokenizer, sampler, RoPE, KV cache, an encrypted client–server channel, and a training/distillation engine — no third-party crypto or ML libraries. A prompt is sealed in the client and stays sealed across the network; only the trusted inference boundary ever opens it. Conversation history is encrypted at rest with PBKDF2 + ChaCha20-Poly1305.

The same engine serves Llama, Qwen (dense and MoE+SSM), and Gemma backends over a Noise-NK-style AEAD channel, exposes an OpenAI-compatible API, and is also available as a C-ABI dynamic library (`libaspida`) for in-process, no-network integration.

## Features

- 🚀 **Native Performance** — Written in Ada/SPARK, compiled to native code
- 🔐 **End-to-End Encryption** — X25519 key exchange + ChaCha20-Poly1305 AEAD, forward-secret per session
- 🎯 **OpenAI-Compatible API** — Drop-in replacement for OpenAI SDKs (`/v1/chat/completions` + SSE streaming)
- ⚡ **GPU Offload** — CUDA kernels for all five K-quants (Q2_K–Q6_K), loaded via `dlopen` with transparent CPU fallback (no link-time CUDA dependency)
- 🧠 **Multiple Backends** — Llama 3.x / Mistral, Qwen3.5 (dense + MoE+SSM), Gemma 4 (dense) behind one unified API
- 📦 **GGUF Support** — Reads 12 ggml formats (F32/F16/BF16, Q8_0/Q4_0/Q5_0, Q2_K–Q6_K, Q8_K); the trainer exports six (Q8_0/Q4_0/Q5_0/Q4_K/Q5_K/Q6_K) so a model trained here is served here
- 🔧 **C ABI** — `libaspida.dylib` exposes the engine to foreign hosts (Swift/C) for in-process inference with no server and no network
- 🔒 **SPARK-Verified Core** — ChaCha20, SHA-256, HKDF & PBKDF2 proved to AoRTE + functional contracts (`make prove`); flow analysis (init / data deps / non-aliasing) across the rest of the crypto library (`make prove-flow`). X25519, Poly1305 & AEAD remain on flow analysis pending field-arithmetic annotations.

## Supported Models

Aspida dispatches on the GGUF `general.architecture` field through a one-row-per-architecture registry — any GGUF carrying a supported architecture loads without code changes. Quantization support is arch-independent: every backend reads all 12 ggml formats below.

| GGUF architecture | Backend | Model families | Status |
|---|---|---|---|
| `llama` | Dense Llama — GQA + RMSNorm + SwiGLU + NeoX RoPE | Llama 3.1 / 3.2, Mistral, and any dense GQA+RMSNorm+SwiGLU+RoPE model | ✅ Validated — Llama-3.2-1B (Q3_K_L) bit-correct vs llama.cpp; Llama-8B & 70B served on NVIDIA GPU |
| `qwen2` | Qwen — dense path | Qwen2 0.5B / 1.5B / 7B | ✅ Supported (routed to the Qwen dense path) |
| `qwen35` | Qwen — dense path | Qwen3.5 dense, Hura-9b | ✅ Validated — Hura-9b end-to-end via the C ABI (Swift bridge): reasoning/code prompts emit coherent multi-hundred-token turns |
| `qwen35moe` | Qwen — MoE top-k router + shared expert + gated DeltaNet/SSM hybrid | Qwen3.5-MoE 35B-A3B | ✅ Validated — tensor shapes and hyperparameters verified against the real GGUF |
| `gemma4` | Gemma — PLE + shared-KV + sliding-window attention + dual RoPE + logit softcap | Gemma 3n E4B, 12B, 26B (PLE and non-PLE/MQA variants) | ✅ Validated — greedy decode bit-identical to llama.cpp on real E4B + 12B models |

**Quantization (read):** F32, F16, BF16, Q8_0, Q4_0, Q5_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K — 12 ggml formats, with fused decode+dot paths for all five K-quants. Unsupported types (IQ\*, ternary, Q4_1/Q5_1/Q8_1) are rejected at load with a clear error — never silently zero-filled.

**Quantization (write):** the from-scratch trainer exports Q8_0, Q4_0, Q5_0, Q4_K, Q5_K, Q6_K — so a model trained in Aspida is served by Aspida.

**Not supported (intentional):** MoE gemma4 (e.g. supergemma-26B, 128 routed experts) is rejected at load with a clear error; only the dense gemma4 path is implemented.

## Quick Start

### Prerequisites

- Alire GNAT 15+ toolchain (the Makefile auto-detects it; the system GNAT is not used)
- CUDA Toolkit 12+ (optional, for GPU offload)
- Make, Git

On a bare Ubuntu/Debian box, `scripts/setup-linux.sh` installs Alire and the
GNAT/gprbuild toolchain into the exact location the Makefile looks for.

### Build

```bash
# Clone the repository
git clone https://github.com/chabanov/aspida.git
cd aspida

# Build the server
make server

# (Optional) Build the CUDA shim — not a Make target; build_so.sh pins the
# llama.cpp/ggml commit it links against (see gpu/README.md)
GG=/path/to/llama.cpp ./build_so.sh        # then run with ASPIDA_GPU=1 + ASPIDA_GPU_LIB=…
```

### Run

```bash
# Start the encrypted server with a model (8765 is the default port).
# On first run it generates its static keypair and writes the public half
# to server_pub.hex — clients pin that key.
QWEN_MODEL_PATH=/path/to/model.gguf ./obj/secure_server 8765

# Or use the convenience target
make serve
```

### Connect

The server speaks the AEAD channel protocol, **not** HTTP — an OpenAI SDK
pointed straight at it gets ciphertext. Pick a client:

```bash
# Interactive encrypted client (does the handshake itself)
make chat
```

```bash
# Or, for OpenAI SDKs: run the local proxy, which does the handshake for you
# and tunnels every request over the channel. It binds 127.0.0.1 ONLY.
#   openai_proxy <server_host> <server_port> <server_pub_hex> [local_port]
./obj/openai_proxy 127.0.0.1 8765 "$(cat server_pub.hex)" 8080

# Now any OpenAI SDK works against http://localhost:8080/v1 (any api_key).
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
| `/api/chat` | POST | Ollama-native chat (same channel, Ollama's request/response shape) |

`/api/chat` is there so tools that already speak Ollama work unchanged: it
accepts Ollama's body (`options.num_predict`, `think`) and answers in Ollama's
shape (`message`, `thinking`). A bare model name means the `:latest` tag, as
Ollama does.

Responses follow the OpenAI schema: real `usage` (`prompt_tokens` /
`completion_tokens` / `total_tokens`) and a correct `finish_reason` —
`"stop"` for a natural end-of-turn, `"length"` when the token cap was hit (so a
truncated reply is never reported as complete). Streaming emits a final chunk
carrying `finish_reason` + `usage`. Client `max_tokens` is honored.

### Example Request

There are two client paths. **Both are end-to-end encrypted on the wire** — the difference is whether you want OpenAI-SDK compatibility (a same-machine loopback hop) or zero plaintext anywhere.

**Path A — OpenAI-compatible proxy (use any OpenAI SDK).** You talk to a local proxy on `127.0.0.1`; it seals your request into the AEAD channel and relays it to the (possibly remote) server. The `http://` below is loopback-only on your own machine — it never touches the network. The network hop is encrypted.

```bash
# The proxy binds 127.0.0.1 only. The prompt is sealed before it leaves this machine.
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ASPIDA_CLIENT_TOKEN" \
  -d '{
    "model": "qwen3.5-35b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

```
your app/SDK ──http(loopback, same machine)──► openai_proxy ──► AEAD-sealed channel ──► secure_server
                                                  (X25519 handshake + ChaCha20-Poly1305 records, pinned server key)
```

**Path B — native client (no plaintext anywhere).** The client performs the Noise-NK handshake itself and seals the prompt before any bytes hit the socket; the server only decrypts inside the trusted inference boundary. No loopback hop, no proxy, no plaintext on any link.

```bash
make chat                       # native encrypted REPL (SESSION=<id> to resume)
./obj/secure_client <host> <port>   # or call the binary directly
```

```
native/browser client ──AEAD-sealed channel──► secure_server   (handshake done in the client; no plaintext on the wire)
```

> **Why does Path A use `http://`?** TLS would be redundant: the proxy binds
> `127.0.0.1` only, so that hop never leaves your machine. The actual
> network transit is handled by the AEAD-encrypted channel, not TLS. If you
> want no same-machine plaintext hop at all, use Path B.

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
| `ASPIDA_TEMP` / `ASPIDA_TOP_P` / `ASPIDA_TOP_K` / `ASPIDA_MIN_P` | Temperature / nucleus / top-k / min-p |
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
- **SPARK proof, honestly scoped** — the crypto root, ChaCha20, SHA-256, HKDF and
  PBKDF2 are machine-proved free of runtime errors and against functional
  contracts (`make prove`). Everything else — including X25519, Poly1305 and the
  AEAD layer — passes SPARK **flow** analysis only; those field-arithmetic proofs
  are future work. Read any "formally verified" claim with that scope
  (ARCHITECTURE.md §6)
- **Constant-time where it matters** — X25519's ladder is branch-free on secrets
  (no `if` on a secret; verified via `objdump`), comparisons are constant-time and
  keys are zeroized. Poly1305/AEAD are **not** hardened against timing analysis
- **Forward secrecy** — Ephemeral key exchange per session
- **Trust boundary** — Network traffic is AEAD-sealed end to end; the only
  plaintext is the same-machine loopback hop on the OpenAI proxy (binds
  `127.0.0.1` only). Native/browser clients have no plaintext hop at all.

## Performance

- **Streaming matvec** — No full FP32 materialization, quantized weights stay quantized
- **Fused K-quant kernels** — All five K-quants (Q2_K–Q6_K) decode+dot in one stack-local pass
- **Thread pool** — Persistent workers avoid spawn/join overhead
- **GPU kernels** — Block-level parallelism for all five K-quants (Q2_K–Q6_K), scalar / warp-per-row / warp-batched variants
- **Stack allocation** — Hot-path decode uses stack-allocated blocks

## Roadmap

- [x] Gemma validation (gemma4 E4B — real-model smoke test)
- [x] Quantization-aware training (fake-quant + STE, demonstrated 2-bit robustness)
- [x] Full GGML quant coverage — read all standard formats (Q2_K–Q6_K, Q4_0/Q5_0/Q8_0); export six formats from the trainer
- [x] GPU kernels for Q2_K/Q3_K — CUDA matvec/matmul (kinds 3/4), validated on an NVIDIA GPU vs CPU reference (`gpu/test_matvec.cu`)
- [x] mRoPE positional encoding — per-section rotation (`Apply_Sections`) implemented
- [x] Gated DeltaNet/SSM hybrid — implemented in the Qwen MoE backend
- [ ] Standalone Mamba selective-scan GGUF (separate from the DeltaNet/SSM hybrid above)
- [ ] Multi-GPU support

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
