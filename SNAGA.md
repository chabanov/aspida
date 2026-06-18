# Project: Aspida

> LLM inference engine на Ada/SPARK з GPU offload та OpenAI-compatible API

## Опис

Aspida — високопродуктивний LLM inference engine, написаний на Ada/SPARK. Підтримує прямий запуск моделей з GGUF форматів (Qwen3.5-MoE+SSM, Gemma 3n E4B, Llama 3.x) з GPU прискоренням та криптографічно захищеним каналом зв'язку.

## Архітектура

```
src/
├── llm/           — Inference engine (52 файли)
│   ├── llm_engine.ads/adb       — Unified API, architecture auto-detect
│   ├── llm_backend.ads/adb      — Polymorphic Model_Backend interface
│   ├── llm_qwen.ads/adb         — Qwen3.5-MoE+SSM backend
│   ├── llm_llama.ads/adb        — Llama 3.x / Mistral (dense GQA)
│   ├── llm_gemma.ads/adb        — Gemma 3n E4B backend
│   ├── llm_sampler.ads/adb      — Unified sampling (temp, top-k, top-p)
│   ├── llm_gguf.ads/adb         — GGUF parser (metadata + tensors)
│   ├── llm_dequant.ads/adb      — Q4_K/Q5_K/Q6_K/Q8_K dequantization
│   ├── llm_weight.ads/adb       — Weight abstraction (dense + quantized)
│   ├── llm_tensor.ads/adb       — Tensor operations
│   ├── llm_pool.ads/adb         — Persistent thread pool
│   ├── llm_rope.ads/adb         — RoPE/mRoPE positional encoding
│   ├── llm_tokenizer.ads/adb    — BPE tokenizer (SentencePiece support)
│   ├── llm_catalog.ads/adb      — Model discovery from filesystem
│   └── llm_gpu.ads/adb          — CUDA offload (runtime dlopen)
│
├── crypto/        — Cryptographic primitives (14 файлів)
│   ├── crypto.ads               — SPARK contracts, const-time ops
│   ├── crypto-chacha20.ads/adb  — Stream cipher
│   ├── crypto-poly1305.ads/adb  — MAC
│   ├── crypto-aead.ads/adb      — ChaCha20-Poly1305 AEAD
│   ├── crypto-x25519.ads/adb    — Key exchange
│   ├── crypto-hkdf.ads/adb      — Key derivation
│   ├── crypto-pbkdf2.ads/adb    — Password-based KDF
│   ├── crypto-sha256.ads/adb    — Hash
│   └── crypto-mem.ads/adb       — Secure zeroing
│
├── secure/        — Secure channel (6 файлів)
│   ├── protocol.ads             — Wire protocol (Prompt/Token/Done)
│   ├── secure_channel.ads/adb   — X25519 + ChaCha20-Poly1305
│   └── encrypting_sink.ads/adb  — Streaming encryption
│
├── server/        — HTTP/WebSocket servers (10 файлів)
│   ├── openai.ads/adb           — OpenAI API parser
│   ├── openai_proxy.adb         — /v1/chat/completions endpoint
│   ├── ws_bridge.adb            — WebSocket bridge for browsers
│   └── secure_server.adb        — Concurrent handler pool
│
├── session/       — Session management (4 файли)
│   ├── session_store.ads/adb    — In-memory session cache
│   └── at_rest.ads/adb          — At-rest encryption
│
└── train/         — Training/distillation infrastructure
```

## Підтримувані моделі

| Backend | Архітектура | Квантизація | Статус |
|---------|-------------|-------------|--------|
| **LLM_Qwen** | MoE + SSM (Qwen3.5-35B-A3B) | Q5_K, Q8_K | ✅ Працює |
| **LLM_Llama** | Dense GQA (Llama 3.x, Mistral) | Q4_K, Q6_K | ✅ Працює |
| **LLM_Gemma** | Gemma 3n E4B | Q4_K, Q5_K | ⚠️ Валідація |

## GPU Offload

- Runtime dlopen CUDA shim (`libaspidagpu.so`)
- Q4_K/Q6_K kernels з block-level parallelism
- Активація: `ASPIDA_GPU=1`
- Автоматичний fallback на CPU

## OpenAI API

| Endpoint | Метод | Опис |
|----------|-------|------|
| `/v1/chat/completions` | POST | Streaming chat (SSE) |
| `/v1/models` | GET | Model catalog |
| Auth | Header | `ASPIDA_CLIENT_TOKEN` |

**Compliance:**
- Token accounting (prompt/completion tokens)
- `finish_reason`: "stop" / "length"
- `max_tokens` default: 1M

## Команди

```bash
# Build
make llm-build          # Зібрати inference engine
make gpu                # Зібрати CUDA shim
make openai-proxy       # Зібрати OpenAI proxy server

# Run
make llm-run            # Запустити ChatML REPL
make serve              # Запустити OpenAI-compatible server
make chat               # Підключитись до running server

# Test
make llm-test           # Запустити тести
make test-crypto        # Тести криптографії
make gpu-test           # Тести GPU kernels

# Clean
make llm-clean          # Очистити build artifacts
```

## Ключові файли

| Файл | Опис |
|------|------|
| `src/llm/llm_engine.ads` | Unified API, architecture detection |
| `src/llm/llm_backend.ads` | Polymorphic interface contract |
| `src/llm/llm_gguf.adb` | GGUF parser (header, metadata, tensors) |
| `src/llm/llm_dequant.adb` | Q5_K/Q6_K dequant, QMatVec streaming |
| `src/llm/llm_gpu.ads` | CUDA offload dispatch |
| `src/server/openai_proxy.adb` | OpenAI API server |
| `src/secure/secure_channel.ads` | Encrypted communication |
| `gpu/gpu_matvec.cu` | CUDA kernels for Q4_K/Q6_K |
| `Makefile` | Build targets |
| `server.gpr` | GNAT project file |

## Конвенції

### API Design
- `Model_Backend` interface — поліморфізм через class-wide types
- `Chat(User_Text)` — single-turn, `Chat(Conversation)` — multi-turn
- `Token_Sink` — streaming callback interface

### Error Handling
- `Model_Load_Error` — explicit exception for load failures
- Bounds checking on all tensor operations
- SPARK contracts on crypto primitives

### Threading
- `LLM_Pool` — persistent worker threads (avoid spawn/join overhead)
- `LLM_Step_Lock` — serialize inference across concurrent clients
- Handler pool (8 tasks) + bounded queue (64 pending)

### Memory
- `Weight` abstraction — keep quantized bytes alive, no full FP32 materialise
- `QMatVec` — streaming matvec without intermediate tensors
- Stack-allocated `Block256` for decode hot path

## Змінні середовища

| Змінна | Опис |
|--------|------|
| `ASPIDA_GPU` | Enable GPU offload (any non-empty value) |
| `ASPIDA_MODELS_DIR` | Directory to scan for *.gguf models |
| `ASPIDA_CLIENT_TOKEN` | Auth token for OpenAI API |

## Статус реалізації

**Готово:**
- ✅ GGUF parser (F32/F16/Q4_K/Q5_K/Q6_K/Q8_K)
- ✅ BPE tokenizer (SentencePiece support)
- ✅ ChatML multi-turn conversation
- ✅ Streaming UI with throughput metrics
- ✅ Thread pool + concurrent server
- ✅ GPU offload (Q4_K/Q6_K)
- ✅ OpenAI-compatible API
- ✅ Secure channel (X25519, ChaCha20-Poly1305)
- ✅ Model catalog discovery

**В процесі:**
- 🔄 Gemma 3n validation
- 🔄 SSM selective scan (Mamba)
- 🔄 mRoPE positional encoding

---
*Auto-generated by Snaga. Last updated: 2026-07-06*
