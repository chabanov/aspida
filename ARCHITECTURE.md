# Aspida — Architecture

Aspida is an end-to-end encrypted LLM inference engine written from scratch in
Ada/SPARK — its own GGUF parser, quantized kernels, tokenizer, sampler, and
cryptographic channel, with **no third-party crypto or ML libraries**. A prompt
is sealed on the client and stays sealed across the network; it is opened only
inside the inference boundary, and the conversation is sealed again at rest.

This document describes how the pieces fit together. It reflects the current
state of the code, not aspirations.

---

## 1. Request lifecycle (end to end)

```
  client (CLI / browser / OpenAI SDK)
     │  prompt sealed in an AEAD-encrypted, pinned-key channel
     ▼
  transport ── ws_bridge / openai_proxy ──► secure_server  (only loopback hop is plaintext, on the proxy)
     │  per-direction AEAD records over a forward-secret handshake
     ▼
  LLM_Engine  ── architecture registry ──►  backend (Qwen / Llama / Gemma)
     │                                          │
     │  continuous-batch scheduler (Llama)      ├─ GGUF weights (mmap-style, on-demand)
     │                                          ├─ quantized matvec (CPU SIMD / GPU shim)
     ▼                                          └─ KV cache + RoPE + sampler
  tokens stream back through the same sealed channel; turns persisted sealed at rest
```

The server terminates the channel and therefore sees plaintext **only** within
that one trusted boundary (a trusted-server topology). Encryption protects the
prompt/response **in transit** and the conversation history **at rest**.

**Two client paths, one honest caveat:**

- **Native CLI / browser client** — perform the handshake themselves and seal
  the prompt *in the client*. There is **no plaintext hop anywhere** outside the
  application; this is the fully end-to-end path.
- **OpenAI-SDK clients** — speak plain HTTP and cannot do our handshake, so they
  talk to a **local proxy** (`openai_proxy`) that binds `127.0.0.1` **only**.
  The SDK→proxy hop is plaintext **on loopback (same machine, never on the
  network)**; the proxy then seals each request into the AEAD channel for the
  network hop to the server. So encryption is real and end-to-end *on the wire* —
  the single plaintext segment is the same-machine loopback step that exists for
  SDK compatibility. It is not a network exposure (the proxy is unreachable off
  the loopback interface), but it is the reason this path is not "no plaintext
  anywhere."

---

## 2. Source layout

| Directory | Role |
|---|---|
| `src/llm/` | Inference engine: GGUF loader, backends, quant kernels, tokenizer, sampler, RoPE, GPU shim, KV-cache/paging |
| `src/crypto/` | From-scratch primitives + AEAD (X25519, ChaCha20-Poly1305, SHA-256, HMAC, HKDF, PBKDF2) |
| `src/secure/` | Secure channel (Noise-NK-style handshake) + record protocol |
| `src/server/` | Concurrent encrypted server, OpenAI-compatible API, local proxy, native client |
| `src/session/` | Session store + at-rest sealing |
| `src/train/` | From-scratch training/distillation engine + BPE tokenizer trainer + GGUF export |

Build is GPR-based (`shared.gpr` carries the strict profile `-gnatwa -gnatwe
-gnatX`); `server.gpr`, `train.gpr`, `crypto*.gpr`, `secure_tests.gpr`,
`tests/llm_tests.gpr` build the binaries and test suites.

---

## 3. Inference engine

### Backends (unified `LLM_Backend.Model_Backend` interface)

| Backend | Architecture | Status |
|---|---|---|
| `LLM_Qwen` | Qwen3.5 MoE (top-k router + shared expert) + gated DeltaNet/SSM hybrid | ✅ |
| `LLM_Llama` | Dense GQA + SwiGLU, interleaved RoPE (Llama 3.x, Mistral) | ✅ |
| `LLM_Gemma` | gemma4 / 3n E4B: PLE, shared-KV, sliding-window attention, dual RoPE, logit softcap | ✅ |

`LLM_Engine` detects `general.architecture` from the GGUF and dispatches to the
matching backend via a one-row registry — adding an architecture needs no
changes to the engine internals.

### GGUF + quantization

The GGUF loader reads metadata + tensor descriptors and accesses tensor data on
demand (no full-model copy). Twelve dequantization types are implemented to the
ggml byte layout — every standard weight format ggml emits:
**F32, F16, BF16, Q8_0, Q4_0, Q5_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K**
(only the IQ*/imatrix and ternary families are unsupported). An unimplemented
type is rejected at load time with a clear error (never a silent zero-fill). A
streaming `QMatVec` computes `y = W·x` one row at a time so the full FP32 weight
is never materialized; all five K-quants (Q2_K–Q6_K) have fused decode+dot paths
with a multi-lane FP reduction for auto-vectorization.

The training engine's exporter (`src/llm/llm_quant.adb`) is the inverse: it
**writes** the six weight-quant formats the engine reads — Q8_0, Q4_0, Q5_0,
Q4_K, Q5_K, Q6_K — each validated by round-trip against the dequantizer. So a
model trained here can be served here in any of those formats.

### Tokenizer & sampler

`LLM_Tokenizer` implements byte-level BPE (GPT-2 byte↔unicode bijection) and
SentencePiece with `<0xHH>` byte fallback, driven by the GGUF vocab + merges.
`LLM_Sampler` provides temperature / top-k / top-p / repetition-penalty
(greedy at `temp ≤ 0`), seedable.

### RoPE & long context

`LLM_RoPE` supports the split-half (Qwen/Gemma) and interleaved (Llama)
conventions. Three context-extension methods are implemented and unit-tested
(all no-ops at scale 1.0): **linear (Position Interpolation)**, **NTK-aware base
scaling**, and **YaRN** (per-dimension extrapolate/interpolate ramp + attention
temperature), read from `rope.scaling.{type,factor,original_context_length}`.

### GPU offload

`LLM_GPU` `dlopen`s a CUDA shim (`libaspidagpu.so`) at runtime; matvec routes
through it with transparent CPU fallback if absent. Enabled with `ASPIDA_GPU=1`.

---

## 4. Serving & concurrency

- **Secure channel** (`src/secure`, `src/crypto`): a Noise-NK-style handshake
  (ephemeral key exchange, pinned server static key, forward secrecy, low-order
  point rejection), then per-direction AEAD records with monotonic nonces and a
  frame-size cap. Optional shared-secret **client authentication** (`Tag_Auth`,
  `ASPIDA_CLIENT_TOKEN`). The browser client performs the same handshake in
  hand-written JS; `ws_bridge` relays ciphertext only.
- **Concurrent server** (`secure_server`): a bounded handler pool; the Llama
  backend runs a **continuous-batch scheduler** so concurrent sessions share
  batched forward passes. Global new-connection **rate limiting**
  (`ASPIDA_RATE_MAX`/`ASPIDA_RATE_WINDOW`, opt-in).
- **OpenAI-compatible API** (`openai`, `openai_proxy`): `POST /v1/chat/
  completions` (non-stream + SSE streaming), `GET /v1/models` (full catalog).
  Real token `usage`, correct `finish_reason` (`stop` vs `length`), client
  `max_tokens` honored.
- **Context management** (competitor-standard): model-aware window
  (`min(model ctx, ASPIDA_CTX)`), turn-aware trimming that pins the system
  prompt + recent turns, optional `context_length_exceeded` strict mode, and
  **context-shift** ("infinite generation" — evict oldest KV, keep attention
  sinks, re-rotate retained keys). A paged KV-cache allocator with refcounted
  prefix sharing (`KV_Pool`) is implemented and tested as the foundation for
  future paged attention.
- **Sessions** (`session_store`, `at_rest`): per-session history sealed at rest
  (password-derived key); session ids validated against path traversal.

---

## 5. Training engine (`src/train`)

A from-scratch autograd core (explicit forward/backward pairs, finite-difference
gradient-checked), AdamW with optional gradient clipping and **optimizer-state
checkpointing**, logit-level **knowledge distillation** (teacher → student), a
generic multi-layer transformer `Student`, a **BPE tokenizer trainer**, and a
GGUF exporter — so a model trained from scratch (with its own learned
vocabulary) loads and runs in this same engine. CUDA training kernels are
reachable via a C-ABI shim (`ASPIDA_TRAIN_GPU`).

---

## 6. Security model

The trust boundary is the server: it sees plaintext only while computing the
forward pass. What Aspida guarantees today:

- **In transit:** all **network** traffic is AEAD-sealed, on a channel whose
  server identity is pinned and whose session keys are forward-secret. Native
  and browser clients seal in the client (no plaintext hop); OpenAI-SDK clients
  have one same-machine loopback plaintext hop on the local proxy (see §1).
- **At rest:** conversation history is encrypted with a password-derived key.
- **Provenance:** the entire crypto stack is implemented in Ada/SPARK with no
  third-party dependencies, with constant-time comparison and key zeroization
  where it matters.

**SPARK proof scope (stated honestly):** the whole codebase compiles in SPARK
mode (flow analysis: data/initialization/aliasing). Absence-of-runtime-error
and selected functional proofs (`--mode=all`) currently cover the core
primitives (ChaCha20, SHA-256, and the crypto root); extending machine-checked
proofs across the session/AEAD layer is ongoing. Claims of "formally verified"
should be read with this scope.

---

## 7. Configuration (environment)

| Variable | Effect |
|---|---|
| `QWEN_MODEL_PATH` | Active model path (any supported GGUF) |
| `ASPIDA_GPU` | Enable GPU offload |
| `ASPIDA_CTX` | Served context window (≤ model context) |
| `ASPIDA_MAX_TOKENS` | Per-turn generation cap (default 2048) |
| `ASPIDA_NO_CTX_SHIFT` | Disable context-shift |
| `ASPIDA_STRICT_CTX` | Return `context_length_exceeded` instead of trimming |
| `ASPIDA_ROPE_NTK` | NTK-aware context extension for an unscaled model |
| `ASPIDA_CLIENT_TOKEN` | Require client auth token |
| `ASPIDA_BIND` | Restrict the listener address |
| `ASPIDA_RATE_MAX` / `ASPIDA_RATE_WINDOW` | New-connection rate limit |
| `ASPIDA_TEMP` / `ASPIDA_TOP_P` / `ASPIDA_TOP_K` / `ASPIDA_REPEAT_PENALTY` / `ASPIDA_SEED` | Sampling |

---

## 8. Design principles

1. **Own the whole stack** — no third-party crypto or ML libraries; every layer
   is readable and auditable Ada/SPARK.
2. **Ada/SPARK as the source of guarantees** — strict warnings-as-errors, SPARK
   contracts, and explicit fail-loud behavior over silent degradation.
3. **Fail loud, not silent** — unsupported quant, malformed input, or an
   over-window prompt produce clear errors, not garbage.
4. **Validated changes** — backends and kernels are checked against references
   (gradient checks, dense-vs-fused matvec equivalence, real-model generation).
