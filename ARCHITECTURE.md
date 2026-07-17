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
`LLM_Sampler` provides temperature / top-k / top-p / min-p / repetition-penalty
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
reachable via a C-ABI shim (`ASPIDA_TRAIN_GPU`). GPU-resident *training* (Stage 1)
is proven by `gpu/train_mlp.cu` — weights + AdamW + data resident on the device,
loss collapses, validated on an NVIDIA GPU; multi-GPU/distributed training is the
documented roadmap in `gpu/README.md`.

**Distillation teachers.** `Distill.Teacher` is a small interface (`Vocab` +
`Forward → per-position logits`); any served model implements it via a thin
adapter — `Teacher_Llama`, `Teacher_Qwen`, `Teacher_Gemma` — backed by each
backend's `Forward_Logits`. So an existing Llama / Qwen-MoE / Gemma model
teaches a new student through the very engine that serves it.

**Multi-teacher (ensemble) distillation.** `Distill.Capture_Ensemble` lets
several models teach one student at once: it averages each teacher's
temperature-scaled softmax in *probability* space (optionally weighted) and
keeps the top-K of that blend. Combining in probability space is the only sound
merge — a union of per-teacher top-K sets is lossy/unbounded and averaging raw
logits across models is meaningless. Because the result is an ordinary `Sample`,
the KL-training loop and on-disk dataset are unchanged. All teachers must share
the student's tokenizer/vocabulary (`Vocab_Mismatch` otherwise). Note that
per-teacher weighted KL is *mathematically identical* to a single KL against the
ensemble target (same gradient), so the ensemble path subsumes it.

**Exceeding the teacher (verifier-driven).** Pure imitation is bounded by the
teacher; to do better the student needs a signal the teacher's distribution
lacks. An **executable verifier** is exactly that. `Code_DSL` is a tiny
program-synthesis task whose `Verify` *runs* a candidate program on test inputs
(functional correctness). Two engines build on it: **verifier-filtered
distillation** (`code_distill`) trains the student only on a noisy teacher's
verified-correct outputs — it reaches 100% where the teacher sits at 40% and
naive imitation collapses to 20% (it learns the teacher's systematic error);
**verifier-bootstrapped self-improvement** (`code_iterate`, STaR-style) removes
the teacher entirely — the model proposes (grammar-constrained sampling), the
verifier filters, correct programs accumulate, a fresh student retrains and
becomes the next proposer (keep-best across rounds), climbing from a random
proposer to full coverage. The student's ceiling is the verifier's quality.
`make distill-demos` runs both; both are model-free.

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
and functional proofs (`--mode=all --level=2`) cover exactly five units — the
crypto root, ChaCha20, SHA-256, HKDF and PBKDF2 — which is what `make prove`
runs. X25519, Poly1305 and the AEAD layer are flow-analysed but **not**
AoRTE-proved; their field arithmetic is future work, and Poly1305/AEAD are not
hardened against timing analysis either. X25519 *is* constant-time by design
(branch-free on secrets). Claims of "formally verified" should be read with this
scope.

---

## 7. Configuration (environment)

The knobs an operator needs. (The engine reads more, but the rest are
diagnostic or experimental switches — `ASPIDA_*_PROF`, `ASPIDA_NO_*`,
`ASPIDA_BATCH_*` — plus a few the CUDA shim reads directly via `getenv`.)

**Model & serving**

| Variable | Effect |
|---|---|
| `QWEN_MODEL_PATH` | Active model path (any supported GGUF). Historical name: it accepts any architecture. No built-in default |
| `ASPIDA_MODELS_DIR` | `:`-separated scan paths for the `/v1/models` catalog. A **deployment pin**: when set, it is the whole list and the discovery fallbacks are skipped |
| `ASPIDA_MAX_LOADED_MODELS` | Resident model slots (default 3, clamped 1..64). Slot 1 is the pinned default and is never evicted |
| `ASPIDA_AUTORELOAD` | Allow runtime model switching (exits for a supervisor to restart; used by `make serve`) |
| `ASPIDA_MAX_TOKENS` | Per-turn generation cap (default 2048) |

**GPU**

| Variable | Effect |
|---|---|
| `ASPIDA_GPU` | Enable GPU offload (transparent CPU fallback if the shim is absent) |
| `ASPIDA_GPU_LIB` | Path to `libaspidagpu.so` (default `./libaspidagpu.so`) |

**Context**

| Variable | Effect |
|---|---|
| `ASPIDA_CTX` | Served context window (≤ model context) |
| `ASPIDA_NO_CTX_SHIFT` | Disable context-shift |
| `ASPIDA_STRICT_CTX` | Return `context_length_exceeded` instead of trimming |
| `ASPIDA_ROPE_NTK` | NTK-aware context extension for an unscaled model |

**Network & anti-DoS**

| Variable | Effect |
|---|---|
| `ASPIDA_BIND` | Restrict the listener address (e.g. `127.0.0.1`) |
| `ASPIDA_CLIENT_TOKEN` | Require client auth token |
| `ASPIDA_RATE_MAX` / `ASPIDA_RATE_WINDOW` | New-connection rate limit (default off) |
| `ASPIDA_HANDSHAKE_TIMEOUT` | Seconds a peer may stall before its handshake is dropped (default 10) |
| `ASPIDA_IDLE_TIMEOUT` | Seconds a connection may stay silent (default 600) |
| `ASPIDA_SEND_TIMEOUT` | Seconds a blocked send may pin a handler when the peer stops reading (default 5) |

**At rest**

| Variable | Effect |
|---|---|
| `ASPIDA_STORE_PASSWORD` | Seal session history at rest (PBKDF2, 600k iterations + ChaCha20-Poly1305). Unset ⇒ history is stored in the clear |

**Sampling & diagnostics**

| Variable | Effect |
|---|---|
| `ASPIDA_TEMP` / `ASPIDA_TOP_P` / `ASPIDA_TOP_K` / `ASPIDA_MIN_P` / `ASPIDA_REPEAT_PENALTY` / `ASPIDA_REPEAT_LAST_N` / `ASPIDA_SEED` | Sampling defaults (a client request overrides them; server-side clamps still apply) |
| `ASPIDA_PROF` | Print coarse per-step timings (attention vs matvec vs RoPE vs FFN) |
| `ASPIDA_DBG` | Verbose tensor dumps |

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
