# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Gemma 3n E4B backend (validation in progress)
- Training/distillation infrastructure
- WebSocket bridge for browser clients
- Full GGML quant **read** coverage: Q3_K and Q2_K decoders (validated against
  real Llama-3.2-1B Q3_K and TinyLlama Q2_K) — the engine now dequantizes every
  standard GGML weight format
- Quant **write**: from-scratch quantizers for Q5_0, Q4_K, Q5_K, Q6_K (joining
  Q8_0/Q4_0), all round-trip-tested; the trainer exports six formats
- Fused decode+dot CPU matvec for all five K-quants (Q2_K–Q6_K)
- GPU CUDA kernels for Q3_K/Q2_K (matvec + batched matmul), validated on an L40S
- `requantize` CLI — convert any GGUF between formats, incl. `q4_k_m`/`q5_k_m`
  mixes (sensitive tensors → Q6_K), metadata preserved byte-exact
- `test_kquant_synth` — model-free CI test for the Q2_K/Q3_K decoders
- Min-p sampling (`min_p` API field / `ASPIDA_MIN_P` env) — keep tokens with
  prob ≥ min_p·p_max; `test_sampler` unit test

### Changed
- Refactored backend interface to polymorphic Model_Backend

## [0.3.0] - 2026-07-06

### Added
- OpenAI-compatible API (`/v1/chat/completions`, `/v1/models`)
- Model catalog discovery from filesystem
- Token accounting (prompt/completion tokens)
- `finish_reason` field in responses
- GPU offload for Q4_K/Q6_K quantization
- CUDA kernels with block-level parallelism
- Runtime dlopen of CUDA shim (no build-time dependency)
- Llama 3.x backend (dense GQA, SwiGLU, RoPE)
- Unified sampling (temperature, top-k, top-p, repeat penalty)
- xorshift64* PRNG for reproducible sampling

### Changed
- Removed legacy TypeScript generator code
- Consolidated build system (single server.gpr)
- Improved HTTP robustness (full buffer writes, error codes)

### Fixed
- GGUF parser: correct U8/I8/U16/I16 type reads
- Read_Exact: loop over short reads
- SPARK contracts on crypto primitives

## [0.2.0] - 2026-07-05

### Added
- Multi-turn ChatML conversation support
- Message_Array type for conversation history
- Streaming UI with throughput metrics
- Token_Sink callback interface
- Persistent thread pool (LLM_Pool)
- Stack-allocated block decode for Q5_K/Q6_K
- 4-lane FMA accumulator for vectorization
- Weight abstraction (dense + quantized)
- QMatVec streaming (no full FP32 materialization)
- Secure channel (X25519, ChaCha20-Poly1305)
- Session management with at-rest encryption
- Concurrent server with handler pool

### Changed
- Replaced spawn/join with persistent workers
- Optimized Q5_K decode (hoist scale lookups)

### Security
- Added SPARK contracts to crypto core
- Constant-time equality checks
- Secure memory wiping

## [0.1.0] - 2026-07-04

### Added
- Initial release
- GGUF parser (header, metadata, tensor info)
- Q5_K/Q8_K dequantization
- BPE tokenizer with SentencePiece support
- Qwen3.5-MoE+SSM backend
- Basic transformer blocks (attention, MoE, SSM)
- RoPE positional encoding
- Single-turn ChatML chat
- Basic REPL interface

### Infrastructure
- GNAT project files
- Makefile with build targets
- Initial test suite

---

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| 0.3.0 | 2026-07-06 | OpenAI API, GPU offload, Llama backend |
| 0.2.0 | 2026-07-05 | Multi-turn chat, thread pool, secure channel |
| 0.1.0 | 2026-07-04 | Initial release, Qwen backend, GGUF parser |

---

[Unreleased]: https://github.com/chabanov/aspida/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/chabanov/aspida/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/chabanov/aspida/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/chabanov/aspida/releases/tag/v0.1.0
