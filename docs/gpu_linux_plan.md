# Aspida — GPU backend + Linux build plan (with cost estimates)

Status: planning. Goal: take the pure-Ada inference engine from CPU-only (scalar,
~0.1 tok/s) to a GPU-accelerated, Linux-deployable engine fit for provider-grade
serving — while keeping the "own engine, no third-party inference libraries"
principle (own kernels; only the vendor compiler/driver are external) and the
encrypted-only chat path.

---

## 0. Where we are (facts)
- Pure Ada/SPARK; scalar CPU. ~10 s/tok (gemma 9B), ~80 s/tok (Llama-70B). Inference serialized by `Infer_Lock` (1 stream at a time).
- 4 backends behind the unified `LLM_Backend.Model_Backend` interface (qwen35moe, qwen2, gemma4, llama-dense).
- Quant dequant in Ada: F32/F16/BF16/Q8_0/Q4_K/Q5_K/Q6_K.
- Build is **macOS-only**: `shared.gpr` linker flags `-Wl,-syslibroot,$SDKROOT` + `-L$SDKROOT/usr/lib`; `Makefile` uses `xcrun`.
- Hot compute lives in: `llm_weight` (MatVec/QMatVec), `llm_dequant`, `llm_tensor`, per-layer forward in each backend.
- Encrypted server (`secure_server`) is the sole chat entry; GPU work must preserve that.

The GPU droplet won't help the *current* engine — it's CPU-only. GPU boxes are a
**dev target** to build a GPU compute path, not a drop-in speedup.

---

## STREAM A — Linux build (prerequisite, low risk)

GPU droplets are Linux (Ubuntu). The code is portable Ada; only the build glue is macOS-specific.

### Tasks
1. **`shared.gpr`**: add an `OS` scenario var (`darwin|linux`). Gate the syslibroot/SDKROOT linker switches to `darwin`; `linux` gets none (GNAT links libc directly).
2. **`Makefile`**: detect OS (`uname`); skip `xcrun`/`SDKROOT` on Linux; keep `-march=native` (build on the target box).
3. **Toolchain on the droplet**: Alire (`alr`) + `gnat_native` + `gprbuild`, or distro GNAT (FSF). Pin versions.
4. **Portability check**: `getentropy` (glibc ≥2.25 — present), `GNAT.Sockets` (portable), `Ada.Directories`/`Stream_IO` (portable). Crypto is pure Ada → portable. Expect zero source changes; if any, isolate behind the `OS` scenario.
5. **Repro**: a `Dockerfile` (ubuntu + alire + build) and/or a `scripts/build-linux.sh`. Optional but recommended for clean rebuilds.
6. **Validate**: build `server.gpr` + `tests` on Linux; run model-free tests; run one model (gemma) end-to-end through the encrypted server on the box.

### Effort & cost
- **Effort:** 1–3 dev-days.
- **Infra:** no GPU needed. A `s-4vcpu-8gb` droplet ($0.08/hr, ~$48/mo) or local Linux VM. Build/test cost ≈ **$5–15** (a few hours of a cheap droplet, destroyed after).
- **Risk:** low. Pure-Ada portability is high; the only real work is the linker-flag scenario + toolchain setup.

---

## STREAM B — GPU backend (the big effort)

### Architecture decision
Keep Ada for orchestration (loader, tokenizer, sampler, chat templates, KV-cache
bookkeeping, crypto). Move the **hot math to GPU kernels** we write ourselves,
called from Ada over the C ABI. Vendor compiler/driver are the only externals —
**no cuBLAS/cuDNN/vLLM/llama.cpp** (that would break the own-engine principle).

```
Ada (LLM_*, secure_server)
   │  pragma Import (C, ...)   ← thin FFI
   ▼
C/CUDA host layer (device alloc, H2D/D2H, kernel launch)
   ▼
Own GPU kernels (.cu / .hip): dequant-GEMM, attention, RMSNorm, RoPE, SwiGLU, softmax
```

The `Model_Backend` interface is the seam: add a GPU-backed wrapper (weights
uploaded to VRAM once at load; per-token forward runs on device; logits copied
back; sampler stays CPU or moves to GPU later).

### NVIDIA (CUDA) vs AMD (ROCm/HIP)
- **CUDA / NVIDIA** — far more docs, examples, tooling; the pragmatic choice for from-scratch kernel work. Dev on `gpu-4000adax1-20gb` ($0.76/hr); real model + benchmark on `gpu-h100x1-80gb` ($3.39/hr, 80 GB fits 70B).
- **AMD MI300X** — `gpu-mi300x1-192gb` is the cheapest big VRAM ($1.99/hr, 192 GB), but ROCm/HIP is less mature. HIP's API ≈ CUDA, so CUDA-first then port is viable.
- **Recommendation:** develop CUDA-first (NVIDIA), keep kernels HIP-portable; revisit MI300X for cost once stable.

### Kernels to write (hardest first)
1. **Quantized matvec/GEMM** (Q4_K/Q5_K/Q6_K/Q8_0 weights × FP16 activations, on-the-fly dequant). Dominant cost; the hard one. Start with matvec (decode, batch=1), then GEMM (prefill/batching).
2. **Attention**: QKᵀ, scaled softmax, ×V, with GQA + KV cache + RoPE (incl. gemma dual-rope/rope_freqs, llama rope_freqs).
3. **RMSNorm**, **SwiGLU/GeGLU**, element-wise add/mul, embedding lookup+dequant.
4. (Later) on-device sampling.

### Phasing
- **Phase 0 — Foundations (1 wk):** Linux build (Stream A) + CUDA toolchain on droplet + Ada↔C↔CUDA "hello kernel" (vector add) proving the FFI + a microbench harness.
- **Phase 1 — One dense model on GPU (4–8 wks):** Llama (simplest graph). Implement core kernels; validate numerically vs the CPU engine (argmax/"Paris" match, then logit closeness); benchmark tok/s. **Milestone: 70B at usable speed (target ≫ 0.1 tok/s; realistic first pass 5–30 tok/s decode depending on kernel quality).**
- **Phase 2 — Other archs (3–6 wks):** gemma (PLE, dual-rope, shared-KV), then qwen (MoE + gated delta-net — exotic, the hardest port).
- **Phase 3 — Provider-grade throughput (4–8 wks):** continuous batching (serve many requests/GPU), FP16/BF16 paths, paged KV cache, remove the single-stream `Infer_Lock`. **This is what makes it a service, not a demo.**
- **Phase 4 — Productionize (2–4 wks):** GPU worker behind the encrypted server, health/metrics, graceful shutdown, multi-worker. (Overlaps the earlier ops-gap list.)

### Honest risk
Writing competitive quant-GEMM/attention kernels from scratch is genuinely hard
(llama.cpp/vLLM are large, multi-year efforts). Expect the first GPU pass to be
**correct but well below state-of-the-art tok/s**, improving over Phases 1→3. This
is the single biggest cost and schedule risk; it is the price of the
own-engine/no-third-party principle.

---

## Cost summary

### A. GPU server rental (hourly billing — spin up for sessions, snapshot, destroy when idle)

| Box | VRAM | $/hr | 80 h/mo (part-time dev) | 160 h/mo (full dev) | 24×7 |
|---|---|---|---|---|---|
| `gpu-4000adax1-20gb` (dev/small) | 20 GB | $0.76 | $61 | $122 | $565 |
| `gpu-mi300x1-192gb` (AMD, big VRAM) | 192 GB | $1.99 | $159 | $318 | $1 481 |
| `gpu-h100x1-80gb` (CUDA, fits 70B) | 80 GB | $3.39 | $271 | $542 | $2 522 |
| `gpu-h200x1-141gb` | 141 GB | $3.44 | $275 | $550 | $2 559 |

- **Snapshot to persist setup between sessions:** ~$0.06/GB·mo → ~100 GB ≈ **$6/mo**.
- **Always-on build/repo box (optional):** `s-2vcpu-4gb` ≈ **$24/mo**.

### B. Recommended dev cadence & monthly infra burn
- **Phase 0 (Linux + FFI):** mostly non-GPU; ~20 h on 4000 Ada → **~$15** + cheap droplet. One-time.
- **Phases 1–2 (kernels, one→all models):** active dev on H100 part-time. Budget **~$300–550/mo** GPU (80–160 h) + $6 snapshot + $24 build box ≈ **~$330–580/mo**. Use the $0.76 4000 Ada for early kernel iteration to cut this; reserve H100 for 70B runs/benchmarks.
- **Phase 3 (batching/throughput):** more H100 hours for load tests → **~$500–800/mo** in heavy-benchmark months.

### C. Indicative totals (infra only, hourly model, destroy-when-idle)
- **MVP path** (Linux + Llama-on-GPU working): ~2–3 months → **~$700–1 600** GPU+infra total.
- **Full provider-grade** (all archs + batching): ~6 months → **~$2 500–4 500** GPU+infra total.
- 24×7 reserved GPU instead of hourly would be **~5–10×** these numbers — avoid until you actually serve traffic.

### D. Engineering effort (separate from infra $)
- Stream A: **1–3 days.**
- Stream B MVP (Phase 0–1, one model on GPU): **~5–9 weeks** (1 GPU/Ada dev).
- Stream B full (Phases 2–4): **+3–5 months.**
Convert to € at your blended dev rate; infra above is the only out-of-pocket cloud cost.

---

## Recommended first move (cheapest, de-risks everything)
1. Do **Stream A (Linux build)** on a cheap non-GPU droplet — ~1–3 days, ~$10. Unblocks everything; no GPU spend.
2. Spin up `gpu-4000adax1-20gb` ($0.76/hr) for **Phase 0** — prove the Ada↔CUDA FFI + run the current CPU engine on Linux/GPU box to baseline. Destroy when done (~$15).
3. Only then commit to H100/MI300X hours for Phase 1 kernel work.

All boxes are hourly — never leave a GPU idle-but-running. Snapshot + destroy between sessions.
