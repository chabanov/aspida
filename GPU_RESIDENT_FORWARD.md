# GPU-resident forward pass — the path to ollama speed parity

Status: **design + staged plan** (2026-07-12). Target: serve Hura
("Hura", `qwen35moe`) on aspida's own CUDA at parity with the ollama baseline
(**124 tok/s** Q8 on the an NVIDIA GPU), preserving the E2EE differentiator and the
SPARK-proven correctness that is aspida's reason to exist.

---

## 1. The measured problem — it is NOT bandwidth, it is per-matvec ping-pong

The current GPU path (`gpu/gpu_matvec.cu`, wired at `src/llm/llm_weight.adb`
`MatVec`/`MatVec_Expert`) offloads **one matvec at a time**:

```
per matvec:  H2D copy x  →  launch k_q*k_w  →  D2H copy y  (blocks = implicit sync)
```

Weights are already cached resident in VRAM (keyed by host pointer — the whole
model stays uploaded after token 1). But the **activation** `x` goes host→device
and `y` comes device→host on *every single weight*, and the D2H is a
serialization point. One MoE decode layer issues **28 quantized GEMVs**
(1 router + 8×3 routed experts + 3 shared); ×36 layers = **~1000 blocking PCIe
round-trips + ~1000 tiny kernel launches per token**. The GPU sits at **3–7 %
utilization** — idle, waiting on the bus and on launch latency.

Measured on the an NVIDIA GPU with `Q4_K_M`: **2.2 tok/s** (100 tok / 45.5 s).
`gpu/experiments/phase1/bench.cu` independently confirms the ceiling: even with
weights fully resident, a per-matvec-launch forward tops out ~2.25 tok/s.

**Root cause:** as long as the norms / softmax / RoPE / SwiGLU / router-topk /
attention run on the CPU *between* the matvecs, each matvec is *forced* to
round-trip its activation. There is no fix inside the "offload one matvec"
model. The activation must **stay resident on the device for the whole forward
pass**, with every op running on-GPU, so the only host↔device traffic per token
is: **token id in, logits out.**

Theoretical headroom confirms this is worth doing: 35B-A3B activates ~3 B params
/ token; at Q4_K (~0.56 B/param) that is ~1.7 GB of weight reads / token. On the
an NVIDIA GPU (~864 GB/s) the weight read alone is ~2 ms → a bandwidth ceiling near
**~500 tok/s**. We are 250× below it purely on overhead.

---

## 2. The fix — a resident decode forward, hidden state never leaves VRAM

Keep in device memory, allocated once per loaded model:
- the model weights (already cached by `gpu_matvec.cu`'s `g_wcache`),
- the **hidden state** `H[dim]` (f32),
- every per-layer **KV cache** (full-attn layers) and **delta-net recurrent
  state** `S_All` + conv window `Conv_Hist` (delta-net layers),
- the token embedding table and LM head (today these are dense-f32 and bypass
  the GPU entirely).

Per token: upload the token id, run the entire 36-layer stack as back-to-back
kernel launches with **no intervening `cudaMemcpy`**, download only the logits
(or the sampled argmax). Then eliminate launch latency with a **CUDA graph**:
capture the per-token kernel sequence once, replay it each step.

This is exactly how the reference implementations do it — see §6.

---

## 3. What already exists to reuse (do not rewrite)

aspida already contains most of the parts, written and tested:

| Need | Reuse | File |
|---|---|---|
| Resident quant GEMV (Q4_K…Q2_K), warp-per-row, weight cache-by-ptr | `k_q*k_w`, `aspida_gpu_matvec`, `g_wcache`, `upload_weight` | `gpu/gpu_matvec.cu` |
| Batched quant GEMV (continuous batching / prefill) | `k_q*k_wb`, `aspida_gpu_matmul` | `gpu/gpu_matvec.cu` |
| Resident-buffer lifecycle + dlopen/dlsym Ada binding pattern | `struct Stu` allocate-once; `student_gpu.ads/.adb` | `gpu/student_shim.cu`, `src/train/student_gpu.*` |
| RMSNorm (CPU-bit-exact, double ascending sum) | `k_rmsnorm` | `gpu/experiments/phase1/ops.cu` |
| NeoX RoPE with freq_factors | `k_rope` | `gpu/experiments/phase1/ops.cu` |
| Multi-head RoPE | `k_rope_mh_fwd` | `gpu/student_kernels.cuh` |
| SiLU / SwiGLU | `k_silu`, `k_swiglu` | `phase1/ops.cu`, `phase1/layer.cu` |
| GQA single-token decode attention | `k_attn` (fix `scores[4096]` cap) | `gpu/experiments/phase1/ops.cu` |
| Multi-head causal attention (prefill) | `k_mha_fwd` | `gpu/student_kernels.cuh` |
| Residual add | `k_add` | `gpu/experiments/phase1/layer.cu` |
| Tensor-core f16 GEMM (prefill throughput) | `k_wmma` + `k_f2h` | `gpu/test_gemm_wmma.cu` |
| Assembled full Llama layer (composition reference) | `layer.cu main()` | `gpu/experiments/phase1/layer.cu` |

**Must write new:** (a) fused **MoE FFN** decode block (router→softmax→top-8→
8×SwiGLU-expert→combine→shared) — no MoE kernel exists yet; (b) fused
**delta-net** block (causal conv1d + per-head gated recurrence against `S_All`) —
unique to this architecture, no equivalent kernel; (c) **incremental KV-cache
decode attention** for GQA full-attn (existing `k_mha_fwd` is full-sequence
prefill, no cache); (d) on-device **embedding gather** + **LM-head GEMV** for the
dense-f32 endpoints; (e) the resident session ABI + Ada binding; (f) CUDA-graph
capture/replay.

Build stays the single-`.cu` / one-`nvcc` model: a new `gpu/qwen_resident.cu`
plus `#include`s of the shared kernels, dlopen'd by a new `LLM_Qwen_GPU` binding.
`--fmad=false` and `-arch=native` are preserved (bit-exactness + portability).

---

## 4. Ada's role — quality and control, honestly

The FLOPs are CUDA; that is standard (llama.cpp, ollama are C++/CUDA too — §6).
aspida's edge is not "Ada does the matmul", it is:

- **Ada/SPARK owns correctness.** The existing CPU decode path (`LLM_MoE`,
  `LLM_DeltaNet_Blk`, `LLM_FullAttn`, all `pragma Suppress (All_Checks)` hot
  kernels) is the **bit-exact oracle**. Every new CUDA kernel is validated
  against it to the last ULP before it is allowed on the hot path — the same
  discipline that already gates `gpu_matvec.cu` (`--fmad=false`, `test_matvec.cu`
  decodes all 5 K-quants on the CPU and diffs).
- **Ada owns the ABI and the residency lifecycle** — strongly-typed
  access-to-C-procedure bindings (`LLM_GPU` pattern), `Available` fallback, and
  `Free_Weight`-before-`Free_Bytes` teardown ordering that the C side cannot get
  wrong because Ada sequences it.
- **Ada owns orchestration** — sampler, session/KV lifetime, the E2EE Noise
  channel, model switching. The CUDA session is a leaf the Ada engine drives.

So "use all the language's capabilities" = SPARK-proven glue + a CPU oracle that
makes the GPU path provably equal to the reference, not weaker C with no checks.

---

## 5. Staged plan — each increment is independently shippable and measured

Every increment exits on **two gates**: (1) bit-exact vs the CPU path on a fixed
prompt (max |Δlogit| below the `--fmad=false` tolerance already used by
`test_matvec.cu`), and (2) a measured tok/s improvement. Never promote a kernel
that fails gate 1.

**Increment 1 — Fused resident MoE-FFN decode.** `aspida_gpu_moe_decode`: hidden
state enters once, stays on device through router GEMV → stable softmax → greedy
top-8 → 8×(gate,up,down + SwiGLU) → weighted combine → shared expert, returns the
FFN output once. Collapses **28 round-trips/layer → 1** for all 36 layers — the
single largest chunk of the GEMVs. Oracle: `LLM_MoE.Forward`. *Expected: the
biggest single jump, MoE is the bulk of decode compute.*

**Increment 2 — Fused resident attention.** (2a) delta-net block: conv1d+SiLU,
per-head gated recurrence against resident `S_All`, gated RMSNorm, out-proj —
oracle `LLM_DeltaNet_Blk.Step`. (2b) GQA full-attn with **incremental KV cache**
resident on device (append K/V for the new position, decode-attend over 0..pos) —
oracle `LLM_FullAttn.Step`. After this, attention state never round-trips.

**Increment 3 — End-to-end resident forward + CUDA graph.** Chain embedding
gather → 36 layers (norms, residuals, incr. 1+2 blocks) → final RMSNorm → LM-head
GEMV entirely on device; `H` never leaves VRAM. Only token id in / logits out.
Capture the per-token kernel sequence as a **CUDA graph**, replay per step to
kill launch latency. Oracle: full `Decode_Tokens.Decode`.

**Increment 4 — Tune to parity.** MMQ-style int8 tensor-core quant GEMM for the
big projections (llama.cpp `mul_mat_q`), flash-decode attention, f16/WMMA GEMM
for prefill, per-op occupancy tuning. Push toward the ~124 tok/s baseline and
beyond for batched serving (the `_wb` batched kernels already exist).

### Targets (honest)
- Increment 1 alone: expect a **multi-× jump** from 2.2 tok/s (MoE is ~28/38 of
  the GEMVs), realistically into the low tens of tok/s.
- Increment 3 (resident + graph): the real inflection — into the **tens–hundreds**.
- Parity (≥124 tok/s) is Increment 4's stretch goal. Reaching or beating it is
  plausible because the bandwidth ceiling is ~500 tok/s and ollama's llama.cpp is
  general-purpose where we are specialized to one architecture.

---

## 6. Standards & competitor reference (what we mirror)

The technique set is well-established; we follow it rather than invent:
- **llama.cpp `ggml-cuda`** — `mul_mat_vec_q` (dequant fused into the GEMV, our
  `k_q*k_w` already does this), `mul_mat_q` / MMQ (int8 tensor-core quant GEMM,
  Increment 4), the fused flash-attention decode kernel, and **CUDA graphs** for
  the decode loop (Increment 3). This is the direct reference for a
  quantized-MoE decode on GPU.
- **Resident weights + activations, single graph replay per token** is the
  standard decode design in llama.cpp and TensorRT-LLM; §2 is that pattern.
- **K-quant block layouts** already match llama.cpp byte-for-byte
  (`gpu_matvec.cu` decoders), so our dequant math is reference-compatible.

---

## 7. Where this runs

Dev requires a GPU; both prod GPUs are busy serving paid features (the GPU host =
Hura/ollama, aspida-media = image-gen). Increments are developed **test-driven on
a dedicated dev GPU droplet** (an NVIDIA GPU) against the CPU oracle, then the winning
build is promoted. Cost is controlled by spinning the dev box up per work
session, snapshotting, and destroying it when idle. Prod (ollama) stays the
instant rollback until a resident build clears every bit-exactness gate *and*
reaches usable tok/s.
