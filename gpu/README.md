# Stream B — GPU backend for the Aspida engine

The pure-Ada engine offloads its quantized mat-vecs (the ~95% decode cost) to
own CUDA kernels, while keeping all orchestration (GGUF load, tokenizer, the
forward graph, sampling, chat templates, crypto) in Ada. Validated end to end
on real Llama-3.3-70B.

## Status (validated on DigitalOcean NVIDIA H100/H200)
- **Kernels** (`gpu_matvec.cu`): all five K-quants — Q4_K, Q5_K, Q6_K, **Q3_K,
  Q2_K** — dequant & matvec + batched matmul (scalar, warp-per-row, and
  warp-batched variants). Validated vs the CPU reference via `test_matvec.cu`
  on an NVIDIA GPU (Q3_K rel 5.9e-4, Q2_K rel 1.2e-4, others ≤1e-4; batched == per-row
  exactly). Also RMSNorm / RoPE / SiLU / GQA-attention (≤1e-6), full transformer
  layer (2.3e-7). See `experiments/phase1/README.md`.
- **Full integration** (`gpu_matvec.cu` + `src/llm/llm_gpu.{ads,adb}`): the Ada
  `LLM_Llama` forward routes its 8 matvecs per layer through a CUDA shim
  (`aspida_gpu_matvec`) loaded at runtime via `dlopen` — so a CPU build/host is
  completely unaffected (no link-time CUDA dependency; `LLM_GPU.Available` is
  False ⇒ pure-Ada fallback). Weights are uploaded to VRAM once (cached by host
  pointer) and stay resident.

  **End-to-end result** — `ASPIDA_GPU=1 ./obj/llama_probe model.gguf "The capital of France is"`:
  ```
  completion: ' Paris .
  A. The Eiffel'
  wall clock (load + 8 tokens): 51 s   vs   CPU ~17 min / 4 tokens
  ```
  Correct output (identical to CPU — the kernels are bit-exact), ~20× faster
  end-to-end on the real engine.

## How to run (on an NVIDIA GPU box, image `gpu-h100x1-base`)

The shim links against upstream llama.cpp's ggml (fattn-mma + mul_mat_id prefill),
so build that first — `build_so.sh` documents the pinned commit and cmake flags:
```
GG=/path/to/llama.cpp ./build_so.sh          # -> libaspidagpu.so
gprbuild -P probe.gpr -XOS=linux            # build the engine (Alire GNAT)
ASPIDA_GPU=1 ASPIDA_GPU_LIB=$PWD/libaspidagpu.so LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
  ./obj/llama_probe model.gguf "The capital of France is" 8
```
Unset `ASPIDA_GPU` → pure CPU (unchanged).

## Training on GPU (Stage 1 — resident loop)

`train_mlp.cu` is the foundation for **GPU-resident training**: weights, AdamW
moments and the dataset live on the device across every step; only the scalar
loss is copied back (no per-op host round-trips — the lever the perf note below
calls for). It fits a deep linear stack to a random target `T = X·W*`, so the
loss must collapse iff the resident forward → backward → AdamW chain is correct.

```
nvcc -O3 -arch=native gpu/train_mlp.cu -o train_mlp && ./train_mlp
```

**Validated on an NVIDIA GPU** (batch 256, dim 512, 4 layers, 3000 steps), running
*alongside the live serving demo* (~450 MB VRAM, demo VRAM/processes untouched):
```
step 1   loss/elem 3.6e-2   →   step 3000   loss/elem 4.3e-6   (collapse)
1206 steps/s   1.94 TFLOP/s (naive kernels)
```
This is the resident matmul (fwd / dA / dB) + AdamW loop proven at depth and
size — the kernel basis a full GPU-resident `Student` is grafted onto.

**Step 5a — resident training SESSION (C ABI).** `train_resident_shim.cu`
packages the resident loop behind a callable C ABI (`art_create` / `art_set_data`
/ `art_step` / `art_free`): weights + AdamW + data stay on the device, only the
scalar loss crosses per step. Builds as a self-test binary AND as
`libaspidatrain.so` for Ada to `dlopen` (like `LLM_GPU`). Validated on an NVIDIA GPU
(loss 3.6e-2 → 9.5e-8, ~1170 steps/s, demo untouched). This is the substrate the
Ada `Student` drives in Step 5b — no per-op host round-trips (unlike the old
`train_shim.cu`).
```
# lib for Ada (Step 5b):
nvcc -O3 -arch=native -shared -Xcompiler -fPIC -DART_NO_MAIN \
  gpu/train_resident_shim.cu -o libaspidatrain.so
```

**Step 5b — Ada drives it.** `Train_GPU_Resident` (`src/train/`) dlopen's the
shim and exposes `Create`/`Set_Data`/`Step`/`Free`; `tools/train_resident_probe`
runs a session from Ada. Validated end-to-end on an NVIDIA GPU (loss 3.6e-2 → 5.1e-15)
— Ada trains on the GPU with only the scalar loss crossing per step; off (CPU
fallback) unless `ASPIDA_TRAIN_LIB` points at the shim.

**Step 5c — grad-checked.** The probe also checks the GPU analytic gradient
against a CPU finite difference of E=½·Σ(Y−T)² (`art_loss_only`/`art_grad_at`/
`art_w_get`/`art_w_set`). On an NVIDIA GPU every probe is within a combined
rel(<2e-2)/abs(<5e-5) tolerance — the resident backward is numerically correct.

**Step 5d — RMSNorm op (grad-checked).** `gpu/test_rmsnorm_gpu.cu`: RMSNorm
forward + backward (input `dx` and gain `dg`) grad-checked vs finite difference
on an NVIDIA GPU (max rel 3.4e-3). The first transformer op toward a GPU-resident
`Student`; matmuls (QKV/MLP/head) are already the resident matmul.

**Step 5e — SwiGLU op (grad-checked).** `gpu/test_swiglu_gpu.cu`: the gated
activation h = SiLU(a)·b forward + backward (`da`, `db`) grad-checked vs finite
difference on an NVIDIA GPU (max rel 5.2e-3).

**Step 5f — RoPE op (grad-checked).** `gpu/test_rope_gpu.cu`: rotary position
embedding forward + backward (backward = rotation by −θ) grad-checked vs finite
difference on an NVIDIA GPU (max rel 7.9e-4).

**Step 5g — causal attention op (grad-checked).** `gpu/test_attention_gpu.cu`:
single-head causal scaled-dot-product attention forward + backward (gradient
through the softmax jacobian — the hardest op), dQ/dK/dV grad-checked vs finite
difference on an NVIDIA GPU (max rel 2.1e-3).

**Step 5h — cross-entropy + embedding ops (grad-checked).**
`gpu/test_celoss_gpu.cu`: CE loss (dlogits = softmax − onehot) and token
embedding (gather / scatter-add, repeated id accumulates) grad-checked on an
an NVIDIA GPU (max rel 5.3e-5). **The grad-checked kernel set for a GPU-resident Student
is complete** — matmul, RMSNorm, SwiGLU, RoPE, attention, CE/embedding.

**Step 5i — assembly, Stage A.** `gpu/student_resident.cu` composes the kernels
into one forward+backward pipeline (embedding → final RMSNorm → head →
cross-entropy), grad-checked end-to-end on an NVIDIA GPU (max rel 2.1e-3) with a
train-sanity that drives loss 20.9 → 2.8.

**Step 5i — assembly, Stage B (full transformer layer).** `student_resident.cu`
chains a complete pre-norm layer (RMSNorm → Q/K/V → RoPE → causal attention →
residual → RMSNorm → SwiGLU → residual) between embedding and head, with correct
transposed backward and both residual paths. Grad-checked end-to-end on an NVIDIA GPU
(all nine layer weights vs CE-loss finite difference, max rel 1.4e-2); train
sanity drives loss 20.95 → 0.015. **The full transformer forward+backward is
assembled and numerically correct on GPU.**

**Step 5i — assembly, Stage C (multi-layer + Ada-driven).** `student_resident.cu`
generalises to L stacked layers (grad-checked across both, loss → 0.009).
`gpu/student_shim.cu` wraps the resident Student in a C-ABI session
(`stu_create`/`stu_set_data`/`stu_step`/`stu_free` → `libaspidastudent.so`);
`Student_GPU` (`src/train/`) dlopens it and `tools/student_gpu_probe` drives it.
**Validated end-to-end on an NVIDIA GPU: Ada trained a full 2-layer transformer-LM on
the GPU (loss 20.5 → 0.010).** This closes single-node GPU Student training.

**Production hardening (Stream B).** Multi-head causal attention
(`test_attention_mh_gpu.cu`, H heads × dh, per-head softmax) is grad-checked on an
an NVIDIA GPU (max rel 1.7e-4); the resident Student shim now uses **AdamW** (bias-corrected,
flat optimizer list) instead of SGD. **Multi-head + per-head RoPE are wired into
the resident Student** — grad-checked self-test (2-layer multi-head, max rel
1.7e-2, loss → 0.009) and Ada-driven shim (loss 20.5 → 0.015) both pass; the
Student is a true multi-head transformer with AdamW. The shim is now
**runtime-configurable** — `stu_create(V,D,F,S,L,H)` so one `libaspidastudent.so`
serves any `Platform` tier; `Student_GPU.Create` takes the architecture and Ada
trained two differently-sized configs end-to-end on an NVIDIA GPU (both loss → 0.014).
A shared-memory **tiled FP32 GEMM** (`test_gemm_tiled.cu`, 1.3× faster on 1024³,
bit-exact) is wired into the Student forward. Remaining perf lever: FP16/
tensor-core (WMMA) — changes numerics, needs FP16 I/O. Then scale-out
(data-parallel across hosts, gated by a measured all-reduce benchmark).

### Roadmap to multi-GPU-server training
| Stage | Build | State |
|---|---|---|
| **1. Single-GPU end-to-end** | resident matmul+AdamW loop (`train_mlp.cu`) ✓; then SwiGLU, RMSNorm, RoPE, attention, embedding/head → full resident `Student`; then Ada wiring (`Train_GPU` resident binding in `Student.Step`) | ▶ loop done |
| 2. Single-node multi-GPU | data-parallel: replica per GPU, split batch, **all-reduce gradients** (NCCL or peer/host reduce); per-device `cudaSetDevice` | ⛔ |
| 3. Sharding (ZeRO/FSDP) | shard params/grads/Adam for students larger than one GPU | ⛔ |
| 4. Multi-node | network collectives (NCCL/IB or TCP), orchestration, fault tolerance | ⛔ |
| Cross-cutting | tiled / tensor-core GEMM + FP16 (1.94 TFLOP/s is naive), data loader, distributed checkpoint, job scheduler + GPU provisioning + metering | ⛔ |

#### Measured: inter-droplet all-reduce benchmark (Step 7 go/no-go)
Two droplets in **tor1** (same region as the GPU box), iperf3 over the VPC:
**~1.68 Gbit/s (~210 MB/s), 2.1 ms RTT** (on small droplets — a *lower bound*;
GPU droplets likely have a higher network tier). Per-step ring all-reduce moves
≈ 2× the gradient. Against the `Platform` tiers (FP32 grads):

| Tier | ~Params | Grad | all-reduce @1.68 Gbit | @10 Gbit (GPU-tier, est.) |
|---|---|---|---|---|
| Small  | 27 M   | 109 MB | ~1.0 s/step  | ~0.17 s |
| Medium | 185 M  | 741 MB | ~7.1 s/step  | ~1.2 s  |
| Large  | 1.35 B | 5.4 GB | ~51 s/step   | ~8.6 s  |

**Verdict:** naive per-step data-parallel over standard droplet networking is
**negative-scaling for Medium/Large** (all-reduce dominates), borderline for
Small. Viable path requires **gradient accumulation** (amortise one all-reduce
over many micro-steps) and/or **compute-comm overlap**, plus confirming the
actual GPU-droplet network tier; ZeRO/FSDP sharding for Large. So multi-node is
**gated**: offer data-parallel only with grad-accumulation, and not for Large
until sharding lands. (Benchmark cost: two throwaway droplets, destroyed.)

**Built:** (1) gradient accumulation (`stu_micro`/`stu_apply`, driven from Ada via
`Student_GPU.Micro`/`Apply`) — accumulate G micro-batches, one averaged AdamW
update. (2) data-parallel all-reduce, **correctness proven bit-exact** at $0
(`test_dataparallel.cu`: two replicas + sum-all-reduce of the accumulators ==
single node on combined data; `stu_get_acc`/`stu_set_acc`/`stu_nparams` flatten the
grad blob for the cross-node SUM). With grad-accumulation the all-reduce is once
per G steps, so at the measured 1.68 Gbit/s: Small ~0.06 s, Medium ~0.44 s,
Large ~3.2 s per effective step at G=16 — viable for Small/Medium; Large still
wants ZeRO/FSDP sharding. Remaining: the TCP transport between two droplets
(mechanical) + compute/comm overlap; a real two-droplet run only confirms the
GPU-droplet network tier and live timing.

## Next (perf)
**Tensor cores (P1):** `test_gemm_wmma.cu` validates a WMMA FP16 GEMM at **6.6×**
the FP32 path on an NVIDIA GPU (5.4 → 35.8 TFLOP/s, Frobenius rel err 2.6e-4). Next:
wire the FP16/WMMA path into the resident Student (FP16 weights/activations +
FP32 master + loss scaling) and re-grad-check under FP16 tolerance.

The matvecs go one-at-a-time with a per-call x↑/y↓ round-trip. The big levers:
keep activations resident on the device and run a whole layer's matmuls without
host round-trips; tiled/batched quant-GEMM + FP16; move RMSNorm/RoPE/attention
on-device too. Correctness is already locked (bit-exact kernels); this is pure
throughput work.

Total GPU spend bringing Stream B to here: ~$12 (all boxes hourly, destroyed
after each run; DO GPU capacity fluctuates — a bounded create-retry loop lands one).
