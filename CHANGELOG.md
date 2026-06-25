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
- gemma4 dense forward validated to greedy bit-parity with llama.cpp on real
  E4B (PLE) and 12B (non-PLE/MQA) models
- gemma4 loader now rejects MoE variants (routed experts, e.g. supergemma-26B)
  with a clear error instead of silently running them dense and emitting garbage
- **Multi-teacher (ensemble) distillation** — `Distill.Capture_Ensemble`
  averages several teachers' per-position distributions in probability space
  (optionally weighted) and keeps the top-K of the blend; the merged result is
  an ordinary `Sample`, so KL training and the on-disk dataset are unchanged.
  Shared-vocabulary mismatch raises `Vocab_Mismatch`. (`test_multi_teacher`,
  incl. a proof that per-teacher weighted KL ≡ ensemble KL at the gradient level)
- **Real-model teachers for every backend** — `LLM_Qwen.Forward_Logits` and
  `LLM_Gemma.Forward_Logits` (per-position logits) plus `Teacher_Qwen` /
  `Teacher_Gemma` adapters join the existing `Teacher_Llama`, so any loaded
  Llama / Qwen-MoE / Gemma model can teach a student. `tools/teacher_probe`
  runtime-validates each on a real model (Llama-3.2-1B, Gemma-4-E4B,
  Qwen3.5-35B-A3B all pass)
- **Verifier-driven distillation (student ≥ teacher).** With an executable
  verifier, a student can exceed its teacher because the verifier carries
  information the teacher's distribution does not:
  - `tools/cyber_train` — closes teach→train→serve from scratch on a real
    domain (BPE on a defensive-cyber corpus → small student → GGUF → re-served
    bit-faithfully).
  - `Code_DSL` + `tools/code_distill` — verifier-filtered distillation: a
    student trained only on a noisy teacher's *verified-correct* outputs reaches
    100% vs the teacher's 40% (naive imitation 20%).
  - `tools/code_iterate` — verifier-bootstrapped self-improvement (STaR-style,
    grammar-constrained sampling, continual training + reject-and-restore so the
    solve-count is monotone): from a random proposer to full coverage with NO
    teacher, driven only by the verifier.
  - `Verifier` interface (`src/train/verifier.ads`) — pluggable executable
    oracle (token-sequence and source-text variants); `Code_DSL.DSL_Verifier`
    and the real-interpreter `Exec_Verifier` (Track 2) both implement it.
  - `tools/exec_distill` — Track 2: a REAL interpreter (python3) runs candidate
    code against tests, filtering a noisy teacher's real solutions (validated on
    Linux/droplet too). `make distill-demos` runs all three.
  - `tools/gen_verify` — Track 2 Phase 2: a real model (`LLM_Llama.Chat`)
    generates code, `Exec_Verifier` runs it; Llama-3.1-8B scored 5/5 verified on
    the droplet (real model → real interpreter, demo untouched).
- **Training-platform control-plane (Step 1, reviewed)** — `PLATFORM.md`
  blueprint + `src/train/platform.ads/adb`: job contract, **exact fixed-point
  pricing (no Float)**, hard **spend cap** + escrow deposit, **failed-job charge
  policy** (provider-cost only, no margin on failure), overflow-guarded sizing,
  student-size tiers, and a **rigorous delivery gate** (`Make_Report`: verified
  domain + held-out N ≥ 50 + student ≥ teacher + margin). Tested (`test_platform`,
  18 checks, in `make train-test`). Hardened after a 4-engineer review panel
  (ML/distributed/security/product): guarantee rescoped/conditional, security +
  sandbox + tenancy + legal sections added, multi-GPU gated behind a measured
  benchmark, real-domain "beats-teachers" proof pulled forward as the go/no-go.
- **Job admission gate + persona/attestation (Step 4 engine slice)** —
  `Platform.Job_Spec` now carries a persona (student identity/copyright name +
  system-behaviour prompt) and a `Teacher_Attested` flag; `Platform.Admit`
  refuses to provision a job that isn't teacher-license-attested, lacks a persona,
  or exceeds budget (`Allow`/`Reject_*`). Tested. Legal *policy* (allowed
  teachers, attestation strength) remains a business decision (PLATFORM.md §Legal).
- **Rigorous eval (Step 2 gate integrity)** — `Exec_Verifier` now has held-out
  (hidden) tests separate from the visible tests: best-of-N SELECTS on visible,
  the gate SCORES on hidden (`Eval_Correct`). `test_eval` proves an overfit
  solution that hard-codes the visible answers passes selection but FAILS the
  held-out eval — so the "beats teachers" guarantee can't be gamed. The real-
  model number on a gate-size (N≥50) benchmark remains GPU-gated.
- **Exec_Verifier hardening + pluggable sandbox (Step 3)** — model-generated
  code runs from a unique scratch path, via direct exec (no shell-string), output
  suppressed, wall-clock bounded by `timeout`, and **wrapped by a configurable
  isolator** (`ASPIDA_VERIFY_SANDBOX` command prefix — validated end-to-end with
  a benign `/usr/bin/env` wrapper). The hard isolator recipe (firejail/container:
  non-root, no-net, blocked metadata, rlimits, seccomp) is in PLATFORM.md
  §Security; required before multi-tenant use.
- **GPU-resident training SESSION (Step 5a)** — `gpu/train_resident_shim.cu`:
  a C-ABI session (`art_create`/`art_set_data`/`art_step`/`art_free`) keeping
  weights + AdamW + data resident on the device (only the scalar loss copied
  per step), built as both a self-test and `libaspidatrain.so` for Ada `dlopen`.
  Validated on an L40S (loss 3.6e-2 → 9.5e-8, ~1170 steps/s) alongside the live
  demo. The substrate the Ada `Student` will drive (Step 5b) — no per-op host
  round-trips.
- **Ada drives GPU-resident training (Step 5b)** — `src/train/train_gpu_resident.ads/adb`
  dlopen the resident-session shim (like `LLM_GPU`; path `ASPIDA_TRAIN_LIB`),
  exposing `Create`/`Set_Data`/`Step`/`Free` with graceful CPU fallback.
  `tools/train_resident_probe` drives a session from Ada; validated **end-to-end
  on an L40S** (loss 3.6e-2 → 5.1e-15) — Ada now trains on the GPU with only the
  scalar loss crossing the FFI per step, demo untouched. (Builds + loads
  gracefully with no shim, so it runs in `make train-test`.)
- **Provisioning orchestrator (Step 7)** — `tools/provision_and_train.sh`: the
  "engineer picks N droplets → we provision → distributed-train → tear down" step.
  Creates GPU droplets (create-retry for fluctuating capacity), deploys the
  resident-Student shim + `dp_node`, runs real TCP all-reduce data-parallel over
  the VPC, collects both ranks' loss, and **always tears the droplets down** (exit
  trap). `--dry-run` prints the plan at $0 (validated); `--run` executes (2-droplet
  prototype, matching `dp_node`; N>2 needs ring all-reduce).
- **Networked all-reduce — real TCP data-parallel (Step 7)** — `gpu/dp_node.cu`:
  two processes each run a resident Student, exchange + SUM the gradient
  accumulators over a TCP socket each round (the all-reduce), then apply.
  Validated at $0 over localhost on the L40S — rank 0 and rank 1 stay
  **bit-identical** (|Δloss| = 0), proving the networked data-parallel path. The
  same binary across two droplets is the real multi-node run (only the peer IP
  changes). Distributed training is now real end-to-end: grad-accumulation + TCP
  all-reduce + proven correctness; ZeRO/FSDP sharding for Large remains.
- **Data-parallel all-reduce — correctness proven (Step 7)** —
  `gpu/student_shim.cu` exposes `stu_nparams`/`stu_get_acc`/`stu_set_acc` to
  flatten/restore the gradient accumulators for a cross-node SUM; deterministic
  init (`stu_create` resets the seed) keeps replicas in sync. `gpu/test_dataparallel.cu`
  validates it at $0 on one GPU (two replicas, real sum-all-reduce between micro
  and apply): two nodes on different data shards converge **bit-identically** to a
  single node trained on the combined data (|R0−R1| = |R0−Rref| = 0). The only
  thing a real two-droplet run adds is moving the accumulator blob over TCP.
- **Gradient accumulation (Step 7 enabler)** — `student_shim.cu` adds
  `stu_micro` (accumulate one micro-batch's grads) + `stu_apply` (average over G +
  one AdamW update); `Student_GPU.Micro`/`Apply` drive it from Ada. Validated
  end-to-end on an L40S (G=4 micro/update, loss collapses). This is what makes
  multi-droplet data-parallel viable per the benchmark below: the all-reduce sums
  the accumulators across nodes once per G micro-steps, amortising the comm cost.
- **Scale-out go/no-go — measured all-reduce benchmark (Step 7)** — provisioned
  two throwaway droplets in `tor1` (the GPU region) and measured inter-droplet VPC
  bandwidth: **~1.68 Gbit/s, 2.1 ms RTT** (small-droplet lower bound). Against the
  `Platform` tiers, per-step ring all-reduce is ~1.0 s (Small) / 7 s (Medium) /
  51 s (Large) — so naive per-step data-parallel is **negative-scaling for
  Medium/Large**; the viable path is gradient-accumulation + compute/comm overlap
  (and ZeRO/FSDP for Large). Multi-node is gated accordingly (gpu/README.md).
  Benchmark droplets destroyed; demo untouched.
- **MVP turnkey orchestrator (Step 6)** — `src/train/turnkey.ads/adb`: sequences a
  job through the validated control-plane (Admit → train → quality-gate →
  Final_Charge), with training and held-out evaluation injected as callbacks (the
  GPU engine `Student_GPU` and `Exec_Verifier` wire in for real runs; export+E2EE
  serve plug in after a Delivered outcome). `test_turnkey` exercises every branch
  model-free (12 checks): delivered → metered charge; gate-fail → provider-cost
  only; not-attested → rejected, zero charge; train-fail → aborted at the cap.
  `tools/turnkey_demo` runs the loop on **real components** (no train/eval stubs,
  model-free, $0): real verifier-filtered training + a real 60-instance held-out
  eval through `Turnkey` — a verified student (100%) beats a noisy teacher (12%),
  job Delivered, charged $6.50 within cap. The whole platform vertical end-to-end.
- **Delivery glue — GGUF export + serve after Delivered** — `Turnkey` gains a
  `Deliverer` callback + `Delivery` artifact (GGUF path + E2EE endpoint), invoked
  ONLY on a Delivered outcome (`test_turnkey` asserts it runs on Delivered and not
  on gate-fail/reject/abort). `tools/turnkey_serve_demo` proves it on a REAL
  trained student: train (loss→3e-5) → gate (student 98% vs teacher 8%, N=100) →
  Delivered → `Student.Export_GGUF` (real 337 KB file) → **loaded back by the
  inference engine `LLM_Llama` (serve-ready)** → endpoint issued. Closes the
  delivery end of the platform.
- **Distillation in the GPU engine (P3)** — soft-target (KL) cross-entropy
  (`k_ce_soft`: loss = −Σ Q·log softmax, grad = softmax − Q) grad-checked
  standalone (`test_celoss_soft.cu`, max rel 3.5e-3) and wired into the resident
  Student (`stu_set_distill` → soft-target mode). End-to-end on an L40S
  (`test_distill_gpu.cu`): the GPU Student distilled toward a teacher distribution
  Q reaches **exactly the teacher entropy H(Q)** (loss 55.1 → 26.26 = H(Q), rel
  gap 0.000) — its softmax becomes Q. Real-teacher distributions now train the
  GPU Student in the same loop; verifier-filtering is example selection upstream.
- **P6 — UARP integration: API-key auth (engineer's own key) ✅** — the platform
  rides on the existing **UARP** backend (snaga.ai — auth, billing/stripe/plans/
  markup, runs/agents/model-catalog, 420 endpoints); the Aspida engine is the
  training/inference compute behind it, so the platform API/auth/billing are
  UARP's, not rebuilt. `src/train/platform_auth.ads/adb` (`Platform_Auth`) validates
  an engineer's `uarp_..._...` key against UARP `GET /api/v1/me` (via `curl` — no
  Ada TLS client; key passed through a 0600 config file, not argv) and returns
  their `user_id` for ownership/billing; verified keys cached for the process.
  `tools/platform_auth_probe` — validated end-to-end against prod: a real key →
  AUTHORIZED (user_id parsed), an invalid key → REJECTED. One key works across
  snaga.ai and the training platform.
- **P6 — control-plane state spine** — `src/train/job_store.ads/adb`: `src/train/job_store.ads/adb`:
  a job registry with a guarded lifecycle (Submit→Quoted, Fund→Funded,
  Start→Running, Finish→Delivered/Failed_Gate/Aborted_Cap); invalid transitions
  raise `Bad_Transition`, unknown ids `Not_Found`. Stores each job's quote and
  final Turnkey outcome. `test_job_store` (10 checks, model-free). The submit/
  status/run API + auth + persistence + scheduler layer over this is the rest of P6.
- **P5 — verifier sandbox (real isolation) ✅** — `tools/verify_sandbox.sh`: a
  hard sandbox for untrusted verifier code using only preinstalled util-linux
  (no apt footprint) — `unshare -n` (private netns: NO egress, blocks the
  cloud-metadata IP), `setpriv` (drop to nobody), `timeout -KILL`, and ulimits
  (address space / CPU / PIDs / file size). Validated on Linux: benign code runs
  (rc 0), runs as `nobody`, network egress to 169.254.169.254 is **blocked**
  (rc 1, no reach), and a `while True` runaway is **killed at the timeout**
  (rc 137). Wired via `ASPIDA_VERIFY_SANDBOX` (Exec_Verifier already routes
  every execution through it, Step 3). Required before multi-tenant use.
- **P4 — student BEATS teacher on a real domain (SVG icons) ✅** — the make-or-break
  proof. `tools/svg_icons.py`: 64×64 icon grammar (324 icons) + a real executable
  oracle (render the candidate SVG with cairosvg, accept on RGB pixel-MSE vs the
  target). `tools/train_svg.adb`: a byte-level Student trained on verifier-filtered
  data (spec→SVG), exported to GGUF, **generated by the real engine**. Final on 65
  held-out icons (combos unseen in training): **STUDENT 65/65 = 100% vs TEACHER
  50/65 = 76.9%** — the verifier-filtered student generalises and beats the noisy
  teacher on a genuine renderable domain (not the toy DSL). Passes the platform
  gate (`Make_Report`: verified, N=65≥50, student ≥ teacher+margin → Delivered).
  Done at **~$0** (local CPU training + engine generation; cairosvg verify on the
  droplet), well under the $50 cap; demo untouched.
- **Data pipeline (P2)** — `src/train/data_pipeline.ads/adb`: turns a domain
  corpus into next-token training windows for the GPU Student, with data-parallel
  sharding. Byte-level tokenisation (vocab-free, deterministic; BPE swaps in
  without changing the windowing/sharding contract), `Window_Count`/`Window`
  (Ids + one-shifted Tgts), and `Shard` (balanced contiguous slices per rank).
  `test_data_pipeline` (11 checks, in `make train-test`): ingest, window+shift,
  and a 10/3 shard that is contiguous, disjoint, and covers all. Model-free, $0.
- **Tensor-core FP16 GEMM (P1, perf)** — `gpu/test_gemm_wmma.cu`: WMMA kernel
  (FP16 inputs, FP32 accumulate) validated on an L40S vs the FP32 reference —
  **6.6× faster (5.4 → 35.8 TFLOP/s), Frobenius rel err 2.6e-4**. This is the
  production perf lever (matmuls are ~90% of training cost). **Forward path wired
  into the resident Student** (`mm_w`): tensor-core FP16 (FP32 accumulate) when
  dims are 16-aligned, FP32 fallback otherwise; FP32 master weights + FP32
  backward, so no loss scaling. Validated — an aligned config (S=16, dims ×16)
  trains through the FP16 forward (loss collapses); non-aligned configs use the
  FP32 path. Next: WMMA for the backward (transposed) matmuls, validated at real
  tier scale (P4).
- **Stream B production hardening — multi-head attention + AdamW** — multi-head
  causal attention (`gpu/test_attention_mh_gpu.cu`, H heads × dh, per-head softmax)
  grad-checked on an L40S (dQ/dK/dV across both heads, max rel 1.7e-4); and the
  resident Student shim (`gpu/student_shim.cu`) upgraded from SGD to **AdamW**
  (flat optimizer list over all parameters, bias-corrected). **Multi-head + per-head
  RoPE are now wired into the resident Student** (both the grad-checked self-test —
  2-layer multi-head, max rel 1.7e-2, loss → 0.009 — and the Ada-driven shim, loss
  20.5 → 0.015) — the Student is a true multi-head transformer with AdamW. The
  shim is now **runtime-configurable** (`stu_create(V,D,F,S,L,H)` — one
  `libaspidastudent.so` serves any `Platform` tier; `Student_GPU.Create` takes the
  architecture; Ada trained two differently-sized configs end-to-end on an L40S,
  both loss → 0.014). A shared-memory **tiled FP32 GEMM** (`gpu/test_gemm_tiled.cu`,
  validated 1.3× faster on 1024³, bit-exact) is wired into the Student forward.
  Remaining perf lever: FP16/tensor-core (WMMA) — a further step that changes
  numerics (FP16 accumulate, not bit-parity) and needs FP16 I/O throughout.
- **GPU training engine COMPLETE — Ada drives a full transformer (Step 5i, Stage C)** —
  `gpu/student_resident.cu` generalised to L stacked layers (grad-checked across
  both layers on an L40S, max rel 8.9e-3, loss → 0.009); `gpu/student_shim.cu`
  exposes the whole resident Student as a C-ABI session (`stu_create`/`stu_set_data`/
  `stu_step`/`stu_free`) built as `libaspidastudent.so`; `src/train/student_gpu.ads/adb`
  (`Student_GPU`) dlopens it and `tools/student_gpu_probe` drives it. **Validated
  end-to-end on an L40S: Ada trained a full 2-layer transformer-LM on the GPU**
  (loss 20.5 → 0.010), only the scalar loss crossing the FFI, demo untouched.
  This closes Step 5 (single-node GPU Student training): resident loop (5a–5c),
  all kernels grad-checked (5d–5h), full Student assembled + Ada-driven (5i).
- **Resident Student assembly — Stage B: full transformer layer (Step 5i)** —
  `gpu/student_resident.cu` now chains a complete pre-norm transformer layer
  (RMSNorm → Q/K/V → RoPE → causal attention → residual → RMSNorm → SwiGLU →
  residual) between embedding and head, with the correct transposed backward and
  both residual gradient paths. Grad-checked **end-to-end** on an L40S — all nine
  layer weights (Wq/Wk/Wv/Wo/g1/g2/Wgate/Wup/Wdown) match a finite difference of
  the CE loss (max rel 1.4e-2) — and a train-sanity SGD drives loss 20.95 →
  0.015. The full transformer forward+backward is assembled and numerically
  correct on GPU; remaining: multi-layer loop (mechanical) + Ada wiring (Stage C).
- **Resident Student assembly — Stage A (Step 5i)** — `gpu/student_resident.cu`:
  the grad-checked kernels composed into one forward+backward pipeline
  (embedding → final RMSNorm → head → cross-entropy), grad-checked **end-to-end**
  on an L40S (dE/dgf/dWh vs a finite difference of the CE loss; all probes ok,
  max rel 2.1e-3) and a train-sanity SGD that drives loss 20.9 → 2.8. The
  skeleton the transformer layers plug into (Stage B: RMSNorm+attn+RoPE+residual
  +RMSNorm+SwiGLU+residual).
- **Cross-entropy + embedding GPU ops, grad-checked (Step 5h)** —
  `gpu/test_celoss_gpu.cu`: cross-entropy loss (dlogits = softmax − onehot) and
  token embedding (gather forward / scatter-add backward, with a repeated id to
  exercise accumulation), grad-checked on an L40S (all probes ok, max rel 5.3e-5).
  **This completes the grad-checked kernel set for a GPU-resident `Student`**
  (matmul + RMSNorm + SwiGLU + RoPE + attention + CE/embedding); next is assembly
  of the validated kernels into the full resident forward/backward loop.
- **Causal attention GPU op, grad-checked (Step 5g)** — `gpu/test_attention_gpu.cu`:
  single-head causal scaled-dot-product attention forward + backward (the hardest
  op — gradient flows through the softmax jacobian), grad-checked for dQ/dK/dV vs
  a finite difference of E=½·Σ(out−t)² on an L40S (all probes ok, max rel 2.1e-3).
  A first round flagged two small-gradient dK probes; improving the finite-diff
  precision (double-accumulated loss + larger ε) made them converge to the
  analytic values — confirming FP32 finite-diff noise, not a backward bug.
- **RoPE GPU op, grad-checked (Step 5f)** — `gpu/test_rope_gpu.cu`: rotary
  position embedding forward + backward (backward = rotation by −θ, since the
  rotation is orthogonal), grad-checked vs finite difference on an L40S (all
  probes ok, max rel 7.9e-4). No learnable params.
- **SwiGLU GPU op, grad-checked (Step 5e)** — `gpu/test_swiglu_gpu.cu`: the gated
  activation h = SiLU(a)·b forward + backward (gate-grad `da`, up-grad `db`),
  grad-checked vs finite difference on an L40S (all probes ok, max rel 5.2e-3;
  abs floor handles near-zero grads). MLP matmuls are already the resident matmul.
- **RMSNorm GPU op, grad-checked (Step 5d)** — `gpu/test_rmsnorm_gpu.cu`:
  RMSNorm forward + backward CUDA kernels (input grad `dx` and gain grad `dg`),
  grad-checked against a finite difference of E=½·Σ(y−t)². Validated on an L40S
  (all probes ok, max rel 3.4e-3). First transformer op grafted toward a
  GPU-resident `Student`; the matmuls (QKV/MLP/head) are already the resident
  matmul. Remaining ops: RoPE, attention (softmax), SwiGLU, embedding/head + CE.
- **GPU backward grad-checked (Step 5c)** — `train_resident_shim.cu` exposes
  forward-only loss, the analytic gradient (no AdamW), and single-weight
  read/perturb; `train_resident_probe` compares the GPU gradient against a CPU
  finite difference of E=½·Σ(Y−T)². Validated on an L40S: every probe within a
  combined rel(<2e-2)/abs(<5e-5) tolerance (abs floor handles FP32 cancellation
  on near-zero gradients) — the resident backward is numerically correct, not
  just loss-collapsing. Next: graft the full `Student` (RMSNorm/RoPE/attn/SwiGLU).
- **GPU-resident training loop (Stage 1)** — `gpu/train_mlp.cu`: weights + AdamW
  moments + data resident on the device across all steps (only the scalar loss
  copied back). Validated on an L40S (loss collapses 3.6e-2 → 4.3e-6; ~1200
  steps/s) running alongside the live demo. Foundation for GPU-resident `Student`
  training; multi-GPU/distributed stages are the documented roadmap (`gpu/README.md`).

### Changed
- Refactored backend interface to polymorphic Model_Backend
- Distillation `Forward_Logits` and `Distill.Capture`/`Capture_Ensemble` now
  heap-allocate the `[N×vocab]` logit buffers (extended-return on the backends),
  so large-vocabulary teachers (e.g. Gemma's 262k) no longer overflow the
  primary stack

### Fixed
- Stack overflow capturing distillation logits from large-vocabulary teachers
  (Gemma 262k): the megabyte `[N×vocab]` buffers are now heap/secondary-stack
- GGUF loader hardened against malformed/untrusted files: tensor offset+size
  validated against the actual file length; element-count/byte-size computed by
  a single checked path; `n_dims`, value-type codes and `general.alignment`
  validated; `Open` closes its fd and raises `Malformed_GGUF` on any error
- `Student.Load` / `Distill.Read` now fail loud (`Bad_Checkpoint` / `Bad_Dataset`,
  closing the file) on truncated/corrupt input instead of leaking a raw
  exception and a file descriptor
- Secure-channel frame-length check moved before the `Natural` conversion (a
  ≥2³¹ length no longer escapes as `Constraint_Error`); handshake frames capped
- Session store: master password captured then scrubbed from the environment
  (no longer inherited by children); sealed files/dir created `0600`/`0700`;
  `fsync` before the atomic rename; 128-bit session ids
- `ws_bridge` relay idle-timeout; `openai_proxy` re-establishes its channel on a
  mid-request error (no cross-request response confusion)

### Security
- Best-effort zeroization extended to X25519 field-element temporaries,
  Poly1305 state and the AEAD recomputed tag (constant-time discipline and full
  SPARK proof of the core crypto units preserved; flow analysis clean)

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
