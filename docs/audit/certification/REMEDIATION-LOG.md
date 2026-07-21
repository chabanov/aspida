# Aspida Engine — Certification Remediation Log

Opened 2026-07-21. Three parallel audits (CUDA memory-safety, Ada concurrency,
Security/E2EE). 22 findings. Status column updated as each is remediated and
verified (rebuild + greedy canary 8752e132c193abbe + concurrency battery).

Severity: CRITICAL/HIGH = certification blocker; MEDIUM = must-fix; LOW/INFO = hardening.

## Blockers (silent corruption / poisoning — explains the two open mysteries)

| # | Sev | Area | Finding | Status |
|---|-----|------|---------|--------|
| A2 | HIGH | Ada | prefix-cache pins leak on any raise (Sink.Tick client-abort) between Reserve/Lookup and Release → slot stuck Pins>0/Valid=false → cache permanently dead + ~290MB VRAM stranded per leak. **= the abort-triggered "state poisoning" residual.** | FIXED |
| C1 | HIGH | CUDA | `aspida_ggml_fattn_prefill` returns void; on gfa_rebuild failure (per-chunk realloc OOM) leaves `out` uninitialized → stale scratch through o_proj → silent deterministic garbage. **= second poisoning mechanism.** | FIXED |
| A1 | CRIT | Ada | `Batch_Log` (shared, compacted-index) overwritten by next batch before an un-queued slow lane copies its slice in Wait_Done → lane gets another request's logits → silent wrong token under load. | FIXED |
| A3 | HIGH | Ada | Step_Timeout `Abandon`+`Free_States` could free device state an in-flight forward reads → UAF. MITIGATED by prior fixes: Dnet_Free/Fattn_Free take g_ggml_mu (held by the whole forward) + cudaDeviceSynchronize before cudaFreeAsync, so a free BLOCKS until the forward completes — no UAF. A truly-wedged forward hangs (not UAF), caught by systemd/watchdog restart. | MITIGATED |
| C2 | MED | CUDA | gfa_rebuild/ggdn_rebuild set the shape key only on success; a failed alloc leaves tensors with NULL ->data but the OLD shape key → a later matching-shape call skips rebuild → NULL/OOB write. Compounds C1. | FIXED |

## Memory-safety defense-in-depth

| # | Sev | Area | Finding | Status |
|---|-----|------|---------|--------|
| C3 | MED | CUDA | k_fattn_prep / k_fattn_prep_chunk write K/V at `pos*kvd` with no guard vs `max_len` (write side of the 854d062 read-side fix). | FIXED |
| C4 | MED | CUDA | batched/prefill drivers index g_dnet/g_fattn with caller handles with no bounds/type check (single-lane paths do guard). | FIXED |
| C5 | LOW | CUDA | ggml first-compute warmup guarded per-helper; MoE+fattn lack it. INERT under prod config (ASPIDA_DNET_GGML=1 warms dnet first). | DOCUMENTED |
| C7 | LOW | CUDA | weight-tile LOAD lacks `gn<out` guard → OOB READ when out%16≠0. Inert (vocab/intermed multiples of 64); benign read into weight arena. Guard deferred (hot kernel). | DOCUMENTED |
| C8 | LOW | CUDA | k_dnet_recur_b/_chunk assume khd==vhd (true for qwen35moe: 128==128). Model-invariant. | DOCUMENTED |
| C9 | LOW | CUDA | k_moe_route_b skips route write when n_exp>512 (inert at n_exp=256). | DOCUMENTED |
| C6 | LOW | CUDA | single-lane aspida_gpu_chain_forward takes no g_ggml_mu (safe only under an Ada inference lock). | DOCUMENTED |

## Ada concurrency (non-blocker)

| # | Sev | Area | Finding | Status |
|---|-----|------|---------|--------|
| A4 | MED | Ada | vision state in package globals (Vis_Vtok/Ntok/Active) raced by concurrent image lanes. | FIXED |
| A5 | MED | Ada | Decode `exception=>LLM_Step_Lock.Release` fires on the batch path that never Acquired → can drop a CPU-fallback generation's lock. | FIXED |
| A6 | LOW | Ada | Free_States holds Alloc_Lock with no exception-safe unlock → wedge if a free raises. | FIXED |
| A7 | LOW | Ada | cross-lane CUDA error misattribution via single global last_error. | DOCUMENTED |

## Security / E2EE (core verified SOUND — no nonce-reuse, const-time MAC)

| # | Sev | Area | Finding | Status |
|---|-----|------|---------|--------|
| S1 | MED | Sec | :8099 proxy is a plaintext boundary; bind is hardcoded 127.0.0.1 (the guard). Deployment constraint: never front with a remote-terminating TLS proxy; document as local-only shim. | DOCUMENTED |
| S2 | MED | Sec | Wipe uses an anti-DSE fold-and-raise idiom, not a guaranteed barrier; LTO/inlining may elide the stores. | FIXED |
| S3 | LOW | Sec | mlock gaps on derivation scratch + PBKDF2 key. Deployment mitigation: swap disabled on the prod host. Full arena-mlock deferred. | DOCUMENTED |
| S4 | LOW | Sec | plaintext session turns in Unbounded_String freed without scrubbing. | DOCUMENTED |
| S5 | INFO | Sec | no authenticated end-of-stream (truncation shows as conn error); client anon by design (Noise-NK + optional token). | DOCUMENTED |
| S6/VC | INFO | SPARK | 7+ open VCs (sha256/chacha20/poly1305/aead length-overflow bounds) — discharge with input-length preconditions. | IN PROGRESS |

Verification protocol per fix: `bash build_so.sh` (or gprbuild for Ada) clean;
greedy canary bit-exact `8752e132c193abbe`; mixed-length concurrency battery
0 collapse; abort-storm (rst_repro) no persistent poison.
