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
| A3 | HIGH | Ada | Step_Timeout `Abandon`+unwind `Free_States` frees device KV/state while a genuinely-wedged `Chain_Forward_Batch` is still reading them → device use-after-free. | FIXED |
| C2 | MED | CUDA | gfa_rebuild/ggdn_rebuild set the shape key only on success; a failed alloc leaves tensors with NULL ->data but the OLD shape key → a later matching-shape call skips rebuild → NULL/OOB write. Compounds C1. | FIXED |

## Memory-safety defense-in-depth

| # | Sev | Area | Finding | Status |
|---|-----|------|---------|--------|
| C3 | MED | CUDA | k_fattn_prep / k_fattn_prep_chunk write K/V at `pos*kvd` with no guard vs `max_len` (write side of the 854d062 read-side fix). | FIXED |
| C4 | MED | CUDA | batched/prefill drivers index g_dnet/g_fattn with caller handles with no bounds/type check (single-lane paths do guard). | FIXED |
| C5 | MED | CUDA | ggml first-compute warmup transient guarded per-helper; MoE+fattn lack it → config-fragile first-token garbage. | FIXED |
| C7 | LOW | CUDA | k_q8_wmma / k_moe_*_grouped weight-tile LOAD lacks the `gn<out` guard the store has → OOB read when out%16≠0 (benign at current dims). | FIXED |
| C8 | LOW | CUDA | k_dnet_recur_b/_chunk assume khd==vhd (true for this model; no guard). | FIXED |
| C9 | LOW | CUDA | k_moe_route_b returns without writing route when n_exp>512 → garbage idx (inert at n_exp=256). | FIXED |
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
| S1 | MED | Sec | :8099 proxy is a plaintext trust boundary; must never be fronted by a remote-terminating TLS proxy. Loopback bind is the guard — add public-bind refusal. | FIXED |
| S2 | MED | Sec | Wipe uses an anti-DSE fold-and-raise idiom, not a guaranteed barrier; LTO/inlining may elide the stores. | FIXED |
| S3 | LOW | Sec | mlock gaps: DH/HKDF derivation scratch + PBKDF2 key are swappable before wipe. | FIXED |
| S4 | LOW | Sec | plaintext session turns in Unbounded_String freed without scrubbing. | DOCUMENTED |
| S5 | INFO | Sec | no authenticated end-of-stream (truncation shows as conn error); client anon by design (Noise-NK + optional token). | DOCUMENTED |
| S6/VC | INFO | SPARK | 7+ open VCs (sha256/chacha20/poly1305/aead length-overflow bounds) — discharge with input-length preconditions. | IN PROGRESS |

Verification protocol per fix: `bash build_so.sh` (or gprbuild for Ada) clean;
greedy canary bit-exact `8752e132c193abbe`; mixed-length concurrency battery
0 collapse; abort-storm (rst_repro) no persistent poison.
