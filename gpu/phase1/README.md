# Stream B · Phase 1 — Q4_K GPU kernels (increment 1)

The foundational and hardest GPU primitive: dequantizing Q4_K weights and doing
the quantized mat-vec **on the GPU**, validated bit-for-bit against the CPU
engine (`LLM_Dequant.Dequant_Q4_K` / `LLM_Weight.MatVec`).

## Files
- `qk.cu` — CUDA kernels (`k_dequant`, `k_matvec`) + C-ABI host wrappers
  (`q4k_dequant_host`, `q4k_matvec_host`) + a self-test `main`. Mirrors the CPU
  Q4_K layout (144 B block, `get_scale_min_k4`, value = `d*sc*q − dmin*m`) and
  the mat-vec orientation (ne0=in, ne1=out; per-row ascending dot).
- `gen_ref.adb` — CPU-side fixture generator (built via `genref.gpr`): emits
  `dq_in/dq_exp` (one token_embd block + expected dequant) and
  `mv_w/mv_x/mv_y` (a real `blk.0.attn_k.weight` Q4_K tensor, a deterministic
  input, and the CPU MatVec output).

## Result (validated on a DO NVIDIA H100, CUDA 13.1)
Real Llama-3.3-70B Q4_K weights, `blk.0.attn_k.weight` = [in 8192, out 1024]:

```
nvcc -O2 --fmad=false -arch=native qk.cu -o qk_test
[dequant] max|gpu-cpu|=0.000e+00 exact_mismatches=0/256 -> BIT-EXACT
[matvec ] out=1024 in=8192  max_abs=0.000e+00 max_rel=0.000e+00 -> OK
PHASE1 KERNELS: PASS
```

Both kernels are **bit-identical** to the CPU engine: dequant has no summation,
and with `--fmad=false` + one-thread-per-row ascending accumulation the mat-vec
matches the CPU scalar reduction exactly. This de-risks the quant-GEMM (the
dominant cost) — the GPU path produces the same numbers as the validated CPU
forward.

## Reproduce
1. CPU: `gprbuild -P genref.gpr ... && obj/gen_ref <llama.gguf>` → 5 `.bin` files.
2. GPU box (nvcc): `nvcc -O2 --fmad=false -arch=native qk.cu -o qk_test && ./qk_test 8192 1024`.

## Increment 2 — RMSNorm / RoPE / SiLU / attention (`ops.cu`)
The remaining per-token primitives for a dense forward, validated on a DO
NVIDIA H200 (`nvcc --fmad=false`):

```
[rmsnorm] n=4096 max_abs=3.6e-6  max_rel=2.5e-6  -> OK   (vs CPU LLM_RMSNorm fixture)
[rope   ] dim=128 max_abs=6.0e-8 max_rel=4.2e-6  -> OK   (NeoX + real rope_freqs, vs CPU fixture)
[silu   ] n=4096 max_abs=0.0                      -> OK   (bit-exact)
[attn   ] nh8 nkv2 hd128 seq40 max_abs=2.6e-8    -> OK   (GQA causal, vs C reference)
PHASE1 OPS: PASS
```

`k_rmsnorm` / `k_rope` match the CPU engine (`LLM_RMSNorm.Forward`,
`LLM_RoPE.Apply` incl. freq_factors) to ~1e-6; `k_silu` is bit-exact; `k_attn`
(single-token GQA decode) matches a C reference to float noise (2.6e-8).
Attention is judged on absolute error — relative blows up on near-zero outputs.

So the **complete kernel set for a dense Llama forward** (quant-GEMM, attention,
RMSNorm, RoPE, SiLU) now exists on GPU and is numerically faithful to the
validated CPU engine. Remaining for a running GPU model: tiled/batched GEMM for
throughput, then wire a GPU `Model_Backend` (weights resident in VRAM, KV cache
on device) and validate a full Llama forward end-to-end + benchmark tok/s.

## Throughput benchmark (`bench.cu`)
Per-token decode at Llama-3.3-70B scale (the 7 Q4_K projections × 80 layers +
output projection; weights resident in VRAM; synthetic weights — content
irrelevant to timing), on a DO **H100**:

```
Llama-70B Q4_K decode (matmul-only), current un-tiled k_matvec:
  per-token: 444 ms  ->  2.25 tok/s   (CPU engine ~0.0125 tok/s @ ~80 s/tok)
```

**~180× faster than the CPU engine** — and this is the un-optimized
correctness-first `k_matvec` (one thread per output row, no tiling / shared
memory / FP16). A tiled/batched quant-GEMM is expected to add a large further
multiple. The throughput question (the whole point of the GPU backend) is
answered: the direction is sound.

## Full layer composition (`layer.cu`) — validated
The whole blk.0 transformer layer assembled from the kernels (rmsnorm ×2, the
Q4_K + **Q6_K** projections, GQA attention, SwiGLU, residuals) at pos 0, checked
against the CPU-engine reference (gen_ref) on real Llama-3.3-70B weights:

```
[layer] D=8192 FFN=28672 nh=64 nkv=8  max_abs=2.31e-07 -> OK
PHASE1 LAYER: PASS
```

Matches to float noise (2.3e-7). Note the model is **mixed-quant Q4_K_M**:
`attn_v` and `ffn_down` (and `output`) are **Q6_K**, the rest Q4_K — so this also
validated the Q6_K kernel (`k_matvec_q6k`, llama.cpp `block_q6_K` layout). A full
forward is 80 of these layers stacked, so the GPU forward's correctness is now
de-risked end to end; the layer loop is just repetition.

## Remaining for a running GPU model
1. Tiled/batched quant-GEMM (the main perf lever) + FP16 activations.
2. Wire a GPU `Model_Backend`: upload weights to VRAM once, KV cache on device,
   assemble the full Llama forward from these kernels, validate end-to-end vs
   the CPU engine (logits / "Paris"), then measure real decode tok/s.

GPU cost so far (Phase 0 + 1.1 + 1.2 + bench): ~$3.7 total (boxes destroyed after each run).
