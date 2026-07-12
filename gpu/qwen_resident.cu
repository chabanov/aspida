// qwen_resident.cu — resident-GPU decode blocks for the Qwen/MoE backend.
//
// See GPU_RESIDENT_FORWARD.md. This eliminates the per-matvec host<->device
// ping-pong (the measured 2.2 tok/s ceiling) by running a whole decode block on
// the device with the activation resident. Increment 1 = the fused MoE FFN.
//
// Build (added to the existing single-nvcc model; --fmad=false keeps the quant
// math bit-exact vs the SPARK CPU oracle, -arch=native keeps it portable):
//   nvcc -O3 --fmad=false -arch=native -shared -Xcompiler -fPIC \
//        gpu/gpu_matvec.cu gpu/qwen_resident.cu -o libaspidagpu.so
//
// Reuses gpu_matvec.cu: the resident weight cache (upload_weight / g_wcache,
// keyed by host pointer) and the warp-per-row K-quant GEMV kernels (k_q*k_w).
// Those symbols are shared via a small internal header so both TUs see them.
//
// STATUS: ABI frozen to match src/llm/llm_qwen_gpu.ads. The kernel bodies are
// implemented and validated test-driven on the GPU dev box, each gated
// bit-exact against LLM_MoE.Forward before promotion. This file documents the
// exact fused structure (mapped to the CPU reference) so that work is mechanical.

#include <cuda_fp16.h>
#include <cstdint>

extern "C" {

// Fused MoE FFN decode:  y[dim] = MoE(x[dim]).
//
// Oracle: LLM_MoE.Forward (src/llm/llm_moe.adb). Steps, all on-device with the
// activation resident (one H2D of x, one D2H of y — vs 28 round-trips today):
//
//   1. router logits:   rl[n_exp]      = router_w[n_exp,dim] . x          (quant GEMV)
//   2. stable softmax over rl[n_exp]                                       (1 block)
//   3. greedy top-k + renormalise -> idx[k], w[k]                         (small)
//   4. for each selected expert e in idx[0..k):                          (per-expert)
//        g = gate_w[e] . x ; u = up_w[e] . x                            (3D quant GEMV,
//        h = silu(g) * u                                                  stride=gb/n_exp)
//        y_e = down_w[e] . h
//        acc += w[e] * y_e
//   5. shared expert:  h = silu(shg.x)*shu.x ; ys = shd.h                 (2D quant GEMV)
//        gate = (sgi_len>1) ? sigmoid(sum(sgi[d]*x[d])) : 1
//        acc += gate * ys
//   6. y = acc
//
// gate/up expert blocks are 3D [n_exp, intermed, dim]; down is
// [n_exp, dim, intermed]; per-expert byte stride = <bytes> / n_exp. kind codes
// 0=Q4_K 1=Q6_K 2=Q5_K 3=Q3_K 4=Q2_K. Every weight pointer is a host address
// already resident in g_wcache (shared with aspida_gpu_matvec) — no re-upload.
void aspida_gpu_moe_decode(
    const float *x, int dim, int n_exp, int top_k, int intermed,
    const void *router_w, long router_bytes, int router_kind,
    const void *gate_w,   long gate_bytes,   int gate_kind,
    const void *up_w,     long up_bytes,     int up_kind,
    const void *down_w,   long down_bytes,   int down_kind,
    const void *shg_w,    long shg_bytes,    int shg_kind,
    const void *shu_w,    long shu_bytes,    int shu_kind,
    const void *shd_w,    long shd_bytes,    int shd_kind,
    const float *shared_gate_inp, int gate_inp_len,
    float *y);

}  // extern "C"
