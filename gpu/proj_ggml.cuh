// proj_ggml.cuh — aspida prefill Q8_0 dense projections via llama.cpp's
// mul_mat_q (MMQ int8 tensor-core path), through the ggml public API.
// Targets the k_q8_reg<4,4> tier (B>=64 prefill projections) which nsys shows
// at ~30% of GPU-busy on a SIMT kernel (~34 TFLOPS effective) — MMQ uses int8
// tensor cores like llama.cpp.  NOT bit-exact vs k_q8_reg (int8 accumulation
// order differs) — same acceptance as the MoE MMQ path: eval-gated.
// llama.cpp / ggml are MIT licensed, (c) 2023-2024 The ggml authors.
//
// Weight handling: ggml has no public device-ptr buffer wrap, so the first
// call per weight D2D-copies the Q8_0 bytes into a ggml-owned tensor (byte
// layout identical: block_q8_0 {fp16 d; int8 qs[32]} == aspida 34-byte block).
// ~1.2GB duplicate VRAM across all projection weights — cheap on H200.
//
// Requires fattn_ggml.cuh (shared backend g_gfa) + moe_ggml.cuh (upload map
// pattern) to be included first.
#pragma once
#include <unordered_map>

struct GgmlProjW { ggml_context *ctx; ggml_backend_buffer_t buf; ggml_tensor *t; };
static std::unordered_map<const void *, GgmlProjW> g_proj_wmap;
static thread_local ggml_gallocr_t g_proj_ga = nullptr;   // P3: per-thread

//  Wrap (copy) a device Q8_0 weight [out rows x in cols] into a ggml tensor.
static ggml_tensor *proj_ggml_weight(const uint8_t *dw, int in, int out) {
    auto it = g_proj_wmap.find(dw);
    if (it != g_proj_wmap.end()) return it->second.t;
    if ((in % 32) != 0 || !aspida_ggml_be()) return nullptr;
    struct ggml_init_params ip = { ggml_tensor_overhead() * 2, nullptr, true };
    ggml_context *ctx = ggml_init(ip);
    ggml_tensor *t = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, in, out);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, aspida_ggml_wbe());   // P3
    if (!buf) { ggml_free(ctx); g_proj_wmap[dw] = {nullptr,nullptr,nullptr}; return nullptr; }
    size_t bytes = (size_t) out * (in / 32) * 34;
    cudaMemcpy(t->data, dw, bytes, cudaMemcpyDeviceToDevice);
    g_proj_wmap[dw] = { ctx, buf, t };
    return t;
}

//  y[B][out] = x[B][in] * W^T via ggml mul_mat (quantize_q8_1 + MMQ).
//  Returns false to fall back to the aspida kernels.
static bool aspida_ggml_proj(const uint8_t *dw, int in, int out,
                             const float *dx, float *dy, int B, cudaStream_t st) {
    std::lock_guard<std::recursive_mutex> ggml_lk (g_ggml_mu);
    set_stage("ggml-proj B=%d in=%d out=%d", B, in, out);
    ggml_tensor *w = proj_ggml_weight(dw, in, out);
    if (!w) return false;
    if (!g_proj_ga) g_proj_ga = ggml_gallocr_new(g_wbuft ? g_wbuft : (aspida_ggml_wbe(), g_wbuft));
    struct ggml_init_params ip = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true };
    ggml_context *ctx = ggml_init(ip);
    ggml_tensor *x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in, B); ggml_set_input(x);
    ggml_tensor *y = ggml_mul_mat(ctx, w, x);                       // [out, B] f32
    ggml_set_output(y);
    ggml_cgraph *gr = ggml_new_graph(ctx);
    ggml_build_forward_expand(gr, y);
    if (!ggml_gallocr_alloc_graph(g_proj_ga, gr)) { ggml_free(ctx); return false; }
    cudaStreamSynchronize(st);                        // x ready on st
    cudaMemcpy(x->data, dx, (size_t) B * in * 4, cudaMemcpyDeviceToDevice);
    //  First-compute warmup transient (same as dnet_ggml): absorb once.
    static thread_local bool proj_warmed = false;   // P3: warm each thread's backend
    if (!proj_warmed) { ggml_backend_graph_compute(aspida_ggml_be(), gr); cudaDeviceSynchronize(); proj_warmed = true; }
    ggml_backend_graph_compute(aspida_ggml_be(), gr);
    cudaDeviceSynchronize();                          // y ready before st copies
    op_trace("ggml-proj");
    cudaMemcpyAsync(dy, y->data, (size_t) B * out * 4, cudaMemcpyDeviceToDevice, st);
    cudaStreamSynchronize(st);   // y lives in the shared gallocr pool — consume before unlock
    ggml_free(ctx);
    return true;
}
