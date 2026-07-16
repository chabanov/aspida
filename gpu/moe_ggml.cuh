// moe_ggml.cuh — aspida prefill MoE via llama.cpp's mul_mat_id (MMQ int8
// tensor-core path), through the ggml public API.  Measured 2.9x faster than
// the grouped wmma kernels at hura dims (dim=2048, 256 experts, top_k=8,
// intermed=512, Q8_0, P=1024): 5.71ms -> ~1.95ms per MoE layer-chunk.
// llama.cpp / ggml are MIT licensed, (c) 2023-2024 The ggml authors.
//
// Two pieces:
//  1. LOADER: routed expert gate/up/down Q8_0 blobs are allocated as ggml
//     tensors (ne=[k,m,n_experts]) at model load — aspida's expert layout
//     [e][rows][k] with bpe = m*bpr is byte-identical to ggml's as-tensor, and
//     ggml block_q8_0 {fp16 d; int8 qs[32]} == aspida's 34-byte Q8_0 block.
//     tensor->data is a plain device pointer (ggml CUDA buffers are
//     cudaMalloc'd), so the UNCHANGED decode kernels keep reading it directly.
//     No duplication: these bytes are uploaded ONCE, into ggml-owned memory.
//  2. PREFILL: per layer-chunk, a small ggml graph
//     mm_id(gate) + mm_id(up) + swiglu_split + mm_id(down)
//     over x [dim,1,P] (broadcast => token quantized once) and ids [top_k,P].
//     The shared expert + the deterministic weighted combine stay on the
//     aspida side.  Graph nodes are rebuilt per call (cheap, llama.cpp does
//     the same per step); the gallocr reuses its buffer for equal shapes.
//
// Requires fattn_ggml.cuh to be included first (shares its ggml backend).
#pragma once
#include "ggml.h"
#include "ggml-cuda.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <unordered_map>

//  Single shared ggml backend (same primary CUDA context as the engine).
static ggml_backend_t aspida_ggml_be() {
    if (!g_gfa.be) {
        g_gfa.be = ggml_backend_cuda_init(0);
        if (g_gfa.be) g_gfa.buft = ggml_backend_cuda_buffer_type(0);
    }
    return g_gfa.be;
}

//  ---- loader: host Q8_0 expert blob -> ggml-owned tensor -------------------
struct GgmlWeight { ggml_context *ctx; ggml_backend_buffer_t buf; ggml_tensor *t; };
static std::unordered_map<const void *, GgmlWeight> g_ggml_wcache;

//  w = host blob of n_mats experts, each m rows x k cols Q8_0 (bytes must
//  equal n_mats*m*(k/32)*34).  Returns the tensor or nullptr (fall back to
//  the aspida upload path).  Cached by host pointer like upload_weight.
static ggml_tensor *aspida_ggml_upload_q8(const void *w, long bytes,
                                          int64_t k, int64_t m, int64_t n_mats) {
    auto it = g_ggml_wcache.find(w);
    if (it != g_ggml_wcache.end()) return it->second.t;
    if (k <= 0 || m <= 0 || n_mats <= 0 || (k % 32) != 0) return nullptr;
    if ((long) (n_mats * m * (k / 32) * 34) != bytes) return nullptr;
    if (!aspida_ggml_be()) return nullptr;
    struct ggml_init_params ip = { ggml_tensor_overhead() * 2, nullptr, true };
    ggml_context *ctx = ggml_init(ip);
    ggml_tensor *t = ggml_new_tensor_3d(ctx, GGML_TYPE_Q8_0, k, m, n_mats);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, g_gfa.be);
    if (!buf) { ggml_free(ctx); return nullptr; }
    ggml_backend_tensor_set(t, w, 0, (size_t) bytes);   // H2D into ggml memory
    g_ggml_wcache[w] = { ctx, buf, t };
    return t;
}

//  Model-unload eviction (called from aspida_gpu_free_weight).
static void aspida_ggml_free_weight(const void *w) {
    auto it = g_ggml_wcache.find(w);
    if (it == g_ggml_wcache.end()) return;
    ggml_backend_buffer_free(it->second.buf);
    ggml_free(it->second.ctx);
    g_ggml_wcache.erase(it);
}

//  ---- prefill: the mm_id graph --------------------------------------------
struct GgmlMoE {
    ggml_gallocr_t ga = nullptr;
    //  inputs are re-created per call in a throwaway node context; the gallocr
    //  keeps (and reuses) the actual device buffer across calls.
};
static GgmlMoE g_gmoe;

//  Routed-experts MoE for one layer-chunk.  x_b: fp32 [P][dim] (device, ready
//  on stream st).  ids_dev: i32 [P][top_k] (device, ready on st) — expert ids.
//  Returns device pointer to out fp32 [P][top_k][dim] (valid until the next
//  call), or nullptr on failure (caller falls back to the grouped kernels).
static const float *aspida_ggml_moe_prefill(
        ggml_tensor *gW, ggml_tensor *uW, ggml_tensor *dW,
        const float *x_b, const int32_t *ids_dev,
        int dim, int intermed, int top_k, int P, cudaStream_t st) {
    if (!gW || !uW || !dW || !aspida_ggml_be()) return nullptr;
    if (!g_gmoe.ga) g_gmoe.ga = ggml_gallocr_new(g_gfa.buft);
    //  node context: tensor structs only (no_alloc); ~10 nodes
    struct ggml_init_params ip = { ggml_tensor_overhead() * 32 + ggml_graph_overhead(), nullptr, true };
    ggml_context *ctx = ggml_init(ip);
    ggml_tensor *x   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, dim, 1, P);  ggml_set_input(x);
    ggml_tensor *ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, top_k, P);   ggml_set_input(ids);
    ggml_tensor *gate = ggml_mul_mat_id(ctx, gW, x, ids);       // [intermed, top_k, P]
    ggml_tensor *up   = ggml_mul_mat_id(ctx, uW, x, ids);       // [intermed, top_k, P]
    ggml_tensor *h    = ggml_swiglu_split(ctx, gate, up);       // silu(gate)*up
    ggml_tensor *out  = ggml_mul_mat_id(ctx, dW, h, ids);       // [dim, top_k, P]
    ggml_set_output(out);
    ggml_cgraph *gr = ggml_new_graph(ctx);
    ggml_build_forward_expand(gr, out);
    if (!ggml_gallocr_alloc_graph(g_gmoe.ga, gr)) { ggml_free(ctx); return nullptr; }
    //  inputs were produced on the aspida stream; make them visible, then copy
    //  into the ggml-allocated input tensors.
    cudaStreamSynchronize(st);
    cudaMemcpy(x->data,  x_b,     (size_t) P * dim * 4,   cudaMemcpyDeviceToDevice);
    cudaMemcpy(ids->data, ids_dev, (size_t) P * top_k * 4, cudaMemcpyDeviceToDevice);
    ggml_backend_graph_compute(g_gfa.be, gr);   // MMQ int8 tensor-core mul_mat_id
    cudaDeviceSynchronize();                    // out ready for the combine on st
    const float *res = (const float *) out->data;
    ggml_free(ctx);        // frees node structs only; data lives in the gallocr buffer
    return res;
}
