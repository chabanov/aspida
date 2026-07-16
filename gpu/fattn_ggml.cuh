// fattn_ggml.cuh — aspida prefill full-attention via llama.cpp's fattn-mma
// (ncols2=8 GQA column-packing + cp_async pipelining), through the ggml public
// API (link-and-call).  Proven 6.5x faster than the wmma3 kernel at hd=256/40k
// on the an NVIDIA GPU.  See fattn_bench / fattn_mma_wrapper (Phase A) for the isolated
// validation.  llama.cpp / ggml are MIT licensed, (c) 2023-2024 The ggml authors.
//
// Layout mapping (aspida native -> ggml FA):
//   Q    aspida [t][h][d]           -> ggml q  ne=[hd,P,nq]   ([h][t][d])  repack
//   K/V  aspida cache [pos][kvh*hd] -> ggml k/v ne=[hd,len,nkv]([kvh][pos][d]) repack
//   mask ggml m f16 ne=[len,P], causal (s<=pos_start+t)?0:-inf  (built on GPU)
//   out  ggml FA out ne=[hd,nq,P]  == aspida [t][h*hd+d]  (ZERO-COPY, then gate)
// Repack is used (not zero-copy views) because ggml has no public CUDA
// buffer_from_ptr; the extra traffic is ~12% and the path is still 6.5x.
#pragma once
#include <cuda_fp16.h>
#include <cmath>
#include "ggml.h"
#include "ggml-cuda.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

// ---- repack / mask / gate kernels ----
__global__ void k_ggml_repack_Q(const float *__restrict__ qa, float *__restrict__ qg,
                                 int P, int nq, int hd) {
    size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) P * nq * hd) return;
    int d = i % hd; size_t r = i / hd; int t = r % P, h = r / P;   // ggml [h][t][d]
    qg[i] = qa[(size_t) t * nq * hd + (size_t) h * hd + d];        // aspida [t][h][d]
}
__global__ void k_ggml_repack_KV(const __half *__restrict__ Ka, __half *__restrict__ kg,
                                 int len, int nkv, int hd) {
    size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) len * nkv * hd) return;
    int d = i % hd; size_t r = i / hd; int s = r % len, kvh = r / len;  // ggml [kvh][s][d]
    kg[i] = Ka[(size_t) s * nkv * hd + (size_t) kvh * hd + d];          // aspida [s][kvh][d]
}
__global__ void k_ggml_build_mask(__half *__restrict__ m, int len, int P, int pos_start) {
    size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) P * len) return;
    int s = i % len, t = i / len;                       // ggml m ne=[len,P] -> (s,t)
    int qpos = pos_start + t;
    m[i] = __float2half((s <= qpos) ? 0.f : -INFINITY);
}
__global__ void k_ggml_apply_gate(float *__restrict__ out, const float *__restrict__ gate, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] *= 1.f / (1.f + expf(-gate[i]));
}

// ---- persistent ggml FA state (rebuilt when the (P,len) shape changes) ----
struct GgmlFA {
    ggml_backend_t be = nullptr;
    ggml_backend_buffer_type_t buft = nullptr;
    ggml_context *ctx = nullptr;
    ggml_gallocr_t ga = nullptr;
    ggml_cgraph *gr = nullptr;
    ggml_tensor *q = nullptr, *k = nullptr, *v = nullptr, *m = nullptr, *out = nullptr;
    int P = -1, len = -1, nq = 0, nkv = 0, hd = 0;
};
static GgmlFA g_gfa;

static bool gfa_rebuild(GgmlFA &s, int nq, int nkv, int hd, int P, int len) {
    if (!s.be) {
        s.be = ggml_backend_cuda_init(0);          // shares aspida's runtime-API
        if (!s.be) { fprintf(stderr, "[ggmlFA] cuda_init failed\n"); return false; }
        s.buft = ggml_backend_cuda_buffer_type(0); // primary context (device 0)
    }
    if (s.ctx) { ggml_free(s.ctx); s.ctx = nullptr; }
    if (s.ga)  { ggml_gallocr_free(s.ga); s.ga = nullptr; }
    struct ggml_init_params ip = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
    s.ctx = ggml_init(ip);
    s.q = ggml_new_tensor_4d(s.ctx, GGML_TYPE_F32, hd, P,   nq,  1); ggml_set_input(s.q);
    s.k = ggml_new_tensor_4d(s.ctx, GGML_TYPE_F16, hd, len, nkv, 1); ggml_set_input(s.k);
    s.v = ggml_new_tensor_4d(s.ctx, GGML_TYPE_F16, hd, len, nkv, 1); ggml_set_input(s.v);
    s.m = ggml_new_tensor_4d(s.ctx, GGML_TYPE_F16, len, P, 1, 1);    ggml_set_input(s.m);
    s.out = ggml_flash_attn_ext(s.ctx, s.q, s.k, s.v, s.m, 1.f / sqrtf((float) hd), 0.f, 0.f);
    ggml_flash_attn_ext_set_prec(s.out, GGML_PREC_F32);  // f32 accumulate (the benched fast path)
    s.gr = ggml_new_graph(s.ctx);
    ggml_build_forward_expand(s.gr, s.out);
    s.ga = ggml_gallocr_new(s.buft);
    if (!ggml_gallocr_alloc_graph(s.ga, s.gr)) { fprintf(stderr, "[ggmlFA] alloc_graph failed\n"); return false; }
    s.P = P; s.len = len; s.nq = nq; s.nkv = nkv; s.hd = hd;
    return true;
}

// Prefill full-attention for one chunk: P queries at positions [pos_start,
// pos_start+P), each causally attending keys [0, pos_start+t].  q_all is the
// prepped/rotated Q [t][h][d] (fp32); fsK/fsV the resident fp16 cache
// [pos][nkv*hd]; gate the per-dim sigmoid gate [t][nq*hd]; out is [t][nq*hd].
// Runs on the aspida stream `st` for ordering with the surrounding chain.
static void aspida_ggml_fattn_prefill(
        const float *q_all, const __half *fsK, const __half *fsV,
        const float *gate, float *out,
        int nq, int nkv, int hd, int P, int pos_start, cudaStream_t st) {
    GgmlFA &s = g_gfa;
    int len = pos_start + P;
    if (s.P != P || s.len != len || s.nq != nq || s.nkv != nkv || s.hd != hd) {
        if (!gfa_rebuild(s, nq, nkv, hd, P, len)) return;
    }
    // prep_chunk wrote fsK/fsV/q_all/gate on `st`; make them visible before the
    // repacks (which run on the default stream) read them.
    cudaStreamSynchronize(st);
    const int bs = 256;
    k_ggml_repack_Q <<<((size_t) P * nq * hd + bs - 1) / bs, bs>>>(q_all, (float *) s.q->data, P, nq, hd);
    k_ggml_repack_KV<<<((size_t) len * nkv * hd + bs - 1) / bs, bs>>>(fsK, (__half *) s.k->data, len, nkv, hd);
    k_ggml_repack_KV<<<((size_t) len * nkv * hd + bs - 1) / bs, bs>>>(fsV, (__half *) s.v->data, len, nkv, hd);
    k_ggml_build_mask<<<((size_t) P * len + bs - 1) / bs, bs>>>((__half *) s.m->data, len, P, pos_start);
    cudaDeviceSynchronize();                 // inputs ready before ggml (own stream)
    ggml_backend_graph_compute(s.be, s.gr);  // <-- fattn-mma (ncols2=8 + cp_async)
    // ggml FA output layout == aspida [t][h*hd+d]; copy out then fold the gate.
    cudaMemcpy(out, s.out->data, (size_t) P * nq * hd * sizeof(float), cudaMemcpyDeviceToDevice);
    k_ggml_apply_gate<<<((size_t) P * nq * hd + bs - 1) / bs, bs, 0, st>>>(out, gate, P * nq * hd);
}
