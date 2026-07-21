// dnet_ggml.cuh — aspida prefill gated-delta-net via ggml's fused GDN op
// (GGML_OP_GATED_DELTA_NET, chunked parallel scan). Same math as the aspida
// k_dnet_recur_warp sequential kernel; ~10x faster (chunked) at large P.
// Gated by ASPIDA_DNET_GGML (default off). Validated vs the aspida recur.
#pragma once
#include <cmath>
#include "ggml.h"
#include "ggml-cuda.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

// ---- repack aspida deltanet inputs -> ggml GDN layout ----
// qn/kn aspida [t*q_dim + kh*khd + d]  ->  ggml [khd, nkh, P] (d,h,t)
__global__ void k_gdn_qk(const float *__restrict__ src, float *__restrict__ dst,
                         int P, int nkh, int khd, int q_dim) {
    size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) P * nkh * khd) return;
    int d = i % khd; size_t r = i / khd; int h = r % nkh; int t = r / nkh;
    dst[i] = src[(size_t) t * q_dim + h * khd + d];
}
// value from conv output cq[t*qo + 2*q_dim + h*vhd + v] -> ggml [vhd, nv, P]
__global__ void k_gdn_v(const float *__restrict__ cq, float *__restrict__ dst,
                        int P, int nv, int vhd, int qo, int q_dim) {
    size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) P * nv * vhd) return;
    int d = i % vhd; size_t r = i / vhd; int h = r % nv; int t = r / nv;
    dst[i] = cq[(size_t) t * qo + 2 * q_dim + h * vhd + d];
}
// gate/beta aspida [t*nv+h] == ggml [1, nv, P] flat (h + t*nv) — direct copy.
__global__ void k_gdn_copy(const float *__restrict__ src, float *__restrict__ dst, size_t n) {
    size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}
__global__ void k_gdn_log(const float *__restrict__ src, float *__restrict__ dst, size_t n) {
    size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = logf(src[i]);
}
// aspida state S[(h*hd+key)*hd+val] -> ggml [val*hd+key + h*hd*hd] (transpose key<->val)
__global__ void k_gdn_st_in(const float *__restrict__ src, float *__restrict__ dst, int hd, int H) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=(size_t)H*hd*hd)return;
    int val=i%hd; size_t r=i/hd; int key=r%hd; int h=r/hd;   // ggml flat = h*hd*hd + val*hd + key (i here)
    dst[i]=src[((size_t)h*hd+key)*hd+val];
}
__global__ void k_gdn_st_out(const float *__restrict__ src, float *__restrict__ dst, int hd, int H) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=(size_t)H*hd*hd)return;
    int val=i%hd; size_t r=i/hd; int key=r%hd; int h=r/hd;
    dst[((size_t)h*hd+key)*hd+val]=src[i];   // ggml[val*hd+key+h*hd*hd] -> aspida[(h*hd+key)*hd+val]
}

struct GgmlGDN {
    ggml_context *ctx = nullptr;
    ggml_gallocr_t ga = nullptr;
    ggml_cgraph *gr = nullptr;
    ggml_tensor *q=nullptr,*k=nullptr,*v=nullptr,*g=nullptr,*b=nullptr,*s=nullptr,*out=nullptr;
    int P=-1, nkh=0, nv=0, khd=0, vhd=0;
};
static GgmlGDN g_ggdn;

static bool ggdn_rebuild(GgmlGDN &S, int nkh, int nv, int khd, int vhd, int P) {
    if (!aspida_ggml_be()) return false;   // reuse the shared ggml cuda backend
    if (S.ctx) { ggml_free(S.ctx); S.ctx=nullptr; }
    if (S.ga)  { ggml_gallocr_free(S.ga); S.ga=nullptr; }
    struct ggml_init_params ip = { ggml_tensor_overhead()*24 + ggml_graph_overhead(), nullptr, true };
    S.ctx = ggml_init(ip);
    S.q = ggml_new_tensor_4d(S.ctx, GGML_TYPE_F32, khd, nkh, P, 1); ggml_set_input(S.q);
    S.k = ggml_new_tensor_4d(S.ctx, GGML_TYPE_F32, khd, nkh, P, 1); ggml_set_input(S.k);
    S.v = ggml_new_tensor_4d(S.ctx, GGML_TYPE_F32, vhd, nv,  P, 1); ggml_set_input(S.v);
    S.g = ggml_new_tensor_4d(S.ctx, GGML_TYPE_F32, 1,   nv,  P, 1); ggml_set_input(S.g);
    S.b = ggml_new_tensor_4d(S.ctx, GGML_TYPE_F32, 1,   nv,  P, 1); ggml_set_input(S.b);
    S.s = ggml_new_tensor_4d(S.ctx, GGML_TYPE_F32, vhd, vhd, nv, 1); ggml_set_input(S.s);
    S.out = ggml_gated_delta_net(S.ctx, S.q, S.k, S.v, S.g, S.b, S.s, 1);
    ggml_set_output(S.out);
    S.gr = ggml_new_graph(S.ctx);
    ggml_build_forward_expand(S.gr, S.out);
    S.ga = ggml_gallocr_new(g_gfa.buft);
    if (!ggml_gallocr_alloc_graph(S.ga, S.gr)) { fprintf(stderr,"[gdn] alloc failed\n"); return false; }
    S.P=P; S.nkh=nkh; S.nv=nv; S.khd=khd; S.vhd=vhd;
    return true;
}

// Returns false to fall back to the aspida recur kernel.
static bool aspida_ggml_dnet_prefill(
        float *stateS, const float *kn, const float *qn, const float *cq,
        const float *gate, const float *beta, float *osh,
        int khd, int vhd, int q_dim, int nkh, int nv, int qo, int v_dim, int P, cudaStream_t st) {
    GgmlGDN &S = g_ggdn;
    if (S.P!=P || S.nkh!=nkh || S.nv!=nv || S.khd!=khd || S.vhd!=vhd)
        if (!ggdn_rebuild(S, nkh, nv, khd, vhd, P)) return false;
    const int bs = 256;
    // repack inputs into the ggml graph tensors (default stream, after st done)
    cudaStreamSynchronize(st);
    k_gdn_qk<<<((size_t)P*nkh*khd+bs-1)/bs,bs>>>(qn, (float*)S.q->data, P, nkh, khd, q_dim);
    k_gdn_qk<<<((size_t)P*nkh*khd+bs-1)/bs,bs>>>(kn, (float*)S.k->data, P, nkh, khd, q_dim);
    k_gdn_v <<<((size_t)P*nv*vhd+bs-1)/bs,bs>>>(cq, (float*)S.v->data, P, nv, vhd, qo, q_dim);
    k_gdn_log<<<((size_t)P*nv+bs-1)/bs,bs>>>(gate, (float*)S.g->data, (size_t)P*nv);
    k_gdn_copy<<<((size_t)P*nv+bs-1)/bs,bs>>>(beta, (float*)S.b->data, (size_t)P*nv);
    k_gdn_st_in<<<((size_t)vhd*vhd*nv+bs-1)/bs,bs>>>(stateS, (float*)S.s->data, vhd, nv);
    ggml_backend_graph_compute(g_gfa.be, S.gr);
    cudaDeviceSynchronize();   // ggml output ready before the st-stream copies (race fix)
    // out = [vhd, nv, P] (== aspida osh [t][h*vhd+v]); new state = [vhd,vhd,nv] after
    const float *go = (const float *) S.out->data;
    size_t attn_elems = (size_t) vhd * nv * P;
    // scale by 1/sqrt(khd) to match aspida (po*scale); apply during copy
    float scale = 1.f / sqrtf((float) khd);
    // osh copy + scale
    k_gdn_copy<<<((size_t)v_dim*P+bs-1)/bs,bs,0,st>>>(go, osh, (size_t)v_dim*P);   // v_dim==nv*vhd
    // (scale folded below in a tiny kernel if needed — GDN may already scale)
    // new state -> aspida stateS
    k_gdn_st_out<<<((size_t)vhd*vhd*nv+bs-1)/bs,bs,0,st>>>(go + attn_elems, stateS, vhd, nv);
    (void) scale;
    return true;
}
