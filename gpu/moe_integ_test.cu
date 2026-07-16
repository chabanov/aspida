// moe_bench.cu — standalone bench of aspida's expert-grouped MoE prefill path
// at REAL hura dims: dim=2048, n_exp=256, top_k=8 (+1 shared), intermed=512,
// Q8_0 weights, P=PCH=1024.  Kernels ported VERBATIM from gpu_matvec.cu
// (k_moe_group / k_moe_gu_grouped / k_moe_down_grouped / k_moe_combine) and
// launched exactly as aspida_gpu_chain_prefill's grouped branch does.
// Compares against an fp32 host reference (probe rows) and reports ms +
// effective weight-read GB/s + %peak.  Prod-safe: ~860MB weights, one layer.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include "fattn_ggml.cuh"
#include "moe_ggml.cuh"

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

#define MOE_MAXK 8
struct MoeRoute { int idx[MOE_MAXK]; float w[MOE_MAXK + 1]; };
__device__ __forceinline__ float f16(const uint8_t *p){ __half h; *reinterpret_cast<uint16_t*>(&h)=(uint16_t)p[0]|((uint16_t)p[1]<<8); return __half2float(h);}

static const int DIM=2048, NEXP=256, TOPK=8, IMD=512, PCHB=1024;
static const float PEAK_GBS = 864.0f;

// ---------------- verbatim aspida kernels ----------------
__global__ void k_moe_group(const MoeRoute *__restrict__ route_b, int P, int top_k,
                            int *__restrict__ mg_pos, int *__restrict__ mg_k,
                            int *__restrict__ mg_cnt, int Pstride) {
    int p = blockIdx.x * blockDim.x + threadIdx.x; if (p >= P) return;
    const MoeRoute *route = route_b + p;
    for (int k = 0; k < top_k; ++k) {
        int e = route->idx[k];
        int s = atomicAdd(&mg_cnt[e], 1);
        if (s < Pstride) { mg_pos[(size_t) e * Pstride + s] = p; mg_k[(size_t) e * Pstride + s] = k; }
    }
}

__global__ void k_moe_gu_grouped(
    const uint8_t *__restrict__ gdw, const uint8_t *__restrict__ udw,
    const float *__restrict__ x_b, float *__restrict__ h_b,
    const int *__restrict__ mg_pos, const int *__restrict__ mg_k,
    const int *__restrict__ mg_cnt,
    long g_bpe, long u_bpe, int dim, int intermed, int Pstride,
    int top_k, int P, int mode) {
    int e = blockIdx.z;
    int cnt = (mode == 0) ? mg_cnt[e] : P;
    int m0 = blockIdx.y * 16; if (m0 >= cnt) return;
    int n0 = blockIdx.x * 16;
    int lane = threadIdx.x & 31;
    int nb = dim / 32; size_t bpr = (size_t) nb * 34;
    const uint8_t *gW = (mode == 0) ? gdw + (size_t) e * g_bpe : gdw;
    const uint8_t *uW = (mode == 0) ? udw + (size_t) e * u_bpe : udw;
    __shared__ half As[16 * 32];
    __shared__ half Gs[32 * 16], Us[32 * 16];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> cg, cu;
    nvcuda::wmma::fill_fragment(cg, 0.0f); nvcuda::wmma::fill_fragment(cu, 0.0f);
    for (int kb = 0; kb < nb; ++kb) {
        for (int m = 0; m < 16; ++m) {
            int gm = m0 + m;
            int p = (gm < cnt) ? ((mode == 0) ? mg_pos[(size_t) e * Pstride + gm] : gm) : -1;
            float xv = (p >= 0) ? x_b[(size_t) p * dim + kb * 32 + lane] : 0.0f;
            As[(size_t) m * 32 + lane] = __float2half(xv);
        }
        for (int n = 0; n < 16; ++n) {
            int gn = n0 + n;
            const uint8_t *blG = gW + (size_t) gn * bpr + (size_t) kb * 34;
            float dG = f16(blG); const int8_t *qG = (const int8_t *) (blG + 2);
            Gs[(size_t) lane * 16 + n] = __float2half(dG * (float) qG[lane]);
            const uint8_t *blU = uW + (size_t) gn * bpr + (size_t) kb * 34;
            float dU = f16(blU); const int8_t *qU = (const int8_t *) (blU + 2);
            Us[(size_t) lane * 16 + n] = __float2half(dU * (float) qU[lane]);
        }
        __syncwarp();
        #pragma unroll
        for (int k16 = 0; k16 < 2; ++k16) {
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> af;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> bg, bu;
            nvcuda::wmma::load_matrix_sync(af, As + k16 * 16, 32);
            nvcuda::wmma::load_matrix_sync(bg, Gs + (size_t) k16 * 16 * 16, 16);
            nvcuda::wmma::load_matrix_sync(bu, Us + (size_t) k16 * 16 * 16, 16);
            nvcuda::wmma::mma_sync(cg, af, bg, cg);
            nvcuda::wmma::mma_sync(cu, af, bu, cu);
        }
        __syncwarp();
    }
    __shared__ float Cg[16 * 16], Cu[16 * 16];
    nvcuda::wmma::store_matrix_sync(Cg, cg, 16, nvcuda::wmma::mem_row_major);
    nvcuda::wmma::store_matrix_sync(Cu, cu, 16, nvcuda::wmma::mem_row_major);
    for (int idx = lane; idx < 256; idx += 32) {
        int m = idx / 16, n = idx % 16, gm = m0 + m;
        if (gm < cnt && (n0 + n) < intermed) {
            int p = (mode == 0) ? mg_pos[(size_t) e * Pstride + gm] : gm;
            int k = (mode == 0) ? mg_k[(size_t) e * Pstride + gm] : top_k;
            float g = Cg[idx], u = Cu[idx];
            h_b[((size_t) p * (MOE_MAXK + 1) + k) * intermed + n0 + n] = (g / (1.f + expf(-g))) * u;
        }
    }
}

__global__ void k_moe_down_grouped(
    const uint8_t *__restrict__ ddw, const float *__restrict__ h_b, float *__restrict__ d_buf,
    const int *__restrict__ mg_pos, const int *__restrict__ mg_k, const int *__restrict__ mg_cnt,
    long d_bpe, int intermed, int dim, int Pstride, int top_k, int P, int mode) {
    int e = blockIdx.z;
    int cnt = (mode == 0) ? mg_cnt[e] : P;
    int m0 = blockIdx.y * 16; if (m0 >= cnt) return;
    int n0 = blockIdx.x * 16;
    int lane = threadIdx.x & 31;
    int nb = intermed / 32; size_t bpr = (size_t) nb * 34;
    const uint8_t *W = (mode == 0) ? ddw + (size_t) e * d_bpe : ddw;
    __shared__ half As[16 * 32];
    __shared__ half Bs[32 * 16];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> cd;
    nvcuda::wmma::fill_fragment(cd, 0.0f);
    for (int kb = 0; kb < nb; ++kb) {
        for (int m = 0; m < 16; ++m) {
            int gm = m0 + m; float hv = 0.0f;
            if (gm < cnt) {
                int p = (mode == 0) ? mg_pos[(size_t) e * Pstride + gm] : gm;
                int k = (mode == 0) ? mg_k[(size_t) e * Pstride + gm] : top_k;
                hv = h_b[((size_t) p * (MOE_MAXK + 1) + k) * intermed + kb * 32 + lane];
            }
            As[(size_t) m * 32 + lane] = __float2half(hv);
        }
        for (int n = 0; n < 16; ++n) {
            const uint8_t *bl = W + (size_t) (n0 + n) * bpr + (size_t) kb * 34;
            float d = f16(bl); const int8_t *q = (const int8_t *) (bl + 2);
            Bs[(size_t) lane * 16 + n] = __float2half(d * (float) q[lane]);
        }
        __syncwarp();
        #pragma unroll
        for (int k16 = 0; k16 < 2; ++k16) {
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> af;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> bf;
            nvcuda::wmma::load_matrix_sync(af, As + k16 * 16, 32);
            nvcuda::wmma::load_matrix_sync(bf, Bs + (size_t) k16 * 16 * 16, 16);
            nvcuda::wmma::mma_sync(cd, af, bf, cd);
        }
        __syncwarp();
    }
    __shared__ float Cs[16 * 16];
    nvcuda::wmma::store_matrix_sync(Cs, cd, 16, nvcuda::wmma::mem_row_major);
    for (int idx = lane; idx < 256; idx += 32) {
        int m = idx / 16, n = idx % 16, gm = m0 + m;
        if (gm < cnt && (n0 + n) < dim) {
            int p = (mode == 0) ? mg_pos[(size_t) e * Pstride + gm] : gm;
            int k = (mode == 0) ? mg_k[(size_t) e * Pstride + gm] : top_k;
            d_buf[((size_t) p * (MOE_MAXK + 1) + k) * dim + n0 + n] = Cs[idx];
        }
    }
}

__global__ void k_moe_combine(const float *__restrict__ d_buf, const MoeRoute *__restrict__ route_b,
                              float *__restrict__ y_b, int top_k, int dim, int P) {
    int p = blockIdx.y; if (p >= P) return;
    int d = blockIdx.x * blockDim.x + threadIdx.x; if (d >= dim) return;
    const MoeRoute *r = route_b + p;
    const float *D = d_buf + (size_t) p * (MOE_MAXK + 1) * dim;
    float acc = 0.f;
    for (int k = 0; k < top_k; ++k) acc += r->w[k] * D[(size_t) k * dim + d];
    acc += r->w[MOE_MAXK] * D[(size_t) top_k * dim + d];
    y_b[(size_t) p * dim + d] = acc;
}

// ---- host helpers ----
static std::mt19937 rng(42);
static float frand(){ return std::uniform_real_distribution<float>(-1.f,1.f)(rng); }

// fill a Q8_0 weight [rows][cols] (row-major, cols%32==0): per 32-block 2B fp16 scale + 32 int8
static void gen_q8(std::vector<uint8_t> &w, int rows, int cols){
    int nb = cols/32; size_t bpr = (size_t)nb*34;
    w.resize((size_t)rows*bpr);
    std::uniform_int_distribution<int> qd(-127,127);
    for (int r=0;r<rows;++r) for (int b=0;b<nb;++b){
        uint8_t *bl = w.data() + (size_t)r*bpr + (size_t)b*34;
        float scale = 0.005f + 0.01f*std::uniform_real_distribution<float>(0.f,1.f)(rng);
        __half hs = __float2half(scale);
        memcpy(bl, &hs, 2);
        int8_t *q = (int8_t*)(bl+2);
        for (int i=0;i<32;++i) q[i] = (int8_t)qd(rng);
    }
}
// dequant one row on host
static void deq_row(const uint8_t *w, int cols, int row, float *out){
    int nb=cols/32; size_t bpr=(size_t)nb*34;
    for (int b=0;b<nb;++b){ const uint8_t*bl=w+(size_t)row*bpr+(size_t)b*34;
        __half hs; memcpy(&hs,bl,2); float d=__half2float(hs);
        const int8_t*q=(const int8_t*)(bl+2);
        for(int i=0;i<32;++i) out[b*32+i]=d*(float)q[i]; }
}


//  ids repack + ggml-path combine (verbatim from gpu_matvec.cu Phase B)
__global__ void k_moe_ids(const MoeRoute *__restrict__ route_b, int32_t *__restrict__ ids,
                          int top_k, int P) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= P * top_k) return;
    int p = i / top_k, k = i - p * top_k;
    ids[i] = route_b[p].idx[k];
}
__global__ void k_moe_combine_ggml(const float *__restrict__ g_out,
                                   const float *__restrict__ d_buf,
                                   const MoeRoute *__restrict__ route_b,
                                   float *__restrict__ y_b, int top_k, int dim, int P) {
    int p = blockIdx.y; if (p >= P) return;
    int d = blockIdx.x * blockDim.x + threadIdx.x; if (d >= dim) return;
    const MoeRoute *r = route_b + p;
    const float *G = g_out + (size_t) p * top_k * dim;
    float acc = 0.f;
    for (int k = 0; k < top_k; ++k) acc += r->w[k] * G[(size_t) k * dim + d];
    acc += r->w[MOE_MAXK] * d_buf[((size_t) p * (MOE_MAXK + 1) + top_k) * dim + d];
    y_b[(size_t) p * dim + d] = acc;
}

int main(int argc,char**argv){
    int P = argc>1?atoi(argv[1]):1024;
    int iters = argc>2?atoi(argv[2]):20;
    printf("=== INTEGRATED ggml-MoE path (moe_ggml.cuh)  dim=%d n_exp=%d top_k=%d(+1sh) intermed=%d Q8_0 P=%d ===\n",
           DIM,NEXP,TOPK,IMD,P);
    long g_bpe = (long)IMD * (DIM/32) * 34;
    long d_bpe = (long)DIM * (IMD/32) * 34;
    std::vector<uint8_t> hg, hu, hd, hsg, hsu, hsd;
    gen_q8(hg, NEXP*IMD, DIM); gen_q8(hu, NEXP*IMD, DIM);
    gen_q8(hd, NEXP*DIM, IMD);
    gen_q8(hsg, IMD, DIM); gen_q8(hsu, IMD, DIM); gen_q8(hsd, DIM, IMD);
    // ROUTED weights -> ggml tensors (the new loader path)
    ggml_tensor *gt = aspida_ggml_upload_q8(hg.data(), (long)hg.size(), DIM, IMD, NEXP);
    ggml_tensor *ut = aspida_ggml_upload_q8(hu.data(), (long)hu.size(), DIM, IMD, NEXP);
    ggml_tensor *dt = aspida_ggml_upload_q8(hd.data(), (long)hd.size(), IMD, DIM, NEXP);
    if (!gt||!ut||!dt) { printf("ggml upload FAILED\n"); return 1; }
    printf("ggml weight tensors allocated (routed %.1f MB); tensor->data=%p\n",
           (double)(hg.size()+hu.size()+hd.size())/1048576.0, gt->data);
    // shared expert stays aspida-owned
    uint8_t *dsg,*dsu,*dsd;
    CK(cudaMalloc(&dsg,hsg.size())); CK(cudaMalloc(&dsu,hsu.size())); CK(cudaMalloc(&dsd,hsd.size()));
    CK(cudaMemcpy(dsg,hsg.data(),hsg.size(),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dsu,hsu.data(),hsu.size(),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dsd,hsd.data(),hsd.size(),cudaMemcpyHostToDevice));
    // activations + routing
    std::vector<float> hx((size_t)P*DIM); for (auto &v:hx) v=frand()*0.1f;
    std::vector<MoeRoute> hroute(P);
    { std::vector<int> perm(NEXP); for(int i=0;i<NEXP;++i) perm[i]=i;
      for (int p=0;p<P;++p){ std::shuffle(perm.begin(),perm.end(),rng);
        float ws=0; for(int k=0;k<TOPK;++k){ hroute[p].idx[k]=perm[k];
            hroute[p].w[k]=0.1f+std::uniform_real_distribution<float>(0.f,1.f)(rng); ws+=hroute[p].w[k]; }
        for(int k=0;k<TOPK;++k) hroute[p].w[k]/=ws;
        hroute[p].w[MOE_MAXK]=1.0f; } }
    float *dx,*dhb,*dmoe,*dy; MoeRoute *droute; int32_t *dids;
    int *mg_pos,*mg_k,*mg_cnt;   // shared-mode kernels take (ignore) them
    CK(cudaMalloc(&dx,(size_t)P*DIM*4));
    CK(cudaMalloc(&dhb,(size_t)P*(MOE_MAXK+1)*IMD*4));
    CK(cudaMalloc(&dmoe,(size_t)P*(MOE_MAXK+1)*DIM*4));
    CK(cudaMalloc(&dy,(size_t)P*DIM*4));
    CK(cudaMalloc(&droute,(size_t)P*sizeof(MoeRoute)));
    CK(cudaMalloc(&dids,(size_t)P*TOPK*4));
    CK(cudaMalloc(&mg_pos,4)); CK(cudaMalloc(&mg_k,4)); CK(cudaMalloc(&mg_cnt,4));
    CK(cudaMemcpy(dx,hx.data(),(size_t)P*DIM*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(droute,hroute.data(),(size_t)P*sizeof(MoeRoute),cudaMemcpyHostToDevice));
    cudaStream_t st; CK(cudaStreamCreate(&st));
    // ---- the NEW dispatch flow (mirrors gpu_matvec.cu) ----
    auto launch_new=[&]()->const float*{
        k_moe_ids<<<((size_t)P*TOPK+255)/256,256,0,st>>>(droute, dids, TOPK, P);
        dim3 grSh((IMD+15)/16,(P+15)/16,1);
        k_moe_gu_grouped<<<grSh,32,0,st>>>(dsg,dsu,dx,dhb,mg_pos,mg_k,mg_cnt,0,0,DIM,IMD,PCHB,TOPK,P,1);
        dim3 grDs((DIM+15)/16,(P+15)/16,1);
        k_moe_down_grouped<<<grDs,32,0,st>>>(dsd,dhb,dmoe,mg_pos,mg_k,mg_cnt,0,IMD,DIM,PCHB,TOPK,P,1);
        const float *gout = aspida_ggml_moe_prefill(gt,ut,dt, dx, dids, DIM, IMD, TOPK, P, st);
        if(!gout) return nullptr;
        dim3 grC((DIM+255)/256,P,1);
        k_moe_combine_ggml<<<grC,256,0,st>>>(gout,dmoe,droute,dy,TOPK,DIM,P);
        return gout;
    };
    if(!launch_new()){ printf("ggml prefill FAILED\n"); return 1; }
    CK(cudaDeviceSynchronize());
    // ---- correctness vs fp32 host reference (same as moe_bench) ----
    std::vector<float> hy((size_t)P*DIM);
    CK(cudaMemcpy(hy.data(),dy,(size_t)P*DIM*4,cudaMemcpyDeviceToHost));
    std::vector<float> row(DIM), rowd(IMD);
    double mxabs=0, mxref=0, sse=0, ssref=0;
    for (int pi : {0, P/2, P-1}) {
        std::vector<float> y(DIM,0.f);
        for (int slot=0; slot<=TOPK; ++slot) {
            const uint8_t *G,*U,*D; float wgt;
            if (slot<TOPK){ int e=hroute[pi].idx[slot];
                G=hg.data()+(size_t)e*g_bpe; U=hu.data()+(size_t)e*g_bpe; D=hd.data()+(size_t)e*d_bpe; wgt=hroute[pi].w[slot]; }
            else { G=hsg.data(); U=hsu.data(); D=hsd.data(); wgt=hroute[pi].w[MOE_MAXK]; }
            std::vector<float> h(IMD);
            for (int r=0;r<IMD;++r){ deq_row(G,DIM,r,row.data());
                float gdot=0; for(int c=0;c<DIM;++c) gdot+=row[c]*hx[(size_t)pi*DIM+c];
                deq_row(U,DIM,r,row.data());
                float udot=0; for(int c=0;c<DIM;++c) udot+=row[c]*hx[(size_t)pi*DIM+c];
                h[r]=(gdot/(1.f+expf(-gdot)))*udot; }
            for (int r=0;r<DIM;++r){ deq_row(D,IMD,r,rowd.data());
                float dd2=0; for(int c=0;c<IMD;++c) dd2+=rowd[c]*h[c];
                y[r]+=wgt*dd2; }
        }
        for (int r=0;r<DIM;++r){ double e=fabs(hy[(size_t)pi*DIM+r]-y[r]);
            if(e>mxabs)mxabs=e; if(fabs(y[r])>mxref)mxref=fabs(y[r]);
            sse+=e*e; ssref+=y[r]*y[r]; }
    }
    printf("correctness (3 probe pos): max_abs=%.3e max|ref|=%.3e scaled=%.3e NMSE=%.3e\n",
           mxabs,mxref,mxabs/(mxref+1e-30),sqrt(sse/(ssref+1e-30)));
    // ---- timing ----
    cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    launch_new(); CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(e0));
    for(int i=0;i<iters;++i) launch_new();
    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
    float ms; cudaEventElapsedTime(&ms,e0,e1); ms/=iters;
    double wbytes = (double)NEXP*(2.0*g_bpe+d_bpe) + (2.0*g_bpe+d_bpe);
    double gbs = wbytes/(ms*1e-3)/1e9;
    printf("NEW ggml-MoE layer-chunk: %.3f ms  weight-read %.1f GB/s  %.1f%% peak  (old grouped: 5.71 ms -> %.2fx)\n",
           ms, gbs, gbs/PEAK_GBS*100.0, 5.71/ms);
    return 0;
}
