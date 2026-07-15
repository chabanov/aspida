// Standalone microbench for the delta-net projection ("dproj") launches of one
// chunked-prefill layer at P=256, real hura Qwen3.6-35B-A3B Q8_0 dims.
//
// Reproduces EXACTLY the dproj launch set of aspida_gpu_chain_prefill:
//   k_norm1_b | qkv(wmma) | alpha(wmma) | beta(wmma) | gate(wmma)
//   | k_dnet_conv_chunk | k_dnet_gates_b
// plus the out projection (ow, wmma) which lives in the dnet REMAINDER, so we
// can attribute the full dnet block. Kernels copied VERBATIM from gpu_matvec.cu.
//
// Also benches candidate optimized GEMM kernels (multi-warp tiles) vs k_q8_wmma
// and validates their output (max rel err) against it.
//
// Build (on the an NVIDIA GPU box, sm_89):
//   nvcc -O3 -arch=sm_89 -lineinfo gpu/bench_dnet_proj.cu -o /tmp/bench_dnet
// Run (exclusive):
//   flock /tmp/aspida_bench.lock -c '/tmp/bench_dnet'
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <functional>

#define CK(x) do{ cudaError_t e=(x); if(e){ printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

// ---- real dims (from GGUF /opt/aspida-models/hura:latest, block0 delta-net) ----
static const int DIM   = 2048;
static const int QO    = 8192;   // attn_qkv rows
static const int NV    = 32;     // ssm_alpha/beta rows, num_v_heads
static const int VDIM  = 4096;   // attn_gate rows, v_dim
static const int KERNEL= 4;      // ssm.conv_kernel
static const int P     = 256;    // prefill chunk
static const int NL_DN = 30;     // delta-net layers (of 40)

//==================== VERBATIM kernels from gpu_matvec.cu ====================
__device__ __forceinline__ float f16(const uint8_t *p){ __half h; *reinterpret_cast<uint16_t*>(&h)=(uint16_t)p[0]|((uint16_t)p[1]<<8); return __half2float(h);}

#define WMM_TM 16
#define WMM_TN 16
#define WMM_TK 32
__global__ void k_q8_wmma(const uint8_t *__restrict__ w, const float *__restrict__ x,
                          float *__restrict__ y, int in, int out, int B) {
    int n0 = blockIdx.x * WMM_TN, m0 = blockIdx.y * WMM_TM;
    int lane = threadIdx.x & 31;
    int nb = in / 32; size_t bpr = (size_t) nb * 34;
    __shared__ half As[WMM_TM * WMM_TK];
    __shared__ half Bs[WMM_TK * WMM_TN];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> cf;
    nvcuda::wmma::fill_fragment(cf, 0.0f);
    for (int kb = 0; kb < nb; ++kb) {
        for (int n = 0; n < WMM_TN; ++n) {
            int gn = n0 + n;
            const uint8_t *bl = w + (size_t) gn * bpr + (size_t) kb * 34;
            float d = f16(bl); const int8_t *qs = (const int8_t *) (bl + 2);
            Bs[(size_t) lane * WMM_TN + n] = __float2half(d * (float) qs[lane]);
        }
        for (int m = 0; m < WMM_TM; ++m) {
            int gm = m0 + m;
            float xv = (gm < B) ? x[(size_t) gm * in + kb * 32 + lane] : 0.0f;
            As[(size_t) m * WMM_TK + lane] = __float2half(xv);
        }
        __syncwarp();
        #pragma unroll
        for (int k16 = 0; k16 < 2; ++k16) {
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> af;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> bf;
            nvcuda::wmma::load_matrix_sync(af, As + k16 * 16, WMM_TK);
            nvcuda::wmma::load_matrix_sync(bf, Bs + (size_t) k16 * 16 * WMM_TN, WMM_TN);
            nvcuda::wmma::mma_sync(cf, af, bf, cf);
        }
        __syncwarp();
    }
    __shared__ float Cs[WMM_TM * WMM_TN];
    nvcuda::wmma::store_matrix_sync(Cs, cf, WMM_TN, nvcuda::wmma::mem_row_major);
    for (int idx = lane; idx < WMM_TM * WMM_TN; idx += 32) {
        int m = idx / WMM_TN, n = idx % WMM_TN, gm = m0 + m, gn = n0 + n;
        if (gm < B && gn < out) y[(size_t) gm * out + gn] = Cs[idx];
    }
}

__global__ void k_norm1_b(const float *__restrict__ x, const float *__restrict__ w,
                          float *__restrict__ y, int n, int B) {
    int b = blockIdx.x; if (b >= B) return;
    const float *xb = x + (size_t) b * n; float *yb = y + (size_t) b * n;
    __shared__ float red[256]; __shared__ float rms;
    int tid = threadIdx.x, nt = blockDim.x; float ls = 0.f;
    for (int i = tid; i < n; i += nt) ls += xb[i] * xb[i];
    red[tid] = ls; __syncthreads();
    for (int o = nt / 2; o > 0; o >>= 1) { if (tid < o) red[tid] += red[tid + o]; __syncthreads(); }
    if (tid == 0) rms = sqrtf(red[0] / (float) n + 1e-6f);
    __syncthreads();
    for (int i = tid; i < n; i += nt) yb[i] = (xb[i] / rms) * w[i];
}

__global__ void k_dnet_conv_chunk(const float *__restrict__ qkv, float *__restrict__ hist,
                                  const float *__restrict__ convw, float *__restrict__ cq,
                                  int qo, int kernel, int P) {
    int c = blockIdx.x * blockDim.x + threadIdx.x; if (c >= qo) return;
    for (int t = 0; t < P; ++t) {
        const float *x = qkv + (size_t) t * qo;
        float *o = cq + (size_t) t * qo;
        float acc = x[c] * convw[c * kernel + (kernel - 1)];
        for (int k = 0; k < kernel - 1; ++k)
            acc += hist[(size_t) k * qo + c] * convw[c * kernel + k];
        o[c] = acc / (1.f + expf(-acc));
        for (int k = 0; k + 1 < kernel - 1; ++k)
            hist[(size_t) k * qo + c] = hist[(size_t) (k + 1) * qo + c];
        if (kernel >= 2) hist[(size_t) (kernel - 2) * qo + c] = x[c];
    }
}

__global__ void k_dnet_gates_b(const float *__restrict__ ar, const float *__restrict__ br,
                               const float *__restrict__ a, const float *__restrict__ dt,
                               float *__restrict__ gate, float *__restrict__ beta, int nv, int B) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x, lane = gid / nv, h = gid % nv;
    if (lane >= B) return;
    const float *ar_l = ar + (size_t) lane * nv, *br_l = br + (size_t) lane * nv;
    float *gate_l = gate + (size_t) lane * nv, *beta_l = beta + (size_t) lane * nv;
    float xx = ar_l[h] + dt[h];
    float sp = xx > 20.f ? xx : (xx < -20.f ? expf(xx) : logf(1.f + expf(xx)));
    gate_l[h] = expf(a[h] * sp);
    beta_l[h] = 1.f / (1.f + expf(-br_l[h]));
}

//==================== OPTIMIZED CANDIDATE: multi-warp Q8 GEMM ====================
// Block = WARPS warps. Tile = (WM rows of B) x (WN cols of out). Each warp owns
// a WFM x WFN grid of 16x16 wmma fragments. The K-strip (32 vals) is dequantized
// ONCE per block into shared (cooperative, no per-lane redundant scale decode:
// one lane decodes a whole 16-col block's scale), then all warps consume it.
// This amortizes dequant + boosts tensor-core occupancy vs one-warp-per-16x16.
//
//   WM = 64 (4 row-frags), WN = 64 (4 col-frags), 8 warps -> each warp 2x2 frags.
#define OWARPS 8
#define OWM 64
#define OWN 64
template<int WM,int WN,int WARPS>
__global__ void k_q8_wmma_mw(const uint8_t *__restrict__ w, const float *__restrict__ x,
                             float *__restrict__ y, int in, int out, int B) {
    using namespace nvcuda;
    const int RF = WM/16, CF = WN/16;          // row/col frags in the tile
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int FPW = (RF*CF)/WARPS;             // frags per warp (contiguous)
    int m0 = blockIdx.y * WM, n0 = blockIdx.x * WN;
    int nb = in / 32; size_t bpr = (size_t) nb * 34;
    __shared__ half As[WM * 32];               // x strip [WM][32]
    __shared__ half Bs[32 * WN];               // w strip [32][WN]
    wmma::fragment<wmma::accumulator,16,16,16,float> cf[FPW];
    #pragma unroll
    for (int i=0;i<FPW;i++) wmma::fill_fragment(cf[i], 0.0f);
    const int tid = threadIdx.x, nt = WARPS*32;
    for (int kb = 0; kb < nb; ++kb) {
        // load B (weight) strip: WN cols x 32 k. cols split across threads.
        for (int c = tid; c < WN; c += nt) {
            int gn = n0 + c;
            const uint8_t *bl = w + (size_t) gn * bpr + (size_t) kb * 34;
            float d = f16(bl); const int8_t *qs = (const int8_t *)(bl + 2);
            #pragma unroll
            for (int k = 0; k < 32; ++k) Bs[(size_t)k * WN + c] = __float2half(d * (float) qs[k]);
        }
        // load A (x) strip: WM rows x 32 k. one thread per (row) fills 32 (vectorized by lane over k in wmma load); simplest: threads cover WM*32.
        for (int e = tid; e < WM * 32; e += nt) {
            int m = e >> 5, k = e & 31, gm = m0 + m;
            As[(size_t)m * 32 + k] = (gm < B) ? __float2half(x[(size_t)gm * in + kb*32 + k]) : __float2half(0.0f);
        }
        __syncthreads();
        // each warp does its FPW fragments
        #pragma unroll
        for (int f = 0; f < FPW; ++f) {
            int fi = warp * FPW + f;           // linear frag idx in RFxCF (row-major)
            int fr = fi / CF, fc = fi % CF;
            wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> af;
            wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> bf;
            #pragma unroll
            for (int k16 = 0; k16 < 2; ++k16) {
                wmma::load_matrix_sync(af, As + fr*16*32 + k16*16, 32);
                wmma::load_matrix_sync(bf, Bs + (size_t)k16*16*WN + fc*16, WN);
                wmma::mma_sync(cf[f], af, bf, cf[f]);
            }
        }
        __syncthreads();
    }
    __shared__ float Cs[WM * WN];
    #pragma unroll
    for (int f = 0; f < FPW; ++f) {
        int fi = warp * FPW + f; int fr = fi / CF, fc = fi % CF;
        wmma::store_matrix_sync(Cs + fr*16*WN + fc*16, cf[f], WN, wmma::mem_row_major);
    }
    __syncthreads();
    for (int e = tid; e < WM*WN; e += nt) {
        int m = e / WN, n = e % WN, gm = m0 + m, gn = n0 + n;
        if (gm < B && gn < out) y[(size_t)gm*out + gn] = Cs[(size_t)m*WN + n];
    }
}

// Scatter the fused [P,PROJ] output into separate contiguous qkv/ar/br/z buffers.
__global__ void k_proj_scatter(const float* __restrict__ comb, int P, int proj,
                               float* __restrict__ qkv, int qo,
                               float* __restrict__ ar,  float* __restrict__ br, int nv,
                               float* __restrict__ z,   int vdim) {
    size_t gid = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (gid >= (size_t)P*proj) return;
    int t = gid / proj, c = gid % proj;
    const float* row = comb + (size_t)t*proj; float v = row[c];
    if (c < qo)                 qkv[(size_t)t*qo + c] = v;
    else if (c < qo+nv)         ar [(size_t)t*nv + (c-qo)] = v;
    else if (c < qo+2*nv)       br [(size_t)t*nv + (c-qo-nv)] = v;
    else                        z  [(size_t)t*vdim + (c-qo-2*nv)] = v;
}

//==================== host helpers ====================
static void fill_q8(std::vector<uint8_t>& buf, int in, int out) {
    int nb = in/32; buf.resize((size_t)out*nb*34);
    for (size_t i=0;i<buf.size();++i) buf[i] = (uint8_t)(rand() & 0xff);
    // set scale halfs to a small sane value so dequant ~O(1)
    for (int r=0;r<out;++r) for (int b=0;b<nb;++b){
        uint8_t* bl = buf.data() + ((size_t)r*nb+b)*34;
        // half ~ 0.01: 0x211f approx; just use a fixed small half bit pattern
        __half h = __float2half(0.01f); uint16_t u=*reinterpret_cast<uint16_t*>(&h);
        bl[0]=u&0xff; bl[1]=u>>8;
    }
}

int main(int argc, char** argv){
    srand(1234);
    int freeMiB=0; { size_t fr,tt; cudaMemGetInfo(&fr,&tt); freeMiB=fr>>20; }
    printf("VRAM free ~%d MiB\n", freeMiB);

    // weights
    std::vector<uint8_t> hqkv,hga,hal,hbe,how;
    fill_q8(hqkv, DIM, QO);
    fill_q8(hga,  DIM, VDIM);
    fill_q8(hal,  DIM, NV);
    fill_q8(hbe,  DIM, NV);
    fill_q8(how,  VDIM, DIM);
    uint8_t *dqkv_w,*dga_w,*dal_w,*dbe_w,*dow_w;
    CK(cudaMalloc(&dqkv_w,hqkv.size())); CK(cudaMemcpy(dqkv_w,hqkv.data(),hqkv.size(),cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dga_w,hga.size()));   CK(cudaMemcpy(dga_w,hga.data(),hga.size(),cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dal_w,hal.size()));   CK(cudaMemcpy(dal_w,hal.data(),hal.size(),cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dbe_w,hbe.size()));   CK(cudaMemcpy(dbe_w,hbe.data(),hbe.size(),cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dow_w,how.size()));   CK(cudaMemcpy(dow_w,how.data(),how.size(),cudaMemcpyHostToDevice));

    // activations / buffers
    float *Hb,*nxb,*dqkv,*dcq,*dar,*dbr,*dz,*dg,*db,*dor,*aob;
    float *anorm,*conv,*aw,*dtw;
    CK(cudaMalloc(&Hb,(size_t)P*DIM*4)); CK(cudaMalloc(&nxb,(size_t)P*DIM*4));
    CK(cudaMalloc(&dqkv,(size_t)P*QO*4)); CK(cudaMalloc(&dcq,(size_t)P*QO*4));
    CK(cudaMalloc(&dar,(size_t)P*NV*4)); CK(cudaMalloc(&dbr,(size_t)P*NV*4));
    CK(cudaMalloc(&dz,(size_t)P*VDIM*4)); CK(cudaMalloc(&dg,(size_t)P*NV*4));
    CK(cudaMalloc(&db,(size_t)P*NV*4)); CK(cudaMalloc(&dor,(size_t)P*VDIM*4));
    CK(cudaMalloc(&aob,(size_t)P*DIM*4));
    CK(cudaMalloc(&anorm,(size_t)DIM*4)); CK(cudaMalloc(&conv,(size_t)QO*KERNEL*4));
    CK(cudaMalloc(&aw,(size_t)NV*4)); CK(cudaMalloc(&dtw,(size_t)NV*4));
    float *hist; CK(cudaMalloc(&hist,(size_t)(KERNEL-1)*QO*4)); CK(cudaMemset(hist,0,(size_t)(KERNEL-1)*QO*4));
    // init activations to small randoms
    { std::vector<float> t((size_t)P*DIM); for(auto&v:t)v=(rand()%1000-500)/1000.f;
      CK(cudaMemcpy(Hb,t.data(),t.size()*4,cudaMemcpyHostToDevice)); }
    { std::vector<float> t((size_t)DIM,1.0f); CK(cudaMemcpy(anorm,t.data(),t.size()*4,cudaMemcpyHostToDevice)); }
    { std::vector<float> t((size_t)QO*KERNEL); for(auto&v:t)v=(rand()%1000-500)/1000.f;
      CK(cudaMemcpy(conv,t.data(),t.size()*4,cudaMemcpyHostToDevice)); }
    { std::vector<float> t((size_t)NV,-0.5f); CK(cudaMemcpy(aw,t.data(),t.size()*4,cudaMemcpyHostToDevice));
      CK(cudaMemcpy(dtw,t.data(),t.size()*4,cudaMemcpyHostToDevice)); }

    cudaStream_t st; CK(cudaStreamCreate(&st));
    cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    const int ITERS=50, WARM=5;

    auto wmma_launch=[&](uint8_t*w,float*x,float*y,int in,int out){
        dim3 g((out+WMM_TN-1)/WMM_TN,(P+WMM_TM-1)/WMM_TM);
        k_q8_wmma<<<g,32,0,st>>>(w,x,y,in,out,P);
    };
    auto mw_launch=[&](uint8_t*w,float*x,float*y,int in,int out){
        dim3 g((out+OWN-1)/OWN,(P+OWM-1)/OWM);
        k_q8_wmma_mw<OWM,OWN,OWARPS><<<g,OWARPS*32,0,st>>>(w,x,y,in,out,P);
    };

    // per-launch timing helper
    auto bench=[&](const char* name, std::function<void()> fn)->float{
        for(int i=0;i<WARM;i++) fn();
        CK(cudaStreamSynchronize(st));
        cudaEventRecord(e0,st);
        for(int i=0;i<ITERS;i++) fn();
        cudaEventRecord(e1,st); cudaEventSynchronize(e1);
        float ms=0; cudaEventElapsedTime(&ms,e0,e1); ms/=ITERS;
        printf("  %-22s %8.4f ms/layer   x30 = %7.3f ms\n", name, ms, ms*NL_DN);
        return ms;
    };

    printf("\n=== per-launch (baseline kernels, real dims, P=%d) ===\n",P);
    float t_norm = bench("k_norm1_b",      [&]{ k_norm1_b<<<P,256,0,st>>>(Hb,anorm,nxb,DIM,P); });
    float t_qkv  = bench("qkv wmma",       [&]{ wmma_launch(dqkv_w,nxb,dqkv,DIM,QO); });
    float t_al   = bench("alpha wmma",     [&]{ wmma_launch(dal_w,nxb,dar,DIM,NV); });
    float t_be   = bench("beta wmma",      [&]{ wmma_launch(dbe_w,nxb,dbr,DIM,NV); });
    float t_ga   = bench("gate wmma",      [&]{ wmma_launch(dga_w,nxb,dz,DIM,VDIM); });
    float t_conv = bench("conv_chunk",     [&]{ k_dnet_conv_chunk<<<(QO+255)/256,256,0,st>>>(dqkv,hist,conv,dcq,QO,KERNEL,P); });
    float t_gate = bench("gates_b",        [&]{ k_dnet_gates_b<<<((size_t)P*NV+255)/256,256,0,st>>>(dar,dbr,aw,dtw,dg,db,NV,P); });
    float dproj = t_norm+t_qkv+t_al+t_be+t_ga+t_conv+t_gate;
    printf("  ------------------------------------------------------------\n");
    printf("  dproj/layer = %.4f ms   x30 = %.3f ms   (measured E2E dproj=31.5)\n", dproj, dproj*NL_DN);
    float t_ow = bench("out wmma (remainder)",[&]{ wmma_launch(dow_w,dor,aob,VDIM,DIM); });
    printf("  remainder(ow) x30 = %.3f ms\n", t_ow*NL_DN);

    printf("\n=== LEVER 1: FUSED input proj (qkv|al|be|ga -> one wmma, out=%d) ===\n", QO+2*NV+VDIM);
    // build fused weight = concat of the four row-blocks (Q8_0 row-major -> byte concat)
    int PROJ = QO + NV + NV + VDIM;
    std::vector<uint8_t> hproj; hproj.reserve(hqkv.size()+hal.size()+hbe.size()+hga.size());
    hproj.insert(hproj.end(),hqkv.begin(),hqkv.end());
    hproj.insert(hproj.end(),hal.begin(),hal.end());
    hproj.insert(hproj.end(),hbe.begin(),hbe.end());
    hproj.insert(hproj.end(),hga.begin(),hga.end());
    uint8_t* dproj_w; CK(cudaMalloc(&dproj_w,hproj.size())); CK(cudaMemcpy(dproj_w,hproj.data(),hproj.size(),cudaMemcpyHostToDevice));
    float* dcomb; CK(cudaMalloc(&dcomb,(size_t)P*PROJ*4));
    float in_sum = t_qkv+t_al+t_be+t_ga;
    float t_fused = bench("fused proj wmma",  [&]{ wmma_launch(dproj_w,nxb,dcomb,DIM,PROJ); });
    printf("  4 separate (qkv+al+be+ga) = %.4f ms/layer  x30 = %.3f ms\n", in_sum, in_sum*NL_DN);
    printf("  fused single launch       = %.4f ms/layer  x30 = %.3f ms   (%.2fx)\n", t_fused, t_fused*NL_DN, in_sum/t_fused);

    // Integration cost A: fused launch + scatter into separate buffers.
    float t_scat = bench("proj_scatter",  [&]{ k_proj_scatter<<<((size_t)P*PROJ+255)/256,256,0,st>>>(dcomb,P,PROJ,dqkv,QO,dar,dbr,NV,dz,VDIM); });
    printf("  NET fused+scatter = %.4f ms/layer  x30 = %.3f ms  vs 4-separate %.3f ms  (save %.3f ms)\n",
           t_fused+t_scat, (t_fused+t_scat)*NL_DN, in_sum*NL_DN, (in_sum-(t_fused+t_scat))*NL_DN);

    printf("\n=== LEVER 2: alpha+beta fused (out=64) vs 2x(out=32) ===\n");
    std::vector<uint8_t> halbe; halbe.insert(halbe.end(),hal.begin(),hal.end()); halbe.insert(halbe.end(),hbe.begin(),hbe.end());
    uint8_t* dalbe; CK(cudaMalloc(&dalbe,halbe.size())); CK(cudaMemcpy(dalbe,halbe.data(),halbe.size(),cudaMemcpyHostToDevice));
    float* dalbe_o; CK(cudaMalloc(&dalbe_o,(size_t)P*2*NV*4));
    float t_albe = bench("alpha|beta wmma o=64",[&]{ wmma_launch(dalbe,nxb,dalbe_o,DIM,2*NV); });
    printf("  2 separate = %.4f ms  vs fused64 = %.4f ms\n", t_al+t_be, t_albe);

    printf("\n=== LEVER 3: multi-warp GEMM tile sweep (qkv) ===\n");
    auto mwL=[&](int wm,int wn,int wp,uint8_t*w,float*x,float*y,int in,int out){
        dim3 g((out+wn-1)/wn,(P+wm-1)/wm);
        if(wm==64&&wn==64) k_q8_wmma_mw<64,64,8><<<g,8*32,0,st>>>(w,x,y,in,out,P);
        else if(wm==32&&wn==128) k_q8_wmma_mw<32,128,8><<<g,8*32,0,st>>>(w,x,y,in,out,P);
        else if(wm==32&&wn==64) k_q8_wmma_mw<32,64,4><<<g,4*32,0,st>>>(w,x,y,in,out,P);
        else if(wm==16&&wn==128) k_q8_wmma_mw<16,128,8><<<g,8*32,0,st>>>(w,x,y,in,out,P);
    };
    bench("qkv MW64x64x8", [&]{ mwL(64,64,8,dqkv_w,nxb,dqkv,DIM,QO); });
    bench("qkv MW32x128x8",[&]{ mwL(32,128,8,dqkv_w,nxb,dqkv,DIM,QO); });
    bench("qkv MW32x64x4", [&]{ mwL(32,64,4,dqkv_w,nxb,dqkv,DIM,QO); });
    bench("qkv MW16x128x8",[&]{ mwL(16,128,8,dqkv_w,nxb,dqkv,DIM,QO); });

    // ---- correctness: MW vs baseline for qkv ----
    float *y_ref,*y_opt; CK(cudaMalloc(&y_ref,(size_t)P*QO*4)); CK(cudaMalloc(&y_opt,(size_t)P*QO*4));
    wmma_launch(dqkv_w,nxb,y_ref,DIM,QO);
    mw_launch(dqkv_w,nxb,y_opt,DIM,QO);
    CK(cudaStreamSynchronize(st));
    std::vector<float> a((size_t)P*QO),b((size_t)P*QO);
    CK(cudaMemcpy(a.data(),y_ref,a.size()*4,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(b.data(),y_opt,b.size()*4,cudaMemcpyDeviceToHost));
    double maxrel=0,maxabs=0; for(size_t i=0;i<a.size();++i){ double d=fabs(a[i]-b[i]); maxabs=fmax(maxabs,d);
        double den=fmax(1e-4,fabs(a[i])); maxrel=fmax(maxrel,d/den);}
    printf("\n  MW vs baseline qkv: max_abs=%.3e max_rel=%.3e\n", maxabs, maxrel);

    // ---- correctness: fused+scatter qkv/ar/br/z == 4 separate launches ----
    {
        float *sq,*sa,*sb,*sz, *fq,*fa,*fb,*fz;
        CK(cudaMalloc(&sq,(size_t)P*QO*4)); CK(cudaMalloc(&sa,(size_t)P*NV*4));
        CK(cudaMalloc(&sb,(size_t)P*NV*4)); CK(cudaMalloc(&sz,(size_t)P*VDIM*4));
        CK(cudaMalloc(&fq,(size_t)P*QO*4)); CK(cudaMalloc(&fa,(size_t)P*NV*4));
        CK(cudaMalloc(&fb,(size_t)P*NV*4)); CK(cudaMalloc(&fz,(size_t)P*VDIM*4));
        wmma_launch(dqkv_w,nxb,sq,DIM,QO); wmma_launch(dal_w,nxb,sa,DIM,NV);
        wmma_launch(dbe_w,nxb,sb,DIM,NV);  wmma_launch(dga_w,nxb,sz,DIM,VDIM);
        wmma_launch(dproj_w,nxb,dcomb,DIM,PROJ);
        k_proj_scatter<<<((size_t)P*PROJ+255)/256,256,0,st>>>(dcomb,P,PROJ,fq,QO,fa,fb,NV,fz,VDIM);
        CK(cudaStreamSynchronize(st));
        auto cmp=[&](const char*nm,float*x,float*y,size_t n){
            std::vector<float> A(n),B(n); CK(cudaMemcpy(A.data(),x,n*4,cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(B.data(),y,n*4,cudaMemcpyDeviceToHost));
            double mr=0; for(size_t i=0;i<n;++i){double d=fabs(A[i]-B[i]),den=fmax(1e-4,fabs(A[i])); mr=fmax(mr,d/den);}
            printf("  %-6s max_rel=%.3e\n", nm, mr); };
        printf("\n=== correctness: fused+scatter vs 4 separate ===\n");
        cmp("qkv",sq,fq,(size_t)P*QO); cmp("alpha",sa,fa,(size_t)P*NV);
        cmp("beta",sb,fb,(size_t)P*NV); cmp("z",sz,fz,(size_t)P*VDIM);
    }
    printf("\nDONE\n");
    return 0;
}
